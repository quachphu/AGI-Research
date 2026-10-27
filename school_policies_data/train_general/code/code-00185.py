        Higher "temperatures" mean that larger jumps in function value will be
        accepted.  For best results `T` should be comparable to the
        separation (in function value) between local minima.
    stepsize : float, optional
        Maximum step size for use in the random displacement.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        `scipy.optimize.minimize` Some important options could be:

        method : str
            The minimization method (e.g. ``"L-BFGS-B"``)
        args : tuple
            Extra arguments passed to the objective function (`func`) and
            its derivatives (Jacobian, Hessian).

    take_step : callable ``take_step(x)``, optional
        Replace the default step-taking routine with this routine. The default
        step-taking routine is a random displacement of the coordinates, but
        other step-taking algorithms may be better for some systems.
        `take_step` can optionally have the attribute ``take_step.stepsize``.
        If this attribute exists, then `basinhopping` will adjust
        ``take_step.stepsize`` in order to try to optimize the global minimum
        search.
    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
        Define a test which will be used to judge whether to accept the
        step. This will be used in addition to the Metropolis test based on
        "temperature" `T`. The acceptable return values are True,
        False, or ``"force accept"``. If any of the tests return False
        then the step is rejected. If the latter, then this will override any
        other tests in order to accept the step. This can be used, for example,
        to forcefully escape from a local minimum that `basinhopping` is
        trapped in.
    callback : callable, ``callback(x, f, accept)``, optional
        A callback function which will be called for all minima found. ``x``
        and ``f`` are the coordinates and function value of the trial minimum,
        and ``accept`` is whether that minimum was accepted. This can
        be used, for example, to save the lowest N minima found. Also,
        `callback` can be used to specify a user defined stop criterion by
        optionally returning True to stop the `basinhopping` routine.
    interval : integer, optional
        interval for how often to update the `stepsize`
    disp : bool, optional
        Set to True to print status messages
    niter_success : integer, optional
        Stop the run if the global minimum candidate remains the same for this
        number of iterations.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        The random numbers generated only affect the default Metropolis
        `accept_test` and the default `take_step`. If you supply your own
        `take_step` and `accept_test`, and these functions use random
        number generation, then those functions are responsible for the state
        of their random number generator.
    target_accept_rate : float, optional
        The target acceptance rate that is used to adjust the `stepsize`.
        If the current acceptance rate is greater than the target,
        then the `stepsize` is increased. Otherwise, it is decreased.
        Range is (0, 1). Default is 0.5.

        .. versionadded:: 1.8.0

    stepwise_factor : float, optional
        The `stepsize` is multiplied or divided by this stepwise factor upon
        each update. Range is (0, 1). Default is 0.9.

        .. versionadded:: 1.8.0

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination. The ``OptimizeResult`` object returned by the
        selected minimizer at the lowest minimum is also contained within this
        object and can be accessed through the ``lowest_optimization_result``
        attribute.  See `OptimizeResult` for a description of other attributes.

    See Also
    --------
    minimize :
        The local minimization function called once for each basinhopping step.
        `minimizer_kwargs` is passed to this routine.

    Notes
    -----
    Basin-hopping is a stochastic algorithm which attempts to find the global
    minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
    [4]_. The algorithm in its current form was described by David Wales and
    Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.

    The algorithm is iterative with each cycle composed of the following
    features

    1) random perturbation of the coordinates

    2) local minimization

    3) accept or reject the new coordinates based on the minimized function
       value

    The acceptance test used here is the Metropolis criterion of standard Monte
    Carlo algorithms, although there are many other possibilities [3]_.

    This global minimization method has been shown to be extremely efficient
    for a wide variety of problems in physics and chemistry. It is
    particularly useful when the function has many minima separated by large
    barriers. See the `Cambridge Cluster Database
    <https://www-wales.ch.cam.ac.uk/CCD.html>`_ for databases of molecular
    systems that have been optimized primarily using basin-hopping. This
    database includes minimization problems exceeding 300 degrees of freedom.

    See the free software program `GMIN <https://www-wales.ch.cam.ac.uk/GMIN>`_
    for a Fortran implementation of basin-hopping. This implementation has many
    variations of the procedure described above, including more
    advanced step taking algorithms and alternate acceptance criterion.

    For stochastic global optimization there is no way to determine if the true
    global minimum has actually been found. Instead, as a consistency check,
    the algorithm can be run from a number of different random starting points
    to ensure the lowest minimum found in each example has converged to the
    global minimum. For this reason, `basinhopping` will by default simply
    run for the number of iterations `niter` and return the lowest minimum
    found. It is left to the user to ensure that this is in fact the global
    minimum.

    Choosing `stepsize`:  This is a crucial parameter in `basinhopping` and
    depends on the problem being solved. The step is chosen uniformly in the
    region from x0-stepsize to x0+stepsize, in each dimension. Ideally, it
    should be comparable to the typical separation (in argument values) between
    local minima of the function being optimized. `basinhopping` will, by
    default, adjust `stepsize` to find an optimal value, but this may take
    many iterations. You will get quicker results if you set a sensible
    initial value for ``stepsize``.

    Choosing `T`: The parameter `T` is the "temperature" used in the
    Metropolis criterion. Basinhopping steps are always accepted if
    ``func(xnew) < func(xold)``. Otherwise, they are accepted with
    probability::

        exp( -(func(xnew) - func(xold)) / T )

    So, for best results, `T` should to be comparable to the typical
    difference (in function values) between local minima. (The height of
    "walls" between local minima is irrelevant.)

    If `T` is 0, the algorithm becomes Monotonic Basin-Hopping, in which all
    steps that increase energy are rejected.

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
        Cambridge, UK.
    .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
        the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
        110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.
    .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the
        multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,
        1987, 84, 6611.
    .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,
        crystals, and biomolecules, Science, 1999, 285, 1368.
    .. [5] Olson, B., Hashmi, I., Molloy, K., and Shehu1, A., Basin Hopping as
        a General and Versatile Optimization Framework for the Characterization
        of Biological Macromolecules, Advances in Artificial Intelligence,
        Volume 2012 (2012), Article ID 674832, :doi:`10.1155/2012/674832`

    Examples
    --------
    The following example is a 1-D minimization problem, with many
    local minima superimposed on a parabola.

    >>> import numpy as np
    >>> from scipy.optimize import basinhopping
    >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    >>> x0 = [1.]

    Basinhopping, internally, uses a local minimization algorithm. We will use
    the parameter `minimizer_kwargs` to tell basinhopping which algorithm to
    use and how to set up that minimizer. This parameter will be passed to
    `scipy.optimize.minimize`.

    >>> minimizer_kwargs = {"method": "BFGS"}
    >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200)
    >>> # the global minimum is:
    >>> ret.x, ret.fun
    -0.1951, -1.0009

    Next consider a 2-D minimization problem. Also, this time, we
    will use gradient information to significantly speed up the search.

    >>> def func2d(x):
    ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
    ...                                                            0.2) * x[0]
    ...     df = np.zeros(2)
    ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    ...     df[1] = 2. * x[1] + 0.2
    ...     return f, df

    We'll also use a different local minimization algorithm. Also, we must tell
    the minimizer that our function returns both energy and gradient (Jacobian).

    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    >>> x0 = [1.0, 1.0]
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200)
    >>> print("global minimum: x = [%.4f, %.4f], f(x) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun))
    global minimum: x = [-0.1951, -0.1000], f(x) = -1.0109

    Here is an example using a custom step-taking routine. Imagine you want
    the first coordinate to take larger steps than the rest of the coordinates.
    This can be implemented like so:

    >>> class MyTakeStep:
    ...    def __init__(self, stepsize=0.5):
    ...        self.stepsize = stepsize
    ...        self.rng = np.random.default_rng()
    ...    def __call__(self, x):
    ...        s = self.stepsize
    ...        x[0] += self.rng.uniform(-2.*s, 2.*s)
    ...        x[1:] += self.rng.uniform(-s, s, x[1:].shape)
    ...        return x

    Since ``MyTakeStep.stepsize`` exists basinhopping will adjust the magnitude
    of `stepsize` to optimize the search. We'll use the same 2-D function as
    before

    >>> mytakestep = MyTakeStep()
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200, take_step=mytakestep)
    >>> print("global minimum: x = [%.4f, %.4f], f(x) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun))
    global minimum: x = [-0.1951, -0.1000], f(x) = -1.0109

    Now, let's do an example using a custom callback function which prints the
    value of every minimum found

    >>> def print_fun(x, f, accepted):
    ...         print("at minimum %.4f accepted %d" % (f, int(accepted)))

    We'll run it for only 10 basinhopping steps this time.

    >>> rng = np.random.default_rng()
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=10, callback=print_fun, rng=rng)
    at minimum 0.4159 accepted 1
    at minimum -0.4317 accepted 1
    at minimum -1.0109 accepted 1
    at minimum -0.9073 accepted 1
    at minimum -0.4317 accepted 0
    at minimum -0.1021 accepted 1
    at minimum -0.7425 accepted 1
    at minimum -0.9073 accepted 1
    at minimum -0.4317 accepted 0
    at minimum -0.7425 accepted 1
    at minimum -0.9073 accepted 1

    The minimum at -1.0109 is actually the global minimum, found already on the
    8th iteration.

    """ # numpy/numpydoc#87  # noqa: E501
    if target_accept_rate <= 0. or target_accept_rate >= 1.:
        raise ValueError('target_accept_rate has to be in range (0, 1)')
    if stepwise_factor <= 0. or stepwise_factor >= 1.:
        raise ValueError('stepwise_factor has to be in range (0, 1)')

    x0 = np.array(x0)

    # set up the np.random generator
    rng = check_random_state(rng)

    # set up minimizer
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()
    wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
                                         **minimizer_kwargs)

    # set up step-taking algorithm
    if take_step is not None:
        if not callable(take_step):
            raise TypeError("take_step must be callable")
        # if take_step.stepsize exists then use AdaptiveStepsize to control
        # take_step.stepsize
        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepsize(
                take_step, interval=interval,
                accept_rate=target_accept_rate,
                factor=stepwise_factor,
                verbose=disp)
        else:
            take_step_wrapped = take_step
    else:
        # use default
        displace = RandomDisplacement(stepsize=stepsize, rng=rng)
        take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
                                             accept_rate=target_accept_rate,
                                             factor=stepwise_factor,
                                             verbose=disp)

    # set up accept tests
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        accept_tests = [accept_test]

    # use default
    metropolis = Metropolis(T, rng=rng)
    accept_tests.append(metropolis)

    if niter_success is None:
        niter_success = niter + 2

    bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
                            accept_tests, disp=disp)

    # The wrapped minimizer is called once during construction of
    # BasinHoppingRunner, so run the callback
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)

    # start main iteration loop
    count, i = 0, 0
    message = ["requested number of basinhopping iterations completed"
               " successfully"]
    for i in range(niter):
        new_global_min = bh.one_cycle()

        if callable(callback):
            # should we pass a copy of x?
            val = callback(bh.xtrial, bh.energy_trial, bh.accept)
            if val is not None:
                if val:
                    message = ["callback function requested stop early by"
                               "returning True"]
                    break

        count += 1
        if new_global_min:
            count = 0
        elif count > niter_success:
            message = ["success condition satisfied"]
            break

    # prepare return object
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    res.success = res.lowest_optimization_result.success
    return res
<|endoftext|>from typing import (  # noqa: UP035
    Any, Callable, Iterable
)

import numpy as np
from scipy.optimize import OptimizeResult
from ._constraints import old_bound_to_new, Bounds
from ._direct import direct as _direct  # type: ignore

__all__ = ['direct']

ERROR_MESSAGES = (
    "Number of function evaluations done is larger than maxfun={}",
    "Number of iterations is larger than maxiter={}",
    "u[i] < l[i] for some i",
    "maxfun is too large",
    "Initialization failed",
    "There was an error in the creation of the sample points",
    "An error occurred while the function was sampled",
    "Maximum number of levels has been reached.",
    "Forced stop",
    "Invalid arguments",
    "Out of memory",
)

SUCCESS_MESSAGES = (
    ("The best function value found is within a relative error={} "
     "of the (known) global optimum f_min"),
    ("The volume of the hyperrectangle containing the lowest function value "
     "found is below vol_tol={}"),
    ("The side length measure of the hyperrectangle containing the lowest "
     "function value found is below len_tol={}"),
)


def direct(
    func: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float64]]],
        float | np.floating[Any] | np.integer[Any] | np.bool_,
    ],
    bounds: Iterable | Bounds,
    *,
    args: tuple = (),
    eps: float = 1e-4,
    maxfun: int | None = None,
    maxiter: int = 1000,
    locally_biased: bool = True,
    f_min: float = -np.inf,
    f_min_rtol: float = 1e-4,
    vol_tol: float = 1e-16,
    len_tol: float = 1e-6,
    callback: Callable[
        [np.ndarray[tuple[int], np.dtype[np.float64]]],
        object,
    ] | None = None,
) -> OptimizeResult:
    """
    Finds the global minimum of a function using the
    DIRECT algorithm.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
        ``func(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args`` is a tuple of
        the fixed parameters needed to completely specify the function.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. ``(min, max)`` pairs for each element in ``x``.

    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    eps : float, optional
        Minimal required difference of the objective function values
        between the current best hyperrectangle and the next potentially
        optimal hyperrectangle to be divided. In consequence, `eps` serves as a
        tradeoff between local and global search: the smaller, the more local
        the search becomes. Default is 1e-4.
    maxfun : int or None, optional
        Approximate upper bound on objective function evaluations.
        If `None`, will be automatically set to ``1000 * N`` where ``N``
        represents the number of dimensions. Will be capped if necessary to
        limit DIRECT's RAM usage to app. 1GiB. This will only occur for very
        high dimensional problems and excessive `max_fun`. Default is `None`.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    locally_biased : bool, optional
        If `True` (default), use the locally biased variant of the
        algorithm known as DIRECT_L. If `False`, use the original unbiased
        DIRECT algorithm. For hard problems with many local minima,
        `False` is recommended.
    f_min : float, optional
        Function value of the global optimum. Set this value only if the
        global optimum is known. Default is ``-np.inf``, so that this
        termination criterion is deactivated.
    f_min_rtol : float, optional
        Terminate the optimization once the relative error between the
        current best minimum `f` and the supplied global minimum `f_min`
        is smaller than `f_min_rtol`. This parameter is only used if
        `f_min` is also set. Must lie between 0 and 1. Default is 1e-4.
    vol_tol : float, optional
        Terminate the optimization once the volume of the hyperrectangle
        containing the lowest function value is smaller than `vol_tol`
        of the complete search space. Must lie between 0 and 1.
        Default is 1e-16.
    len_tol : float, optional
        If ``locally_biased=True``, terminate the optimization once half of
        the normalized maximal side length of the hyperrectangle containing
        the lowest function value is smaller than `len_tol`.
        If ``locally_biased=False``, terminate the optimization once half of
        the normalized diagonal of the hyperrectangle containing the lowest
        function value is smaller than `len_tol`. Must lie between 0 and 1.
        Default is 1e-6.
    callback : callable, optional
        A callback function with signature ``callback(xk)`` where ``xk``
        represents the best function value found so far.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    Notes
    -----
    DIviding RECTangles (DIRECT) is a deterministic global
    optimization algorithm capable of minimizing a black box function with
    its variables subject to lower and upper bound constraints by sampling
    potential solutions in the search space [1]_. The algorithm starts by
    normalising the search space to an n-dimensional unit hypercube.
    It samples the function at the center of this hypercube and at 2n
    (n is the number of variables) more points, 2 in each coordinate
    direction. Using these function values, DIRECT then divides the
    domain into hyperrectangles, each having exactly one of the sampling
    points as its center. In each iteration, DIRECT chooses, using the `eps`
    parameter which defaults to 1e-4, some of the existing hyperrectangles
    to be further divided. This division process continues until either the
    maximum number of iterations or maximum function evaluations allowed
    are exceeded, or the hyperrectangle containing the minimal value found
    so far becomes small enough. If `f_min` is specified, the optimization
    will stop once this function value is reached within a relative tolerance.
    The locally biased variant of DIRECT (originally called DIRECT_L) [2]_ is
    used by default. It makes the search more locally biased and more
    efficient for cases with only a few local minima.

    A note about termination criteria: `vol_tol` refers to the volume of the
    hyperrectangle containing the lowest function value found so far. This
    volume decreases exponentially with increasing dimensionality of the
    problem. Therefore `vol_tol` should be decreased to avoid premature
    termination of the algorithm for higher dimensions. This does not hold
    for `len_tol`: it refers either to half of the maximal side length
    (for ``locally_biased=True``) or half of the diagonal of the
    hyperrectangle (for ``locally_biased=False``).

    This code is based on the DIRECT 2.0.4 Fortran code by Gablonsky et al. at
    https://ctk.math.ncsu.edu/SOFTWARE/DIRECTv204.tar.gz .
    This original version was initially converted via f2c and then cleaned up
    and reorganized by Steven G. Johnson, August 2007, for the NLopt project.
    The `direct` function wraps the C implementation.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Jones, D.R., Perttunen, C.D. & Stuckman, B.E. Lipschitzian
        optimization without the Lipschitz constant. J Optim Theory Appl
        79, 157-181 (1993).
    .. [2] Gablonsky, J., Kelley, C. A Locally-Biased form of the DIRECT
        Algorithm. Journal of Global Optimization 21, 27-37 (2001).

    Examples
    --------
    The following example is a 2-D problem with four local minima: minimizing
    the Styblinski-Tang function
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> from scipy.optimize import direct, Bounds
    >>> def styblinski_tang(pos):
    ...     x, y = pos
    ...     return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)
    >>> bounds = Bounds([-4., -4.], [4., 4.])
    >>> result = direct(styblinski_tang, bounds)
    >>> result.x, result.fun, result.nfev
    array([-2.90321597, -2.90321597]), -78.3323279095383, 2011

    The correct global minimum was found but with a huge number of function
    evaluations (2011). Loosening the termination tolerances `vol_tol` and
    `len_tol` can be used to stop DIRECT earlier.

    >>> result = direct(styblinski_tang, bounds, len_tol=1e-3)
    >>> result.x, result.fun, result.nfev
    array([-2.9044353, -2.9044353]), -78.33230330754142, 207

    """
    # convert bounds to new Bounds class if necessary
    if not isinstance(bounds, Bounds):
        if isinstance(bounds, list) or isinstance(bounds, tuple):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
        else:
            message = ("bounds must be a sequence or "
                       "instance of Bounds class")
            raise ValueError(message)

    lb = np.ascontiguousarray(bounds.lb, dtype=np.float64)
    ub = np.ascontiguousarray(bounds.ub, dtype=np.float64)

    # validate bounds
    # check that lower bounds are smaller than upper bounds
    if not np.all(lb < ub):
        raise ValueError('Bounds are not consistent min < max')
    # check for infs
    if (np.any(np.isinf(lb)) or np.any(np.isinf(ub))):
        raise ValueError("Bounds must not be inf.")

    # validate tolerances
    if (vol_tol < 0 or vol_tol > 1):
        raise ValueError("vol_tol must be between 0 and 1.")
    if (len_tol < 0 or len_tol > 1):
        raise ValueError("len_tol must be between 0 and 1.")
    if (f_min_rtol < 0 or f_min_rtol > 1):
        raise ValueError("f_min_rtol must be between 0 and 1.")

    # validate maxfun and maxiter
    if maxfun is None:
        maxfun = 1000 * lb.shape[0]
    if not isinstance(maxfun, int):
        raise ValueError("maxfun must be of type int.")
    if maxfun < 0:
        raise ValueError("maxfun must be > 0.")
    if not isinstance(maxiter, int):
        raise ValueError("maxiter must be of type int.")
    if maxiter < 0:
        raise ValueError("maxiter must be > 0.")

    # validate boolean parameters
    if not isinstance(locally_biased, bool):
        raise ValueError("locally_biased must be True or False.")

    def _func_wrap(x, args=None):
        x = np.asarray(x)
        if args is None:
            f = func(x)
        else:
            f = func(x, *args)
        # always return a float
        return np.asarray(f).item()

    # TODO: fix disp argument
    x, fun, ret_code, nfev, nit = _direct(
        _func_wrap,
        np.asarray(lb), np.asarray(ub),
        args,
        False, eps, maxfun, maxiter,
        locally_biased,
        f_min, f_min_rtol,
        vol_tol, len_tol, callback
    )

    format_val = (maxfun, maxiter, f_min_rtol, vol_tol, len_tol)
    if ret_code > 2:
        message = SUCCESS_MESSAGES[ret_code - 3].format(
                    format_val[ret_code - 1])
    elif 0 < ret_code <= 2:
        message = ERROR_MESSAGES[ret_code - 1].format(format_val[ret_code - 1])
    elif 0 > ret_code > -100:
        message = ERROR_MESSAGES[abs(ret_code) + 1]
    else:
        message = ERROR_MESSAGES[ret_code + 99]

    return OptimizeResult(x=np.asarray(x), fun=fun, status=ret_code,
                          success=ret_code > 2, message=message,
                          nfev=nfev, nit=nit)
<|endoftext|>"""Simplex method for  linear programming

The *simplex* method uses a traditional, full-tableau implementation of
Dantzig's simplex algorithm [1]_, [2]_ (*not* the Nelder-Mead simplex).
This algorithm is included for backwards compatibility and educational
purposes.

    .. versionadded:: 0.15.0

Warnings
--------

The simplex method may encounter numerical difficulties when pivot
values are close to the specified tolerance. If encountered try
remove any redundant constraints, change the pivot strategy to Bland's
rule or increase the tolerance value.

Alternatively, more robust methods maybe be used. See
:ref:`'interior-point' <optimize.linprog-interior-point>` and
:ref:`'revised simplex' <optimize.linprog-revised_simplex>`.

References
----------
.. [1] Dantzig, George B., Linear programming and extensions. Rand
       Corporation Research Study Princeton Univ. Press, Princeton, NJ,
       1963
.. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
       Mathematical Programming", McGraw-Hill, Chapter 4.
"""

import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve


def _pivot_col(T, tol=1e-9, bland=False):
    """
    Given a linear programming simplex tableau, determine the column
    of the variable to enter the basis.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    tol : float
        Elements in the objective row larger than -tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the column (select the
        first column with a negative coefficient in the objective row,
        regardless of magnitude).

    Returns
    -------
    status: bool
        True if a suitable pivot column was found, otherwise False.
        A return of False indicates that the linear programming simplex
        algorithm is complete.
    col: int
        The index of the column of the pivot element.
        If status is False, col will be returned as nan.
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    if bland:
        # ma.mask is sometimes 0d
        return True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
    return True, np.ma.nonzero(ma == ma.min())[0][0]


def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):
    """
    Given a linear programming simplex tableau, determine the row for the
    pivot operation.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a Problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    basis : array
        A list of the current basic variables.
    pivcol : int
        The index of the pivot column.
    phase : int
        The phase of the simplex algorithm (1 or 2).
    tol : float
        Elements in the pivot column smaller than tol will not be considered
        for pivoting. Nominally this value is zero, but numerical issues
        cause a tolerance about zero to be necessary.
    bland : bool
        If True, use Bland's rule for selection of the row (if more than one
        row can be used, choose the one with the lowest variable index).

    Returns
    -------
    status: bool
        True if a suitable pivot row was found, otherwise False. A return
        of False indicates that the linear programming problem is unbounded.
    row: int
        The index of the row of the pivot element. If status is False, row
        will be returned as nan.
    """
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    if ma.count() == 0:
        return False, np.nan
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.nonzero(q == q.min())[0]
    if bland:
        return True, min_rows[np.argmin(np.take(basis, min_rows))]
    return True, min_rows[0]


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
    """
    Pivot the simplex tableau inplace on the element given by (pivrow, pivol).
    The entering variable corresponds to the column given by pivcol forcing
    the variable basis[pivrow] to leave the basis.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    basis : 1-D array
        An array of the indices of the basic variables, such that basis[i]
        contains the column corresponding to the basic variable for row i.
        Basis is modified in place by _apply_pivot.
    pivrow : int
        Row index of the pivot.
    pivcol : int
        Column index of the pivot.
    """
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]

    # The selected pivot should never lead to a pivot value less than the tol.
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        message = (
            f"The pivot operation produces a pivot value of:{pivval: .1e}, "
            "which is only slightly greater than the specified "
            f"tolerance{tol: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.")
        warn(message, OptimizeWarning, stacklevel=5)


def _solve_simplex(T, n, basis, callback, postsolve_args,
                   maxiter=1000, tol=1e-9, phase=2, bland=False, nit0=0,
                   ):
    """
    Solve a linear programming problem in "standard form" using the Simplex
    Method. Linear Programming is intended to solve the following problem form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    n : int
        The number of true variables in the problem.
    basis : 1-D array
        An array of the indices of the basic variables, such that basis[i]
        contains the column corresponding to the basic variable for row i.
        Basis is modified in place by _solve_simplex
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback must accept a
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True only when a phase has completed successfully. This
                will be False for most iterations.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the optimization being executed. In phase 1 a basic
                feasible solution is sought and the T has an additional row
                representing an alternate objective function.
            status : int
                An integer representing the exit status of the optimization::

                     0 : Optimization terminated successfully
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered

            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.
    maxiter : int
        The maximum number of iterations to perform before aborting the
        optimization.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    phase : int
        The phase of the optimization being executed. In phase 1 a basic
        feasible solution is sought and the T has an additional row
        representing an alternate objective function.
    bland : bool
        If True, choose pivots using Bland's rule [3]_. In problems which
        fail to converge due to cycling, using Bland's rule can provide
        convergence at the expense of a less optimal path about the simplex.
    nit0 : int
        The initial iteration number used to keep an accurate iteration total
        in a two-phase problem.

    Returns
    -------
    nit : int
        The number of iterations. Used to keep an accurate iteration total
        in the two-phase problem.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered

    """
    nit = nit0
    status = 0
    message = ''
    complete = False

    if phase == 1:
        m = T.shape[1]-2
    elif phase == 2:
        m = T.shape[1]-1
    else:
        raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")

    if phase == 2:
        # Check if any artificial variables are still in the basis.
        # If yes, check if any coefficients from this row and a column
        # corresponding to one of the non-artificial variable is non-zero.
        # If found, pivot at this term. If not, start phase 2.
        # Do this for all artificial variables in the basis.
        # Ref: "An Introduction to Linear Programming and Game Theory"
        # by Paul R. Thie, Gerard E. Keough, 3rd Ed,
        # Chapter 3.7 Redundant Systems (pag 102)
        for pivrow in [row for row in range(basis.size)
                       if basis[row] > T.shape[1] - 2]:
            non_zero_row = [col for col in range(T.shape[1] - 1)
                            if abs(T[pivrow, col]) > tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1

    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not complete:
        # Find the pivot column
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            # Find the pivot row
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            if not pivrow_found:
                status = 3
                complete = True

        if callback is not None:
            solution[:] = 0
            solution[basis[:n]] = T[:n, -1]
            x = solution[:m]
            x, fun, slack, con = _postsolve(
                x, postsolve_args
            )
            res = OptimizeResult({
                'x': x,
                'fun': fun,
                'slack': slack,
                'con': con,
                'status': status,
                'message': message,
                'nit': nit,
                'success': status == 0 and complete,
                'phase': phase,
                'complete': complete,
                })
            callback(res)

        if not complete:
            if nit >= maxiter:
                # Iteration limit exceeded
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return nit, status


def _linprog_simplex(c, c0, A, b, callback, postsolve_args,
                     maxiter=1000, tol=1e-9, disp=False, bland=False,
                     **unknown_options):
    """
    Minimize a linear objective function subject to linear equality and
    non-negativity constraints using the two phase simplex method.
    Linear programming is intended to solve problems of the following form:

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
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector
            fun : float
                Current value of the objective function
            success : bool
                True when an algorithm has completed successfully.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``
            phase : int
                The phase of the algorithm being executed.
            status : int
                An integer representing the status of the optimization::

                     0 : Algorithm proceeding nominally
                     1 : Iteration limit reached
                     2 : Problem appears to be infeasible
                     3 : Problem appears to be unbounded
                     4 : Serious numerical difficulties encountered
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform.
    disp : bool
        If True, print exit status message to sys.stdout
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    bland : bool
        If True, use Bland's anti-cycling rule [3]_ to choose pivots to
        prevent cycling. If False, choose pivots which should lead to a
        converged solution more quickly. The latter method is subject to
        cycling (non-convergence) in rare instances.
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

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.


    Notes
    -----
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
    """
    _check_unknown_options(unknown_options)

    status = 0
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}

    n, m = A.shape

    # All constraints must have b >= 0.
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1

    # As all constraints are equality constraints the artificial variables
    # will also be basic variables.
    av = np.arange(n) + m
    basis = av.copy()

    # Format the phase one tableau by adding artificial variables and stacking
    # the constraints, the objective row and pseudo-objective row.
    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    nit1, status = _solve_simplex(T, n, basis, callback=callback,
                                  postsolve_args=postsolve_args,
                                  maxiter=maxiter, tol=tol, phase=1,
                                  bland=bland
                                  )
    # if pseudo objective is zero, remove the last row from the tableau and
    # proceed to phase 2
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
        # Remove the pseudo-objective row from the tableau
        T = T[:-1, :]
        # Remove the artificial variable columns from the tableau
        T = np.delete(T, av, 1)
    else:
        # Failure to find a feasible starting point
        status = 2
        messages[status] = (
            "Phase 1 of the simplex method failed to find a feasible "
            "solution. The pseudo-objective function evaluates to "
            f"{abs(T[-1, -1]):.1e} "
            f"which exceeds the required tolerance of {tol} for a solution to be "
            "considered 'close enough' to zero to be a basic solution. "
            "Consider increasing the tolerance to be greater than "
            f"{abs(T[-1, -1]):.1e}. "
            "If this tolerance is unacceptably large the problem may be "
            "infeasible."
        )

    if status == 0:
        # Phase 2
        nit2, status = _solve_simplex(T, n, basis, callback=callback,
                                      postsolve_args=postsolve_args,
                                      maxiter=maxiter, tol=tol, phase=2,
                                      bland=bland, nit0=nit1
                                      )

    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]

    return x, status, messages[status], int(nit2)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.


from scipy._lib.deprecation import _sub_module_deprecation


__all__ = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="moduleTNC",
                                   private_modules=["_moduleTNC"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'OptimizeResult',
    'fmin_tnc',
    'zeros',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="tnc",
                                   private_modules=["_tnc"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014
"""
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import (check_random_state, MapWrapper, _FunctionWrapper,
                              rng_integers, _transition_to_rng)

from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
                                         NonlinearConstraint, LinearConstraint)
from scipy.sparse import issparse

__all__ = ['differential_evolution']


_MACHEPS = np.finfo(np.float64).eps


@_transition_to_rng("seed", position_num=9)
def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, rng=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0, updating='immediate',
                           workers=1, constraints=(), x0=None, *,
                           integrality=None, vectorized=False):
    r"""Finds the global minimum of a multivariate function.

    The differential evolution method [1]_ is stochastic in nature. It does
    not use gradient methods to find the minimum, and can search large areas
    of candidate space, but often requires larger numbers of function
    evaluations than conventional gradient-based techniques.

    The algorithm is due to Storn and Price [2]_.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function. The number of parameters, N, is equal
        to ``len(x)``.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

        1. Instance of `Bounds` class.
        2. ``(min, max)`` pairs for each element in ``x``, defining the
           finite lower and upper bounds for the optimizing argument of
           `func`.

        The total number of bounds is used to determine the number of
        parameters, N. If there are parameters whose bounds are equal the total
        number of free parameters is ``N - N_equal``.

    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : {str, callable}, optional
        The differential evolution strategy to use. Should be one of:

        - 'best1bin'
        - 'best1exp'
        - 'rand1bin'
        - 'rand1exp'
        - 'rand2bin'
        - 'rand2exp'
        - 'randtobest1bin'
        - 'randtobest1exp'
        - 'currenttobest1bin'
        - 'currenttobest1exp'
        - 'best2exp'
        - 'best2bin'

        The default is 'best1bin'. Strategies that may be implemented are
        outlined in 'Notes'.
        Alternatively the differential evolution strategy can be customized by
        providing a callable that constructs a trial vector. The callable must
        have the form ``strategy(candidate: int, population: np.ndarray, rng=None)``,
        where ``candidate`` is an integer specifying which entry of the
        population is being evolved, ``population`` is an array of shape
        ``(S, N)`` containing all the population members (where S is the
        total population size), and ``rng`` is the random number generator
        being used within the solver.
        ``candidate`` will be in the range ``[0, S)``.
        ``strategy`` must return a trial vector with shape ``(N,)``. The
        fitness of this trial vector is compared against the fitness of
        ``population[candidate]``.

        .. versionchanged:: 1.12.0
            Customization of evolution strategy via a callable.

    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * (N - N_equal)``
    popsize : int, optional
        A multiplier for setting the total population size. The population has
        ``popsize * (N - N_equal)`` individuals. This keyword is overridden if
        an initial population is supplied via the `init` keyword. When using
        ``init='sobol'`` the population size is calculated as the next power
        of 2 after ``popsize * (N - N_equal)``.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by :math:`F`.
        If specified as a float it should be in the range [0, 2).
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    disp : bool, optional
        Prints the evaluated `func` at every iteration.
    callback : callable, optional
        A callable called after each iteration. Has the signature::

            callback(intermediate_result: OptimizeResult)

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the best solution
        found so far and the objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`.

        The callback also supports a signature like::

            callback(x, convergence: float=val)

        ``val`` represents the fractional value of the population convergence.
        When ``val`` is greater than ``1.0``, the function halts.

        Introspection is used to determine which of the signatures is invoked.

        Global minimization will halt if the callback raises ``StopIteration``
        or returns ``True``; any polishing is still carried out.

        .. versionchanged:: 1.12.0
            callback accepts the ``intermediate_result`` keyword.

    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. If a constrained problem is
        being studied then the `trust-constr` method is used instead. For large
        problems with many constraints, polishing can take a long time due to
        the Jacobian computations.

        .. versionchanged:: 1.15.0
            If `workers` is specified then the map-like callable that wraps
            `func` is supplied to `minimize` instead of it using `func`
            directly. This allows the caller to control how and where the
            invocations actually run.

    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:

        - 'latinhypercube'
        - 'sobol'
        - 'halton'
        - 'random'
        - array specifying the initial population. The array should have
          shape ``(S, N)``, where S is the total population size and N is
          the number of parameters.

        `init` is clipped to `bounds` before use.

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space.

        'sobol' and 'halton' are superior alternatives and maximize even more
        the parameter space. 'sobol' will enforce an initial population
        size which is calculated as the next power of 2 after
        ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit
        less efficient. See `scipy.stats.qmc` for more details.

        'random' initializes the population randomly - this has the drawback
        that clustering can occur, preventing the whole of parameter space
        being covered. Use of an array to specify a population could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    updating : {'immediate', 'deferred'}, optional
        If ``'immediate'``, the best solution vector is continuously updated
        within a single generation [4]_. This can lead to faster convergence as
        trial vectors can take advantage of continuous improvements in the best
        solution.
        With ``'deferred'``, the best solution vector is updated once per
        generation. Only ``'deferred'`` is compatible with parallelization or
        vectorization, and the `workers` and `vectorized` keywords can
        over-ride this option.

        .. versionadded:: 1.2.0

    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel
        (uses `multiprocessing.Pool <multiprocessing>`).
        Supply -1 to use all available CPU cores.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        This option will override the `updating` keyword to
        ``updating='deferred'`` if ``workers != 1``.
        This option overrides the `vectorized` keyword if ``workers != 1``.
        Requires that `func` be pickleable.

        .. versionadded:: 1.2.0

    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        Constraints on the solver, over and above those applied by the `bounds`
        kwd. Uses the approach by Lampinen [5]_.

        .. versionadded:: 1.4.0

    x0 : None or array-like, optional
        Provides an initial guess to the minimization. Once the population has
        been initialized this vector replaces the first (best) member. This
        replacement is done even if `init` is given an initial population.
        ``x0.shape == (N,)``.

        .. versionadded:: 1.7.0

    integrality : 1-D array, optional
        For each decision variable, a boolean value indicating whether the
        decision variable is constrained to integer values. The array is
        broadcast to ``(N,)``.
        If any decision variables are constrained to be integral, they will not
        be changed during polishing.
        Only integer values lying between the lower and upper bounds are used.
        If there are no integer values lying between the bounds then a
        `ValueError` is raised.

        .. versionadded:: 1.9.0

    vectorized : bool, optional
        If ``vectorized is True``, `func` is sent an `x` array with
        ``x.shape == (N, S)``, and is expected to return an array of shape
        ``(S,)``, where `S` is the number of solution vectors to be calculated.
        If constraints are applied, each of the functions used to construct
        a `Constraint` object should accept an `x` array with
        ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where
        `M` is the number of constraint components.
        This option is an alternative to the parallelization offered by
        `workers`, and may help in optimization speed by reducing interpreter
        overhead from multiple function calls. This keyword is ignored if
        ``workers != 1``.
        This option will override the `updating` keyword to
        ``updating='deferred'``.
        See the notes section for further discussion on when to use
        ``'vectorized'``, and when to use ``'workers'``.

        .. versionadded:: 1.9.0

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully,
        ``message`` which describes the cause of the termination,
        ``population`` the solution vectors present in the population, and
        ``population_energies`` the value of the objective function for each
        entry in ``population``.
        See `OptimizeResult` for a description of other attributes. If `polish`
        was employed, and a lower minimum was obtained by the polishing, then
        OptimizeResult also contains the ``jac`` attribute.
        If the eventual solution does not satisfy the applied constraints
        ``success`` will be `False`.

    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the
    population the algorithm mutates each candidate solution by mixing with
    other candidate solutions to create a trial candidate. There are several
    strategies [3]_ for creating trial candidates, which suit some problems
    more than others. The 'best1bin' strategy is a good starting point for
    many systems. In this strategy two members of the population are randomly
    chosen. Their difference is used to mutate the best member (the 'best' in
    'best1bin'), :math:`x_0`, so far:

    .. math::

        b' = x_0 + F \cdot (x_{r_0} - x_{r_1})

    where :math:`F` is the `mutation` parameter.
    A trial vector is then constructed. Starting with a randomly chosen ith
    parameter the trial is sequentially filled (in modulo) with parameters
    from ``b'`` or the original candidate. The choice of whether to use ``b'``
    or the original candidate is made with a binomial distribution (the 'bin'
    in 'best1bin') - a random number in [0, 1) is generated. If this number is
    less than the `recombination` constant then the parameter is loaded from
    ``b'``, otherwise it is loaded from the original candidate. The final
    parameter is always loaded from ``b'``. Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.

    The other strategies available are outlined in Qiang and
    Mitchell (2014) [3]_.


    - ``rand1`` : :math:`b' = x_{r_0} + F \cdot (x_{r_1} - x_{r_2})`
    - ``rand2`` : :math:`b' = x_{r_0} + F \cdot (x_{r_1} + x_{r_2} - x_{r_3} - x_{r_4})`
    - ``best1`` : :math:`b' = x_0 + F \cdot (x_{r_0} - x_{r_1})`
    - ``best2`` : :math:`b' = x_0 + F \cdot (x_{r_0} + x_{r_1} - x_{r_2} - x_{r_3})`
    - ``currenttobest1`` : :math:`b' = x_i + F \cdot (x_0 - x_i + x_{r_0} - x_{r_1})`
    - ``randtobest1`` : :math:`b' = x_{r_0} + F \cdot (x_0 - x_{r_0} + x_{r_1} - x_{r_2})`

    where the integers :math:`r_0, r_1, r_2, r_3, r_4` are chosen randomly
    from the interval [0, NP) with `NP` being the total population size and
    the original candidate having index `i`. The user can fully customize the
    generation of the trial candidates by supplying a callable to ``strategy``.

    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.

    By default the best solution vector is updated continuously within a single
    iteration (``updating='immediate'``). This is a modification [4]_ of the
    original differential evolution algorithm which can lead to faster
    convergence as trial vectors can immediately benefit from improved
    solutions. To use the original Storn and Price behaviour, updating the best
    solution once per iteration, set ``updating='deferred'``.
    The ``'deferred'`` approach is compatible with both parallelization and
    vectorization (``'workers'`` and ``'vectorized'`` keywords). These may
    improve minimization speed by using computer resources more efficiently.
    The ``'workers'`` distribute calculations over multiple processors. By
    default the Python `multiprocessing` module is used, but other approaches
    are also possible, such as the Message Passing Interface (MPI) used on
    clusters [6]_ [7]_. The overhead from these approaches (creating new
    Processes, etc) may be significant, meaning that computational speed
    doesn't necessarily scale with the number of processors used.
    Parallelization is best suited to computationally expensive objective
    functions. If the objective function is less expensive, then
    ``'vectorized'`` may aid by only calling the objective function once per
    iteration, rather than multiple times for all the population members; the
    interpreter overhead is reduced.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Differential evolution, Wikipedia,
           http://en.wikipedia.org/wiki/Differential_evolution
    .. [2] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [3] Qiang, J., Mitchell, C., A Unified Differential Evolution Algorithm
            for Global Optimization, 2014, https://www.osti.gov/servlets/purl/1163659
    .. [4] Wormington, M., Panaccione, C., Matney, K. M., Bowen, D. K., -
           Characterization of structures from X-ray scattering data using
           genetic algorithms, Phil. Trans. R. Soc. Lond. A, 1999, 357,
           2827-2848
    .. [5] Lampinen, J., A constraint handling approach for the differential
           evolution algorithm. Proceedings of the 2002 Congress on
           Evolutionary Computation. CEC'02 (Cat. No. 02TH8600). Vol. 2. IEEE,
           2002.
    .. [6] https://mpi4py.readthedocs.io/en/stable/
    .. [7] https://schwimmbad.readthedocs.io/en/latest/
 

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.

    >>> import numpy as np
    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    Now repeat, but with parallelization.

    >>> result = differential_evolution(rosen, bounds, updating='deferred',
    ...                                 workers=2)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    Let's do a constrained minimization.

    >>> from scipy.optimize import LinearConstraint, Bounds

    We add the constraint that the sum of ``x[0]`` and ``x[1]`` must be less
    than or equal to 1.9.  This is a linear constraint, which may be written
    ``A @ x <= 1.9``, where ``A = array([[1, 1]])``.  This can be encoded as
    a `LinearConstraint` instance:

    >>> lc = LinearConstraint([[1, 1]], -np.inf, 1.9)

    Specify limits using a `Bounds` object.

    >>> bounds = Bounds([0., 0.], [2., 2.])
    >>> result = differential_evolution(rosen, bounds, constraints=lc,
    ...                                 rng=1)
    >>> result.x, result.fun
    (array([0.96632622, 0.93367155]), 0.0011352416852625719)

    Next find the minimum of the Ackley function
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds, rng=1)
    >>> result.x, result.fun
    (array([0., 0.]), 4.440892098500626e-16)

    The Ackley function is written in a vectorized manner, so the
    ``'vectorized'`` keyword can be employed. Note the reduced number of
    function evaluations.

    >>> result = differential_evolution(
    ...     ackley, bounds, vectorized=True, updating='deferred', rng=1
    ... )
    >>> result.x, result.fun
    (array([0., 0.]), 4.440892098500626e-16)

    The following custom strategy function mimics 'best1bin':

    >>> def custom_strategy_fn(candidate, population, rng=None):
    ...     parameter_count = population.shape(-1)
    ...     mutation, recombination = 0.7, 0.9
    ...     trial = np.copy(population[candidate])
    ...     fill_point = rng.choice(parameter_count)
    ...
    ...     pool = np.arange(len(population))
    ...     rng.shuffle(pool)
    ...
    ...     # two unique random numbers that aren't the same, and
    ...     # aren't equal to candidate.
    ...     idxs = []
    ...     while len(idxs) < 2 and len(pool) > 0:
    ...         idx = pool[0]
    ...         pool = pool[1:]
    ...         if idx != candidate:
    ...             idxs.append(idx)
    ...
    ...     r0, r1 = idxs[:2]
    ...
    ...     bprime = (population[0] + mutation *
    ...               (population[r0] - population[r1]))
    ...
    ...     crossovers = rng.uniform(size=parameter_count)
    ...     crossovers = crossovers < recombination
    ...     crossovers[fill_point] = True
    ...     trial = np.where(crossovers, bprime, trial)
    ...     return trial

    """# noqa: E501

    # using a context manager means that any created Pool objects are
    # cleared up.
    with DifferentialEvolutionSolver(func, bounds, args=args,
                                     strategy=strategy,
                                     maxiter=maxiter,
                                     popsize=popsize, tol=tol,
                                     mutation=mutation,
                                     recombination=recombination,
                                     rng=rng, polish=polish,
                                     callback=callback,
                                     disp=disp, init=init, atol=atol,
                                     updating=updating,
                                     workers=workers,
                                     constraints=constraints,
                                     x0=x0,
                                     integrality=integrality,
                                     vectorized=vectorized) as solver:
        ret = solver.solve()

    return ret


class DifferentialEvolutionSolver:

    """This class implements the differential evolution solver

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function. The number of parameters, N, is equal
        to ``len(x)``.
    bounds : sequence or `Bounds`
        Bounds for variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. ``(min, max)`` pairs for each element in ``x``, defining the
               finite lower and upper bounds for the optimizing argument of
               `func`.

        The total number of bounds is used to determine the number of
        parameters, N. If there are parameters whose bounds are equal the total
        number of free parameters is ``N - N_equal``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : {str, callable}, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1bin'
            - 'rand1exp'
            - 'rand2bin'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'

        The default is 'best1bin'. Strategies that may be
        implemented are outlined in 'Notes'.

        Alternatively the differential evolution strategy can be customized
        by providing a callable that constructs a trial vector. The callable
        must have the form
        ``strategy(candidate: int, population: np.ndarray, rng=None)``,
        where ``candidate`` is an integer specifying which entry of the
        population is being evolved, ``population`` is an array of shape
        ``(S, N)`` containing all the population members (where S is the
        total population size), and ``rng`` is the random number generator
        being used within the solver.
        ``candidate`` will be in the range ``[0, S)``.
        ``strategy`` must return a trial vector with shape ``(N,)``. The
        fitness of this trial vector is compared against the fitness of
        ``population[candidate]``.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * (N - N_equal)``
    popsize : int, optional
        A multiplier for setting the total population size. The population has
        ``popsize * (N - N_equal)`` individuals. This keyword is overridden if
        an initial population is supplied via the `init` keyword. When using
        ``init='sobol'`` the population size is calculated as the next power
        of 2 after ``popsize * (N - N_equal)``.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(population_energies) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.

    rng : {None, int, `numpy.random.Generator`}, optional
        
        ..versionchanged:: 1.15.0
            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator` this keyword was changed from `seed` to `rng`.
            For an interim period both keywords will continue to work (only specify
            one of them). After the interim period using the `seed` keyword will emit
            warnings. The behavior of the `seed` and `rng` keywords is outlined below.

        If `rng` is passed by keyword, types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a `Generator`.
        If `rng` is already a `Generator` instance, then the provided instance is
        used.
        
        If this argument is passed by position or `seed` is passed by keyword, the
        behavior is:
        
        - If `seed` is None (or `np.random`), the `numpy.random.RandomState`
          singleton is used.
        - If `seed` is an int, a new `RandomState` instance is used,
          seeded with `seed`.
        - If `seed` is already a `Generator` or `RandomState` instance then
          that instance is used.
        
        Specify `seed`/`rng` for repeatable minimizations.
    disp : bool, optional
        Prints the evaluated `func` at every iteration.
    callback : callable, optional
        A callable called after each iteration. Has the signature:

            ``callback(intermediate_result: OptimizeResult)``

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the best solution
        found so far and the objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`.

        The callback also supports a signature like:

            ``callback(x, convergence: float=val)``

        ``val`` represents the fractional value of the population convergence.
         When ``val`` is greater than ``1.0``, the function halts.

        Introspection is used to determine which of the signatures is invoked.

        Global minimization will halt if the callback raises ``StopIteration``
        or returns ``True``; any polishing is still carried out.

        .. versionchanged:: 1.12.0
            callback accepts the ``intermediate_result`` keyword.

    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly. If a constrained problem is
        being studied then the `trust-constr` method is used instead. For large
        problems with many constraints, polishing can take a long time due to
        the Jacobian computations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'sobol'
            - 'halton'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(S, N)``, where S is the total population size and
              N is the number of parameters.
              `init` is clipped to `bounds` before use.

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space.

        'sobol' and 'halton' are superior alternatives and maximize even more
        the parameter space. 'sobol' will enforce an initial population
        size which is calculated as the next power of 2 after
        ``popsize * (N - N_equal)``. 'halton' has no requirements but is a bit
        less efficient. See `scipy.stats.qmc` for more details.

        'random' initializes the population randomly - this has the drawback
        that clustering can occur, preventing the whole of parameter space
        being covered. Use of an array to specify a population could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    updating : {'immediate', 'deferred'}, optional
        If ``'immediate'``, the best solution vector is continuously updated
        within a single generation [4]_. This can lead to faster convergence as
        trial vectors can take advantage of continuous improvements in the best
        solution.
        With ``'deferred'``, the best solution vector is updated once per
        generation. Only ``'deferred'`` is compatible with parallelization or
        vectorization, and the `workers` and `vectorized` keywords can
        over-ride this option.
    workers : int or map-like callable, optional
        If `workers` is an int the population is subdivided into `workers`
        sections and evaluated in parallel
        (uses `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
        Alternatively supply a map-like callable, such as
        `multiprocessing.Pool.map` for evaluating the population in parallel.
        This evaluation is carried out as ``workers(func, iterable)``.
        This option will override the `updating` keyword to
        `updating='deferred'` if `workers != 1`.
        Requires that `func` be pickleable.
    constraints : {NonLinearConstraint, LinearConstraint, Bounds}
        Constraints on the solver, over and above those applied by the `bounds`
        kwd. Uses the approach by Lampinen.
    x0 : None or array-like, optional
        Provides an initial guess to the minimization. Once the population has
        been initialized this vector replaces the first (best) member. This
        replacement is done even if `init` is given an initial population.
        ``x0.shape == (N,)``.
    integrality : 1-D array, optional
        For each decision variable, a boolean value indicating whether the
        decision variable is constrained to integer values. The array is
        broadcast to ``(N,)``.
        If any decision variables are constrained to be integral, they will not
        be changed during polishing.
        Only integer values lying between the lower and upper bounds are used.
        If there are no integer values lying between the bounds then a
        `ValueError` is raised.
    vectorized : bool, optional
        If ``vectorized is True``, `func` is sent an `x` array with
        ``x.shape == (N, S)``, and is expected to return an array of shape
        ``(S,)``, where `S` is the number of solution vectors to be calculated.
        If constraints are applied, each of the functions used to construct
        a `Constraint` object should accept an `x` array with
        ``x.shape == (N, S)``, and return an array of shape ``(M, S)``, where
        `M` is the number of constraint components.
        This option is an alternative to the parallelization offered by
        `workers`, and may help in optimization speed. This keyword is
        ignored if ``workers != 1``.
        This option will override the `updating` keyword to
        ``updating='deferred'``.
    """ # noqa: E501

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(S, N) where N is the number of parameters and S>5")

    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, rng=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, updating='immediate',
                 workers=1, constraints=(), x0=None, *, integrality=None,
                 vectorized=False):

        if callable(strategy):
            # a callable strategy is going to be stored in self.strategy anyway
            pass
        elif strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = _wrap_callback(callback, "differential_evolution")
        self.polish = polish

        # set the updating / parallelisation options
        if updating in ['immediate', 'deferred']:
            self._updating = updating

        self.vectorized = vectorized

        # want to use parallelisation, but updating is immediate
        if workers != 1 and updating == 'immediate':
            warnings.warn("differential_evolution: the 'workers' keyword has"
                          " overridden updating='immediate' to"
                          " updating='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'

        if vectorized and workers != 1:
            warnings.warn("differential_evolution: the 'workers' keyword"
                          " overrides the 'vectorized' keyword", stacklevel=2)
            self.vectorized = vectorized = False

        if vectorized and updating == 'immediate':
            warnings.warn("differential_evolution: the 'vectorized' keyword"
                          " has overridden updating='immediate' to updating"
                          "='deferred'", UserWarning, stacklevel=2)
            self._updating = 'deferred'

        # an object with a map method.
        if vectorized:
            def maplike_for_vectorized_func(func, x):
                # send an array (N, S) to the user func,
                # expect to receive (S,). Transposition is required because
                # internally the population is held as (S, N)
                return np.atleast_1d(func(x.T))
            workers = maplike_for_vectorized_func

        self._mapwrapper = MapWrapper(workers)

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        # we create a wrapped function to allow the use of map (and Pool.map
        # in the future)
        self.func = _FunctionWrapper(func, args)
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        if isinstance(bounds, Bounds):
            self.limits = np.array(new_bounds_to_old(bounds.lb,
                                                     bounds.ub,
                                                     len(bounds.lb)),
                                   dtype=float).T
        else:
            self.limits = np.array(bounds, dtype='float').T

        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing finite '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        with np.errstate(divide='ignore'):
            # if lb == ub then the following line will be 1/0, which is why
            # we ignore the divide by zero warning. The result from 1/0 is
            # inf, so replace those values by 0.
            self.__recip_scale_arg2 = 1 / self.__scale_arg2
            self.__recip_scale_arg2[~np.isfinite(self.__recip_scale_arg2)] = 0

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(rng)

        # Which parameters are going to be integers?
        if np.any(integrality):
            # # user has provided a truth value for integer constraints
            integrality = np.broadcast_to(
                integrality,
                self.parameter_count
            )
            integrality = np.asarray(integrality, bool)
            # For integrality parameters change the limits to only allow
            # integer values lying between the limits.
            lb, ub = np.copy(self.limits)

            lb = np.ceil(lb)
            ub = np.floor(ub)
            if not (lb[integrality] <= ub[integrality]).all():
                # there's a parameter that doesn't have an integer value
                # lying between the limits
                raise ValueError("One of the integrality constraints does not"
                                 " have any possible integer values between"
                                 " the lower/upper bounds.")
            nlb = np.nextafter(lb[integrality] - 0.5, np.inf)
            nub = np.nextafter(ub[integrality] + 0.5, -np.inf)

            self.integrality = integrality
            self.limits[0, self.integrality] = nlb
            self.limits[1, self.integrality] = nub
        else:
            self.integrality = False

        # check for equal bounds
        eb = self.limits[0] == self.limits[1]
        eb_count = np.count_nonzero(eb)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        # 202301 - reduced population size to account for parameters with
        # equal bounds. If there are no varying parameters set N to at least 1
        self.num_population_members = max(
            5,
            popsize * max(1, self.parameter_count - eb_count)
        )
        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        self._nfev = 0
        # check first str otherwise will fail to compare str with array
        if isinstance(init, str):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'sobol':
                # must be Ns = 2**m for Sobol'
                n_s = int(2 ** np.ceil(np.log2(self.num_population_members)))
                self.num_population_members = n_s
                self.population_shape = (self.num_population_members,
                                         self.parameter_count)
                self.init_population_qmc(qmc_engine='sobol')
            elif init == 'halton':
                self.init_population_qmc(qmc_engine='halton')
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        if x0 is not None:
            # scale to within unit interval and
            # ensure parameters are within bounds.
            x0_scaled = self._unscale_parameters(np.asarray(x0))
            if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
                raise ValueError(
                    "Some entries in x0 lay outside the specified bounds"
                )
            self.population[0] = x0_scaled

        # infrastructure for constraints
        self.constraints = constraints
        self._wrapped_constraints = []

        if hasattr(constraints, '__len__'):
            # sequence of constraints, this will also deal with default
            # keyword parameter
            for c in constraints:
                self._wrapped_constraints.append(
                    _ConstraintWrapper(c, self.x)
                )
        else:
            self._wrapped_constraints = [
                _ConstraintWrapper(constraints, self.x)
            ]
        self.total_constraints = np.sum(
            [c.num_constr for c in self._wrapped_constraints]
        )
        self.constraint_violation = np.zeros((self.num_population_members, 1))
        self.feasible = np.ones(self.num_population_members, bool)

        # an array to shuffle when selecting candidates. Create it here
        # rather than repeatedly creating it in _select_samples.
        self._random_population_index = np.arange(self.num_population_members)
        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.uniform(size=self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_qmc(self, qmc_engine):
        """Initializes the population with a QMC method.

        QMC methods ensures that each parameter is uniformly
        sampled over its range.

        Parameters
        ----------
        qmc_engine : str
            The QMC method to use for initialization. Can be one of
            ``latinhypercube``, ``sobol`` or ``halton``.

        """
        from scipy.stats import qmc

        rng = self.random_number_generator

        # Create an array for population of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'sobol':
            sampler = qmc.Sobol(d=self.parameter_count, seed=rng)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=self.parameter_count, seed=rng)
        else:
            raise ValueError(self.__init_error_msg)

        self.population = sampler.random(n=self.num_population_members)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initializes the population at random. This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.uniform(size=self.population_shape)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initializes the population with a user specified population.

        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (S, N), where N is the number of parameters.
            The population is clipped to the lower and upper bounds.
        """
        # make sure you're using a float array
        popn = np.asarray(init, dtype=np.float64)

        if (np.size(popn, 0) < 5 or
                popn.shape[1] != self.parameter_count or
                len(popn.shape) != 2):
            raise ValueError("The population supplied needs to have shape"
                             " (S, len(x)), where S > 4.")

        # scale values and clip to bounds, assigning to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)

        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        # reset population energies
        self.population_energies = np.full(self.num_population_members,
                                           np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        """
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        if np.any(np.isinf(self.population_energies)):
            return np.inf
        return (np.std(self.population_energies) /
                (np.abs(np.mean(self.population_energies)) + _MACHEPS))

    def converged(self):
        """
        Return True if the solver has converged.
        """
        if np.any(np.isinf(self.population_energies)):
            return False

        return (np.std(self.population_energies) <=
                self.atol +
                self.tol * np.abs(np.mean(self.population_energies)))

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully,
            ``message`` which describes the cause of the termination,
            ``population`` the solution vectors present in the population, and
            ``population_energies`` the value of the objective function for
            each entry in ``population``.
            See `OptimizeResult` for a description of other attributes. If
            `polish` was employed, and a lower minimum was obtained by the
            polishing, then OptimizeResult also contains the ``jac`` attribute.
            If the eventual solution does not satisfy the applied constraints
            ``success`` will be `False`.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only work out population energies for feasible solutions
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

        # do the optimization.
        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                if self._nfev > self.maxfun:
                    status_message = _status_message['maxfev']
                elif self._nfev == self.maxfun:
                    status_message = ('Maximum number of function evaluations'
                                      ' has been reached.')
                break

            if self.disp:
                print(f"differential_evolution step {nit}: f(x)="
                      f" {self.population_energies[0]}"
                      )

            if self.callback:
                c = self.tol / (self.convergence + _MACHEPS)
                res = self._result(nit=nit, message="in progress")
                res.convergence = c
                try:
                    warning_flag = bool(self.callback(res))
                except StopIteration:
                    warning_flag = True

                if warning_flag:
                    status_message = 'callback function requested stop early'

            # should the solver terminate?
            if warning_flag or self.converged():
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = self._result(
            nit=nit, message=status_message, warning_flag=warning_flag
        )

        if self.polish and not np.all(self.integrality):
            # can't polish if all the parameters are integers
            if np.any(self.integrality):
                # set the lower/upper bounds equal so that any integrality
                # constraints work.
                limits, integrality = self.limits, self.integrality
                limits[0, integrality] = DE_result.x[integrality]
                limits[1, integrality] = DE_result.x[integrality]

            polish_method = 'L-BFGS-B'

            if self._wrapped_constraints:
                polish_method = 'trust-constr'

                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.):
                    warnings.warn("differential evolution didn't find a "
                                  "solution satisfying the constraints, "
                                  "attempting to polish from the least "
                                  "infeasible solution",
                                  UserWarning, stacklevel=2)
            if self.disp:
                print(f"Polishing solution with '{polish_method}'")
            result = minimize(lambda x:
                                list(self._mapwrapper(self.func, np.atleast_2d(x)))[0],
                              np.copy(DE_result.x),
                              method=polish_method,
                              bounds=self.limits.T,
                              constraints=self.constraints)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            # Polishing solution is only accepted if there is an improvement in
            # cost function, the polishing was successful and the solution lies
            # within the bounds.
            if (result.fun < DE_result.fun and
                    result.success and
                    np.all(result.x <= self.limits[1]) and
                    np.all(self.limits[0] <= result.x)):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        if self._wrapped_constraints:
            DE_result.constr = [c.violation(DE_result.x) for
                                c in self._wrapped_constraints]
            DE_result.constr_violation = np.max(
                np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = ("The solution does not satisfy the "
                                     f"constraints, MAXCV = {DE_result.maxcv}")

        return DE_result

    def _result(self, **kwds):
        # form an intermediate OptimizeResult
        nit = kwds.get('nit', None)
        message = kwds.get('message', None)
        warning_flag = kwds.get('warning_flag', False)
        result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=message,
            success=(warning_flag is not True),
            population=self._scale_parameters(self.population),
            population_energies=self.population_energies
        )
        if self._wrapped_constraints:
            result.constr = [c.violation(result.x)
                             for c in self._wrapped_constraints]
            result.constr_violation = np.max(np.concatenate(result.constr))
            result.maxcv = result.constr_violation
            if result.maxcv > 0:
                result.success = False

        return result

    def _calculate_population_energies(self, population):
        """
        Calculate the energies of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        energies : ndarray
            An array of energies corresponding to each population member. If
            maxfun will be exceeded during this call, then the number of
            function evaluations will be reduced and energies will be
            right-padded with np.inf. Has shape ``(np.size(population, 0),)``
        """
        num_members = np.size(population, 0)
        # S is the number of function evals left to stay under the
        # maxfun budget
        S = min(num_members, self.maxfun - self._nfev)

        energies = np.full(num_members, np.inf)

        parameters_pop = self._scale_parameters(population)
        try:
            calc_energies = list(
                self._mapwrapper(self.func, parameters_pop[0:S])
            )
            calc_energies = np.squeeze(calc_energies)
        except (TypeError, ValueError) as e:
            # wrong number of arguments for _mapwrapper
            # or wrong length returned from the mapper
            raise RuntimeError(
                "The map-like callable must be of the form f(func, iterable), "
                "returning a sequence of numbers the same length as 'iterable'"
            ) from e

        if calc_energies.size != S:
            if self.vectorized:
                raise RuntimeError("The vectorized function must return an"
                                   " array of shape (S,) when given an array"
                                   " of shape (len(x), S)")
            raise RuntimeError("func(x, *args) must return a scalar value")

        energies[0:S] = calc_energies

        if self.vectorized:
            self._nfev += 1
        else:
            self._nfev += S

        return energies

    def _promote_lowest_energy(self):
        # swaps 'best solution' into first population entry

        idx = np.arange(self.num_population_members)
        feasible_solutions = idx[self.feasible]
        if feasible_solutions.size:
            # find the best feasible solution
            idx_t = np.argmin(self.population_energies[feasible_solutions])
            l = feasible_solutions[idx_t]
        else:
            # no solution was feasible, use 'best' infeasible solution, which
            # will violate constraints the least
            l = np.argmin(np.sum(self.constraint_violation, axis=1))

        self.population_energies[[0, l]] = self.population_energies[[l, 0]]
        self.population[[0, l], :] = self.population[[l, 0], :]
        self.feasible[[0, l]] = self.feasible[[l, 0]]
        self.constraint_violation[[0, l], :] = (
        self.constraint_violation[[l, 0], :])

    def _constraint_violation_fn(self, x):
        """
        Calculates total constraint violation for all the constraints, for a
        set of solutions.

        Parameters
        ----------
        x : ndarray
            Solution vector(s). Has shape (S, N), or (N,), where S is the
            number of solutions to investigate and N is the number of
            parameters.

        Returns
        -------
        cv : ndarray
            Total violation of constraints. Has shape ``(S, M)``, where M is
            the total number of constraint components (which is not necessarily
            equal to len(self._wrapped_constraints)).
        """
        # how many solution vectors you're calculating constraint violations
        # for
        S = np.size(x) // self.parameter_count
        _out = np.zeros((S, self.total_constraints))
        offset = 0
        for con in self._wrapped_constraints:
            # the input/output of the (vectorized) constraint function is
            # {(N, S), (N,)} --> (M, S)
            # The input to _constraint_violation_fn is (S, N) or (N,), so
            # transpose to pass it to the constraint. The output is transposed
            # from (M, S) to (S, M) for further use.
            c = con.violation(x.T).T

            # The shape of c should be (M,), (1, M), or (S, M). Check for
            # those shapes, as an incorrect shape indicates that the
            # user constraint function didn't return the right thing, and
            # the reshape operation will fail. Intercept the wrong shape
            # to give a reasonable error message. I'm not sure what failure
            # modes an inventive user will come up with.
            if c.shape[-1] != con.num_constr or (S > 1 and c.shape[0] != S):
                raise RuntimeError("An array returned from a Constraint has"
                                   " the wrong shape. If `vectorized is False`"
                                   " the Constraint should return an array of"
                                   " shape (M,). If `vectorized is True` then"
                                   " the Constraint must return an array of"
                                   " shape (M, S), where S is the number of"
                                   " solution vectors and M is the number of"
                                   " constraint components in a given"
                                   " Constraint object.")

            # the violation function may return a 1D array, but is it a
            # sequence of constraints for one solution (S=1, M>=1), or the
            # value of a single constraint for a sequence of solutions
            # (S>=1, M=1)
            c = np.reshape(c, (S, con.num_constr))
            _out[:, offset:offset + con.num_constr] = c
            offset += con.num_constr

        return _out

    def _calculate_population_feasibilities(self, population):
        """
        Calculate the feasibilities of a population.

        Parameters
        ----------
        population : ndarray
            An array of parameter vectors normalised to [0, 1] using lower
            and upper limits. Has shape ``(np.size(population, 0), N)``.

        Returns
        -------
        feasible, constraint_violation : ndarray, ndarray
            Boolean array of feasibility for each population member, and an
            array of the constraint violation for each population member.
            constraint_violation has shape ``(np.size(population, 0), M)``,
            where M is the number of constraints.
        """
        num_members = np.size(population, 0)
        if not self._wrapped_constraints:
            # shortcut for no constraints
            return np.ones(num_members, bool), np.zeros((num_members, 1))

        # (S, N)
        parameters_pop = self._scale_parameters(population)

        if self.vectorized:
            # (S, M)
            constraint_violation = np.array(
                self._constraint_violation_fn(parameters_pop)
            )
        else:
            # (S, 1, M)
            constraint_violation = np.array([self._constraint_violation_fn(x)
                                             for x in parameters_pop])
            # if you use the list comprehension in the line above it will
            # create an array of shape (S, 1, M), because each iteration
            # generates an array of (1, M). In comparison the vectorized
            # version returns (S, M). It's therefore necessary to remove axis 1
            constraint_violation = constraint_violation[:, 0]

        feasible = ~(np.sum(constraint_violation, axis=1) > 0)

        return feasible, constraint_violation

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._mapwrapper.__exit__(*args)

    def _accept_trial(self, energy_trial, feasible_trial, cv_trial,
                      energy_orig, feasible_orig, cv_orig):
        """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.

        This test corresponds to section III of Lampinen [1]_.

        Parameters
        ----------
        energy_trial : float
            Energy of the trial solution
        feasible_trial : float
            Feasibility of trial solution
        cv_trial : array-like
            Excess constraint violation for the trial solution
        energy_orig : float
            Energy of the original solution
        feasible_orig : float
            Feasibility of original solution
        cv_orig : array-like
            Excess constraint violation for the original solution

        Returns
        -------
        accepted : bool

        """
        if feasible_orig and feasible_trial:
            return energy_trial <= energy_orig
        elif feasible_trial and not feasible_orig:
            return True
        elif not feasible_trial and (cv_trial <= cv_orig).all():
            # cv_trial < cv_orig would imply that both trial and orig are not
            # feasible
            return True

        return False

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only need to work out population energies for those that are
            # feasible
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

        if self.dither is not None:
            self.scale = self.random_number_generator.uniform(self.dither[0],
                                                              self.dither[1])

        if self._updating == 'immediate':
            # update best solution immediately
            for candidate in range(self.num_population_members):
                if self._nfev > self.maxfun:
                    raise StopIteration

                # create a trial solution
                trial = self._mutate(candidate)

                # ensuring that it's in the range [0, 1)
                self._ensure_constraint(trial)

                # scale from [0, 1) to the actual parameter value
                parameters = self._scale_parameters(trial)

                # determine the energy of the objective function
                if self._wrapped_constraints:
                    cv = self._constraint_violation_fn(parameters)
                    feasible = False
                    energy = np.inf
                    if not np.sum(cv) > 0:
                        # solution is feasible
                        feasible = True
                        energy = self.func(parameters)
                        self._nfev += 1
                else:
                    feasible = True
                    cv = np.atleast_2d([0.])
                    energy = self.func(parameters)
                    self._nfev += 1

                # compare trial and population member
                if self._accept_trial(energy, feasible, cv,
                                      self.population_energies[candidate],
                                      self.feasible[candidate],
                                      self.constraint_violation[candidate]):
                    self.population[candidate] = trial
                    self.population_energies[candidate] = np.squeeze(energy)
                    self.feasible[candidate] = feasible
                    self.constraint_violation[candidate] = cv

                    # if the trial candidate is also better than the best
                    # solution then promote it.
                    if self._accept_trial(energy, feasible, cv,
                                          self.population_energies[0],
                                          self.feasible[0],
                                          self.constraint_violation[0]):
                        self._promote_lowest_energy()

        elif self._updating == 'deferred':
            # update best solution once per generation
            if self._nfev >= self.maxfun:
                raise StopIteration

            # 'deferred' approach, vectorised form.
            # create trial solutions
            trial_pop = self._mutate_many(
                np.arange(self.num_population_members)
            )

            # enforce bounds
            self._ensure_constraint(trial_pop)

            # determine the energies of the objective function, but only for
            # feasible trials
            feasible, cv = self._calculate_population_feasibilities(trial_pop)
            trial_energies = np.full(self.num_population_members, np.inf)

            # only calculate for feasible entries
            trial_energies[feasible] = self._calculate_population_energies(
                trial_pop[feasible])

            # which solutions are 'improved'?
            loc = [self._accept_trial(*val) for val in
                   zip(trial_energies, feasible, cv, self.population_energies,
                       self.feasible, self.constraint_violation)]
            loc = np.array(loc)
            self.population = np.where(loc[:, np.newaxis],
                                       trial_pop,
                                       self.population)
            self.population_energies = np.where(loc,
                                                trial_energies,
                                                self.population_energies)
            self.feasible = np.where(loc,
                                     feasible,
                                     self.feasible)
            self.constraint_violation = np.where(loc[:, np.newaxis],
                                                 cv,
                                                 self.constraint_violation)

            # make sure the best solution is updated if updating='deferred'.
            # put the lowest energy into the best solution position.
            self._promote_lowest_energy()

        return self.x, self.population_energies[0]

    def _scale_parameters(self, trial):
        """Scale from a number between 0 and 1 to parameters."""
        # trial either has shape (N, ) or (L, N), where L is the number of
        # solutions being scaled
        scaled = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        if np.count_nonzero(self.integrality):
            i = np.broadcast_to(self.integrality, scaled.shape)
            scaled[i] = np.round(scaled[i])
        return scaled

    def _unscale_parameters(self, parameters):
        """Scale from parameters to a number between 0 and 1."""
        return (parameters - self.__scale_arg1) * self.__recip_scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """Make sure the parameters lie between the limits."""
        mask = np.bitwise_or(trial > 1, trial < 0)
        if oob := np.count_nonzero(mask):
            trial[mask] = self.random_number_generator.uniform(size=oob)

    def _mutate_custom(self, candidate):
        rng = self.random_number_generator
        msg = (
            "strategy must have signature"
            " f(candidate: int, population: np.ndarray, rng=None) returning an"
            " array of shape (N,)"
        )
        _population = self._scale_parameters(self.population)
        if not len(np.shape(candidate)):
            # single entry in population
            trial = self.strategy(candidate, _population, rng=rng)
            if trial.shape != (self.parameter_count,):
                raise RuntimeError(msg)
        else:
            S = candidate.shape[0]
            trial = np.array(
                [self.strategy(c, _population, rng=rng) for c in candidate],
                dtype=float
            )
            if trial.shape != (S, self.parameter_count):
                raise RuntimeError(msg)
        return self._unscale_parameters(trial)

    def _mutate_many(self, candidates):
        """Create trial vectors based on a mutation strategy."""
        rng = self.random_number_generator

        S = len(candidates)
        if callable(self.strategy):
            return self._mutate_custom(candidates)

        trial = np.copy(self.population[candidates])
        samples = np.array([self._select_samples(c, 5) for c in candidates])

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidates, samples)
        else:
            bprime = self.mutation_func(samples)

        fill_point = rng_integers(rng, self.parameter_count, size=S)
        crossovers = rng.uniform(size=(S, self.parameter_count))
        crossovers = crossovers < self.cross_over_probability
        if self.strategy in self._binomial:
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            i = np.arange(S)
            crossovers[i, fill_point[i]] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            crossovers[..., 0] = True
            for j in range(S):
                i = 0
                init_fill = fill_point[j]
                while (i < self.parameter_count and crossovers[j, i]):
                    trial[j, init_fill] = bprime[j, init_fill]
                    init_fill = (init_fill + 1) % self.parameter_count
                    i += 1

            return trial

    def _mutate(self, candidate):
        """Create a trial vector based on a mutation strategy."""
        rng = self.random_number_generator

        if callable(self.strategy):
            return self._mutate_custom(candidate)

        fill_point = rng_integers(rng, self.parameter_count)
        samples = self._select_samples(candidate, 5)

        trial = np.copy(self.population[candidate])

        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, samples)
        else:
            bprime = self.mutation_func(samples)

        crossovers = rng.uniform(size=self.parameter_count)
        crossovers = crossovers < self.cross_over_probability
        if self.strategy in self._binomial:
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            crossovers[0] = True
            while i < self.parameter_count and crossovers[i]:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """best1bin, best1exp"""
        # samples.shape == (S, 5)
        # or
        # samples.shape(5,)
        r0, r1 = samples[..., :2].T
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """rand1bin, rand1exp"""
        r0, r1, r2 = samples[..., :3].T
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        """randtobest1bin, randtobest1exp"""
        r0, r1, r2 = samples[..., :3].T
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] -
                                self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """currenttobest1bin, currenttobest1exp"""
        r0, r1 = samples[..., :2].T
        bprime = (self.population[candidate] + self.scale *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        """best2bin, best2exp"""
        r0, r1, r2, r3 = samples[..., :4].T
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """rand2bin, rand2exp"""
        r0, r1, r2, r3, r4 = samples[..., :5].T
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement. You can't have the original candidate either.
        """
        self.random_number_generator.shuffle(self._random_population_index)
        idxs = self._random_population_index[:number_samples + 1]
        return idxs[idxs != candidate][:number_samples]


class _ConstraintWrapper:
    """Object to wrap/evaluate user defined constraints.

    Very similar in practice to `PreparedConstraint`, except that no evaluation
    of jac/hess is performed (explicit or implicit).

    If created successfully, it will contain the attributes listed below.

    Parameters
    ----------
    constraint : {`NonlinearConstraint`, `LinearConstraint`, `Bounds`}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables, shape (N,)

    Attributes
    ----------
    fun : callable
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.

    Notes
    -----
    _ConstraintWrapper.fun and _ConstraintWrapper.violation can get sent
    arrays of shape (N, S) or (N,), where S is the number of vectors of shape
    (N,) to consider constraints for.
    """
    def __init__(self, constraint, x0):
        self.constraint = constraint

        if isinstance(constraint, NonlinearConstraint):
            def fun(x):
                x = np.asarray(x)
                return np.atleast_1d(constraint.fun(x))
        elif isinstance(constraint, LinearConstraint):
            def fun(x):
                if issparse(constraint.A):
                    A = constraint.A
                else:
                    A = np.atleast_2d(constraint.A)

                res = A.dot(x)
                # x either has shape (N, S) or (N)
                # (M, N) x (N, S) --> (M, S)
                # (M, N) x (N,)   --> (M,)
                # However, if (M, N) is a matrix then:
                # (M, N) * (N,)   --> (M, 1), we need this to be (M,)
                if x.ndim == 1 and res.ndim == 2:
                    # deal with case that constraint.A is an np.matrix
                    # see gh20041
                    res = np.asarray(res)[:, 0]

                return res
        elif isinstance(constraint, Bounds):
            def fun(x):
                return np.asarray(x)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        self.fun = fun

        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)

        x0 = np.asarray(x0)

        # find out the number of constraints
        f0 = fun(x0)
        self.num_constr = m = f0.size
        self.parameter_count = x0.size

        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)

        self.bounds = (lb, ub)

    def __call__(self, x):
        return np.atleast_1d(self.fun(x))

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables, (N, S), where N is number of
            parameters and S is the number of solutions to be investigated.

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `_ConstraintWrapper.fun`.
            Has shape (M, S) where M is the number of constraint components.
        """
        # expect ev to have shape (num_constr, S) or (num_constr,)
        ev = self.fun(np.asarray(x))

        try:
            excess_lb = np.maximum(self.bounds[0] - ev.T, 0)
            excess_ub = np.maximum(ev.T - self.bounds[1], 0)
        except ValueError as e:
            raise RuntimeError("An array returned from a Constraint has"
                               " the wrong shape. If `vectorized is False`"
                               " the Constraint should return an array of"
                               " shape (M,). If `vectorized is True` then"
                               " the Constraint must return an array of"
                               " shape (M, S), where S is the number of"
                               " solution vectors and M is the number of"
                               " constraint components in a given"
                               " Constraint object.") from e

        v = (excess_lb + excess_ub).T
        return v
<|endoftext|>"""
=====================================================
Optimization and root finding (:mod:`scipy.optimize`)
=====================================================

.. currentmodule:: scipy.optimize

.. toctree::
   :hidden:

   optimize.cython_optimize

SciPy ``optimize`` provides functions for minimizing (or maximizing)
objective functions, possibly subject to constraints. It includes
solvers for nonlinear problems (with support for both local and global
optimization algorithms), linear programming, constrained
and nonlinear least-squares, root finding, and curve fitting.

Common functions and objects, shared across different solvers, are:

.. autosummary::
   :toctree: generated/

   show_options - Show specific options optimization solvers.
   OptimizeResult - The optimization result returned by some optimizers.
   OptimizeWarning - The optimization encountered problems.


Optimization
============

Scalar functions optimization
-----------------------------

.. autosummary::
   :toctree: generated/

   minimize_scalar - Interface for minimizers of univariate functions

The `minimize_scalar` function supports the following methods:

.. toctree::

   optimize.minimize_scalar-brent
   optimize.minimize_scalar-bounded
   optimize.minimize_scalar-golden

Local (multivariate) optimization
---------------------------------

.. autosummary::
   :toctree: generated/

   minimize - Interface for minimizers of multivariate functions.

The `minimize` function supports the following methods:

.. toctree::

   optimize.minimize-neldermead
   optimize.minimize-powell
   optimize.minimize-cg
   optimize.minimize-bfgs
   optimize.minimize-newtoncg
   optimize.minimize-lbfgsb
   optimize.minimize-tnc
   optimize.minimize-cobyla
   optimize.minimize-cobyqa
   optimize.minimize-slsqp
   optimize.minimize-trustconstr
   optimize.minimize-dogleg
   optimize.minimize-trustncg
   optimize.minimize-trustkrylov
   optimize.minimize-trustexact

Constraints are passed to `minimize` function as a single object or
as a list of objects from the following classes:

.. autosummary::
   :toctree: generated/

   NonlinearConstraint - Class defining general nonlinear constraints.
   LinearConstraint - Class defining general linear constraints.

Simple bound constraints are handled separately and there is a special class
for them:

.. autosummary::
   :toctree: generated/

   Bounds - Bound constraints.

Quasi-Newton strategies implementing `HessianUpdateStrategy`
interface can be used to approximate the Hessian in `minimize`
function (available only for the 'trust-constr' method). Available
quasi-Newton methods implementing this interface are:

.. autosummary::
   :toctree: generated/

   BFGS - Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian update strategy.
   SR1 - Symmetric-rank-1 Hessian update strategy.

.. _global_optimization:

Global optimization
-------------------

.. autosummary::
   :toctree: generated/

   basinhopping - Basinhopping stochastic optimizer.
   brute - Brute force searching optimizer.
   differential_evolution - Stochastic optimizer using differential evolution.

   shgo - Simplicial homology global optimizer.
   dual_annealing - Dual annealing stochastic optimizer.
   direct - DIRECT (Dividing Rectangles) optimizer.

Least-squares and curve fitting
===============================

Nonlinear least-squares
-----------------------

.. autosummary::
   :toctree: generated/

   least_squares - Solve a nonlinear least-squares problem with bounds on the variables.

Linear least-squares
--------------------

.. autosummary::
   :toctree: generated/

   nnls - Linear least-squares problem with non-negativity constraint.
   lsq_linear - Linear least-squares problem with bound constraints.
   isotonic_regression - Least squares problem of isotonic regression via PAVA.

Curve fitting
-------------

.. autosummary::
   :toctree: generated/

   curve_fit -- Fit curve to a set of points.

Root finding
============

Scalar functions
----------------
.. autosummary::
   :toctree: generated/

   root_scalar - Unified interface for nonlinear solvers of scalar functions.
   brentq - quadratic interpolation Brent method.
   brenth - Brent method, modified by Harris with hyperbolic extrapolation.
   ridder - Ridder's method.
   bisect - Bisection method.
   newton - Newton's method (also Secant and Halley's methods).
   toms748 - Alefeld, Potra & Shi Algorithm 748.
   RootResults - The root finding result returned by some root finders.

The `root_scalar` function supports the following methods:

.. toctree::

   optimize.root_scalar-brentq
   optimize.root_scalar-brenth
   optimize.root_scalar-bisect
   optimize.root_scalar-ridder
   optimize.root_scalar-newton
   optimize.root_scalar-toms748
   optimize.root_scalar-secant
   optimize.root_scalar-halley



The table below lists situations and appropriate methods, along with
*asymptotic* convergence rates per iteration (and per function evaluation)
for successful convergence to a simple root(*).
Bisection is the slowest of them all, adding one bit of accuracy for each
function evaluation, but is guaranteed to converge.
The other bracketing methods all (eventually) increase the number of accurate
bits by about 50% for every function evaluation.
The derivative-based methods, all built on `newton`, can converge quite quickly
if the initial value is close to the root.  They can also be applied to
functions defined on (a subset of) the complex plane.

+-------------+----------+----------+-----------+-------------+-------------+----------------+
| Domain of f | Bracket? |    Derivatives?      | Solvers     |        Convergence           |
+             +          +----------+-----------+             +-------------+----------------+
|             |          | `fprime` | `fprime2` |             | Guaranteed? |  Rate(s)(*)    |
+=============+==========+==========+===========+=============+=============+================+
| `R`         | Yes      | N/A      | N/A       | - bisection | - Yes       | - 1 "Linear"   |
|             |          |          |           | - brentq    | - Yes       | - >=1, <= 1.62 |
|             |          |          |           | - brenth    | - Yes       | - >=1, <= 1.62 |
|             |          |          |           | - ridder    | - Yes       | - 2.0 (1.41)   |
|             |          |          |           | - toms748   | - Yes       | - 2.7 (1.65)   |
+-------------+----------+----------+-----------+-------------+-------------+----------------+
| `R` or `C`  | No       | No       | No        | secant      | No          | 1.62 (1.62)    |
+-------------+----------+----------+-----------+-------------+-------------+----------------+
| `R` or `C`  | No       | Yes      | No        | newton      | No          | 2.00 (1.41)    |
+-------------+----------+----------+-----------+-------------+-------------+----------------+
| `R` or `C`  | No       | Yes      | Yes       | halley      | No          | 3.00 (1.44)    |
+-------------+----------+----------+-----------+-------------+-------------+----------------+

.. seealso::

   `scipy.optimize.cython_optimize` -- Typed Cython versions of root finding functions

Fixed point finding:

.. autosummary::
   :toctree: generated/

   fixed_point - Single-variable fixed-point solver.

Multidimensional
----------------

.. autosummary::
   :toctree: generated/

   root - Unified interface for nonlinear solvers of multivariate functions.

The `root` function supports the following methods:

.. toctree::

   optimize.root-hybr
   optimize.root-lm
   optimize.root-broyden1
   optimize.root-broyden2
   optimize.root-anderson
   optimize.root-linearmixing
   optimize.root-diagbroyden
   optimize.root-excitingmixing
   optimize.root-krylov
   optimize.root-dfsane
   
Elementwise Minimization and Root Finding
=========================================

.. toctree::
   :maxdepth: 3

   optimize.elementwise

Linear programming / MILP
=========================

.. autosummary::
   :toctree: generated/

   milp -- Mixed integer linear programming.
   linprog -- Unified interface for minimizers of linear programming problems.

The `linprog` function supports the following methods:

.. toctree::

   optimize.linprog-simplex
   optimize.linprog-interior-point
   optimize.linprog-revised_simplex
   optimize.linprog-highs-ipm
   optimize.linprog-highs-ds
   optimize.linprog-highs

The simplex, interior-point, and revised simplex methods support callback
functions, such as:

.. autosummary::
   :toctree: generated/

   linprog_verbose_callback -- Sample callback function for linprog (simplex).

Assignment problems
===================

.. autosummary::
   :toctree: generated/

   linear_sum_assignment -- Solves the linear-sum assignment problem.
   quadratic_assignment -- Solves the quadratic assignment problem.

The `quadratic_assignment` function supports the following methods:

.. toctree::

   optimize.qap-faq
   optimize.qap-2opt

Utilities
=========

Finite-difference approximation
-------------------------------

.. autosummary::
   :toctree: generated/

   approx_fprime - Approximate the gradient of a scalar function.
   check_grad - Check the supplied derivative using finite differences.


Line search
-----------

.. autosummary::
   :toctree: generated/

   bracket - Bracket a minimum, given two starting points.
   line_search - Return a step that satisfies the strong Wolfe conditions.

Hessian approximation
---------------------

.. autosummary::
   :toctree: generated/

   LbfgsInvHessProduct - Linear operator for L-BFGS approximate inverse Hessian.
   HessianUpdateStrategy - Interface for implementing Hessian update strategies

Benchmark problems
------------------

.. autosummary::
   :toctree: generated/

   rosen - The Rosenbrock function.
   rosen_der - The derivative of the Rosenbrock function.
   rosen_hess - The Hessian matrix of the Rosenbrock function.
   rosen_hess_prod - Product of the Rosenbrock Hessian with a vector.

Legacy functions
================

The functions below are not recommended for use in new scripts;
all of these methods are accessible via a newer, more consistent
interfaces, provided by the interfaces above.

Optimization
------------

General-purpose multivariate methods:

.. autosummary::
   :toctree: generated/

   fmin - Nelder-Mead Simplex algorithm.
   fmin_powell - Powell's (modified) conjugate direction method.
   fmin_cg - Non-linear (Polak-Ribiere) conjugate gradient algorithm.
   fmin_bfgs - Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno).
   fmin_ncg - Line-search Newton Conjugate Gradient.

Constrained multivariate methods:

.. autosummary::
   :toctree: generated/

   fmin_l_bfgs_b - Zhu, Byrd, and Nocedal's constrained optimizer.
   fmin_tnc - Truncated Newton code.
   fmin_cobyla - Constrained optimization by linear approximation.
   fmin_slsqp - Minimization using sequential least-squares programming.

Univariate (scalar) minimization methods:

.. autosummary::
   :toctree: generated/

   fminbound - Bounded minimization of a scalar function.
   brent - 1-D function minimization using Brent method.
   golden - 1-D function minimization using Golden Section method.

Least-squares
-------------

.. autosummary::
   :toctree: generated/

   leastsq - Minimize the sum of squares of M equations in N unknowns.

Root finding
------------

General nonlinear solvers:

.. autosummary::
   :toctree: generated/

   fsolve - Non-linear multivariable equation solver.
   broyden1 - Broyden's first method.
   broyden2 - Broyden's second method.
   NoConvergence -  Exception raised when nonlinear solver does not converge.

Large-scale nonlinear solvers:

.. autosummary::
   :toctree: generated/

   newton_krylov
   anderson

   BroydenFirst
   InverseJacobian
   KrylovJacobian

Simple iteration solvers:

.. autosummary::
   :toctree: generated/

   excitingmixing
   linearmixing
   diagbroyden

"""  # noqa: E501

from ._optimize import *
from ._minimize import *
from ._root import *
from ._root_scalar import *
from ._minpack_py import *
from ._zeros_py import *
from ._lbfgsb_py import fmin_l_bfgs_b, LbfgsInvHessProduct
from ._tnc import fmin_tnc
from ._cobyla_py import fmin_cobyla
from ._nonlin import *
from ._slsqp_py import fmin_slsqp
from ._nnls import nnls
from ._basinhopping import basinhopping
from ._linprog import linprog, linprog_verbose_callback
from ._lsap import linear_sum_assignment
from ._differentialevolution import differential_evolution
from ._lsq import least_squares, lsq_linear
from ._isotonic import isotonic_regression
from ._constraints import (NonlinearConstraint,
                           LinearConstraint,
                           Bounds)
from ._hessian_update_strategy import HessianUpdateStrategy, BFGS, SR1
from ._shgo import shgo
from ._dual_annealing import dual_annealing
from ._qap import quadratic_assignment
from ._direct_py import direct
from ._milp import milp

# Deprecated namespaces, to be removed in v2.0.0
from . import (
    cobyla, lbfgsb, linesearch, minpack, minpack2, moduleTNC, nonlin, optimize,
    slsqp, tnc, zeros
)

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'RootResults',
    'bisect',
    'brenth',
    'brentq',
    'newton',
    'ridder',
    'toms748',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="zeros",
                                   private_modules=["_zeros_py"], all=__all__,
                                   attribute=name)
<|endoftext|># TNC Python interface
# @(#) $Jeannot: tnc.py,v 1.11 2005/01/28 18:27:31 js Exp $

# Copyright (c) 2004-2005, Jean-Sebastien Roy (js@jeannot.org)

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
TNC: A Python interface to the TNC non-linear optimizer

TNC is a non-linear optimizer. To use it, you must provide a function to
minimize. The function must take one argument: the list of coordinates where to
evaluate the function; and it must return either a tuple, whose first element is the
value of the function, and whose second argument is the gradient of the function
(as a list of values); or None, to abort the minimization.
"""

from scipy.optimize import _moduleTNC as moduleTNC
from ._optimize import (MemoizeJac, OptimizeResult, _check_unknown_options,
                       _prepare_scalar_function)
from ._constraints import old_bound_to_new
from scipy._lib._array_api import array_namespace
from scipy._lib import array_api_extra as xpx

from numpy import inf, array, zeros

__all__ = ['fmin_tnc']


MSG_NONE = 0  # No messages
MSG_ITER = 1  # One line per iteration
MSG_INFO = 2  # Informational messages
MSG_VERS = 4  # Version info
MSG_EXIT = 8  # Exit reasons
MSG_ALL = MSG_ITER + MSG_INFO + MSG_VERS + MSG_EXIT

MSGS = {
        MSG_NONE: "No messages",
        MSG_ITER: "One line per iteration",
        MSG_INFO: "Informational messages",
        MSG_VERS: "Version info",
        MSG_EXIT: "Exit reasons",
        MSG_ALL: "All messages"
}

INFEASIBLE = -1  # Infeasible (lower bound > upper bound)
LOCALMINIMUM = 0  # Local minimum reached (|pg| ~= 0)
FCONVERGED = 1  # Converged (|f_n-f_(n-1)| ~= 0)
XCONVERGED = 2  # Converged (|x_n-x_(n-1)| ~= 0)
MAXFUN = 3  # Max. number of function evaluations reached
LSFAIL = 4  # Linear search failed
CONSTANT = 5  # All lower bounds are equal to the upper bounds
NOPROGRESS = 6  # Unable to progress
USERABORT = 7  # User requested end of minimization

RCSTRINGS = {
        INFEASIBLE: "Infeasible (lower bound > upper bound)",
        LOCALMINIMUM: "Local minimum reached (|pg| ~= 0)",
        FCONVERGED: "Converged (|f_n-f_(n-1)| ~= 0)",
        XCONVERGED: "Converged (|x_n-x_(n-1)| ~= 0)",
        MAXFUN: "Max. number of function evaluations reached",
        LSFAIL: "Linear search failed",
        CONSTANT: "All lower bounds are equal to the upper bounds",
        NOPROGRESS: "Unable to progress",
        USERABORT: "User requested end of minimization"
}

# Changes to interface made by Travis Oliphant, Apr. 2004 for inclusion in
#  SciPy


def fmin_tnc(func, x0, fprime=None, args=(), approx_grad=0,
             bounds=None, epsilon=1e-8, scale=None, offset=None,
             messages=MSG_ALL, maxCGit=-1, maxfun=None, eta=-1,
             stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1,
             rescale=-1, disp=None, callback=None):
    r"""
    Minimize a function with variables subject to bounds, using
    gradient information in a truncated Newton algorithm. This
    method wraps a C implementation of the algorithm.

    Parameters
    ----------
    func : callable ``func(x, *args)``
        Function to minimize.  Must do one of:

        1. Return f and g, where f is the value of the function and g its
           gradient (a list of floats).

        2. Return the function value but supply gradient function
           separately as `fprime`.

        3. Return the function value and set ``approx_grad=True``.

        If the function returns None, the minimization
        is aborted.
    x0 : array_like
        Initial estimate of minimum.
    fprime : callable ``fprime(x, *args)``, optional
        Gradient of `func`. If None, then either `func` must return the
        function value and the gradient (``f,g = func(x, *args)``)
        or `approx_grad` must be True.
    args : tuple, optional
        Arguments to pass to function.
    approx_grad : bool, optional
        If true, approximate the gradient numerically.
    bounds : list, optional
        (min, max) pairs for each element in x0, defining the
        bounds on that parameter. Use None or +/-inf for one of
        min or max when there is no bound in that direction.
    epsilon : float, optional
        Used if approx_grad is True. The stepsize in a finite
        difference approximation for fprime.
    scale : array_like, optional
        Scaling factors to apply to each variable. If None, the
        factors are up-low for interval bounded variables and
        1+|x| for the others. Defaults to None.
    offset : array_like, optional
        Value to subtract from each variable. If None, the
        offsets are (up+low)/2 for interval bounded variables
        and x for the others.
    messages : int, optional
        Bit mask used to select messages display during
        minimization values defined in the MSGS dict. Defaults to
        MGS_ALL.
    disp : int, optional
        Integer interface to messages. 0 = no message, 5 = all messages
    maxCGit : int, optional
        Maximum number of hessian*vector evaluations per main
        iteration. If maxCGit == 0, the direction chosen is
        -gradient if maxCGit < 0, maxCGit is set to
        max(1,min(50,n/2)). Defaults to -1.
    maxfun : int, optional
        Maximum number of function evaluation. If None, maxfun is
        set to max(100, 10*len(x0)). Defaults to None. Note that this function
        may violate the limit because of evaluating gradients by numerical
        differentiation.
    eta : float, optional
        Severity of the line search. If < 0 or > 1, set to 0.25.
        Defaults to -1.
    stepmx : float, optional
        Maximum step for the line search. May be increased during
        call. If too small, it will be set to 10.0. Defaults to 0.
    accuracy : float, optional
        Relative precision for finite difference calculations. If
        <= machine_precision, set to sqrt(machine_precision).
        Defaults to 0.
    fmin : float, optional
        Minimum function value estimate. Defaults to 0.
    ftol : float, optional
        Precision goal for the value of f in the stopping criterion.
        If ftol < 0.0, ftol is set to 0.0 defaults to -1.
    xtol : float, optional
        Precision goal for the value of x in the stopping
        criterion (after applying x scaling factors). If xtol <
        0.0, xtol is set to sqrt(machine_precision). Defaults to
        -1.
    pgtol : float, optional
        Precision goal for the value of the projected gradient in
        the stopping criterion (after applying x scaling factors).
        If pgtol < 0.0, pgtol is set to 1e-2 * sqrt(accuracy).
        Setting it to 0.0 is not recommended. Defaults to -1.
    rescale : float, optional
        Scaling factor (in log10) used to trigger f value
        rescaling. If 0, rescale at each iteration. If a large
        value, never rescale. If < 0, rescale is set to 1.3.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.

    Returns
    -------
    x : ndarray
        The solution.
    nfeval : int
        The number of function evaluations.
    rc : int
        Return code, see below

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'TNC' `method` in particular.

    Notes
    -----
    The underlying algorithm is truncated Newton, also called
    Newton Conjugate-Gradient. This method differs from
    scipy.optimize.fmin_ncg in that

    1. it wraps a C implementation of the algorithm
    2. it allows each variable to be given an upper and lower bound.

    The algorithm incorporates the bound constraints by determining
    the descent direction as in an unconstrained truncated Newton,
    but never taking a step-size large enough to leave the space
    of feasible x's. The algorithm keeps track of a set of
    currently active constraints, and ignores them when computing
    the minimum allowable step size. (The x's associated with the
    active constraint are kept fixed.) If the maximum allowable
    step size is zero then a new constraint is added. At the end
    of each iteration one of the constraints may be deemed no
    longer active and removed. A constraint is considered
    no longer active is if it is currently active
    but the gradient for that variable points inward from the
    constraint. The specific constraint removed is the one
    associated with the variable of largest index whose
    constraint is no longer active.

    Return codes are defined as follows:

    - ``-1`` : Infeasible (lower bound > upper bound)
    - ``0`` : Local minimum reached (:math:`|pg| \approx 0`)
    - ``1`` : Converged (:math:`|f_n-f_(n-1)| \approx 0`)
    - ``2`` : Converged (:math:`|x_n-x_(n-1)| \approx 0`)
    - ``3`` : Max. number of function evaluations reached
    - ``4`` : Linear search failed
    - ``5`` : All lower bounds are equal to the upper bounds
    - ``6`` : Unable to progress
    - ``7`` : User requested end of minimization

    References
    ----------
    Wright S., Nocedal J. (2006), 'Numerical Optimization'

    Nash S.G. (1984), "Newton-Type Minimization Via the Lanczos Method",
    SIAM Journal of Numerical Analysis 21, pp. 770-778

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

    if disp is not None:  # disp takes precedence over messages
        mesg_num = disp
    else:
        mesg_num = {0:MSG_NONE, 1:MSG_ITER, 2:MSG_INFO, 3:MSG_VERS,
                    4:MSG_EXIT, 5:MSG_ALL}.get(messages, MSG_ALL)
    # build options
    opts = {'eps': epsilon,
            'scale': scale,
            'offset': offset,
            'mesg_num': mesg_num,
            'maxCGit': maxCGit,
            'maxfun': maxfun,
            'eta': eta,
            'stepmx': stepmx,
            'accuracy': accuracy,
            'minfev': fmin,
            'ftol': ftol,
            'xtol': xtol,
            'gtol': pgtol,
            'rescale': rescale,
            'disp': False}

    res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback, **opts)

    return res['x'], res['nfev'], res['status']


def _minimize_tnc(fun, x0, args=(), jac=None, bounds=None,
                  eps=1e-8, scale=None, offset=None, mesg_num=None,
                  maxCGit=-1, eta=-1, stepmx=0, accuracy=0,
                  minfev=0, ftol=-1, xtol=-1, gtol=-1, rescale=-1, disp=False,
                  callback=None, finite_diff_rel_step=None, maxfun=None,
                  **unknown_options):
    """
    Minimize a scalar function of one or more variables using a truncated
    Newton (TNC) algorithm.

    Options
    -------
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    scale : list of floats
        Scaling factors to apply to each variable. If None, the
        factors are up-low for interval bounded variables and
        1+|x] for the others. Defaults to None.
    offset : float
        Value to subtract from each variable. If None, the
        offsets are (up+low)/2 for interval bounded variables
        and x for the others.
    disp : bool
       Set to True to print convergence messages.
    maxCGit : int
        Maximum number of hessian*vector evaluations per main
        iteration. If maxCGit == 0, the direction chosen is
        -gradient if maxCGit < 0, maxCGit is set to
        max(1,min(50,n/2)). Defaults to -1.
    eta : float
        Severity of the line search. If < 0 or > 1, set to 0.25.
        Defaults to -1.
    stepmx : float
        Maximum step for the line search. May be increased during
        call. If too small, it will be set to 10.0. Defaults to 0.
    accuracy : float
        Relative precision for finite difference calculations. If
        <= machine_precision, set to sqrt(machine_precision).
        Defaults to 0.
    minfev : float
        Minimum function value estimate. Defaults to 0.
    ftol : float
        Precision goal for the value of f in the stopping criterion.
        If ftol < 0.0, ftol is set to 0.0 defaults to -1.
    xtol : float
        Precision goal for the value of x in the stopping
        criterion (after applying x scaling factors). If xtol <
        0.0, xtol is set to sqrt(machine_precision). Defaults to
        -1.
    gtol : float
        Precision goal for the value of the projected gradient in
        the stopping criterion (after applying x scaling factors).
        If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
        Setting it to 0.0 is not recommended. Defaults to -1.
    rescale : float
        Scaling factor (in log10) used to trigger f value
        rescaling.  If 0, rescale at each iteration.  If a large
        value, never rescale.  If < 0, rescale is set to 1.3.
    finite_diff_rel_step : None or array_like, optional
        If ``jac in ['2-point', '3-point', 'cs']`` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    maxfun : int
        Maximum number of function evaluations. If None, `maxfun` is
        set to max(100, 10*len(x0)). Defaults to None.
    """
    _check_unknown_options(unknown_options)
    fmin = minfev
    pgtol = gtol

    xp = array_namespace(x0)
    x0 = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
    dtype = xp.float64
    if xp.isdtype(x0.dtype, "real floating"):
        dtype = x0.dtype
    x0 = xp.reshape(xp.astype(x0, dtype), -1)

    n = len(x0)

    if bounds is None:
        bounds = [(None,None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    new_bounds = old_bound_to_new(bounds)

    if mesg_num is not None:
        messages = {0:MSG_NONE, 1:MSG_ITER, 2:MSG_INFO, 3:MSG_VERS,
                    4:MSG_EXIT, 5:MSG_ALL}.get(mesg_num, MSG_ALL)
    elif disp:
        messages = MSG_ALL
    else:
        messages = MSG_NONE

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step,
                                  bounds=new_bounds)
    func_and_grad = sf.fun_and_grad

    """
    low, up   : the bounds (lists of floats)
                if low is None, the lower bounds are removed.
                if up is None, the upper bounds are removed.
                low and up defaults to None
    """
    low = zeros(n)
    up = zeros(n)
    for i in range(n):
        if bounds[i] is None:
            l, u = -inf, inf
        else:
            l,u = bounds[i]
            if l is None:
                low[i] = -inf
            else:
                low[i] = l
            if u is None:
                up[i] = inf
            else:
                up[i] = u

    if scale is None:
        scale = array([])

    if offset is None:
        offset = array([])

    if maxfun is None:
        maxfun = max(100, 10*len(x0))

    rc, nf, nit, x, funv, jacv = moduleTNC.tnc_minimize(
        func_and_grad, x0, low, up, scale,
        offset, messages, maxCGit, maxfun,
        eta, stepmx, accuracy, fmin, ftol,
        xtol, pgtol, rescale, callback
    )
    # the TNC documentation states: "On output, x, f and g may be very
    # slightly out of sync because of scaling". Therefore re-evaluate
    # func_and_grad so they are synced.
    funv, jacv = func_and_grad(x)

    return OptimizeResult(x=x, fun=funv, jac=jacv, nfev=sf.nfev,
                          nit=nit, status=rc, message=RCSTRINGS[rc],
                          success=(-1 < rc < 3))
<|endoftext|>import warnings
import numpy as np
from scipy.sparse import csc_array, vstack, issparse
from scipy._lib._util import VisibleDeprecationWarning
from ._highspy._highs_wrapper import _highs_wrapper  # type: ignore[import-not-found,import-untyped]
from ._constraints import LinearConstraint, Bounds
from ._optimize import OptimizeResult
from ._linprog_highs import _highs_to_scipy_status_message


def _constraints_to_components(constraints):
    """
    Convert sequence of constraints to a single set of components A, b_l, b_u.

    `constraints` could be

    1. A LinearConstraint
    2. A tuple representing a LinearConstraint
    3. An invalid object
    4. A sequence of composed entirely of objects of type 1/2
    5. A sequence containing at least one object of type 3

    We want to accept 1, 2, and 4 and reject 3 and 5.
    """
    message = ("`constraints` (or each element within `constraints`) must be "
               "convertible into an instance of "
               "`scipy.optimize.LinearConstraint`.")
    As = []
    b_ls = []
    b_us = []

    # Accept case 1 by standardizing as case 4
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    else:
        # Reject case 3
        try:
            iter(constraints)
        except TypeError as exc:
            raise ValueError(message) from exc

        # Accept case 2 by standardizing as case 4
        if len(constraints) == 3:
            # argument could be a single tuple representing a LinearConstraint
            try:
                constraints = [LinearConstraint(*constraints)]
            except (TypeError, ValueError, VisibleDeprecationWarning):
                # argument was not a tuple representing a LinearConstraint
                pass

    # Address cases 4/5
    for constraint in constraints:
        # if it's not a LinearConstraint or something that represents a
        # LinearConstraint at this point, it's invalid
        if not isinstance(constraint, LinearConstraint):
            try:
                constraint = LinearConstraint(*constraint)
            except TypeError as exc:
                raise ValueError(message) from exc
        As.append(csc_array(constraint.A))
        b_ls.append(np.atleast_1d(constraint.lb).astype(np.float64))
        b_us.append(np.atleast_1d(constraint.ub).astype(np.float64))

    if len(As) > 1:
        A = vstack(As, format="csc")
        b_l = np.concatenate(b_ls)
        b_u = np.concatenate(b_us)
    else:  # avoid unnecessary copying
        A = As[0]
        b_l = b_ls[0]
        b_u = b_us[0]

    return A, b_l, b_u


def _milp_iv(c, integrality, bounds, constraints, options):
    # objective IV
    if issparse(c):
        raise ValueError("`c` must be a dense array.")
    c = np.atleast_1d(c).astype(np.float64)
    if c.ndim != 1 or c.size == 0 or not np.all(np.isfinite(c)):
        message = ("`c` must be a one-dimensional array of finite numbers "
                   "with at least one element.")
        raise ValueError(message)

    # integrality IV
    if issparse(integrality):
        raise ValueError("`integrality` must be a dense array.")
    message = ("`integrality` must contain integers 0-3 and be broadcastable "
               "to `c.shape`.")
    if integrality is None:
        integrality = 0
    try:
        integrality = np.broadcast_to(integrality, c.shape).astype(np.uint8)
    except ValueError:
        raise ValueError(message)
    if integrality.min() < 0 or integrality.max() > 3:
        raise ValueError(message)

    # bounds IV
    if bounds is None:
        bounds = Bounds(0, np.inf)
    elif not isinstance(bounds, Bounds):
        message = ("`bounds` must be convertible into an instance of "
                   "`scipy.optimize.Bounds`.")
        try:
            bounds = Bounds(*bounds)
        except TypeError as exc:
            raise ValueError(message) from exc

    try:
        lb = np.broadcast_to(bounds.lb, c.shape).astype(np.float64)
        ub = np.broadcast_to(bounds.ub, c.shape).astype(np.float64)
    except (ValueError, TypeError) as exc:
        message = ("`bounds.lb` and `bounds.ub` must contain reals and "
                   "be broadcastable to `c.shape`.")
        raise ValueError(message) from exc

    # constraints IV
    if not constraints:
        constraints = [LinearConstraint(np.empty((0, c.size)),
                                        np.empty((0,)), np.empty((0,)))]
    try:
        A, b_l, b_u = _constraints_to_components(constraints)
    except ValueError as exc:
        message = ("`constraints` (or each element within `constraints`) must "
                   "be convertible into an instance of "
                   "`scipy.optimize.LinearConstraint`.")
        raise ValueError(message) from exc

    if A.shape != (b_l.size, c.size):
        message = "The shape of `A` must be (len(b_l), len(c))."
        raise ValueError(message)
    indptr, indices, data = A.indptr, A.indices, A.data.astype(np.float64)

    # options IV
    options = options or {}
    supported_options = {'disp', 'presolve', 'time_limit', 'node_limit',
                         'mip_rel_gap'}
    unsupported_options = set(options).difference(supported_options)
    if unsupported_options:
        message = (f"Unrecognized options detected: {unsupported_options}. "
                   "These will be passed to HiGHS verbatim.")
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    options_iv = {'log_to_console': options.pop("disp", False),
                  'mip_max_nodes': options.pop("node_limit", None)}
    options_iv.update(options)

    return c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options_iv


def milp(c, *, integrality=None, bounds=None, constraints=None, options=None):
    r"""
    Mixed-integer linear programming

    Solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & b_l \leq A x \leq b_u,\\
        & l \leq x \leq u, \\
        & x_i \in \mathbb{Z}, i \in X_i

    where :math:`x` is a vector of decision variables;
    :math:`c`, :math:`b_l`, :math:`b_u`, :math:`l`, and :math:`u` are vectors;
    :math:`A` is a matrix, and :math:`X_i` is the set of indices of
    decision variables that must be integral. (In this context, a
    variable that can assume only integer values is said to be "integral";
    it has an "integrality" constraint.)

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        b_l <= A @ x <= b_u
        l <= x <= u
        Specified elements of x must be integers

    By default, ``l = 0`` and ``u = np.inf`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1D dense array_like
        The coefficients of the linear objective function to be minimized.
        `c` is converted to a double precision array before the problem is
        solved.
    integrality : 1D dense array_like, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous. `integrality` is converted
        to an array of integers before the problem is solved.

    bounds : scipy.optimize.Bounds, optional
        Bounds on the decision variables. Lower and upper bounds are converted
        to double precision arrays before the problem is solved. The
        ``keep_feasible`` parameter of the `Bounds` object is ignored. If
        not specified, all decision variables are constrained to be
        non-negative.
    constraints : sequence of scipy.optimize.LinearConstraint, optional
        Linear constraints of the optimization problem. Arguments may be
        one of the following:

        1. A single `LinearConstraint` object
        2. A single tuple that can be converted to a `LinearConstraint` object
           as ``LinearConstraint(*constraints)``
        3. A sequence composed entirely of objects of type 1. and 2.

        Before the problem is solved, all values are converted to double
        precision, and the matrices of constraint coefficients are converted to
        instances of `scipy.sparse.csc_array`. The ``keep_feasible`` parameter
        of `LinearConstraint` objects is ignored.
    options : dict, optional
        A dictionary of solver options. The following keys are recognized.

        disp : bool (default: ``False``)
            Set to ``True`` if indicators of optimization status are to be
            printed to the console during optimization.
        node_limit : int, optional
            The maximum number of nodes (linear program relaxations) to solve
            before stopping. Default is no maximum number of nodes.
        presolve : bool (default: ``True``)
            Presolve attempts to identify trivial infeasibilities,
            identify trivial unboundedness, and simplify the problem before
            sending it to the main solver.
        time_limit : float, optional
            The maximum number of seconds allotted to solve the problem.
            Default is no time limit.
        mip_rel_gap : float, optional
            Termination criterion for MIP solver: solver will terminate when
            the gap between the primal objective value and the dual objective
            bound, scaled by the primal objective value, is <= mip_rel_gap.

    Returns
    -------
    res : OptimizeResult
        An instance of :class:`scipy.optimize.OptimizeResult`. The object
        is guaranteed to have the following attributes.

        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimal solution found.

            ``1`` : Iteration or time limit reached.

            ``2`` : Problem is infeasible.

            ``3`` : Problem is unbounded.

            ``4`` : Other; see message for details.

        success : bool
            ``True`` when an optimal solution is found and ``False`` otherwise.

        message : str
            A string descriptor of the exit status of the algorithm.

        The following attributes will also be present, but the values may be
        ``None``, depending on the solution status.

        x : ndarray
            The values of the decision variables that minimize the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        mip_node_count : int
            The number of subproblems or "nodes" solved by the MILP solver.
        mip_dual_bound : float
            The MILP solver's final estimate of the lower bound on the optimal
            solution.
        mip_gap : float
            The difference between the primal objective value and the dual
            objective bound, scaled by the primal objective value.

    Notes
    -----
    `milp` is a wrapper of the HiGHS linear optimization software [1]_. The
    algorithm is deterministic, and it typically finds the global optimum of
    moderately challenging mixed-integer linear programs (when it exists).

    References
    ----------
    .. [1] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [2] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5

    Examples
    --------
    Consider the problem at
    https://en.wikipedia.org/wiki/Integer_programming#Example, which is
    expressed as a maximization problem of two variables. Since `milp` requires
    that the problem be expressed as a minimization problem, the objective
    function coefficients on the decision variables are:

    >>> import numpy as np
    >>> c = -np.array([0, 1])

    Note the negative sign: we maximize the original objective function
    by minimizing the negative of the objective function.

    We collect the coefficients of the constraints into arrays like:

    >>> A = np.array([[-1, 1], [3, 2], [2, 3]])
    >>> b_u = np.array([1, 12, 12])
    >>> b_l = np.full_like(b_u, -np.inf, dtype=float)

    Because there is no lower limit on these constraints, we have defined a
    variable ``b_l`` full of values representing negative infinity. This may
    be unfamiliar to users of `scipy.optimize.linprog`, which only accepts
    "less than" (or "upper bound") inequality constraints of the form
    ``A_ub @ x <= b_u``. By accepting both ``b_l`` and ``b_u`` of constraints
    ``b_l <= A_ub @ x <= b_u``, `milp` makes it easy to specify "greater than"
    inequality constraints, "less than" inequality constraints, and equality
    constraints concisely.

    These arrays are collected into a single `LinearConstraint` object like:

    >>> from scipy.optimize import LinearConstraint
    >>> constraints = LinearConstraint(A, b_l, b_u)

    The non-negativity bounds on the decision variables are enforced by
    default, so we do not need to provide an argument for `bounds`.

    Finally, the problem states that both decision variables must be integers:

    >>> integrality = np.ones_like(c)

    We solve the problem like:

    >>> from scipy.optimize import milp
    >>> res = milp(c=c, constraints=constraints, integrality=integrality)
    >>> res.x
    [2.0, 2.0]

    Note that had we solved the relaxed problem (without integrality
    constraints):

    >>> res = milp(c=c, constraints=constraints)  # OR:
    >>> # from scipy.optimize import linprog; res = linprog(c, A, b_u)
    >>> res.x
    [1.8, 2.8]

    we would not have obtained the correct solution by rounding to the nearest
    integers.

    Other examples are given :ref:`in the tutorial <tutorial-optimize_milp>`.

    """
    args_iv = _milp_iv(c, integrality, bounds, constraints, options)
    c, integrality, lb, ub, indptr, indices, data, b_l, b_u, options = args_iv

    highs_res = _highs_wrapper(c, indptr, indices, data, b_l, b_u,
                               lb, ub, integrality, options)

    res = {}

    # Convert to scipy-style status and message
    highs_status = highs_res.get('status', None)
    highs_message = highs_res.get('message', None)
    status, message = _highs_to_scipy_status_message(highs_status,
                                                     highs_message)
    res['status'] = status
    res['message'] = message
    res['success'] = (status == 0)
    x = highs_res.get('x', None)
    res['x'] = np.array(x) if x is not None else None
    res['fun'] = highs_res.get('fun', None)
    res['mip_node_count'] = highs_res.get('mip_node_count', None)
    res['mip_dual_bound'] = highs_res.get('mip_dual_bound', None)
    res['mip_gap'] = highs_res.get('mip_gap', None)

    return OptimizeResult(res)
<|endoftext|>"""Newton-CG trust-region optimization."""
import math

import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

__all__ = []


def _minimize_trust_ncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                        **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the Newton conjugate gradient trust-region algorithm.

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
        raise ValueError('Jacobian is required for Newton-CG trust-region '
                         'minimization')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Newton-CG trust-region minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  hessp=hessp, subproblem=CGSteihaugSubproblem,
                                  **trust_region_options)


class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""
    def solve(self, trust_radius):
        """
        Solve the subproblem using a conjugate gradient method.

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
        This is algorithm (7.2) of Nocedal and Wright 2nd edition.
        Only the function that computes the Hessian-vector product is required.
        The Hessian itself is not required, and the Hessian does
        not need to be positive semidefinite.
        """

        # get the norm of jacobian and define the origin
        p_origin = np.zeros_like(self.jac)

        # define a default tolerance
        tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.jac_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.jac
        d = -r

        # Search for the min of the approximation of the objective function.
        while True:

            # do an iteration
            Bd = self.hessp(d)
            dBd = np.dot(d, Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = np.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if scipy.linalg.norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = np.dot(r_next, r_next)
            if math.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next
<|endoftext|>"""
Unified interfaces to minimization algorithms.

Functions
---------
- minimize : minimization of a function of several variables.
- minimize_scalar : minimization of a function of one variable.
"""

__all__ = ['minimize', 'minimize_scalar']


from warnings import warn

import numpy as np

# unconstrained minimization
from ._optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
                        _minimize_bfgs, _minimize_newtoncg,
                        _minimize_scalar_brent, _minimize_scalar_bounded,
                        _minimize_scalar_golden, MemoizeJac, OptimizeResult,
                        _wrap_callback, _recover_from_bracket_error)
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr

# constrained minimization
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._cobyqa_py import _minimize_cobyqa
from ._slsqp_py import _minimize_slsqp
from ._constraints import (old_bound_to_new, new_bounds_to_old,
                           old_constraint_to_new, new_constraint_to_old,
                           NonlinearConstraint, LinearConstraint, Bounds,
                           PreparedConstraint)
from ._differentiable_functions import FD_METHODS

MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                    'trust-krylov']

# These methods support the new callback interface (passed an OptimizeResult)
MINIMIZE_METHODS_NEW_CB = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                           'l-bfgs-b', 'trust-constr', 'dogleg', 'trust-ncg',
                           'trust-exact', 'trust-krylov', 'cobyqa']

MINIMIZE_SCALAR_METHODS = ['brent', 'bounded', 'golden']

def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    """Minimization of scalar function of one or more variables.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized::

            fun(x, *args) -> float

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        Type of solver.  Should be one of

        - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
        - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
        - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
        - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
        - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
        - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
        - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
        - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
        - 'COBYQA'      :ref:`(see here) <optimize.minimize-cobyqa>`
        - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
        - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
        - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
        - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
        - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
        - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
        - custom - a callable object, see below for description.

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending on whether or not the problem has constraints or bounds.
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
        trust-exact and trust-constr.
        If it is a callable, it should be a function that returns the gradient
        vector::

            jac(x, *args) -> array_like, shape (n,)

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters. If `jac` is a Boolean and is True, `fun` is
        assumed to return a tuple ``(f, g)`` containing the objective
        function and the gradient.
        Methods 'Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact', and
        'trust-krylov' require that either a callable be supplied, or that
        `fun` return the objective and gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
        trust-ncg, trust-krylov, trust-exact and trust-constr.
        If it is callable, it should return the Hessian matrix::

            hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)

        where ``x`` is a (n,) ndarray and ``args`` is a tuple with the fixed
        parameters.
        The keywords {'2-point', '3-point', 'cs'} can also be used to select
        a finite difference scheme for numerical estimation of the hessian.
        Alternatively, objects implementing the `HessianUpdateStrategy`
        interface can be used to approximate the Hessian. Available
        quasi-Newton methods implementing this interface are:

        - `BFGS`
        - `SR1`

        Not all of the options are available for each of the methods; for
        availability refer to the notes.
    hessp : callable, optional
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, trust-ncg, trust-krylov, trust-constr.
        Only one of `hessp` or `hess` needs to be given. If `hess` is
        provided, then `hessp` will be ignored. `hessp` must compute the
        Hessian times an arbitrary vector::

            hessp(x, p, *args) ->  ndarray shape (n,)

        where ``x`` is a (n,) ndarray, ``p`` is an arbitrary vector with
        dimension (n,) and ``args`` is a tuple with the fixed
        parameters.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
        trust-constr, COBYLA, and COBYQA methods. There are two ways to specify
        the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`. None
           is used to specify no bound.

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.

        Constraints for 'trust-constr' and 'cobyqa' are defined as a single object
        or a list of objects specifying constraints to the optimization problem.
        Available constraints are:

        - `LinearConstraint`
        - `NonlinearConstraint`

        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
        Each dictionary with fields:

        type : str
            Constraint type: 'eq' for equality, 'ineq' for inequality.
        fun : callable
            The function defining the constraint.
        jac : callable, optional
            The Jacobian of `fun` (only for SLSQP).
