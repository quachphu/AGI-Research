            c0, x, revstack, complete, status, message)


def _parse_linprog(lp, options, meth):
    """
    Parse the provided linear programming problem

    ``_parse_linprog`` employs two main steps ``_check_sparse_inputs`` and
    ``_clean_inputs``. ``_check_sparse_inputs`` checks for sparsity in the
    provided constraints (``A_ub`` and ``A_eq) and if these match the provided
    sparsity optional values.

    ``_clean inputs`` checks of the provided inputs. If no violations are
    identified the objective vector, upper bound constraints, equality
    constraints, and simple bounds are returned in the expected format.

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

    options : dict
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.

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

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options('linprog')`.

    """
    if options is None:
        options = {}

    solver_options = {k: v for k, v in options.items()}
    solver_options, A_ub, A_eq = _check_sparse_inputs(solver_options, meth,
                                                      lp.A_ub, lp.A_eq)
    # Convert lists to numpy arrays, etc...
    lp = _clean_inputs(lp._replace(A_ub=A_ub, A_eq=A_eq))
    return lp, solver_options


def _get_Abc(lp, c0):
    """
    Given a linear programming problem of the form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    Return the problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    by adding slack variables and making variable substitutions as necessary.

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
            The bounds of ``x``, lower bounds in the 1st column, upper
            bounds in the 2nd column. The bounds are possibly tightened
            by the presolve procedure.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.

    Returns
    -------
    A : 2-D array
        2-D array such that ``A`` @ ``x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x0 : 1-D array
        Starting values of the independent variables, which will be refined by
        the optimization algorithm

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.

    """
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp

    if sps.issparse(A_eq):
        sparse = True
        A_eq = sps.csr_matrix(A_eq)
        A_ub = sps.csr_matrix(A_ub)

        def hstack(blocks):
            return sps.hstack(blocks, format="csr")

        def vstack(blocks):
            return sps.vstack(blocks, format="csr")

        zeros = sps.csr_matrix
        eye = sps.eye
    else:
        sparse = False
        hstack = np.hstack
        vstack = np.vstack
        zeros = np.zeros
        eye = np.eye

    # Variables lbs and ubs (see below) may be changed, which feeds back into
    # bounds, so copy.
    bounds = np.array(bounds, copy=True)

    # modify problem such that all variables have only non-negativity bounds
    lbs = bounds[:, 0]
    ubs = bounds[:, 1]
    m_ub, n_ub = A_ub.shape

    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)

    # unbounded below: substitute xi = -xi' (unbounded above)
    # if -inf <= xi <= ub, then -ub <= -xi <= inf, so swap and invert bounds
    l_nolb_someub = np.logical_and(lb_none, ub_some)
    i_nolb = np.nonzero(l_nolb_someub)[0]
    lbs[l_nolb_someub], ubs[l_nolb_someub] = (
        -ubs[l_nolb_someub], -lbs[l_nolb_someub])
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    c[i_nolb] *= -1
    if x0 is not None:
        x0[i_nolb] *= -1
    if len(i_nolb) > 0:
        if A_ub.shape[0] > 0:  # sometimes needed for sparse arrays... weird
            A_ub[:, i_nolb] *= -1
        if A_eq.shape[0] > 0:
            A_eq[:, i_nolb] *= -1

    # upper bound: add inequality constraint
    i_newub, = ub_some.nonzero()
    ub_newub = ubs[ub_some]
    n_bounds = len(i_newub)
    if n_bounds > 0:
        shape = (n_bounds, A_ub.shape[1])
        if sparse:
            idxs = (np.arange(n_bounds), i_newub)
            A_ub = vstack((A_ub, sps.csr_matrix((np.ones(n_bounds), idxs),
                                                shape=shape)))
        else:
            A_ub = vstack((A_ub, np.zeros(shape)))
            A_ub[np.arange(m_ub, A_ub.shape[0]), i_newub] = 1
        b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
        b_ub[m_ub:] = ub_newub

    A1 = vstack((A_ub, A_eq))
    b = np.concatenate((b_ub, b_eq))
    c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))
    # unbounded: substitute xi = xi+ + xi-
    l_free = np.logical_and(lb_none, ub_none)
    i_free = np.nonzero(l_free)[0]
    n_free = len(i_free)
    c = np.concatenate((c, np.zeros(n_free)))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros(n_free)))
    A1 = hstack((A1[:, :n_ub], -A1[:, i_free]))
    c[n_ub:n_ub+n_free] = -c[i_free]
    if x0 is not None:
        i_free_neg = x0[i_free] < 0
        x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        x0[i_free[i_free_neg]] = 0

    # add slack variables
    A2 = vstack([eye(A_ub.shape[0]), zeros((A_eq.shape[0], A_ub.shape[0]))])

    A = hstack([A1, A2])

    # lower bound: substitute xi = xi' + lb
    # now there is a constant term in objective
    i_shift = np.nonzero(lb_some)[0]
    lb_shift = lbs[lb_some].astype(float)
    c0 += np.sum(lb_shift * c[i_shift])
    if sparse:
        b = b.reshape(-1, 1)
        A = A.tocsc()
        b -= (A[:, i_shift] @ sps.diags(lb_shift)).sum(axis=1)
        b = b.ravel()
    else:
        b -= (A[:, i_shift] * lb_shift).sum(axis=1)
    if x0 is not None:
        x0[i_shift] -= lb_shift

    return A, b, c, c0, x0


def _round_to_power_of_two(x):
    """
    Round elements of the array to the nearest power of two.
    """
    return 2**np.around(np.log2(x))


def _autoscale(A, b, c, x0):
    """
    Scales the problem according to equilibration from [12].
    Also normalizes the right hand side vector by its maximum element.
    """
    m, n = A.shape

    C = 1
    R = 1

    if A.size > 0:

        R = np.max(np.abs(A), axis=1)
        if sps.issparse(A):
            R = R.toarray().flatten()
        R[R == 0] = 1
        R = 1/_round_to_power_of_two(R)
        A = sps.diags(R)@A if sps.issparse(A) else A*R.reshape(m, 1)
        b = b*R

        C = np.max(np.abs(A), axis=0)
        if sps.issparse(A):
            C = C.toarray().flatten()
        C[C == 0] = 1
        C = 1/_round_to_power_of_two(C)
        A = A@sps.diags(C) if sps.issparse(A) else A*C
        c = c*C

    b_scale = np.max(np.abs(b)) if b.size > 0 else 1
    if b_scale == 0:
        b_scale = 1.
    b = b/b_scale

    if x0 is not None:
        x0 = x0/b_scale*(1/C)
    return A, b, c, x0, C, b_scale


def _unscale(x, C, b_scale):
    """
    Converts solution to _autoscale problem -> solution to original problem.
    """

    try:
        n = len(C)
        # fails if sparse or scalar; that's OK.
        # this is only needed for original simplex (never sparse)
    except TypeError:
        n = len(x)

    return x[:n]*b_scale*C


def _display_summary(message, status, fun, iteration):
    """
    Print the termination summary of the linear program

    Parameters
    ----------
    message : str
            A string descriptor of the exit status of the optimization.
    status : int
        An integer representing the exit status of the optimization::

                0 : Optimization terminated successfully
                1 : Iteration limit reached
                2 : Problem appears to be infeasible
                3 : Problem appears to be unbounded
                4 : Serious numerical difficulties encountered

    fun : float
        Value of the objective function.
    iteration : iteration
        The number of iterations performed.
    """
    print(message)
    if status in (0, 1):
        print(f"         Current function value: {fun: <12.6f}")
    print(f"         Iterations: {iteration:d}")


def _postsolve(x, postsolve_args, complete=False):
    """
    Given solution x to presolved, standard form linear program x, add
    fixed variables back into the problem and undo the variable substitutions
    to get solution to original linear program. Also, calculate the objective
    function value, slack in original upper bound constraints, and residuals
    in original equality constraints.

    Parameters
    ----------
    x : 1-D array
        Solution vector to the standard-form problem.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem, including:

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
            The bounds of ``x``, lower bounds in the 1st column, upper
            bounds in the 2nd column. The bounds are possibly tightened
            by the presolve procedure.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    revstack: list of functions
        the functions in the list reverse the operations of _presolve()
        the function signature is x_org = f(x_mod), where x_mod is the result
        of a presolve step and x_org the value at the start of the step
    complete : bool
        Whether the solution is was determined in presolve (``True`` if so)

    Returns
    -------
    x : 1-D array
        Solution vector to original linear programming problem
    fun: float
        optimal objective value for original problem
    slack : 1-D array
        The (non-negative) slack in the upper bound constraints, that is,
        ``b_ub - A_ub @ x``
    con : 1-D array
        The (nominally zero) residuals of the equality constraints, that is,
        ``b - A_eq @ x``
    """
    # note that all the inputs are the ORIGINAL, unmodified versions
    # no rows, columns have been removed

    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = postsolve_args[0]
    revstack, C, b_scale = postsolve_args[1:]

    x = _unscale(x, C, b_scale)

    # Undo variable substitutions of _get_Abc()
    # if "complete", problem was solved in presolve; don't do anything here
    n_x = bounds.shape[0]
    if not complete and bounds is not None:  # bounds are never none, probably
        n_unbounded = 0
        for i, bi in enumerate(bounds):
            lbi = bi[0]
            ubi = bi[1]
            if lbi == -np.inf and ubi == np.inf:
                n_unbounded += 1
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            else:
                if lbi == -np.inf:
                    x[i] = ubi - x[i]
                else:
                    x[i] += lbi
    # all the rest of the variables were artificial
    x = x[:n_x]

    # If there were variables removed from the problem, add them back into the
    # solution vector
    # Apply the functions in revstack (reverse direction)
    for rev in reversed(revstack):
        x = rev(x)

    fun = x.dot(c)
    with np.errstate(invalid="ignore"):
        slack = b_ub - A_ub.dot(x)  # report slack for ORIGINAL UB constraints
        # report residuals of ORIGINAL EQ constraints
        con = b_eq - A_eq.dot(x)

    return x, fun, slack, con


def _check_result(x, fun, status, slack, con, bounds, tol, message,
                  integrality):
    """
    Check the validity of the provided solution.

    A valid (optimal) solution satisfies all bounds, all slack variables are
    negative and all equality constraint residuals are strictly non-zero.
    Further, the lower-bounds, upper-bounds, slack and residuals contain
    no nan values.

    Parameters
    ----------
    x : 1-D array
        Solution vector to original linear programming problem
    fun: float
        optimal objective value for original problem
    status : int
        An integer representing the exit status of the optimization::

             0 : Optimization terminated successfully
             1 : Iteration limit reached
             2 : Problem appears to be infeasible
             3 : Problem appears to be unbounded
             4 : Serious numerical difficulties encountered

    slack : 1-D array
        The (non-negative) slack in the upper bound constraints, that is,
        ``b_ub - A_ub @ x``
    con : 1-D array
        The (nominally zero) residuals of the equality constraints, that is,
        ``b - A_eq @ x``
    bounds : 2D array
        The bounds on the original variables ``x``
    message : str
        A string descriptor of the exit status of the optimization.
    tol : float
        Termination tolerance; see [1]_ Section 4.5.

    Returns
    -------
    status : int
        An integer representing the exit status of the optimization::

             0 : Optimization terminated successfully
             1 : Iteration limit reached
             2 : Problem appears to be infeasible
             3 : Problem appears to be unbounded
             4 : Serious numerical difficulties encountered

    message : str
        A string descriptor of the exit status of the optimization.
    """
    # Somewhat arbitrary
    tol = np.sqrt(tol) * 10

    if x is None:
        # HiGHS does not provide x if infeasible/unbounded
        if status == 0:  # Observed with HiGHS Simplex Primal
            status = 4
            message = ("The solver did not provide a solution nor did it "
                       "report a failure. Please submit a bug report.")
        return status, message

    contains_nans = (
        np.isnan(x).any()
        or np.isnan(fun)
        or np.isnan(slack).any()
        or np.isnan(con).any()
    )

    if contains_nans:
        is_feasible = False
    else:
        if integrality is None:
            integrality = 0
        valid_bounds = (x >= bounds[:, 0] - tol) & (x <= bounds[:, 1] + tol)
        # When integrality is 2 or 3, x must be within bounds OR take value 0
        valid_bounds |= (integrality > 1) & np.isclose(x, 0, atol=tol)
        invalid_bounds = not np.all(valid_bounds)

        invalid_slack = status != 3 and (slack < -tol).any()
        invalid_con = status != 3 and (np.abs(con) > tol).any()
        is_feasible = not (invalid_bounds or invalid_slack or invalid_con)

    if status == 0 and not is_feasible:
        status = 4
        message = ("The solution does not satisfy the constraints within the "
                   "required tolerance of " + f"{tol:.2E}" + ", yet "
                   "no errors were raised and there is no certificate of "
                   "infeasibility or unboundedness. Check whether "
                   "the slack and constraint residuals are acceptable; "
                   "if not, consider enabling presolve, adjusting the "
                   "tolerance option(s), and/or using a different method. "
                   "Please consider submitting a bug report.")
    elif status == 2 and is_feasible:
        # Occurs if the simplex method exits after phase one with a very
        # nearly basic feasible solution. Postsolving can make the solution
        # basic, however, this solution is NOT optimal
        status = 4
        message = ("The solution is feasible, but the solver did not report "
                   "that the solution was optimal. Please try a different "
                   "method.")

    return status, message
<|endoftext|>"""Revised simplex method for linear programming

The *revised simplex* method uses the method described in [1]_, except
that a factorization [2]_ of the basis matrix, rather than its inverse,
is efficiently maintained and used to solve the linear systems at each
iteration of the algorithm.

.. versionadded:: 1.3.0

References
----------
.. [1] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
.. [2] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.

"""
# Author: Matt Haberland

import numpy as np
from numpy.linalg import LinAlgError

from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult


def _phase_one(A, b, x0, callback, postsolve_args, maxiter, tol, disp,
               maxupdate, mast, pivot):
    """
    The purpose of phase one is to find an initial basic feasible solution
    (BFS) to the original problem.

    Generates an auxiliary problem with a trivial BFS and an objective that
    minimizes infeasibility of the original problem. Solves the auxiliary
    problem using the main simplex routine (phase two). This either yields
    a BFS to the original problem or determines that the original problem is
    infeasible. If feasible, phase one detects redundant rows in the original
    constraint matrix and removes them, then chooses additional indices as
    necessary to complete a basis/BFS for the original problem.
    """

    m, n = A.shape
    status = 0

    # generate auxiliary problem to get initial BFS
    A, b, c, basis, x, status = _generate_auxiliary_problem(A, b, x0, tol)

    if status == 6:
        residual = c.dot(x)
        iter_k = 0
        return x, basis, A, b, residual, status, iter_k

    # solve auxiliary problem
    phase_one_n = n
    iter_k = 0
    x, basis, status, iter_k = _phase_two(c, A, x, basis, callback,
                                          postsolve_args,
                                          maxiter, tol, disp,
                                          maxupdate, mast, pivot,
                                          iter_k, phase_one_n)

    # check for infeasibility
    residual = c.dot(x)
    if status == 0 and residual > tol:
        status = 2

    # drive artificial variables out of basis
    # TODO: test redundant row removal better
    # TODO: make solve more efficient with BGLU? This could take a while.
    keep_rows = np.ones(m, dtype=bool)
    for basis_column in basis[basis >= n]:
        B = A[:, basis]
        try:
            basis_finder = np.abs(solve(B, A))  # inefficient
            pertinent_row = np.argmax(basis_finder[:, basis_column])
            eligible_columns = np.ones(n, dtype=bool)
            eligible_columns[basis[basis < n]] = 0
            eligible_column_indices = np.where(eligible_columns)[0]
            index = np.argmax(basis_finder[:, :n]
                              [pertinent_row, eligible_columns])
            new_basis_column = eligible_column_indices[index]
            if basis_finder[pertinent_row, new_basis_column] < tol:
                keep_rows[pertinent_row] = False
            else:
                basis[basis == basis_column] = new_basis_column
        except LinAlgError:
            status = 4

    # form solution to original problem
    A = A[keep_rows, :n]
    basis = basis[keep_rows]
    x = x[:n]
    m = A.shape[0]
    return x, basis, A, b, residual, status, iter_k


def _get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape

    # options for inclusion are those that aren't already in the basis
    a = np.arange(m+n)
    bl = np.zeros(len(a), dtype=bool)
    bl[basis] = 1
    options = a[~bl]
    options = options[options < n]  # and they have to be non-artificial

    # form basis matrix
    B = np.zeros((m, m))
    B[:, 0:len(basis)] = A[:, basis]

    if (basis.size > 0 and
            np.linalg.matrix_rank(B[:, :len(basis)]) < len(basis)):
        raise Exception("Basis has dependent columns")

    rank = 0  # just enter the loop
    for i in range(n):  # somewhat arbitrary, but we need another way out
        # permute the options, and take as many as needed
        new_basis = np.random.permutation(options)[:m-len(basis)]
        B[:, len(basis):] = A[:, new_basis]  # update the basis matrix
        rank = np.linalg.matrix_rank(B)      # check the rank
        if rank == m:
            break

    return np.concatenate((basis, new_basis))


def _generate_auxiliary_problem(A, b, x0, tol):
    """
    Modifies original problem to create an auxiliary problem with a trivial
    initial basic feasible solution and an objective that minimizes
    infeasibility in the original problem.

    Conceptually, this is done by stacking an identity matrix on the right of
    the original constraint matrix, adding artificial variables to correspond
    with each of these new columns, and generating a cost vector that is all
    zeros except for ones corresponding with each of the new variables.

    A initial basic feasible solution is trivial: all variables are zero
    except for the artificial variables, which are set equal to the
    corresponding element of the right hand side `b`.

    Running the simplex method on this auxiliary problem drives all of the
    artificial variables - and thus the cost - to zero if the original problem
    is feasible. The original problem is declared infeasible otherwise.

    Much of the complexity below is to improve efficiency by using singleton
    columns in the original problem where possible, thus generating artificial
    variables only as necessary, and using an initial 'guess' basic feasible
    solution.
    """
    status = 0
    m, n = A.shape

    if x0 is not None:
        x = x0
    else:
        x = np.zeros(n)

    r = b - A@x  # residual; this must be all zeros for feasibility

    A[r < 0] = -A[r < 0]  # express problem with RHS positive for trivial BFS
    b[r < 0] = -b[r < 0]  # to the auxiliary problem
    r[r < 0] *= -1

    # Rows which we will need to find a trivial way to zero.
    # This should just be the rows where there is a nonzero residual.
    # But then we would not necessarily have a column singleton in every row.
    # This makes it difficult to find an initial basis.
    if x0 is None:
        nonzero_constraints = np.arange(m)
    else:
        nonzero_constraints = np.where(r > tol)[0]

    # these are (at least some of) the initial basis columns
    basis = np.where(np.abs(x) > tol)[0]

    if len(nonzero_constraints) == 0 and len(basis) <= m:  # already a BFS
        c = np.zeros(n)
        basis = _get_more_basis_columns(A, basis)
        return A, b, c, basis, x, status
    elif (len(nonzero_constraints) > m - len(basis) or
          np.any(x < 0)):  # can't get trivial BFS
        c = np.zeros(n)
        status = 6
        return A, b, c, basis, x, status

    # chooses existing columns appropriate for inclusion in initial basis
    cols, rows = _select_singleton_columns(A, r)

    # find the rows we need to zero that we _can_ zero with column singletons
    i_tofix = np.isin(rows, nonzero_constraints)
    # these columns can't already be in the basis, though
    # we are going to add them to the basis and change the corresponding x val
    i_notinbasis = np.logical_not(np.isin(cols, basis))
    i_fix_without_aux = np.logical_and(i_tofix, i_notinbasis)
    rows = rows[i_fix_without_aux]
    cols = cols[i_fix_without_aux]

    # indices of the rows we can only zero with auxiliary variable
    # these rows will get a one in each auxiliary column
    arows = nonzero_constraints[np.logical_not(
                                np.isin(nonzero_constraints, rows))]
    n_aux = len(arows)
    acols = n + np.arange(n_aux)          # indices of auxiliary columns

    basis_ng = np.concatenate((cols, acols))   # basis columns not from guess
    basis_ng_rows = np.concatenate((rows, arows))  # rows we need to zero

    # add auxiliary singleton columns
    A = np.hstack((A, np.zeros((m, n_aux))))
    A[arows, acols] = 1

    # generate initial BFS
    x = np.concatenate((x, np.zeros(n_aux)))
    x[basis_ng] = r[basis_ng_rows]/A[basis_ng_rows, basis_ng]

    # generate costs to minimize infeasibility
    c = np.zeros(n_aux + n)
    c[acols] = 1

    # basis columns correspond with nonzeros in guess, those with column
    # singletons we used to zero remaining constraints, and any additional
    # columns to get a full set (m columns)
    basis = np.concatenate((basis, basis_ng))
    basis = _get_more_basis_columns(A, basis)  # add columns as needed

    return A, b, c, basis, x, status


def _select_singleton_columns(A, b):
    """
    Finds singleton columns for which the singleton entry is of the same sign
    as the right-hand side; these columns are eligible for inclusion in an
    initial basis. Determines the rows in which the singleton entries are
    located. For each of these rows, returns the indices of the one singleton
    column and its corresponding row.
    """
    # find indices of all singleton columns and corresponding row indices
    column_indices = np.nonzero(np.sum(np.abs(A) != 0, axis=0) == 1)[0]
    columns = A[:, column_indices]          # array of singleton columns
    row_indices = np.zeros(len(column_indices), dtype=int)
    nonzero_rows, nonzero_columns = np.nonzero(columns)
    row_indices[nonzero_columns] = nonzero_rows   # corresponding row indices

    # keep only singletons with entries that have same sign as RHS
    # this is necessary because all elements of BFS must be non-negative
    same_sign = A[row_indices, column_indices]*b[row_indices] >= 0
    column_indices = column_indices[same_sign][::-1]
    row_indices = row_indices[same_sign][::-1]
    # Reversing the order so that steps below select rightmost columns
    # for initial basis, which will tend to be slack variables. (If the
    # guess corresponds with a basic feasible solution but a constraint
    # is not satisfied with the corresponding slack variable zero, the slack
    # variable must be basic.)

    # for each row, keep rightmost singleton column with an entry in that row
    unique_row_indices, first_columns = np.unique(row_indices,
                                                  return_index=True)
    return column_indices[first_columns], unique_row_indices


def _find_nonzero_rows(A, tol):
    """
    Returns logical array indicating the locations of rows with at least
    one nonzero element.
    """
    return np.any(np.abs(A) > tol, axis=1)


def _select_enter_pivot(c_hat, bl, a, rule="bland", tol=1e-12):
    """
    Selects a pivot to enter the basis. Currently Bland's rule - the smallest
    index that has a negative reduced cost - is the default.
    """
    if rule.lower() == "mrc":  # index with minimum reduced cost
        return a[~bl][np.argmin(c_hat)]
    else:  # smallest index w/ negative reduced cost
        return a[~bl][c_hat < -tol][0]


def _display_iter(phase, iteration, slack, con, fun):
    """
    Print indicators of optimization status to the console.
    """
    header = True if not iteration % 20 else False

    if header:
        print("Phase",
              "Iteration",
              "Minimum Slack      ",
              "Constraint Residual",
              "Objective          ")

    # :<X.Y left aligns Y digits in X digit spaces
    fmt = '{0:<6}{1:<10}{2:<20.13}{3:<20.13}{4:<20.13}'
    try:
        slack = np.min(slack)
    except ValueError:
        slack = "NA"
    print(fmt.format(phase, iteration, slack, np.linalg.norm(con), fun))


def _display_and_callback(phase_one_n, x, postsolve_args, status,
                          iteration, disp, callback):
    if phase_one_n is not None:
        phase = 1
        x_postsolve = x[:phase_one_n]
    else:
        phase = 2
        x_postsolve = x
    x_o, fun, slack, con = _postsolve(x_postsolve,
                                      postsolve_args)

    if callback is not None:
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                              'con': con, 'nit': iteration,
                              'phase': phase, 'complete': False,
                              'status': status, 'message': "",
                              'success': False})
        callback(res)
    if disp:
        _display_iter(phase, iteration, slack, con, fun)


def _phase_two(c, A, x, b, callback, postsolve_args, maxiter, tol, disp,
               maxupdate, mast, pivot, iteration=0, phase_one_n=None):
    """
    The heart of the simplex method. Beginning with a basic feasible solution,
    moves to adjacent basic feasible solutions successively lower reduced cost.
    Terminates when there are no basic feasible solutions with lower reduced
    cost or if the problem is determined to be unbounded.

    This implementation follows the revised simplex method based on LU
    decomposition. Rather than maintaining a tableau or an inverse of the
    basis matrix, we keep a factorization of the basis matrix that allows
    efficient solution of linear systems while avoiding stability issues
    associated with inverted matrices.
    """
    m, n = A.shape
    status = 0
    a = np.arange(n)                    # indices of columns of A
    ab = np.arange(m)                   # indices of columns of B
    if maxupdate:
        # basis matrix factorization object; similar to B = A[:, b]
        B = BGLU(A, b, maxupdate, mast)
    else:
        B = LU(A, b)

    for iteration in range(iteration, maxiter):

        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status,
                                  iteration, disp, callback)

        bl = np.zeros(len(a), dtype=bool)
        bl[b] = 1

        xb = x[b]       # basic variables
        cb = c[b]       # basic costs

        try:
            v = B.solve(cb, transposed=True)    # similar to v = solve(B.T, cb)
        except LinAlgError:
            status = 4
            break

        # TODO: cythonize?
        c_hat = c - v.dot(A)    # reduced cost
        c_hat = c_hat[~bl]
        # Above is much faster than:
        # N = A[:, ~bl]                 # slow!
        # c_hat = c[~bl] - v.T.dot(N)
        # Can we perform the multiplication only on the nonbasic columns?

        if np.all(c_hat >= -tol):  # all reduced costs positive -> terminate
            break

        j = _select_enter_pivot(c_hat, bl, a, rule=pivot, tol=tol)
        u = B.solve(A[:, j])        # similar to u = solve(B, A[:, j])

        i = u > tol                 # if none of the u are positive, unbounded
        if not np.any(i):
            status = 3
            break

        th = xb[i]/u[i]
        l = np.argmin(th)           # implicitly selects smallest subscript
        th_star = th[l]             # step size

        x[b] = x[b] - th_star*u     # take step
        x[j] = th_star
        B.update(ab[i][l], j)       # modify basis
        b = B.b                     # similar to b[ab[i][l]] =

    else:
        # If the end of the for loop is reached (without a break statement),
        # then another step has been taken, so the iteration counter should
        # increment, info should be displayed, and callback should be called.
        iteration += 1
        status = 1
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status,
                                  iteration, disp, callback)

    return x, b, status, iteration


def _linprog_rs(c, c0, A, b, x0, callback, postsolve_args,
                maxiter=5000, tol=1e-12, disp=False,
                maxupdate=10, mast=False, pivot="mrc",
                **unknown_options):
    """
    Solve the following linear programming problem via a two-phase
    revised simplex algorithm.::

        minimize:     c @ x

        subject to:  A @ x == b
                     0 <= x < oo

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Currently unused.)
    A : 2-D array
        2-D array which, when matrix-multiplied by ``x``, gives the values of
        the equality constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    x0 : 1-D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm. For the revised simplex method, these must
        correspond with a basic feasible solution.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector.
            fun : float
                Current value of the objective function ``c @ x``.
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
                that is, ``b - A_eq @ x``.
            phase : int
                The phase of the algorithm being executed.
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

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform in either phase.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    maxupdate : int
        The maximum number of updates performed on the LU factorization.
        After this many updates is reached, the basis matrix is factorized
        from scratch.
    mast : bool
        Minimize Amortized Solve Time. If enabled, the average time to solve
        a linear system using the basis factorization is measured. Typically,
        the average solve time will decrease with each successive solve after
        initial factorization, as factorization takes much more time than the
        solve operation (and updates). Eventually, however, the updated
        factorization becomes sufficiently complex that the average solve time
        begins to increase. When this is detected, the basis is refactorized
        from scratch. Enable this option to maximize speed at the risk of
        nondeterministic behavior. Ignored if ``maxupdate`` is 0.
    pivot : "mrc" or "bland"
        Pivot rule: Minimum Reduced Cost (default) or Bland's rule. Choose
        Bland's rule if iteration limit is reached and cycling is suspected.
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
         4 : Numerical difficulties encountered
         5 : No constraints; turn presolve on
         6 : Guess x0 cannot be converted to a basic feasible solution

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.
    """

    _check_unknown_options(unknown_options)

    messages = ["Optimization terminated successfully.",
                "Iteration limit reached.",
                "The problem appears infeasible, as the phase one auxiliary "
                "problem terminated successfully with a residual of {0:.1e}, "
                "greater than the tolerance {1} required for the solution to "
                "be considered feasible. Consider increasing the tolerance to "
                "be greater than {0:.1e}. If this tolerance is unacceptably "
                "large, the problem is likely infeasible.",
                "The problem is unbounded, as the simplex algorithm found "
                "a basic feasible solution from which there is a direction "
                "with negative reduced cost in which all decision variables "
                "increase.",
                "Numerical difficulties encountered; consider trying "
                "method='interior-point'.",
                "Problems with no constraints are trivially solved; please "
                "turn presolve on.",
                "The guess x0 cannot be converted to a basic feasible "
                "solution. "
                ]

    if A.size == 0:  # address test_unbounded_below_no_presolve_corrected
        return np.zeros(c.shape), 5, messages[5], 0

    x, basis, A, b, residual, status, iteration = (
        _phase_one(A, b, x0, callback, postsolve_args,
                   maxiter, tol, disp, maxupdate, mast, pivot))

    if status == 0:
        x, basis, status, iteration = _phase_two(c, A, x, basis, callback,
                                                 postsolve_args,
                                                 maxiter, tol, disp,
                                                 maxupdate, mast, pivot,
                                                 iteration)

    return x, status, messages[status].format(residual, tol), iteration
<|endoftext|>import warnings
from . import _minpack

import numpy as np
from numpy import (atleast_1d, triu, shape, transpose, zeros, prod, greater,
                   asarray, inf,
                   finfo, inexact, issubdtype, dtype)
from scipy import linalg
from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError
from scipy._lib._util import _asarray_validated, _lazywhere, _contains_nan
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._optimize import OptimizeResult, _check_unknown_options, OptimizeWarning
from ._lsq import least_squares
# from ._lsq.common import make_strictly_feasible
from ._lsq.least_squares import prepare_bounds
from scipy.optimize._minimize import Bounds

__all__ = ['fsolve', 'leastsq', 'fixed_point', 'curve_fit']


def _check_func(checker, argname, thefunc, x0, args, numinputs,
                output_shape=None):
    res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))
    if (output_shape is not None) and (shape(res) != output_shape):
        if (output_shape[0] != 1):
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = f"{checker}: there is a mismatch between the input and output " \
                  f"shape of the '{argname}' argument"
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += f" '{func_name}'."
            else:
                msg += "."
            msg += f'Shape should be {output_shape} but it is {shape(res)}.'
            raise TypeError(msg)
    if issubdtype(res.dtype, inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return shape(res), dt


def fsolve(func, x0, args=(), fprime=None, full_output=0,
           col_deriv=0, xtol=1.49012e-8, maxfev=0, band=None,
           epsfcn=None, factor=100, diag=None):
    """
    Find the roots of a function.

    Return the roots of the (non-linear) equations defined by
    ``func(x) = 0`` given a starting estimate.

    Parameters
    ----------
    func : callable ``f(x, *args)``
        A function that takes at least one (possibly vector) argument,
        and returns a value of the same length.
    x0 : ndarray
        The starting estimate for the roots of ``func(x) = 0``.
    args : tuple, optional
        Any extra arguments to `func`.
    fprime : callable ``f(x, *args)``, optional
        A function to compute the Jacobian of `func` with derivatives
        across the rows. By default, the Jacobian will be estimated.
    full_output : bool, optional
        If True, return optional outputs.
    col_deriv : bool, optional
        Specify whether the Jacobian function computes derivatives down
        the columns (faster, because there is no transpose operation).
    xtol : float, optional
        The calculation will terminate if the relative error between two
        consecutive iterates is at most `xtol`.
    maxfev : int, optional
        The maximum number of calls to the function. If zero, then
        ``100*(N+1)`` is the maximum where N is the number of elements
        in `x0`.
    band : tuple, optional
        If set to a two-sequence containing the number of sub- and
        super-diagonals within the band of the Jacobi matrix, the
        Jacobi matrix is considered banded (only for ``fprime=None``).
    epsfcn : float, optional
        A suitable step length for the forward-difference
        approximation of the Jacobian (for ``fprime=None``). If
        `epsfcn` is less than the machine precision, it is assumed
        that the relative errors in the functions are of the order of
        the machine precision.
    factor : float, optional
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in the interval
        ``(0.1, 100)``.
    diag : sequence, optional
        N positive entries that serve as a scale factors for the
        variables.

    Returns
    -------
    x : ndarray
        The solution (or the result of the last iteration for
        an unsuccessful call).
    infodict : dict
        A dictionary of optional outputs with the keys:

        ``nfev``
            number of function calls
        ``njev``
            number of Jacobian calls
        ``fvec``
            function evaluated at the output
        ``fjac``
            the orthogonal matrix, q, produced by the QR
            factorization of the final approximate Jacobian
            matrix, stored column wise
        ``r``
            upper triangular matrix produced by QR factorization
            of the same matrix
        ``qtf``
            the vector ``(transpose(q) * fvec)``

    ier : int
        An integer flag.  Set to 1 if a solution was found, otherwise refer
        to `mesg` for more information.
    mesg : str
        If no solution is found, `mesg` details the cause of failure.

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See the ``method='hybr'`` in particular.

    Notes
    -----
    ``fsolve`` is a wrapper around MINPACK's hybrd and hybrj algorithms.

    Examples
    --------
    Find a solution to the system of equations:
    ``x0*cos(x1) = 4,  x1*x0 - x1 = 5``.

    >>> import numpy as np
    >>> from scipy.optimize import fsolve
    >>> def func(x):
    ...     return [x[0] * np.cos(x[1]) - 4,
    ...             x[1] * x[0] - x[1] - 5]
    >>> root = fsolve(func, [1, 1])
    >>> root
    array([6.50409711, 0.90841421])
    >>> np.isclose(func(root), [0.0, 0.0])  # func(root) should be almost 0.0.
    array([ True,  True])

    """
    def _wrapped_func(*fargs):
        """
        Wrapped `func` to track the number of times
        the function has been called.
        """
        _wrapped_func.nfev += 1
        return func(*fargs)

    _wrapped_func.nfev = 0

    options = {'col_deriv': col_deriv,
               'xtol': xtol,
               'maxfev': maxfev,
               'band': band,
               'eps': epsfcn,
               'factor': factor,
               'diag': diag}

    res = _root_hybr(_wrapped_func, x0, args, jac=fprime, **options)
    res.nfev = _wrapped_func.nfev

    if full_output:
        x = res['x']
        info = {k: res.get(k)
                    for k in ('nfev', 'njev', 'fjac', 'r', 'qtf') if k in res}
        info['fvec'] = res['fun']
        return x, info, res['status'], res['message']
    else:
        status = res['status']
        msg = res['message']
        if status == 0:
            raise TypeError(msg)
        elif status == 1:
            pass
        elif status in [2, 3, 4, 5]:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            raise TypeError(msg)
        return res['x']


def _root_hybr(func, x0, args=(), jac=None,
               col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, eps=None,
               factor=100, diag=None, **unknown_options):
    """
    Find the roots of a multivariate function using MINPACK's hybrd and
    hybrj routines (modified Powell method).

    Options
    -------
    col_deriv : bool
        Specify whether the Jacobian function computes derivatives down
        the columns (faster, because there is no transpose operation).
    xtol : float
        The calculation will terminate if the relative error between two
        consecutive iterates is at most `xtol`.
    maxfev : int
        The maximum number of calls to the function. If zero, then
        ``100*(N+1)`` is the maximum where N is the number of elements
        in `x0`.
    band : tuple
        If set to a two-sequence containing the number of sub- and
        super-diagonals within the band of the Jacobi matrix, the
        Jacobi matrix is considered banded (only for ``jac=None``).
    eps : float
        A suitable step length for the forward-difference
        approximation of the Jacobian (for ``jac=None``). If
        `eps` is less than the machine precision, it is assumed
        that the relative errors in the functions are of the order of
        the machine precision.
    factor : float
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in the interval
        ``(0.1, 100)``.
    diag : sequence
        N positive entries that serve as a scale factors for the
        variables.

    """
    _check_unknown_options(unknown_options)
    epsfcn = eps

    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('fsolve', 'func', func, x0, args, n, (n,))
    if epsfcn is None:
        epsfcn = finfo(dtype).eps
    Dfun = jac
    if Dfun is None:
        if band is None:
            ml, mu = -10, -10
        else:
            ml, mu = band[:2]
        if maxfev == 0:
            maxfev = 200 * (n + 1)
        retval = _minpack._hybrd(func, x0, args, 1, xtol, maxfev,
                                 ml, mu, epsfcn, factor, diag)
    else:
        _check_func('fsolve', 'fprime', Dfun, x0, args, n, (n, n))
        if (maxfev == 0):
            maxfev = 100 * (n + 1)
        retval = _minpack._hybrj(func, Dfun, x0, args, 1,
                                 col_deriv, xtol, maxfev, factor, diag)

    x, status = retval[0], retval[-1]

    errors = {0: "Improper input parameters were entered.",
              1: "The solution converged.",
              2: "The number of calls to function has "
                  "reached maxfev = %d." % maxfev,
              3: f"xtol={xtol:f} is too small, no further improvement "
                  "in the approximate\n solution is possible.",
              4: "The iteration is not making good progress, as measured "
                  "by the \n improvement from the last five "
                  "Jacobian evaluations.",
              5: "The iteration is not making good progress, "
                  "as measured by the \n improvement from the last "
                  "ten iterations.",
              'unknown': "An error occurred."}

    info = retval[1]
    info['fun'] = info.pop('fvec')
    sol = OptimizeResult(x=x, success=(status == 1), status=status,
                         method="hybr")
    sol.update(info)
    try:
        sol['message'] = errors[status]
    except KeyError:
        sol['message'] = errors['unknown']

    return sol


LEASTSQ_SUCCESS = [1, 2, 3, 4]
LEASTSQ_FAILURE = [5, 6, 7, 8]


def leastsq(func, x0, args=(), Dfun=None, full_output=False,
            col_deriv=False, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    """
    Minimize the sum of squares of a set of equations.

    ::

        x = arg min(sum(func(y)**2,axis=0))
                 y

    Parameters
    ----------
    func : callable
        Should take at least one (possibly length ``N`` vector) argument and
        returns ``M`` floating point numbers. It must not return NaNs or
        fitting might fail. ``M`` must be greater than or equal to ``N``.
    x0 : ndarray
        The starting estimate for the minimization.
    args : tuple, optional
        Any extra arguments to func are placed in this tuple.
    Dfun : callable, optional
        A function or method to compute the Jacobian of func with derivatives
        across the rows. If this is None, the Jacobian will be estimated.
    full_output : bool, optional
        If ``True``, return all optional outputs (not just `x` and `ier`).
    col_deriv : bool, optional
        If ``True``, specify that the Jacobian function computes derivatives
        down the columns (faster, because there is no transpose operation).
    ftol : float, optional
        Relative error desired in the sum of squares.
    xtol : float, optional
        Relative error desired in the approximate solution.
    gtol : float, optional
        Orthogonality desired between the function vector and the columns of
        the Jacobian.
    maxfev : int, optional
        The maximum number of calls to the function. If `Dfun` is provided,
        then the default `maxfev` is 100*(N+1) where N is the number of elements
        in x0, otherwise the default `maxfev` is 200*(N+1).
    epsfcn : float, optional
        A variable used in determining a suitable step length for the forward-
        difference approximation of the Jacobian (for Dfun=None).
        Normally the actual step length will be sqrt(epsfcn)*x
        If epsfcn is less than the machine precision, it is assumed that the
        relative errors are of the order of the machine precision.
    factor : float, optional
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
    diag : sequence, optional
        N positive entries that serve as a scale factors for the variables.

    Returns
    -------
    x : ndarray
        The solution (or the result of the last iteration for an unsuccessful
        call).
    cov_x : ndarray
        The inverse of the Hessian. `fjac` and `ipvt` are used to construct an
        estimate of the Hessian. A value of None indicates a singular matrix,
        which means the curvature in parameters `x` is numerically flat. To
        obtain the covariance matrix of the parameters `x`, `cov_x` must be
        multiplied by the variance of the residuals -- see curve_fit. Only
        returned if `full_output` is ``True``.
    infodict : dict
        a dictionary of optional outputs with the keys:

        ``nfev``
            The number of function calls
        ``fvec``
            The function evaluated at the output
        ``fjac``
            A permutation of the R matrix of a QR
            factorization of the final approximate
            Jacobian matrix, stored column wise.
            Together with ipvt, the covariance of the
            estimate can be approximated.
        ``ipvt``
            An integer array of length N which defines
            a permutation matrix, p, such that
            fjac*p = q*r, where r is upper triangular
            with diagonal elements of nonincreasing
            magnitude. Column j of p is column ipvt(j)
            of the identity matrix.
        ``qtf``
            The vector (transpose(q) * fvec).

        Only returned if `full_output` is ``True``.
    mesg : str
        A string message giving information about the cause of failure.
        Only returned if `full_output` is ``True``.
    ier : int
        An integer flag. If it is equal to 1, 2, 3 or 4, the solution was
        found. Otherwise, the solution was not found. In either case, the
        optional output variable 'mesg' gives more information.

    See Also
    --------
    least_squares : Newer interface to solve nonlinear least-squares problems
        with bounds on the variables. See ``method='lm'`` in particular.

    Notes
    -----
    "leastsq" is a wrapper around MINPACK's lmdif and lmder algorithms.

    cov_x is a Jacobian approximation to the Hessian of the least squares
    objective function.
    This approximation assumes that the objective function is based on the
    difference between some observed target data (ydata) and a (non-linear)
    function of the parameters `f(xdata, params)` ::

           func(params) = ydata - f(xdata, params)

    so that the objective function is ::

           min   sum((ydata - f(xdata, params))**2, axis=0)
         params

    The solution, `x`, is always a 1-D array, regardless of the shape of `x0`,
    or whether `x0` is a scalar.

    Examples
    --------
    >>> from scipy.optimize import leastsq
    >>> def func(x):
    ...     return 2*(x-3)**2+1
    >>> leastsq(func, 0)
    (array([2.99999999]), 1)

    """
    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
    m = shape[0]

    if n > m:
        raise TypeError(f"Improper input: func input vector length N={n} must"
                        f" not exceed func output vector length M={m}")

    if epsfcn is None:
        epsfcn = finfo(dtype).eps

    if Dfun is None:
        if maxfev == 0:
            maxfev = 200*(n + 1)
        retval = _minpack._lmdif(func, x0, args, full_output, ftol, xtol,
                                 gtol, maxfev, epsfcn, factor, diag)
    else:
        if col_deriv:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
        if maxfev == 0:
            maxfev = 100 * (n + 1)
        retval = _minpack._lmder(func, Dfun, x0, args, full_output,
                                 col_deriv, ftol, xtol, gtol, maxfev,
                                 factor, diag)

    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  f"in the sum of squares\n  are at most {ftol:f}", None],
              2: ["The relative error between two consecutive "
                  f"iterates is at most {xtol:f}", None],
              3: ["Both actual and predicted relative reductions in "
                  f"the sum of squares\n  are at most {ftol:f} and the "
                  "relative error between two consecutive "
                  f"iterates is at \n  most {xtol:f}", None],
              4: ["The cosine of the angle between func(x) and any "
                  f"column of the\n  Jacobian is at most {gtol:f} in "
                  "absolute value", None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: [f"ftol={ftol:f} is too small, no further reduction "
                  "in the sum of squares\n  is possible.",
                  ValueError],
              7: [f"xtol={xtol:f} is too small, no further improvement in "
                  "the approximate\n  solution is possible.",
                  ValueError],
              8: [f"gtol={gtol:f} is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine precision.", ValueError]}

    # The FORTRAN return value (possible return values are >= 0 and <= 8)
    info = retval[-1]

    if full_output:
        cov_x = None
        if info in LEASTSQ_SUCCESS:
            # This was
            # perm = take(eye(n), retval[1]['ipvt'] - 1, 0)
            # r = triu(transpose(retval[1]['fjac'])[:n, :])
            # R = dot(r, perm)
            # cov_x = inv(dot(transpose(R), R))
            # but the explicit dot product was not necessary and sometimes
            # the result was not symmetric positive definite. See gh-4555.
            perm = retval[1]['ipvt']
            n = len(perm)
            r = triu(transpose(retval[1]['fjac'])[:n, :])
            inv_triu = linalg.get_lapack_funcs('trtri', (r,))
            try:
                # inverse of permuted matrix is a permutation of matrix inverse
                invR, trtri_info = inv_triu(r)  # default: upper, non-unit diag
                if trtri_info != 0:  # explicit comparison for readability
                    raise LinAlgError(f'trtri returned info {trtri_info}')
                invR[perm] = invR.copy()
                cov_x = invR @ invR.T
            except (LinAlgError, ValueError):
                pass
        return (retval[0], cov_x) + retval[1:-1] + (errors[info][0], info)
    else:
        if info in LEASTSQ_FAILURE:
            warnings.warn(errors[info][0], RuntimeWarning, stacklevel=2)
        elif info == 0:
            raise errors[info][1](errors[info][0])
        return retval[0], info


def _lightweight_memoizer(f):
    # very shallow memoization to address gh-13670: only remember the first set
    # of parameters and corresponding function value, and only attempt to use
    # them twice (the number of times the function is evaluated at x0).
    def _memoized_func(params):
        if _memoized_func.skip_lookup:
            return f(params)

        if np.all(_memoized_func.last_params == params):
            return _memoized_func.last_val
        elif _memoized_func.last_params is not None:
            _memoized_func.skip_lookup = True

        val = f(params)

        if _memoized_func.last_params is None:
            _memoized_func.last_params = np.copy(params)
            _memoized_func.last_val = val

        return val

    _memoized_func.last_params = None
    _memoized_func.last_val = None
    _memoized_func.skip_lookup = False
    return _memoized_func


def _wrap_func(func, xdata, ydata, transform):
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
    elif transform.size == 1 or transform.ndim == 1:
        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        def func_wrapped(params):
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)
    return func_wrapped


def _wrap_jac(jac, xdata, transform):
    if transform is None:
        def jac_wrapped(params):
            return jac(xdata, *params)
    elif transform.ndim == 1:
        def jac_wrapped(params):
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
    else:
        def jac_wrapped(params):
            return solve_triangular(transform,
                                    np.asarray(jac(xdata, *params)),
                                    lower=True)
    return jac_wrapped


def _initialize_feasible(lb, ub):
    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
              check_finite=None, bounds=(-np.inf, np.inf), method=None,
              jac=None, *, full_output=False, nan_policy=None,
              **kwargs):
    """
    Use non-linear least squares to fit a function, f, to data.

    Assumes ``ydata = f(xdata, *params) + eps``.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : array_like
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, and each element should be float
        convertible if it is an array like object.
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    p0 : array_like, optional
        Initial guess for the parameters (length N). If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    sigma : None or scalar or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:

        - A scalar or 1-D `sigma` should contain values of standard deviations of
          errors in `ydata`. In this case, the optimized function is
          ``chisq = sum((r / sigma) ** 2)``.

        - A 2-D `sigma` should contain the covariance matrix of
          errors in `ydata`. In this case, the optimized function is
          ``chisq = r.T @ inv(sigma) @ r``.

          .. versionadded:: 0.19

        None (default) is equivalent of 1-D `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True if `nan_policy` is not specified
        explicitly and False otherwise.
    bounds : 2-tuple of array_like or `Bounds`, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        There are two ways to specify the bounds:

        - Instance of `Bounds` class.

        - 2-tuple of array_like: Each element of the tuple must be either
          an array with the length equal to the number of parameters, or a
          scalar (in which case the bound is taken to be the same for all
          parameters). Use ``np.inf`` with an appropriate sign to disable
          bounds on all or some parameters.

    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization. See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.

        .. versionadded:: 0.17
    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be estimated numerically.
        String keywords for 'trf' and 'dogbox' methods can be used to select
        a finite difference scheme, see `least_squares`.

        .. versionadded:: 0.18
    full_output : boolean, optional
        If True, this function returns additional information: `infodict`,
        `mesg`, and `ier`.

        .. versionadded:: 1.9
    nan_policy : {'raise', 'omit', None}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is None):

        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values
        * None: no special handling of NaNs is performed
          (except what is done by check_finite); the behavior when NaNs
          are present is implementation-dependent and may change.

        Note that if this value is specified explicitly (not None),
        `check_finite` will be set as False.

        .. versionadded:: 1.11
    **kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized.
    pcov : 2-D array
        The estimated approximate covariance of popt. The diagonals provide
        the variance of the parameter estimate. To compute one standard
        deviation errors on the parameters, use
        ``perr = np.sqrt(np.diag(pcov))``. Note that the relationship between
        `cov` and parameter error estimates is derived based on a linear
        approximation to the model function around the optimum [1]_.
        When this approximation becomes inaccurate, `cov` may not provide an
        accurate measure of uncertainty.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix. Covariance matrices with large condition numbers
        (e.g. computed with `numpy.linalg.cond`) may indicate that results are
        unreliable.
    infodict : dict (returned only if `full_output` is True)
        a dictionary of optional outputs with the keys:

        ``nfev``
            The number of function calls. Methods 'trf' and 'dogbox' do not
            count function calls for numerical Jacobian approximation,
            as opposed to 'lm' method.
        ``fvec``
            The residual values evaluated at the solution, for a 1-D `sigma`
            this is ``(f(x, *popt) - ydata)/sigma``.
        ``fjac``
            A permutation of the R matrix of a QR
            factorization of the final approximate
            Jacobian matrix, stored column wise.
            Together with ipvt, the covariance of the
            estimate can be approximated.
            Method 'lm' only provides this information.
        ``ipvt``
            An integer array of length N which defines
            a permutation matrix, p, such that
            fjac*p = q*r, where r is upper triangular
            with diagonal elements of nonincreasing
            magnitude. Column j of p is column ipvt(j)
            of the identity matrix.
            Method 'lm' only provides this information.
        ``qtf``
            The vector (transpose(q) * fvec).
            Method 'lm' only provides this information.

        .. versionadded:: 1.9
    mesg : str (returned only if `full_output` is True)
        A string message giving information about the solution.

        .. versionadded:: 1.9
    ier : int (returned only if `full_output` is True)
        An integer flag. If it is equal to 1, 2, 3 or 4, the solution was
        found. Otherwise, the solution was not found. In either case, the
        optional output variable `mesg` gives more information.

        .. versionadded:: 1.9

    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.

    RuntimeError
        if the least-squares minimization fails.

    OptimizeWarning
        if covariance of the parameters can not be estimated.

    See Also
    --------
    least_squares : Minimize the sum of squares of nonlinear functions.
    scipy.stats.linregress : Calculate a linear least squares regression for
                             two sets of measurements.

    Notes
    -----
    Users should ensure that inputs `xdata`, `ydata`, and the output of `f`
    are ``float64``, or else the optimization may return incorrect results.

    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
    through `leastsq`. Note that this algorithm can only deal with
    unconstrained problems.

    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
    the docstring of `least_squares` for more information.

    Parameters to be fitted must have similar scale. Differences of multiple
    orders of magnitude can lead to incorrect results. For the 'trf' and
    'dogbox' methods, the `x_scale` keyword argument can be used to scale
    the parameters.

    References
    ----------
    .. [1] K. Vugrin et al. Confidence region estimation techniques for nonlinear
           regression in groundwater flow: Three case studies. Water Resources
           Research, Vol. 43, W03423, :doi:`10.1029/2005WR004804`

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.optimize import curve_fit

    >>> def func(x, a, b, c):
    ...     return a * np.exp(-b * x) + c

    Define the data to be fit with some noise:

    >>> xdata = np.linspace(0, 4, 50)
    >>> y = func(xdata, 2.5, 1.3, 0.5)
    >>> rng = np.random.default_rng()
    >>> y_noise = 0.2 * rng.normal(size=xdata.size)
    >>> ydata = y + y_noise
    >>> plt.plot(xdata, ydata, 'b-', label='data')

    Fit for the parameters a, b, c of the function `func`:

    >>> popt, pcov = curve_fit(func, xdata, ydata)
    >>> popt
    array([2.56274217, 1.37268521, 0.47427475])
    >>> plt.plot(xdata, func(xdata, *popt), 'r-',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    Constrain the optimization to the region of ``0 <= a <= 3``,
    ``0 <= b <= 1`` and ``0 <= c <= 0.5``:

    >>> popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    >>> popt
    array([2.43736712, 1.        , 0.34463856])
    >>> plt.plot(xdata, func(xdata, *popt), 'g--',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend()
    >>> plt.show()

    For reliable results, the model `func` should not be overparametrized;
    redundant parameters can cause unreliable covariance matrices and, in some
    cases, poorer quality fits. As a quick check of whether the model may be
    overparameterized, calculate the condition number of the covariance matrix:

    >>> np.linalg.cond(pcov)
    34.571092161547405  # may vary

    The value is small, so it does not raise much concern. If, however, we were
    to add a fourth parameter ``d`` to `func` with the same effect as ``a``:

    >>> def func2(x, a, b, c, d):
    ...     return a * d * np.exp(-b * x) + c  # a and d are redundant
    >>> popt, pcov = curve_fit(func2, xdata, ydata)
    >>> np.linalg.cond(pcov)
    1.13250718925596e+32  # may vary

    Such a large value is cause for concern. The diagonal elements of the
    covariance matrix, which is related to uncertainty of the fit, gives more
    information:

    >>> np.diag(pcov)
    array([1.48814742e+29, 3.78596560e-02, 5.39253738e-03, 2.76417220e+28])  # may vary

    Note that the first and last terms are much larger than the other elements,
    suggesting that the optimal values of these parameters are ambiguous and
    that only one of these parameters is needed in the model.

    If the optimal parameters of `f` differ by multiple orders of magnitude, the
    resulting fit can be inaccurate. Sometimes, `curve_fit` can fail to find any
    results:

    >>> ydata = func(xdata, 500000, 0.01, 15)
    >>> try:
    ...     popt, pcov = curve_fit(func, xdata, ydata, method = 'trf')
    ... except RuntimeError as e:
    ...     print(e)
    Optimal parameters not found: The maximum number of function evaluations is
    exceeded.

    If parameter scale is roughly known beforehand, it can be defined in
    `x_scale` argument:

    >>> popt, pcov = curve_fit(func, xdata, ydata, method = 'trf',
    ...                        x_scale = [1000, 1, 1])
    >>> popt
    array([5.00000000e+05, 1.00000000e-02, 1.49999999e+01])
    """
    if p0 is None:
        # determine number of parameters by inspecting the function
        sig = _getfullargspec(f)
        args = sig.args
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(p0)
        n = p0.size

    if isinstance(bounds, Bounds):
        lb, ub = bounds.lb, bounds.ub
    else:
        lb, ub = prepare_bounds(bounds, n)
    if p0 is None:
        p0 = _initialize_feasible(lb, ub)

    bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
    if method is None:
        if bounded_problem:
            method = 'trf'
        else:
            method = 'lm'

    if method == 'lm' and bounded_problem:
        raise ValueError("Method 'lm' only works for unconstrained problems. "
                         "Use 'trf' or 'dogbox' instead.")

    if check_finite is None:
        check_finite = True if nan_policy is None else False

    # optimization may produce garbage for float32 inputs, cast them to float64
    if check_finite:
        ydata = np.asarray_chkfinite(ydata, float)
    else:
        ydata = np.asarray(ydata, float)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        if check_finite:
            xdata = np.asarray_chkfinite(xdata, float)
        else:
            xdata = np.asarray(xdata, float)

    if ydata.size == 0:
        raise ValueError("`ydata` must not be empty!")

    # nan handling is needed only if check_finite is False because if True,
    # the x-y data are already checked, and they don't contain nans.
    if not check_finite and nan_policy is not None:
        if nan_policy == "propagate":
            raise ValueError("`nan_policy='propagate'` is not supported "
                             "by this function.")

        policies = [None, 'raise', 'omit']
        x_contains_nan, nan_policy = _contains_nan(xdata, nan_policy,
                                                   policies=policies)
        y_contains_nan, nan_policy = _contains_nan(ydata, nan_policy,
                                                   policies=policies)

        if (x_contains_nan or y_contains_nan) and nan_policy == 'omit':
            # ignore NaNs for N dimensional arrays
            has_nan = np.isnan(xdata)
            has_nan = has_nan.any(axis=tuple(range(has_nan.ndim-1)))
            has_nan |= np.isnan(ydata)

            xdata = xdata[..., ~has_nan]
            ydata = ydata[~has_nan]

            # Also omit the corresponding entries from sigma
            if sigma is not None:
                sigma = np.asarray(sigma)
                if sigma.ndim == 1:
                    sigma = sigma[~has_nan]
                elif sigma.ndim == 2:
                    sigma = sigma[~has_nan, :]
                    sigma = sigma[:, ~has_nan]

    # Determine type of sigma
    if sigma is not None:
        sigma = np.asarray(sigma)

        # if 1-D or a scalar, sigma are errors, define transform = 1/sigma
        if sigma.size == 1 or sigma.shape == (ydata.size,):
            transform = 1.0 / sigma
        # if 2-D, sigma is the covariance matrix,
        # define transform = L such that L L^T = C
        elif sigma.shape == (ydata.size, ydata.size):
            try:
                # scipy.linalg.cholesky requires lower=True to return L L^T = A
                transform = cholesky(sigma, lower=True)
            except LinAlgError as e:
                raise ValueError("`sigma` must be positive definite.") from e
        else:
            raise ValueError("`sigma` has incorrect shape.")
    else:
        transform = None

    func = _lightweight_memoizer(_wrap_func(f, xdata, ydata, transform))

    if callable(jac):
        jac = _lightweight_memoizer(_wrap_jac(jac, xdata, transform))
    elif jac is None and method != 'lm':
        jac = '2-point'

    if 'args' in kwargs:
        # The specification for the model function `f` does not support
        # additional arguments. Refer to the `curve_fit` docstring for
        # acceptable call signatures of `f`.
        raise ValueError("'args' is not a supported keyword argument.")

    if method == 'lm':
        # if ydata.size == 1, this might be used for broadcast.
        if ydata.size != 1 and n > ydata.size:
            raise TypeError(f"The number of func parameters={n} must not"
                            f" exceed the number of data points={ydata.size}")
        res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
        popt, pcov, infodict, errmsg, ier = res
        ysize = len(infodict['fvec'])
        cost = np.sum(infodict['fvec'] ** 2)
        if ier not in [1, 2, 3, 4]:
            raise RuntimeError("Optimal parameters not found: " + errmsg)
    else:
        # Rename maxfev (leastsq) to max_nfev (least_squares), if specified.
        if 'max_nfev' not in kwargs:
            kwargs['max_nfev'] = kwargs.pop('maxfev', None)

        res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                            **kwargs)

        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)

        infodict = dict(nfev=res.nfev, fvec=res.fun)
        ier = res.status
        errmsg = res.message

        ysize = len(res.fun)
        cost = 2 * res.cost  # res.cost is half sum of squares!
        popt = res.x

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)

    warn_cov = False
    if pcov is None or np.isnan(pcov).any():
        # indeterminate covariance
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
            warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning, stacklevel=2)

    if full_output:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov


def check_gradient(fcn, Dfcn, x0, args=(), col_deriv=0):
    """Perform a simple check on the gradient for correctness.

    """

    x = atleast_1d(x0)
    n = len(x)
    x = x.reshape((n,))
    fvec = atleast_1d(fcn(x, *args))
    m = len(fvec)
    fvec = fvec.reshape((m,))
    ldfjac = m
    fjac = atleast_1d(Dfcn(x, *args))
    fjac = fjac.reshape((m, n))
    if col_deriv == 0:
        fjac = transpose(fjac)

    xp = zeros((n,), float)
    err = zeros((m,), float)
    fvecp = None
    _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 1, err)

    fvecp = atleast_1d(fcn(xp, *args))
    fvecp = fvecp.reshape((m,))
    _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 2, err)

    good = (prod(greater(err, 0.5), axis=0))

    return (good, err)


def _del2(p0, p1, d):
    return p0 - np.square(p1 - p0) / d


def _relerr(actual, desired):
    return (actual - desired) / desired


def _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel):
    p0 = x0
    for i in range(maxiter):
        p1 = func(p0, *args)
        if use_accel:
            p2 = func(p1, *args)
            d = p2 - 2.0 * p1 + p0
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            p = p1
        relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p)
        if np.all(np.abs(relerr) < xtol):
            return p
        p0 = p
    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    raise RuntimeError(msg)


def fixed_point(func, x0, args=(), xtol=1e-8, maxiter=500, method='del2'):
    """
    Find a fixed point of the function.

    Given a function of one or more variables and a starting point, find a
    fixed point of the function: i.e., where ``func(x0) == x0``.

    Parameters
    ----------
    func : function
        Function to evaluate.
    x0 : array_like
        Fixed point of function.
    args : tuple, optional
        Extra arguments to `func`.
    xtol : float, optional
        Convergence tolerance, defaults to 1e-08.
    maxiter : int, optional
        Maximum number of iterations, defaults to 500.
    method : {"del2", "iteration"}, optional
        Method of finding the fixed-point, defaults to "del2",
        which uses Steffensen's Method with Aitken's ``Del^2``
        convergence acceleration [1]_. The "iteration" method simply iterates
        the function until convergence is detected, without attempting to
        accelerate the convergence.

    References
    ----------
    .. [1] Burden, Faires, "Numerical Analysis", 5th edition, pg. 80

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import optimize
    >>> def func(x, c1, c2):
    ...    return np.sqrt(c1/(x+c2))
    >>> c1 = np.array([10,12.])
    >>> c2 = np.array([3, 5.])
    >>> optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))
    array([ 1.4920333 ,  1.37228132])

    """
    use_accel = {'del2': True, 'iteration': False}[method]
    x0 = _asarray_validated(x0, as_inexact=True)
    return _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel)
<|endoftext|>"""
Unified interfaces to root finding algorithms for real or complex
scalar functions.

Functions
---------
- root : find a root of a scalar function.
"""
import numpy as np

from . import _zeros_py as optzeros
from ._numdiff import approx_derivative

__all__ = ['root_scalar']

ROOT_SCALAR_METHODS = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748',
                       'newton', 'secant', 'halley']


class MemoizeDer:
    """Decorator that caches the value and derivative(s) of function each
    time it is called.

    This is a simplistic memoizer that calls and caches a single value
    of ``f(x, *args)``.
    It assumes that `args` does not change between invocations.
    It supports the use case of a root-finder where `args` is fixed,
    `x` changes, and only rarely, if at all, does x assume the same value
    more than once."""
    def __init__(self, fun):
        self.fun = fun
        self.vals = None
        self.x = None
        self.n_calls = 0

    def __call__(self, x, *args):
        r"""Calculate f or use cached value if available"""
        # Derivative may be requested before the function itself, always check
        if self.vals is None or x != self.x:
            fg = self.fun(x, *args)
            self.x = x
            self.n_calls += 1
            self.vals = fg[:]
        return self.vals[0]

    def fprime(self, x, *args):
        r"""Calculate f' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[1]

    def fprime2(self, x, *args):
        r"""Calculate f'' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[2]

    def ncalls(self):
        return self.n_calls


def root_scalar(f, args=(), method=None, bracket=None,
                fprime=None, fprime2=None,
                x0=None, x1=None,
                xtol=None, rtol=None, maxiter=None,
                options=None):
    """
    Find a root of a scalar function.

    Parameters
    ----------
    f : callable
        A function to find a root of.

        Suppose the callable has signature ``f0(x, *my_args, **my_kwargs)``, where
        ``my_args`` and ``my_kwargs`` are required positional and keyword arguments.
        Rather than passing ``f0`` as the callable, wrap it to accept
        only ``x``; e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Type of solver.  Should be one of

        - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
        - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
        - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
        - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
        - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
        - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
        - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
        - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`

    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    x0 : float, optional
        Initial guess.
    x1 : float, optional
        A second guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the derivative.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, optional
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the
        first and second derivatives.
        `fprime2` can also be a callable returning the second derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options : dict, optional
        A dictionary of solver options. E.g., ``k``, see
        :obj:`show_options()` for details.

    Returns
    -------
    sol : RootResults
        The solution represented as a ``RootResults`` object.
        Important attributes are: ``root`` the solution , ``converged`` a
        boolean flag indicating if the algorithm exited successfully and
        ``flag`` which describes the cause of the termination. See
        `RootResults` for a description of other attributes.

    See also
    --------
    show_options : Additional options accepted by the solvers
    root : Find a root of a vector function.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    The default is to use the best method available for the situation
    presented.
    If a bracket is provided, it may use one of the bracketing methods.
    If a derivative and an initial value are specified, it may
    select one of the derivative-based methods.
    If no method is judged applicable, it will raise an Exception.

    Arguments for each method are as follows (x=required, o=optional).

    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |
    +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+
    | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`brentq <optimize.root_scalar-brentq>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`brenth <optimize.root_scalar-brenth>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`ridder <optimize.root_scalar-ridder>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`toms748 <optimize.root_scalar-toms748>` | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`secant <optimize.root_scalar-secant>`   | x |  o   |         | x  | o  |        |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`newton <optimize.root_scalar-newton>`   | x |  o   |         | x  |    |   o    |         |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+
    | :ref:`halley <optimize.root_scalar-halley>`   | x |  o   |         | x  |    |   x    |    x    |  o   |  o   |    o    |   o     |
    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+

    Examples
    --------

    Find the root of a simple cubic

    >>> from scipy import optimize
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    >>> def fprime(x):
    ...     return 3*x**2

    The `brentq` method takes as input a bracket

    >>> sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 10, 11)

    The `newton` method takes as input a single point and uses the
    derivative(s).

    >>> sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 22)

    The function can provide the value and derivative(s) in a single call.

    >>> def f_p_pp(x):
    ...     return (x**3 - 1), 3*x**2, 6*x

    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, method='newton'
    ... )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 11)

    >>> sol = optimize.root_scalar(
    ...     f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley'
    ... )
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 7, 8)


    """  # noqa: E501
    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}

    # fun also returns the derivative(s)
    is_memoized = False
    if fprime2 is not None and not callable(fprime2):
        if bool(fprime2):
            f = MemoizeDer(f)
            is_memoized = True
            fprime2 = f.fprime2
            fprime = f.fprime
        else:
            fprime2 = None
    if fprime is not None and not callable(fprime):
        if bool(fprime):
            f = MemoizeDer(f)
            is_memoized = True
            fprime = f.fprime
        else:
            fprime = None

    # respect solver-specific default tolerances - only pass in if actually set
    kwargs = {}
    for k in ['xtol', 'rtol', 'maxiter']:
        v = locals().get(k)
        if v is not None:
            kwargs[k] = v

    # Set any solver-specific options
    if options:
        kwargs.update(options)
    # Always request full_output from the underlying method as _root_scalar
    # always returns a RootResults object
    kwargs.update(full_output=True, disp=False)

    # Pick a method if not specified.
    # Use the "best" method available for the situation.
    if not method:
        if bracket is not None:
            method = 'brentq'
        elif x0 is not None:
            if fprime:
                if fprime2:
                    method = 'halley'
                else:
                    method = 'newton'
            elif x1 is not None:
                method = 'secant'
            else:
                method = 'newton'
    if not method:
        raise ValueError('Unable to select a solver as neither bracket '
                         'nor starting point provided.')

    meth = method.lower()
    map2underlying = {'halley': 'newton', 'secant': 'newton'}

    try:
        methodc = getattr(optzeros, map2underlying.get(meth, meth))
    except AttributeError as e:
        raise ValueError(f'Unknown solver {meth}') from e

    if meth in ['bisect', 'ridder', 'brentq', 'brenth', 'toms748']:
        if not isinstance(bracket, (list, tuple, np.ndarray)):
            raise ValueError(f'Bracket needed for {method}')

        a, b = bracket[:2]
        try:
            r, sol = methodc(f, a, b, args=args, **kwargs)
        except ValueError as e:
            # gh-17622 fixed some bugs in low-level solvers by raising an error
            # (rather than returning incorrect results) when the callable
            # returns a NaN. It did so by wrapping the callable rather than
            # modifying compiled code, so the iteration count is not available.
            if hasattr(e, "_x"):
                sol = optzeros.RootResults(root=e._x,
                                           iterations=np.nan,
                                           function_calls=e._function_calls,
                                           flag=str(e), method=method)
            else:
                raise

    elif meth in ['secant']:
        if x0 is None:
            raise ValueError(f'x0 must not be None for {method}')
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=None, fprime2=None,
                         x1=x1, **kwargs)
    elif meth in ['newton']:
        if x0 is None:
            raise ValueError(f'x0 must not be None for {method}')
        if not fprime:
            # approximate fprime with finite differences

            def fprime(x, *args):
                # `root_scalar` doesn't actually seem to support vectorized
                # use of `newton`. In that case, `approx_derivative` will
                # always get scalar input. Nonetheless, it always returns an
                # array, so we extract the element to produce scalar output.
                # Similarly, `approx_derivative` always passes array input, so
                # we extract the element to ensure the user's function gets
                # scalar input.
                def f_wrapped(x, *args):
                    return f(x[0], *args)
                return approx_derivative(f_wrapped, x, method='2-point', args=args)[0]

        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=None,
                         **kwargs)
    elif meth in ['halley']:
        if x0 is None:
            raise ValueError(f'x0 must not be None for {method}')
        if not fprime:
            raise ValueError(f'fprime must be specified for {method}')
        if not fprime2:
            raise ValueError(f'fprime2 must be specified for {method}')
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=fprime2, **kwargs)
    else:
        raise ValueError(f'Unknown solver {method}')

    if is_memoized:
        # Replace the function_calls count with the memoized count.
        # Avoids double and triple-counting.
        n_calls = f.n_calls
        sol.function_calls = n_calls

    return sol


def _root_scalar_brentq_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above

    """
    pass


def _root_scalar_brenth_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass

def _root_scalar_toms748_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_secant_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    x1 : float, optional
        A second guess. Must be different from `x0`. If not specified,
        a value near `x0` will be chosen.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_newton_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_halley_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, required
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, required
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of 1st and 2nd derivatives along with the objective function.
        `fprime2` can also be a callable returning the 2nd derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_ridder_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_bisect_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  ``f(x, *args)`` must have different
        signs at the two endpoints.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass
<|endoftext|>"""Constraints definition for minimize."""
import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
    VectorFunction, LinearVectorFunction, IdentityVectorFunction)
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse


def _arr_to_scalar(x):
    # If x is a numpy array, return x.item().  This will
    # fail if the array has more than one element.
    return x.item() if isinstance(x, np.ndarray) else x


class NonlinearConstraint:
    """Nonlinear constraint on the variables.

    The constraint has the general inequality form::

        lb <= fun(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and ``fun`` returns a vector with m components.

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    fun : callable
        The function defining the constraint.
        The signature is ``fun(x) -> array_like, shape (m,)``.
    lb, ub : array_like
        Lower and upper bounds on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary.
    jac : {callable,  '2-point', '3-point', 'cs'}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix,
        where element (i, j) is the partial derivative of f[i] with
        respect to x[j]).  The keywords {'2-point', '3-point',
        'cs'} select a finite difference scheme for the numerical estimation.
        A callable must have the following signature::

            jac(x) -> {ndarray, sparse matrix}, shape (m, n)

        Default is '2-point'.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy, None}, optional
        Method for computing the Hessian matrix. The keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme for
        numerical  estimation.  Alternatively, objects implementing
        `HessianUpdateStrategy` interface can be used to approximate the
        Hessian. Currently available implementations are:

        - `BFGS` (default option)
        - `SR1`

        A callable must return the Hessian matrix of ``dot(fun, v)`` and
        must have the following signature:
        ``hess(x, v) -> {LinearOperator, sparse matrix, array_like}, shape (n, n)``.
        Here ``v`` is ndarray with shape (m,) containing Lagrange multipliers.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    finite_diff_rel_step: None or array_like, optional
        Relative step size for the finite difference approximation. Default is
        None, which will select a reasonable value automatically depending
        on a finite difference scheme.
    finite_diff_jac_sparsity: {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations. A zero entry means
        that a corresponding element in the Jacobian is identically zero.
        If provided, forces the use of 'lsmr' trust-region solver.
        If None (default) then dense differencing will be used.

    Notes
    -----
    Finite difference schemes {'2-point', '3-point', 'cs'} may be used for
    approximating either the Jacobian or the Hessian. We, however, do not allow
    its use for approximating both simultaneously. Hence whenever the Jacobian
    is estimated via finite-differences, we require the Hessian to be estimated
    using one of the quasi-Newton strategies.

    The scheme 'cs' is potentially the most accurate, but requires the function
    to correctly handles complex inputs and be analytically continuable to the
    complex plane. The scheme '3-point' is more accurate than '2-point' but
    requires twice as many operations.

    Examples
    --------
    Constrain ``x[0] < sin(x[1]) + 1.9``

    >>> from scipy.optimize import NonlinearConstraint
    >>> import numpy as np
    >>> con = lambda x: x[0] - np.sin(x[1])
    >>> nlc = NonlinearConstraint(con, -np.inf, 1.9)

    """
    def __init__(self, fun, lb, ub, jac='2-point', hess=None,
                 keep_feasible=False, finite_diff_rel_step=None,
                 finite_diff_jac_sparsity=None):
        if hess is None:
            hess = BFGS()
        self.fun = fun
        self.lb = lb
        self.ub = ub
        self.finite_diff_rel_step = finite_diff_rel_step
        self.finite_diff_jac_sparsity = finite_diff_jac_sparsity
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible


class LinearConstraint:
    """Linear constraint on the variables.

    The constraint has the general inequality form::

        lb <= A.dot(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and the matrix A has shape (m, n).

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    A : {array_like, sparse matrix}, shape (m, n)
        Matrix defining the constraint.
    lb, ub : dense array_like, optional
        Lower and upper limits on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no limits).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    def _input_validation(self):
        if self.A.ndim != 2:
            message = "`A` must have exactly two dimensions."
            raise ValueError(message)

        try:
            shape = self.A.shape[0:1]
            self.lb = np.broadcast_to(self.lb, shape)
            self.ub = np.broadcast_to(self.ub, shape)
            self.keep_feasible = np.broadcast_to(self.keep_feasible, shape)
        except ValueError:
            message = ("`lb`, `ub`, and `keep_feasible` must be broadcastable "
                       "to shape `A.shape[0:1]`")
            raise ValueError(message)

    def __init__(self, A, lb=-np.inf, ub=np.inf, keep_feasible=False):
        if not issparse(A):
            # In some cases, if the constraint is not valid, this emits a
            # VisibleDeprecationWarning about ragged nested sequences
            # before eventually causing an error. `scipy.optimize.milp` would
            # prefer that this just error out immediately so it can handle it
            # rather than concerning the user.
            with catch_warnings():
                simplefilter("error")
                self.A = np.atleast_2d(A).astype(np.float64)
        else:
            self.A = A
        if issparse(lb) or issparse(ub):
            raise ValueError("Constraint limits must be dense arrays.")
        self.lb = np.atleast_1d(lb).astype(np.float64)
        self.ub = np.atleast_1d(ub).astype(np.float64)

        if issparse(keep_feasible):
            raise ValueError("`keep_feasible` must be a dense array.")
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def residual(self, x):
        """
        Calculate the residual between the constraint function and the limits

        For a linear constraint of the form::

            lb <= A@x <= ub

        the lower and upper residuals between ``A@x`` and the limits are values
        ``sl`` and ``sb`` such that::

            lb + sl == A@x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        the constraint are satisfied; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of the constraint is not
        satisfied.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        return self.A@x - self.lb, self.ub - self.A@x


class Bounds:
    """Bounds constraint on the variables.

    The constraint has the general inequality form::

        lb <= x <= ub

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    lb, ub : dense array_like, optional
        Lower and upper bounds on independent variables. `lb`, `ub`, and
        `keep_feasible` must be the same shape or broadcastable.
        Set components of `lb` and `ub` equal
        to fix a variable. Use ``np.inf`` with an appropriate sign to disable
        bounds on all or some variables. Note that you can mix constraints of
        different types: interval, one-sided or equality, by setting different
        components of `lb` and `ub` as necessary. Defaults to ``lb = -np.inf``
        and ``ub = np.inf`` (no bounds).
    keep_feasible : dense array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. Must be broadcastable with `lb` and `ub`.
        Default is False. Has no effect for equality constraints.
    """
    def _input_validation(self):
        try:
            res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
            self.lb, self.ub, self.keep_feasible = res
        except ValueError:
            message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
            raise ValueError(message)

    def __init__(self, lb=-np.inf, ub=np.inf, keep_feasible=False):
        if issparse(lb) or issparse(ub):
            raise ValueError("Lower and upper bounds must be dense arrays.")
        self.lb = np.atleast_1d(lb)
        self.ub = np.atleast_1d(ub)

        if issparse(keep_feasible):
            raise ValueError("`keep_feasible` must be a dense array.")
        self.keep_feasible = np.atleast_1d(keep_feasible).astype(bool)
        self._input_validation()

    def __repr__(self):
        start = f"{type(self).__name__}({self.lb!r}, {self.ub!r}"
        if np.any(self.keep_feasible):
            end = f", keep_feasible={self.keep_feasible!r})"
        else:
            end = ")"
        return start + end

    def residual(self, x):
        """Calculate the residual (slack) between the input and the bounds

        For a bound constraint of the form::

            lb <= x <= ub

        the lower and upper residuals between `x` and the bounds are values
        ``sl`` and ``sb`` such that::

            lb + sl == x == ub - sb

        When all elements of ``sl`` and ``sb`` are positive, all elements of
        ``x`` lie within the bounds; a negative element in ``sl`` or ``sb``
        indicates that the corresponding element of ``x`` is out of bounds.

        Parameters
        ----------
        x: array_like
            Vector of independent variables

        Returns
        -------
        sl, sb : array-like
            The lower and upper residuals
        """
        return x - self.lb, self.ub - x


class PreparedConstraint:
    """Constraint prepared from a user defined constraint.

    On creation it will check whether a constraint definition is valid and
    the initial point is feasible. If created successfully, it will contain
    the attributes listed below.

    Parameters
    ----------
    constraint : {NonlinearConstraint, LinearConstraint`, Bounds}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables.
    sparse_jacobian : bool or None, optional
        If bool, then the Jacobian of the constraint will be converted
        to the corresponded format if necessary. If None (default), such
        conversion is not made.
    finite_diff_bounds : 2-tuple, optional
        Lower and upper bounds on the independent variables for the finite
        difference approximation, if applicable. Defaults to no bounds.

    Attributes
    ----------
    fun : {VectorFunction, LinearVectorFunction, IdentityVectorFunction}
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    keep_feasible : ndarray
         Array indicating which components must be kept feasible with a size
         equal to the number of the constraints.
    """
    def __init__(self, constraint, x0, sparse_jacobian=None,
                 finite_diff_bounds=(-np.inf, np.inf)):
        if isinstance(constraint, NonlinearConstraint):
            fun = VectorFunction(constraint.fun, x0,
                                 constraint.jac, constraint.hess,
                                 constraint.finite_diff_rel_step,
                                 constraint.finite_diff_jac_sparsity,
                                 finite_diff_bounds, sparse_jacobian)
        elif isinstance(constraint, LinearConstraint):
            fun = LinearVectorFunction(constraint.A, x0, sparse_jacobian)
        elif isinstance(constraint, Bounds):
            fun = IdentityVectorFunction(x0, sparse_jacobian)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        m = fun.m

        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)
        keep_feasible = np.asarray(constraint.keep_feasible, dtype=bool)

        lb = np.broadcast_to(lb, m)
        ub = np.broadcast_to(ub, m)
        keep_feasible = np.broadcast_to(keep_feasible, m)

        if keep_feasible.shape != (m,):
            raise ValueError("`keep_feasible` has a wrong shape.")

        mask = keep_feasible & (lb != ub)
        f0 = fun.f
        if np.any(f0[mask] < lb[mask]) or np.any(f0[mask] > ub[mask]):
            raise ValueError("`x0` is infeasible with respect to some "
                             "inequality constraint with `keep_feasible` "
                             "set to True.")

        self.fun = fun
        self.bounds = (lb, ub)
        self.keep_feasible = keep_feasible

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `PreparedConstraint.fun`.
        """
        with catch_warnings():
            # Ignore the following warning, it's not important when
            # figuring out total violation
            # UserWarning: delta_grad == 0.0. Check if the approximated
            # function is linear
            filterwarnings("ignore", "delta_grad", UserWarning)
            ev = self.fun.fun(np.asarray(x))

        excess_lb = np.maximum(self.bounds[0] - ev, 0)
        excess_ub = np.maximum(ev - self.bounds[1], 0)

        return excess_lb + excess_ub


def new_bounds_to_old(lb, ub, n):
    """Convert the new bounds representation to the old one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are -np.inf/np.inf they are replaced by
    None.
    """
    lb = np.broadcast_to(lb, n)
    ub = np.broadcast_to(ub, n)

    lb = [float(x) if x > -np.inf else None for x in lb]
    ub = [float(x) if x < np.inf else None for x in ub]

    return list(zip(lb, ub))


def old_bound_to_new(bounds):
    """Convert the old bounds representation to the new one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are None they are replaced by
    -np.inf/np.inf.
    """
    lb, ub = zip(*bounds)

    # Convert occurrences of None to -inf or inf, and replace occurrences of
    # any numpy array x with x.item(). Then wrap the results in numpy arrays.
    lb = np.array([float(_arr_to_scalar(x)) if x is not None else -np.inf
                   for x in lb])
    ub = np.array([float(_arr_to_scalar(x)) if x is not None else np.inf
                   for x in ub])

    return lb, ub


def strict_bounds(lb, ub, keep_feasible, n_vars):
    """Remove bounds which are not asked to be kept feasible."""
    strict_lb = np.resize(lb, n_vars).astype(float)
    strict_ub = np.resize(ub, n_vars).astype(float)
    keep_feasible = np.resize(keep_feasible, n_vars)
    strict_lb[~keep_feasible] = -np.inf
    strict_ub[~keep_feasible] = np.inf
    return strict_lb, strict_ub


def new_constraint_to_old(con, x0):
    """
    Converts new-style constraint objects to old-style constraint dictionaries.
    """
    if isinstance(con, NonlinearConstraint):
        if (con.finite_diff_jac_sparsity is not None or
                con.finite_diff_rel_step is not None or
                not isinstance(con.hess, BFGS) or  # misses user specified BFGS
                con.keep_feasible):
            warn("Constraint options `finite_diff_jac_sparsity`, "
                 "`finite_diff_rel_step`, `keep_feasible`, and `hess`"
                 "are ignored by this method.",
                 OptimizeWarning, stacklevel=3)

        fun = con.fun
        if callable(con.jac):
            jac = con.jac
        else:
            jac = None

    else:  # LinearConstraint
        if np.any(con.keep_feasible):
            warn("Constraint option `keep_feasible` is ignored by this method.",
                 OptimizeWarning, stacklevel=3)

        A = con.A
        if issparse(A):
            A = A.toarray()
        def fun(x):
            return np.dot(A, x)
        def jac(x):
            return A

    # FIXME: when bugs in VectorFunction/LinearVectorFunction are worked out,
    # use pcon.fun.fun and pcon.fun.jac. Until then, get fun/jac above.
    pcon = PreparedConstraint(con, x0)
    lb, ub = pcon.bounds

    i_eq = lb == ub
    i_bound_below = np.logical_xor(lb != -np.inf, i_eq)
    i_bound_above = np.logical_xor(ub != np.inf, i_eq)
    i_unbounded = np.logical_and(lb == -np.inf, ub == np.inf)

    if np.any(i_unbounded):
        warn("At least one constraint is unbounded above and below. Such "
             "constraints are ignored.",
             OptimizeWarning, stacklevel=3)

    ceq = []
    if np.any(i_eq):
        def f_eq(x):
            y = np.array(fun(x)).flatten()
            return y[i_eq] - lb[i_eq]
        ceq = [{"type": "eq", "fun": f_eq}]

        if jac is not None:
            def j_eq(x):
                dy = jac(x)
                if issparse(dy):
                    dy = dy.toarray()
                dy = np.atleast_2d(dy)
                return dy[i_eq, :]
            ceq[0]["jac"] = j_eq

    cineq = []
    n_bound_below = np.sum(i_bound_below)
    n_bound_above = np.sum(i_bound_above)
    if n_bound_below + n_bound_above:
        def f_ineq(x):
            y = np.zeros(n_bound_below + n_bound_above)
            y_all = np.array(fun(x)).flatten()
            y[:n_bound_below] = y_all[i_bound_below] - lb[i_bound_below]
            y[n_bound_below:] = -(y_all[i_bound_above] - ub[i_bound_above])
            return y
        cineq = [{"type": "ineq", "fun": f_ineq}]

        if jac is not None:
            def j_ineq(x):
                dy = np.zeros((n_bound_below + n_bound_above, len(x0)))
                dy_all = jac(x)
                if issparse(dy_all):
                    dy_all = dy_all.toarray()
                dy_all = np.atleast_2d(dy_all)
                dy[:n_bound_below, :] = dy_all[i_bound_below]
                dy[n_bound_below:, :] = -dy_all[i_bound_above]
                return dy
            cineq[0]["jac"] = j_ineq

    old_constraints = ceq + cineq

    if len(old_constraints) > 1:
        warn("Equality and inequality constraints are specified in the same "
             "element of the constraint list. For efficient use with this "
             "method, equality and inequality constraints should be specified "
             "in separate elements of the constraint list. ",
             OptimizeWarning, stacklevel=3)
    return old_constraints


def old_constraint_to_new(ic, con):
    """
    Converts old-style constraint dictionaries to new-style constraint objects.
    """
    # check type
    try:
        ctype = con['type'].lower()
    except KeyError as e:
        raise KeyError('Constraint %d has no type defined.' % ic) from e
    except TypeError as e:
        raise TypeError(
            'Constraints must be a sequence of dictionaries.'
        ) from e
    except AttributeError as e:
        raise TypeError("Constraint's type must be a string.") from e
    else:
        if ctype not in ['eq', 'ineq']:
            raise ValueError(f"Unknown constraint type '{con['type']}'.")
    if 'fun' not in con:
        raise ValueError('Constraint %d has no function defined.' % ic)

    lb = 0
    if ctype == 'eq':
        ub = 0
    else:
        ub = np.inf

    jac = '2-point'
    if 'args' in con:
        args = con['args']
        def fun(x):
            return con["fun"](x, *args)
        if 'jac' in con:
            def jac(x):
                return con["jac"](x, *args)
    else:
        fun = con['fun']
        if 'jac' in con:
            jac = con['jac']

    return NonlinearConstraint(fun, lb, ub, jac)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.optimize` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BroydenFirst',
    'InverseJacobian',
    'KrylovJacobian',
    'anderson',
    'broyden1',
    'broyden2',
    'diagbroyden',
    'excitingmixing',
    'linearmixing',
    'newton_krylov',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="optimize", module="nonlin",
                                   private_modules=["_nonlin"], all=__all__,
                                   attribute=name)
<|endoftext|>import numpy as np
import scipy._lib._elementwise_iterative_method as eim
from scipy._lib._util import _RichResult
from scipy._lib._array_api import array_namespace, xp_ravel, xp_default_dtype

_ELIMITS = -1  # used in _bracket_root
_ESTOPONESIDE = 2  # used in _bracket_root

def _bracket_root_iv(func, xl0, xr0, xmin, xmax, factor, args, maxiter):

    if not callable(func):
        raise ValueError('`func` must be callable.')

    if not np.iterable(args):
        args = (args,)

    xp = array_namespace(xl0)
    xl0 = xp.asarray(xl0)[()]
    if (not xp.isdtype(xl0.dtype, "numeric")
        or xp.isdtype(xl0.dtype, "complex floating")):
        raise ValueError('`xl0` must be numeric and real.')
    if not xp.isdtype(xl0.dtype, "real floating"):
        xl0 = xp.asarray(xl0, dtype=xp_default_dtype(xp))

    # If xr0 is not supplied, fill with a dummy value for the sake of
    # broadcasting. We need to wait until xmax has been validated to
    # compute the default value.
    xr0_not_supplied = False
    if xr0 is None:
        xr0 = xp.nan
        xr0_not_supplied = True

    xmin = -xp.inf if xmin is None else xmin
    xmax = xp.inf if xmax is None else xmax
    factor = 2. if factor is None else factor
    xl0, xr0, xmin, xmax, factor = xp.broadcast_arrays(
        xl0, xp.asarray(xr0), xp.asarray(xmin), xp.asarray(xmax), xp.asarray(factor))

    if (not xp.isdtype(xr0.dtype, "numeric")
        or xp.isdtype(xr0.dtype, "complex floating")):
        raise ValueError('`xr0` must be numeric and real.')

    if (not xp.isdtype(xmin.dtype, "numeric")
        or xp.isdtype(xmin.dtype, "complex floating")):
        raise ValueError('`xmin` must be numeric and real.')

    if (not xp.isdtype(xmax.dtype, "numeric")
        or xp.isdtype(xmax.dtype, "complex floating")):
        raise ValueError('`xmax` must be numeric and real.')

    if (not xp.isdtype(factor.dtype, "numeric")
        or xp.isdtype(factor.dtype, "complex floating")):
        raise ValueError('`factor` must be numeric and real.')
    if not xp.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')

    # Calculate the default value of xr0 if a value has not been supplied.
    # Be careful to ensure xr0 is not larger than xmax.
    if xr0_not_supplied:
        xr0 = xl0 + xp.minimum((xmax - xl0)/ 8, xp.asarray(1.0))
        xr0 = xp.astype(xr0, xl0.dtype, copy=False)

    maxiter = xp.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if (not xp.isdtype(maxiter.dtype, "numeric") or maxiter.shape != tuple()
            or xp.isdtype(maxiter.dtype, "complex floating")):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)

    return func, xl0, xr0, xmin, xmax, factor, args, maxiter, xp


def _bracket_root(func, xl0, xr0=None, *, xmin=None, xmax=None, factor=None,
                  args=(), maxiter=1000):
    """Bracket the root of a monotonic scalar function of one variable

    This function works elementwise when `xl0`, `xr0`, `xmin`, `xmax`, `factor`, and
    the elements of `args` are broadcastable arrays.

    Parameters
    ----------
    func : callable
        The function for which the root is to be bracketed.
        The signature must be::

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with `x`. ``func`` must be an elementwise function: each element
        ``func(x)[i]`` must equal ``func(x[i])`` for all indices ``i``.
    xl0, xr0: float array_like
        Starting guess of bracket, which need not contain a root. If `xr0` is
        not provided, ``xr0 = xl0 + 1``. Must be broadcastable with one another.
    xmin, xmax : float array_like, optional
        Minimum and maximum allowable endpoints of the bracket, inclusive. Must
        be broadcastable with `xl0` and `xr0`.
    factor : float array_like, default: 2
        The factor used to grow the bracket. See notes for details.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.  Must be arrays
        broadcastable with `xl0`, `xr0`, `xmin`, and `xmax`. If the callable to be
        bracketed requires arguments that are not broadcastable with these
        arrays, wrap that callable with `func` such that `func` accepts
        only `x` and broadcastable arrays.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform.

    Returns
    -------
    res : _RichResult
        An instance of `scipy._lib._util._RichResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        xl, xr : float
            The lower and upper ends of the bracket, if the algorithm
            terminated successfully.
        fl, fr : float
            The function value at the lower and upper ends of the bracket.
        nfev : int
            The number of function evaluations required to find the bracket.
            This is distinct from the number of times `func` is *called*
            because the function may evaluated at multiple points in a single
            call.
        nit : int
            The number of iterations of the algorithm that were performed.
        status : int
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm produced a valid bracket.
            - ``-1`` : The bracket expanded to the allowable limits without finding a bracket.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``-5``: The initial bracket does not satisfy `xmin <= xl0 < xr0 < xmax`.
            - ``1`` : The algorithm is proceeding normally (in `callback` only).
            - ``2`` : A bracket was found in the opposite search direction (in `callback` only).

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).

    Notes
    -----
    This function generalizes an algorithm found in pieces throughout
    `scipy.stats`. The strategy is to iteratively grow the bracket ``(l, r)``
     until ``func(l) < 0 < func(r)``. The bracket grows to the left as follows.

    - If `xmin` is not provided, the distance between `xl0` and `l` is iteratively
      increased by `factor`.
    - If `xmin` is provided, the distance between `xmin` and `l` is iteratively
      decreased by `factor`. Note that this also *increases* the bracket size.

    Growth of the bracket to the right is analogous.

    Growth of the bracket in one direction stops when the endpoint is no longer
    finite, the function value at the endpoint is no longer finite, or the
    endpoint reaches its limiting value (`xmin` or `xmax`). Iteration terminates
    when the bracket stops growing in both directions, the bracket surrounds
    the root, or a root is found (accidentally).

    If two brackets are found - that is, a bracket is found on both sides in
    the same iteration, the smaller of the two is returned.
    If roots of the function are found, both `l` and `r` are set to the
    leftmost root.

    """  # noqa: E501
    # Todo:
    # - find bracket with sign change in specified direction
    # - Add tolerance
    # - allow factor < 1?

    callback = None  # works; I just don't want to test it
    temp = _bracket_root_iv(func, xl0, xr0, xmin, xmax, factor, args, maxiter)
    func, xl0, xr0, xmin, xmax, factor, args, maxiter, xp = temp

    xs = (xl0, xr0)
    temp = eim._initialize(func, xs, args)
    func, xs, fs, args, shape, dtype, xp = temp  # line split for PEP8
    xl0, xr0 = xs
    xmin = xp_ravel(xp.astype(xp.broadcast_to(xmin, shape), dtype, copy=False), xp=xp)
    xmax = xp_ravel(xp.astype(xp.broadcast_to(xmax, shape), dtype, copy=False), xp=xp)
    invalid_bracket = ~((xmin <= xl0) & (xl0 < xr0) & (xr0 <= xmax))

    # The approach is to treat the left and right searches as though they were
    # (almost) totally independent one-sided bracket searches. (The interaction
    # is considered when checking for termination and preparing the result
    # object.)
    # `x` is the "moving" end of the bracket
    x = xp.concat(xs)
    f = xp.concat(fs)
    invalid_bracket = xp.concat((invalid_bracket, invalid_bracket))
    n = x.shape[0] // 2

    # `x_last` is the previous location of the moving end of the bracket. If
    # the signs of `f` and `f_last` are different, `x` and `x_last` form a
    # bracket.
    x_last = xp.concat((x[n:], x[:n]))
    f_last = xp.concat((f[n:], f[:n]))
    # `x0` is the "fixed" end of the bracket.
    x0 = x_last
    # We don't need to retain the corresponding function value, since the
    # fixed end of the bracket is only needed to compute the new value of the
    # moving end; it is never returned.
    limit = xp.concat((xmin, xmax))

    factor = xp_ravel(xp.broadcast_to(factor, shape), xp=xp)
    factor = xp.astype(factor, dtype, copy=False)
    factor = xp.concat((factor, factor))

    active = xp.arange(2*n)
    args = [xp.concat((arg, arg)) for arg in args]

    # This is needed due to inner workings of `eim._loop`.
    # We're abusing it a tiny bit.
    shape = shape + (2,)

    # `d` is for "distance".
    # For searches without a limit, the distance between the fixed end of the
    # bracket `x0` and the moving end `x` will grow by `factor` each iteration.
    # For searches with a limit, the distance between the `limit` and moving
    # end of the bracket `x` will shrink by `factor` each iteration.
    i = xp.isinf(limit)
    ni = ~i
    d = xp.zeros_like(x)
    d[i] = x[i] - x0[i]
    d[ni] = limit[ni] - x[ni]

    status = xp.full_like(x, eim._EINPROGRESS, dtype=xp.int32)  # in progress
    status[invalid_bracket] = eim._EINPUTERR
    nit, nfev = 0, 1  # one function evaluation per side performed above

    work = _RichResult(x=x, x0=x0, f=f, limit=limit, factor=factor,
                       active=active, d=d, x_last=x_last, f_last=f_last,
                       nit=nit, nfev=nfev, status=status, args=args,
                       xl=xp.nan, xr=xp.nan, fl=xp.nan, fr=xp.nan, n=n)
    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xr', 'xr'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'),
                      ('fr', 'fr'), ('x', 'x'), ('f', 'f'),
                      ('x_last', 'x_last'), ('f_last', 'f_last')]

    def pre_func_eval(work):
        # Initialize moving end of bracket
        x = xp.zeros_like(work.x)

        # Unlimited brackets grow by `factor` by increasing distance from fixed
        # end to moving end.
        i = xp.isinf(work.limit)  # indices of unlimited brackets
        work.d[i] *= work.factor[i]
        x[i] = work.x0[i] + work.d[i]

        # Limited brackets grow by decreasing the distance from the limit to
        # the moving end.
        ni = ~i  # indices of limited brackets
        work.d[ni] /= work.factor[ni]
        x[ni] = work.limit[ni] - work.d[ni]

        return x

    def post_func_eval(x, f, work):
        # Keep track of the previous location of the moving end so that we can
        # return a narrower bracket. (The alternative is to remember the
        # original fixed end, but then the bracket would be wider than needed.)
        work.x_last = work.x
        work.f_last = work.f
        work.x = x
        work.f = f

    def check_termination(work):
        # Condition 0: initial bracket is invalid
        stop = (work.status == eim._EINPUTERR)

        # Condition 1: a valid bracket (or the root itself) has been found
        sf = xp.sign(work.f)
        sf_last = xp.sign(work.f_last)
        i = ((sf_last == -sf) | (sf_last == 0) | (sf == 0)) & ~stop
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        # Condition 2: the other side's search found a valid bracket.
        # (If we just found a bracket with the rightward search, we can stop
        #  the leftward search, and vice-versa.)
        # To do this, we need to set the status of the other side's search;
        # this is tricky because `work.status` contains only the *active*
        # elements, so we don't immediately know the index of the element we
        # need to set - or even if it's still there. (That search may have
        # terminated already, e.g. by reaching its `limit`.)
        # To facilitate this, `work.active` contains a unit integer index of
        # each search. Index `k` (`k < n)` and `k + n` correspond with a
        # leftward and rightward search, respectively. Elements are removed
        # from `work.active` just as they are removed from `work.status`, so
        # we use `work.active` to help find the right location in
        # `work.status`.
        # Get the integer indices of the elements that can also stop
        also_stop = (work.active[i] + work.n) % (2*work.n)
        # Check whether they are still active. We want to find the indices
        # in work.active where the associated values in work.active are
        # contained in also_stop. xp.searchsorted let's us take advantage
        # of work.active being sorted, but requires some hackery because
        # searchsorted solves the separate but related problem of finding
        # the indices where the values in also_stop should be added to
        # maintain sorted order.
        j = xp.searchsorted(work.active, also_stop)
        # If the location exceeds the length of the `work.active`, they are
        # not there. This happens when a value in also_stop is larger than
        # the greatest value in work.active. This case needs special handling
        # because we cannot simply check that also_stop == work.active[j].
        mask = j < work.active.shape[0]
        # Note that we also have to use the mask to filter also_stop to ensure
        # that also_stop and j will still have the same shape.
        j, also_stop = j[mask], also_stop[mask]
        j = j[also_stop == work.active[j]]
        # Now convert these to boolean indices to use with `work.status`.
        i = xp.zeros_like(stop)
        i[j] = True  # boolean indices of elements that can also stop
        i = i & ~stop
        work.status[i] = _ESTOPONESIDE
        stop[i] = True

        # Condition 3: moving end of bracket reaches limit
        i = (work.x == work.limit) & ~stop
        work.status[i] = _ELIMITS
        stop[i] = True

        # Condition 4: non-finite value encountered
        i = ~(xp.isfinite(work.x) & xp.isfinite(work.f)) & ~stop
        work.status[i] = eim._EVALUEERR
        stop[i] = True

        return stop

    def post_termination_check(work):
        pass

    def customize_result(res, shape):
        n = res['x'].shape[0] // 2

        # To avoid ambiguity, below we refer to `xl0`, the initial left endpoint
        # as `a` and `xr0`, the initial right endpoint, as `b`.
        # Because we treat the two one-sided searches as though they were
        # independent, what we keep track of in `work` and what we want to
        # return in `res` look quite different. Combine the results from the
        # two one-sided searches before reporting the results to the user.
        # - "a" refers to the leftward search (the moving end started at `a`)
        # - "b" refers to the rightward search (the moving end started at `b`)
        # - "l" refers to the left end of the bracket (closer to -oo)
        # - "r" refers to the right end of the bracket (closer to +oo)
        xal = res['x'][:n]
        xar = res['x_last'][:n]
        xbl = res['x_last'][n:]
        xbr = res['x'][n:]

        fal = res['f'][:n]
        far = res['f_last'][:n]
        fbl = res['f_last'][n:]
        fbr = res['f'][n:]

        # Initialize the brackets and corresponding function values to return
        # to the user. Brackets may not be valid (e.g. there is no root,
        # there weren't enough iterations, NaN encountered), but we still need
        # to return something. One option would be all NaNs, but what I've
        # chosen here is the left- and right-most points at which the function
        # has been evaluated. This gives the user some information about what
        # interval of the real line has been searched and shows that there is
        # no sign change between the two ends.
        xl = xp.asarray(xal, copy=True)
        fl = xp.asarray(fal, copy=True)
        xr = xp.asarray(xbr, copy=True)
        fr = xp.asarray(fbr, copy=True)

        # `status` indicates whether the bracket is valid or not. If so,
        # we want to adjust the bracket we return to be the narrowest possible
        # given the points at which we evaluated the function.
        # For example if bracket "a" is valid and smaller than bracket "b" OR
        # if bracket "a" is valid and bracket "b" is not valid, we want to
        # return bracket "a" (and vice versa).
        sa = res['status'][:n]
        sb = res['status'][n:]

        da = xar - xal
        db = xbr - xbl

        i1 = ((da <= db) & (sa == 0)) | ((sa == 0) & (sb != 0))
        i2 = ((db <= da) & (sb == 0)) | ((sb == 0) & (sa != 0))

        xr[i1] = xar[i1]
        fr[i1] = far[i1]
        xl[i2] = xbl[i2]
        fl[i2] = fbl[i2]

        # Finish assembling the result object
        res['xl'] = xl
        res['xr'] = xr
        res['fl'] = fl
        res['fr'] = fr

        res['nit'] = xp.maximum(res['nit'][:n], res['nit'][n:])
        res['nfev'] = res['nfev'][:n] + res['nfev'][n:]
        # If the status on one side is zero, the status is zero. In any case,
        # report the status from one side only.
        res['status'] = xp.where(sa == 0, sa, sb)
        res['success'] = (res['status'] == 0)

        del res['x']
        del res['f']
        del res['x_last']
        del res['f_last']

        return shape[:-1]

    return eim._loop(work, callback, shape, maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval, check_termination,
                     post_termination_check, customize_result, res_work_pairs,
                     xp)


def _bracket_minimum_iv(func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter):

    if not callable(func):
        raise ValueError('`func` must be callable.')

    if not np.iterable(args):
        args = (args,)

    xp = array_namespace(xm0)
    xm0 = xp.asarray(xm0)[()]
    if (not xp.isdtype(xm0.dtype, "numeric")
        or xp.isdtype(xm0.dtype, "complex floating")):
        raise ValueError('`xm0` must be numeric and real.')
    if not xp.isdtype(xm0.dtype, "real floating"):
        xm0 = xp.asarray(xm0, dtype=xp_default_dtype(xp))

    xmin = -xp.inf if xmin is None else xmin
    xmax = xp.inf if xmax is None else xmax

    # If xl0 (xr0) is not supplied, fill with a dummy value for the sake
    # of broadcasting. We need to wait until xmin (xmax) has been validated
    # to compute the default values.
    xl0_not_supplied = False
    if xl0 is None:
        xl0 = xp.nan
        xl0_not_supplied = True

    xr0_not_supplied = False
    if xr0 is None:
        xr0 = xp.nan
        xr0_not_supplied = True

    factor = 2.0 if factor is None else factor
    xl0, xm0, xr0, xmin, xmax, factor = xp.broadcast_arrays(
        xp.asarray(xl0), xm0, xp.asarray(xr0), xp.asarray(xmin),
        xp.asarray(xmax), xp.asarray(factor)
    )

    if (not xp.isdtype(xl0.dtype, "numeric")
        or xp.isdtype(xl0.dtype, "complex floating")):
        raise ValueError('`xl0` must be numeric and real.')

    if (not xp.isdtype(xr0.dtype, "numeric")
        or xp.isdtype(xr0.dtype, "complex floating")):
        raise ValueError('`xr0` must be numeric and real.')

    if (not xp.isdtype(xmin.dtype, "numeric")
        or xp.isdtype(xmin.dtype, "complex floating")):
        raise ValueError('`xmin` must be numeric and real.')

    if (not xp.isdtype(xmax.dtype, "numeric")
        or xp.isdtype(xmax.dtype, "complex floating")):
        raise ValueError('`xmax` must be numeric and real.')

    if (not xp.isdtype(factor.dtype, "numeric")
        or xp.isdtype(factor.dtype, "complex floating")):
        raise ValueError('`factor` must be numeric and real.')
    if not xp.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')

    # Calculate default values of xl0 and/or xr0 if they have not been supplied
    # by the user. We need to be careful to ensure xl0 and xr0 are not outside
    # of (xmin, xmax).
    if xl0_not_supplied:
        xl0 = xm0 - xp.minimum((xm0 - xmin)/16, xp.asarray(0.5))
        xl0 = xp.astype(xl0, xm0.dtype, copy=False)
    if xr0_not_supplied:
        xr0 = xm0 + xp.minimum((xmax - xm0)/16, xp.asarray(0.5))
        xr0 = xp.astype(xr0, xm0.dtype, copy=False)

    maxiter = xp.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if (not xp.isdtype(maxiter.dtype, "numeric") or maxiter.shape != tuple()
            or xp.isdtype(maxiter.dtype, "complex floating")):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)

    return func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter, xp


def _bracket_minimum(func, xm0, *, xl0=None, xr0=None, xmin=None, xmax=None,
                     factor=None, args=(), maxiter=1000):
    """Bracket the minimum of a unimodal scalar function of one variable

    This function works elementwise when `xm0`, `xl0`, `xr0`, `xmin`, `xmax`,
    and the elements of `args` are broadcastable arrays.

    Parameters
    ----------
    func : callable
        The function for which the minimum is to be bracketed.
        The signature must be::

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``. `func` must be an elementwise function: each element
        ``func(x)[i]`` must equal ``func(x[i])`` for all indices `i`.
    xm0: float array_like
        Starting guess for middle point of bracket.
    xl0, xr0: float array_like, optional
        Starting guesses for left and right endpoints of the bracket. Must be
        broadcastable with one another and with `xm0`.
    xmin, xmax : float array_like, optional
        Minimum and maximum allowable endpoints of the bracket, inclusive. Must
        be broadcastable with `xl0`, `xm0`, and `xr0`.
    factor : float array_like, optional
        Controls expansion of bracket endpoint in downhill direction. Works
        differently in the cases where a limit is set in the downhill direction
        with `xmax` or `xmin`. See Notes.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.  Must be arrays
        broadcastable with `xl0`, `xm0`, `xr0`, `xmin`, and `xmax`. If the
        callable to be bracketed requires arguments that are not broadcastable
        with these arrays, wrap that callable with `func` such that `func`
        accepts only ``x`` and broadcastable arrays.
    maxiter : int, optional
        The maximum number of iterations of the algorithm to perform. The number
        of function evaluations is three greater than the number of iterations.

    Returns
    -------
    res : _RichResult
        An instance of `scipy._lib._util._RichResult` with the following
        attributes. The descriptions are written as though the values will be
        scalars; however, if `func` returns an array, the outputs will be
        arrays of the same shape.

        xl, xm, xr : float
            The left, middle, and right points of the bracket, if the algorithm
            terminated successfully.
        fl, fm, fr : float
            The function value at the left, middle, and right points of the bracket.
        nfev : int
            The number of function evaluations required to find the bracket.
        nit : int
            The number of iterations of the algorithm that were performed.
        status : int
            An integer representing the exit status of the algorithm.

            - ``0`` : The algorithm produced a valid bracket.
            - ``-1`` : The bracket expanded to the allowable limits. Assuming
                       unimodality, this implies the endpoint at the limit is a
                       minimizer.
            - ``-2`` : The maximum number of iterations was reached.
            - ``-3`` : A non-finite value was encountered.
            - ``-4`` : ``None`` shall pass.
            - ``-5`` : The initial bracket does not satisfy
                       `xmin <= xl0 < xm0 < xr0 <= xmax`.

        success : bool
            ``True`` when the algorithm terminated successfully (status ``0``).

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
    """  # noqa: E501
    callback = None  # works; I just don't want to test it

    temp = _bracket_minimum_iv(func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter)
    func, xm0, xl0, xr0, xmin, xmax, factor, args, maxiter, xp = temp

    xs = (xl0, xm0, xr0)
    temp = eim._initialize(func, xs, args)
    func, xs, fs, args, shape, dtype, xp = temp

    xl0, xm0, xr0 = xs
    fl0, fm0, fr0 = fs
    xmin = xp.astype(xp.broadcast_to(xmin, shape), dtype, copy=False)
    xmin = xp_ravel(xmin, xp=xp)
    xmax = xp.astype(xp.broadcast_to(xmax, shape), dtype, copy=False)
    xmax = xp_ravel(xmax, xp=xp)
    invalid_bracket = ~((xmin <= xl0) & (xl0 < xm0) & (xm0 < xr0) & (xr0 <= xmax))
    # We will modify factor later on so make a copy. np.broadcast_to returns
    # a read-only view.
    factor = xp.astype(xp.broadcast_to(factor, shape), dtype, copy=True)
    factor = xp_ravel(factor)

    # To simplify the logic, swap xl and xr if f(xl) < f(xr). We should always be
    # marching downhill in the direction from xl to xr.
    comp = fl0 < fr0
    xl0[comp], xr0[comp] = xr0[comp], xl0[comp]
    fl0[comp], fr0[comp] = fr0[comp], fl0[comp]
    # We only need the boundary in the direction we're traveling.
    limit = xp.where(comp, xmin, xmax)

    unlimited = xp.isinf(limit)
    limited = ~unlimited
    step = xp.empty_like(xl0)

    step[unlimited] = (xr0[unlimited] - xm0[unlimited])
    step[limited] = (limit[limited] - xr0[limited])

    # Step size is divided by factor for case where there is a limit.
    factor[limited] = 1 / factor[limited]

    status = xp.full_like(xl0, eim._EINPROGRESS, dtype=xp.int32)
    status[invalid_bracket] = eim._EINPUTERR
    nit, nfev = 0, 3

    work = _RichResult(xl=xl0, xm=xm0, xr=xr0, xr0=xr0, fl=fl0, fm=fm0, fr=fr0,
                       step=step, limit=limit, limited=limited, factor=factor, nit=nit,
                       nfev=nfev, status=status, args=args)

    res_work_pairs = [('status', 'status'), ('xl', 'xl'), ('xm', 'xm'), ('xr', 'xr'),
                      ('nit', 'nit'), ('nfev', 'nfev'), ('fl', 'fl'), ('fm', 'fm'),
                      ('fr', 'fr')]

    def pre_func_eval(work):
        work.step *= work.factor
        x = xp.empty_like(work.xr)
        x[~work.limited] = work.xr0[~work.limited] + work.step[~work.limited]
        x[work.limited] = work.limit[work.limited] - work.step[work.limited]
        # Since the new bracket endpoint is calculated from an offset with the
        # limit, it may be the case that the new endpoint equals the old endpoint,
        # when the old endpoint is sufficiently close to the limit. We use the
        # limit itself as the new endpoint in these cases.
        x[work.limited] = xp.where(
            x[work.limited] == work.xr[work.limited],
            work.limit[work.limited],
            x[work.limited],
        )
        return x

    def post_func_eval(x, f, work):
        work.xl, work.xm, work.xr = work.xm, work.xr, x
        work.fl, work.fm, work.fr = work.fm, work.fr, f

    def check_termination(work):
        # Condition 0: Initial bracket is invalid.
        stop = (work.status == eim._EINPUTERR)

        # Condition 1: A valid bracket has been found.
        i = (
            (work.fl >= work.fm) & (work.fr > work.fm)
            | (work.fl > work.fm) & (work.fr >= work.fm)
        ) & ~stop
        work.status[i] = eim._ECONVERGED
        stop[i] = True

        # Condition 2: Moving end of bracket reaches limit.
        i = (work.xr == work.limit) & ~stop
        work.status[i] = _ELIMITS
        stop[i] = True

        # Condition 3: non-finite value encountered
        i = ~(xp.isfinite(work.xr) & xp.isfinite(work.fr)) & ~stop
        work.status[i] = eim._EVALUEERR
        stop[i] = True

        return stop

    def post_termination_check(work):
        pass

    def customize_result(res, shape):
        # Reorder entries of xl and xr if they were swapped due to f(xl0) < f(xr0).
        comp = res['xl'] > res['xr']
        res['xl'][comp], res['xr'][comp] = res['xr'][comp], res['xl'][comp]
        res['fl'][comp], res['fr'][comp] = res['fr'][comp], res['fl'][comp]
        return shape

    return eim._loop(work, callback, shape,
                     maxiter, func, args, dtype,
                     pre_func_eval, post_func_eval,
                     check_termination, post_termination_check,
                     customize_result, res_work_pairs, xp)
<|endoftext|>import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult
import numpy as np


_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps

__all__ = ['newton', 'bisect', 'ridder', 'brentq', 'brenth', 'toms748',
           'RootResults']

# Must agree with CONVERGED, SIGNERR, CONVERR, ...  in zeros.h
_ECONVERGED = 0
_ESIGNERR = -1  # used in _chandrupatla
_ECONVERR = -2
_EVALUEERR = -3
_ECALLBACK = -4
_EINPROGRESS = 1

CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'


flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR,
            _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}


class RootResults(OptimizeResult):
    """Represents the root finding result.

    Attributes
    ----------
    root : float
        Estimated root location.
    iterations : int
        Number of iterations needed to find the root.
    function_calls : int
        Number of times the function was called.
    converged : bool
        True if the routine converged.
    flag : str
        Description of the cause of termination.
    method : str
        Root finding method used.

    """

    def __init__(self, root, iterations, function_calls, flag, method):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        if flag in flag_map:
            self.flag = flag_map[flag]
        else:
            self.flag = flag
        self.method = method


def results_c(full_output, r, method):
    if full_output:
        x, funcalls, iterations, flag = r
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    else:
        return r


def _results_select(full_output, r, method):
    """Select from a tuple of (root, funccalls, iterations, flag)"""
    x, funcalls, iterations, flag = r
    if full_output:
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    return x


def _wrap_nan_raise(f):

    def f_raise(x, *args):
        fx = f(x, *args)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = (f'The function value at x={x} is NaN; '
                   'solver cannot continue.')
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx

    f_raise._function_calls = 0
    return f_raise


def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50,
           fprime2=None, x1=None, rtol=0.0,
           full_output=False, disp=True):
    """
    Find a root of a real or complex function using the Newton-Raphson
    (or secant or Halley's) method.

    Find a root of the scalar-valued function `func` given a nearby scalar
    starting point `x0`.
    The Newton-Raphson method is used if the derivative `fprime` of `func`
    is provided, otherwise the secant method is used. If the second order
    derivative `fprime2` of `func` is also provided, then Halley's method is
    used.

    If `x0` is a sequence with more than one item, `newton` returns an array:
    the roots of the function from each (scalar) starting point in `x0`.
    In this case, `func` must be vectorized to return a sequence or array of
    the same shape as its first argument. If `fprime` (`fprime2`) is given,
    then its return must also have the same shape: each element is the first
    (second) derivative of `func` with respect to its only variable evaluated
    at each element of its first argument.

    `newton` is for finding roots of a scalar-valued functions of a single
    variable. For problems involving several variables, see `root`.

    Parameters
    ----------
    func : callable
        The function whose root is wanted. It must be a function of a
        single variable of the form ``f(x,a,b,c...)``, where ``a,b,c...``
        are extra arguments that can be passed in the `args` parameter.
    x0 : float, sequence, or ndarray
        An initial estimate of the root that should be somewhere near the
        actual root. If not scalar, then `func` must be vectorized and return
        a sequence or array of the same shape as its first argument.
    fprime : callable, optional
        The derivative of the function when available and convenient. If it
        is None (default), then the secant method is used.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the root's value. If `func` is complex-valued,
        a larger `tol` is recommended as both the real and imaginary parts
        of `x` contribute to ``|x - x0|``.
    maxiter : int, optional
        Maximum number of iterations.
    fprime2 : callable, optional
        The second order derivative of the function when available and
        convenient. If it is None (default), then the normal Newton-Raphson
        or the secant method is used. If it is not None, then Halley's method
        is used.
    x1 : float, optional
        Another estimate of the root that should be somewhere near the
        actual root. Used if `fprime` is not provided.
    rtol : float, optional
        Tolerance (relative) for termination.
    full_output : bool, optional
        If `full_output` is False (default), the root is returned.
        If True and `x0` is scalar, the return value is ``(x, r)``, where ``x``
        is the root and ``r`` is a `RootResults` object.
        If True and `x0` is non-scalar, the return value is ``(x, converged,
        zero_der)`` (see Returns section for details).
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge, with
        the error message containing the number of iterations and current
        function value. Otherwise, the convergence status is recorded in a
        `RootResults` return object.
        Ignored if `x0` is not scalar.
        *Note: this has little to do with displaying, however,
        the `disp` keyword cannot be renamed for backwards compatibility.*

    Returns
    -------
    root : float, sequence, or ndarray
        Estimated location where function is zero.
    r : `RootResults`, optional
        Present if ``full_output=True`` and `x0` is scalar.
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.
    converged : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements converged successfully.
    zero_der : ndarray of bool, optional
        Present if ``full_output=True`` and `x0` is non-scalar.
        For vector functions, indicates which elements had a zero derivative.

    See Also
    --------
    root_scalar : interface to root solvers for scalar functions
    root : interface to root solvers for multi-input, multi-output functions

    Notes
    -----
    The convergence rate of the Newton-Raphson method is quadratic,
    the Halley method is cubic, and the secant method is
    sub-quadratic. This means that if the function is well-behaved
    the actual error in the estimated root after the nth iteration
    is approximately the square (cube for Halley) of the error
    after the (n-1)th step. However, the stopping criterion used
    here is the step size and there is no guarantee that a root
    has been found. Consequently, the result should be verified.
    Safer algorithms are brentq, brenth, ridder, and bisect,
    but they all require that the root first be bracketed in an
    interval where the function changes sign. The brentq algorithm
    is recommended for general use in one dimensional problems
    when such an interval has been found.

    When `newton` is used with arrays, it is best suited for the following
    types of problems:

    * The initial guesses, `x0`, are all relatively the same distance from
      the roots.
    * Some or all of the extra arguments, `args`, are also arrays so that a
      class of similar problems can be solved together.
    * The size of the initial guesses, `x0`, is larger than O(100) elements.
      Otherwise, a naive loop may perform as well or better than a vector.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import optimize

    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    ``fprime`` is not provided, use the secant method:

    >>> root = optimize.newton(f, 1.5)
    >>> root
    1.0000000000000016
    >>> root = optimize.newton(f, 1.5, fprime2=lambda x: 6 * x)
    >>> root
    1.0000000000000016

    Only ``fprime`` is provided, use the Newton-Raphson method:

    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2)
    >>> root
    1.0

    Both ``fprime2`` and ``fprime`` are provided, use Halley's method:

    >>> root = optimize.newton(f, 1.5, fprime=lambda x: 3 * x**2,
    ...                        fprime2=lambda x: 6 * x)
    >>> root
    1.0

    When we want to find roots for a set of related starting values and/or
    function parameters, we can provide both of those as an array of inputs:

    >>> f = lambda x, a: x**3 - a
    >>> fder = lambda x, a: 3 * x**2
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(100)
    >>> a = np.arange(-50, 50)
    >>> vec_res = optimize.newton(f, x, fprime=fder, args=(a, ), maxiter=200)

    The above is the equivalent of solving for each value in ``(x, a)``
    separately in a for-loop, just faster:

    >>> loop_res = [optimize.newton(f, x0, fprime=fder, args=(a0,),
    ...                             maxiter=200)
    ...             for x0, a0 in zip(x, a)]
    >>> np.allclose(vec_res, loop_res)
    True

    Plot the results found for all values of ``a``:

    >>> analytical_result = np.sign(a) * np.abs(a)**(1/3)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(a, analytical_result, 'o')
    >>> ax.plot(a, vec_res, '.')
    >>> ax.set_xlabel('$a$')
    >>> ax.set_ylabel('$x$ where $f(x, a)=0$')
    >>> plt.show()

    """
    if tol <= 0:
        raise ValueError(f"tol too small ({tol:g} <= 0)")
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    if np.size(x0) > 1:
        return _array_newton(func, x0, fprime, args, tol, maxiter, fprime2,
                             full_output)

    # Convert to float (don't use float(x0); this works also for complex x0)
    # Use np.asarray because we want x0 to be a numpy object, not a Python
    # object. e.g. np.complex(1+1j) > 0 is possible, but (1 + 1j) > 0 raises
    # a TypeError
    x0 = np.asarray(x0)[()] * 1.0
    p0 = x0
    funcalls = 0
    if fprime is not None:
        # Newton-Raphson method
        method = "newton"
        for itr in range(maxiter):
            # first evaluate fval
            fval = func(p0, *args)
            funcalls += 1
            # If fval is 0, a root has been found, then terminate
            if fval == 0:
                return _results_select(
                    full_output, (p0, funcalls, itr, _ECONVERGED), method)
            fder = fprime(p0, *args)
            funcalls += 1
            if fder == 0:
                msg = "Derivative was zero."
                if disp:
                    msg += (
                        " Failed to converge after %d iterations, value is %s."
                        % (itr + 1, p0))
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return _results_select(
                    full_output, (p0, funcalls, itr + 1, _ECONVERR), method)
            newton_step = fval / fder
            if fprime2:
                fder2 = fprime2(p0, *args)
                funcalls += 1
                method = "halley"
                # Halley's method:
                #   newton_step /= (1.0 - 0.5 * newton_step * fder2 / fder)
                # Only do it if denominator stays close enough to 1
                # Rationale: If 1-adj < 0, then Halley sends x in the
                # opposite direction to Newton. Doesn't happen if x is close
                # enough to root.
                adj = newton_step * fder2 / fder / 2
                if np.abs(adj) < 1:
                    newton_step /= 1.0 - adj
            p = p0 - newton_step
            if np.isclose(p, p0, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            p0 = p
    else:
        # Secant method
        method = "secant"
        if x1 is not None:
            if x1 == x0:
                raise ValueError("x1 and x0 must be different")
            p1 = x1
        else:
            eps = 1e-4
            p1 = x0 * (1 + eps)
            p1 += (eps if p1 >= 0 else -eps)
        q0 = func(p0, *args)
        funcalls += 1
        q1 = func(p1, *args)
        funcalls += 1
        if abs(q1) < abs(q0):
            p0, p1, q0, q1 = p1, p0, q1, q0
        for itr in range(maxiter):
            if q1 == q0:
                if p1 != p0:
                    msg = f"Tolerance of {p1 - p0} reached."
                    if disp:
                        msg += (
                            " Failed to converge after %d iterations, value is %s."
                            % (itr + 1, p1))
                        raise RuntimeError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                p = (p1 + p0) / 2.0
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERR), method)
            else:
                if abs(q1) > abs(q0):
                    p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
                else:
                    p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            if np.isclose(p, p1, rtol=rtol, atol=tol):
                return _results_select(
                    full_output, (p, funcalls, itr + 1, _ECONVERGED), method)
            p0, q0 = p1, q1
            p1 = p
            q1 = func(p1, *args)
            funcalls += 1

    if disp:
        msg = ("Failed to converge after %d iterations, value is %s."
               % (itr + 1, p))
        raise RuntimeError(msg)

    return _results_select(full_output, (p, funcalls, itr + 1, _ECONVERR), method)


def _array_newton(func, x0, fprime, args, tol, maxiter, fprime2, full_output):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.

    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.
    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = np.array(x0, copy=True)

    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)
    if fprime is not None:
        # Newton-Raphson method
        for iteration in range(maxiter):
            # first evaluate fval
            fval = np.asarray(func(p, *args))
            # If all fval are 0, all roots have been found, then terminate
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = np.asarray(fprime(p, *args))
            nz_der = (fder != 0)
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                break
            # Newton step
            dp = fval[nz_der] / fder[nz_der]
            if fprime2 is not None:
                fder2 = np.asarray(fprime2(p, *args))
                dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, dp, np.float64))
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
    else:
        # Secant method
        dx = np.finfo(float).eps**0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = np.asarray(func(p, *args))
        q1 = np.asarray(func(p1, *args))
        active = np.ones_like(p, dtype=bool)
        for iteration in range(maxiter):
            nz_der = (q1 != q0)
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # Secant Step
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            # only update nonzero derivatives
            p = np.asarray(p, dtype=np.result_type(p, p1, dp, np.float64))
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # don't assign zero derivatives again
            failures[nz_der] = np.abs(dp) >= tol  # not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = np.asarray(func(p1, *args))

    zero_der = ~nz_der & failures  # don't include converged with zero-ders
    if zero_der.any():
        # Secant warnings
        if fprime is None:
            nonzero_dp = (p1 != p)
            # non-zero dp, but infinite newton step
            zero_der_nz_dp = (zero_der & nonzero_dp)
            if zero_der_nz_dp.any():
                rms = np.sqrt(
                    sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2)
                )
                warnings.warn(f'RMS of {rms:g} reached', RuntimeWarning, stacklevel=3)
        # Newton or Halley warnings
        else:
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = f'{all_or_some:s} derivatives were zero'
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = f'{all_or_some:s} failed to converge after {maxiter:d} iterations'
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)

    return p


def bisect(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find root of a function within an interval using bisection.

    Basic bisection routine to find a root of the function `f` between the
    arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.
    Slow but sure.

    Parameters
    ----------
    f : function
        Python function returning a number.  `f` must be continuous, and
        f(a) and f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where x is the root, and r is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in a `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    Examples
    --------

    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.bisect(f, 0, 2)
    >>> root
    1.0

    >>> root = optimize.bisect(f, -2, 0)
    >>> root
    -1.0

    See Also
    --------
    brentq, brenth, bisect, newton
    fixed_point : scalar fixed-point finder
    fsolve : n-dimensional root-finding

    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError(f"xtol too small ({xtol:g} <= 0)")
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._bisect(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "bisect")


def ridder(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find a root of a function in an interval using Ridder's method.

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence.
        In particular, ``r.converged`` is True if the routine converged.

    See Also
    --------
    brentq, brenth, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    Notes
    -----
    Uses [Ridders1979]_ method to find a root of the function `f` between the
    arguments `a` and `b`. Ridders' method is faster than bisection, but not
    generally as fast as the Brent routines. [Ridders1979]_ provides the
    classic description and source of the algorithm. A description can also be
    found in any recent edition of Numerical Recipes.

    The routine used here diverges slightly from standard presentations in
    order to be a bit more careful of tolerance.

    References
    ----------
    .. [Ridders1979]
       Ridders, C. F. J. "A New Algorithm for Computing a
       Single Root of a Real Continuous Function."
       IEEE Trans. Circuits Systems 26, 979-980, 1979.

    Examples
    --------

    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.ridder(f, 0, 2)
    >>> root
    1.0

    >>> root = optimize.ridder(f, -2, 0)
    >>> root
    -1.0
    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError(f"xtol too small ({xtol:g} <= 0)")
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._ridder(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "ridder")


def brentq(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """
    Find a root of a function in a bracketing interval using Brent's method.

    Uses the classic Brent's method to find a root of the function `f` on
    the sign changing interval [a , b]. Generally considered the best of the
    rootfinding routines here. It is a safe version of the secant method that
    uses inverse quadratic extrapolation. Brent's method combines root
    bracketing, interval bisection, and inverse quadratic interpolation. It is
    sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973)
    claims convergence is guaranteed for functions computable within [a,b].

    [Brent1973]_ provides the classic description of the algorithm. Another
    description can be found in a recent edition of Numerical Recipes, including
    [PressEtal1992]_. A third description is at
    http://mathworld.wolfram.com/BrentsMethod.html. It should be easy to
    understand the algorithm just by reading our code. Our code diverges a bit
    from standard presentations: we choose a different formula for the
    extrapolation step.

    Parameters
    ----------
    f : function
        Python function returning a number. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)` must
        have opposite signs.
    a : scalar
        One end of the bracketing interval :math:`[a, b]`.
    b : scalar
        The other end of the bracketing interval :math:`[a, b]`.
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
    leastsq : nonlinear least squares minimizer
    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
    basinhopping, differential_evolution, brute : global optimizers
    fminbound, brent, golden, bracket : local scalar minimizers
    fsolve : N-D root-finding
    brenth, ridder, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    Notes
    -----
    `f` must be continuous.  f(a) and f(b) must have opposite signs.

    References
    ----------
    .. [Brent1973]
       Brent, R. P.,
       *Algorithms for Minimization Without Derivatives*.
       Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.

    .. [PressEtal1992]
       Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.
       *Numerical Recipes in FORTRAN: The Art of Scientific Computing*, 2nd ed.
       Cambridge, England: Cambridge University Press, pp. 352-355, 1992.
       Section 9.3:  "Van Wijngaarden-Dekker-Brent Method."

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brentq(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brentq(f, 0, 2)
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
    r = _zeros._brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "brentq")


def brenth(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    """Find a root of a function in a bracketing interval using Brent's
    method with hyperbolic extrapolation.

    A variation on the classic Brent routine to find a root of the function f
    between the arguments a and b that uses hyperbolic extrapolation instead of
    inverse quadratic extrapolation. Bus & Dekker (1975) guarantee convergence
    for this method, claiming that the upper bound of function evaluations here
    is 4 or 5 times that of bisection.
    f(a) and f(b) cannot have the same signs. Generally, on a par with the
    brent routine, but not as heavily tested. It is a safe version of the
    secant method that uses hyperbolic extrapolation.
    The version here is by Chuck Harris, and implements Algorithm M of
    [BusAndDekker1975]_, where further details (convergence properties,
    additional remarks and such) can be found

    Parameters
    ----------
    f : function
        Python function returning a number. f must be continuous, and f(a) and
        f(b) must have opposite signs.
    a : scalar
        One end of the bracketing interval [a,b].
    b : scalar
        The other end of the bracketing interval [a,b].
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. As with `brentq`, for nice
        functions the method will often satisfy the above condition
        with ``xtol/2`` and ``rtol/2``.
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. As with `brentq`, for nice functions
        the method will often satisfy the above condition with
        ``xtol/2`` and ``rtol/2``.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    See Also
    --------
    fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg : multivariate local optimizers
    leastsq : nonlinear least squares minimizer
    fmin_l_bfgs_b, fmin_tnc, fmin_cobyla : constrained multivariate optimizers
    basinhopping, differential_evolution, brute : global optimizers
    fminbound, brent, golden, bracket : local scalar minimizers
    fsolve : N-D root-finding
    brentq, ridder, bisect, newton : 1-D root-finding
    fixed_point : scalar fixed-point finder

    References
    ----------
    .. [BusAndDekker1975]
       Bus, J. C. P., Dekker, T. J.,
