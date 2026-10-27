        raise LinAlgError("Illegal value encountered in "
                          "the %d term" % (-info,))

    return np.dot(np.dot(u, y), v.conj().transpose())


def solve_continuous_lyapunov(a, q):
    """
    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.

    Uses the Bartels-Stewart algorithm to find :math:`X`.

    Parameters
    ----------
    a : array_like
        A square matrix

    q : array_like
        Right-hand side square matrix

    Returns
    -------
    x : ndarray
        Solution to the continuous Lyapunov equation

    See Also
    --------
    solve_discrete_lyapunov : computes the solution to the discrete-time
        Lyapunov equation
    solve_sylvester : computes the solution to the Sylvester equation

    Notes
    -----
    The continuous Lyapunov equation is a special form of the Sylvester
    equation, hence this solver relies on LAPACK routine ?TRSYL.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
    >>> b = np.array([2, 4, -1])
    >>> q = np.eye(3)
    >>> x = linalg.solve_continuous_lyapunov(a, q)
    >>> x
    array([[ -0.75  ,   0.875 ,  -3.75  ],
           [  0.875 ,  -1.375 ,   5.3125],
           [ -3.75  ,   5.3125, -27.0625]])
    >>> np.allclose(a.dot(x) + x.dot(a.T), q)
    True
    """

    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))

    r_or_c = float

    for ind, _ in enumerate((a, q)):
        if np.iscomplexobj(_):
            r_or_c = complex

        if not np.equal(*_.shape):
            raise ValueError(f"Matrix {'aq'[ind]} should be square.")

    # Shape consistency check
    if a.shape != q.shape:
        raise ValueError("Matrix a and q should have the same shape.")

    # Accommodate empty array
    if a.size == 0:
        tdict = {'s': np.float32, 'd': np.float64,
                 'c': np.complex64, 'z': np.complex128}
        func, = get_lapack_funcs(('trsyl',), arrays=(a, q))
        return np.empty(a.shape, dtype=tdict[func.typecode])

    # Compute the Schur decomposition form of a
    r, u = schur(a, output='real')

    # Construct f = u'*q*u
    f = u.conj().T.dot(q.dot(u))

    # Call the Sylvester equation solver
    trsyl = get_lapack_funcs('trsyl', (r, f))

    dtype_string = 'T' if r_or_c is float else 'C'
    y, scale, info = trsyl(r, r, f, tranb=dtype_string)

    if info < 0:
        raise ValueError('?TRSYL exited with the internal error '
                         f'"illegal value in argument number {-info}.". See '
                         'LAPACK documentation for the ?TRSYL error codes.')
    elif info == 1:
        warnings.warn('Input "a" has an eigenvalue pair whose sum is '
                      'very close to or exactly zero. The solution is '
                      'obtained via perturbing the coefficients.',
                      RuntimeWarning, stacklevel=2)
    y *= scale

    return u.dot(y).dot(u.conj().T)


# For backwards compatibility, keep the old name
solve_lyapunov = solve_continuous_lyapunov


def _solve_discrete_lyapunov_direct(a, q):
    """
    Solves the discrete Lyapunov equation directly.

    This function is called by the `solve_discrete_lyapunov` function with
    `method=direct`. It is not supposed to be called directly.
    """

    lhs = np.kron(a, a.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = solve(lhs, q.flatten())

    return np.reshape(x, q.shape)


def _solve_discrete_lyapunov_bilinear(a, q):
    """
    Solves the discrete Lyapunov equation using a bilinear transformation.

    This function is called by the `solve_discrete_lyapunov` function with
    `method=bilinear`. It is not supposed to be called directly.
    """
    eye = np.eye(a.shape[0])
    aH = a.conj().transpose()
    aHI_inv = inv(aH + eye)
    b = np.dot(aH - eye, aHI_inv)
    c = 2*np.dot(np.dot(inv(a + eye), q), aHI_inv)
    return solve_lyapunov(b.conj().transpose(), -c)


def solve_discrete_lyapunov(a, q, method=None):
    """
    Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.

    Parameters
    ----------
    a, q : (M, M) array_like
        Square matrices corresponding to A and Q in the equation
        above respectively. Must have the same shape.

    method : {'direct', 'bilinear'}, optional
        Type of solver.

        If not given, chosen to be ``direct`` if ``M`` is less than 10 and
        ``bilinear`` otherwise.

    Returns
    -------
    x : ndarray
        Solution to the discrete Lyapunov equation

    See Also
    --------
    solve_continuous_lyapunov : computes the solution to the continuous-time
        Lyapunov equation

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *direct* if ``M`` is less than 10
    and ``bilinear`` otherwise.

    Method *direct* uses a direct analytical solution to the discrete Lyapunov
    equation. The algorithm is given in, for example, [1]_. However, it requires
    the linear solution of a system with dimension :math:`M^2` so that
    performance degrades rapidly for even moderately sized matrices.

    Method *bilinear* uses a bilinear transformation to convert the discrete
    Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`
    where :math:`B=(A-I)(A+I)^{-1}` and
    :math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be
    efficiently solved since it is a special case of a Sylvester equation.
    The transformation algorithm is from Popov (1964) as described in [2]_.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] "Lyapunov equation", Wikipedia,
       https://en.wikipedia.org/wiki/Lyapunov_equation#Discrete_time
    .. [2] Gajic, Z., and M.T.J. Qureshi. 2008.
       Lyapunov Matrix Equation in System Stability and Control.
       Dover Books on Engineering Series. Dover Publications.

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0.2, 0.5],[0.7, -0.9]])
    >>> q = np.eye(2)
    >>> x = linalg.solve_discrete_lyapunov(a, q)
    >>> x
    array([[ 0.70872893,  1.43518822],
           [ 1.43518822, -2.4266315 ]])
    >>> np.allclose(a.dot(x).dot(a.T)-x, -q)
    True

    """
    a = np.asarray(a)
    q = np.asarray(q)
    if method is None:
        # Select automatically based on size of matrices
        if a.shape[0] >= 10:
            method = 'bilinear'
        else:
            method = 'direct'

    meth = method.lower()

    if meth == 'direct':
        x = _solve_discrete_lyapunov_direct(a, q)
    elif meth == 'bilinear':
        x = _solve_discrete_lyapunov_bilinear(a, q)
    else:
        raise ValueError(f'Unknown solver {method}')

    return x


def solve_continuous_are(a, b, q, r, e=None, s=None, balanced=True):
    r"""
    Solves the continuous-time algebraic Riccati equation (CARE).

    The CARE is defined as

    .. math::

          X A + A^H X - X B R^{-1} B^H X + Q = 0

    The limitations for a solution to exist are :

        * All eigenvalues of :math:`A` on the right half plane, should be
          controllable.

        * The associated hamiltonian pencil (See Notes), should have
          eigenvalues sufficiently away from the imaginary axis.

    Moreover, if ``e`` or ``s`` is not precisely ``None``, then the
    generalized version of CARE

    .. math::

          E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0

    is solved. When omitted, ``e`` is assumed to be the identity and ``s``
    is assumed to be the zero matrix with sizes compatible with ``a`` and
    ``b``, respectively.

    Parameters
    ----------
    a : (M, M) array_like
        Square matrix
    b : (M, N) array_like
        Input
    q : (M, M) array_like
        Input
    r : (N, N) array_like
        Nonsingular square matrix
    e : (M, M) array_like, optional
        Nonsingular square matrix
    s : (M, N) array_like, optional
        Input
    balanced : bool, optional
        The boolean that indicates whether a balancing step is performed
        on the data. The default is set to True.

    Returns
    -------
    x : (M, M) ndarray
        Solution to the continuous-time algebraic Riccati equation.

    Raises
    ------
    LinAlgError
        For cases where the stable subspace of the pencil could not be
        isolated. See Notes section and the references for details.

    See Also
    --------
    solve_discrete_are : Solves the discrete-time algebraic Riccati equation

    Notes
    -----
    The equation is solved by forming the extended hamiltonian matrix pencil,
    as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

        [ A    0    B ]             [ E   0    0 ]
        [-Q  -A^H  -S ] - \lambda * [ 0  E^H   0 ]
        [ S^H B^H   R ]             [ 0   0    0 ]

    and using a QZ decomposition method.

    In this algorithm, the fail conditions are linked to the symmetry
    of the product :math:`U_2 U_1^{-1}` and condition number of
    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
    eigenvectors spanning the stable subspace with 2-m rows and partitioned
    into two m-row matrices. See [1]_ and [2]_ for more details.

    In order to improve the QZ decomposition accuracy, the pencil goes
    through a balancing step where the sum of absolute values of
    :math:`H` and :math:`J` entries (after removing the diagonal entries of
    the sum) is balanced following the recipe given in [3]_.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
       Riccati Equations.", SIAM Journal on Scientific and Statistical
       Computing, Vol.2(2), :doi:`10.1137/0902010`

    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
       Equations.", Massachusetts Institute of Technology. Laboratory for
       Information and Decision Systems. LIDS-R ; 859. Available online :
       http://hdl.handle.net/1721.1/1301

    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`

    Examples
    --------
    Given `a`, `b`, `q`, and `r` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[4, 3], [-4.5, -3.5]])
    >>> b = np.array([[1], [-1]])
    >>> q = np.array([[9, 6], [6, 4.]])
    >>> r = 1
    >>> x = linalg.solve_continuous_are(a, b, q, r)
    >>> x
    array([[ 21.72792206,  14.48528137],
           [ 14.48528137,   9.65685425]])
    >>> np.allclose(a.T.dot(x) + x.dot(a)-x.dot(b).dot(b.T).dot(x), -q)
    True

    """

    # Validate input arguments
    a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
                                                     a, b, q, r, e, s, 'care')

    H = np.empty((2*m+n, 2*m+n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, m:2*m] = 0.
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = -a.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, m:2*m] = b.conj().T
    H[2*m:, 2*m:] = r

    if gen_are and e is not None:
        J = block_diag(e, e.conj().T, np.zeros_like(r, dtype=r_or_c))
    else:
        J = block_diag(np.eye(2*m), np.zeros_like(r, dtype=r_or_c))

    if balanced:
        # xGEBAL does not remove the diagonals before scaling. Also
        # to avoid destroying the Symplectic structure, we follow Ref.3
        M = np.abs(H) + np.abs(J)
        np.fill_diagonal(M, 0.)
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        # do we need to bother?
        if not np.allclose(sca, np.ones_like(sca)):
            # Now impose diag(D,inv(D)) from Benner where D is
            # square root of s_i/s_(n+i) for i=0,....
            sca = np.log2(sca)
            # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
            s = np.round((sca[m:2*m] - sca[:m])/2)
            sca = 2 ** np.r_[s, -s, sca[2*m:]]
            # Elementwise multiplication via broadcasting.
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale

    # Deflate the pencil to 2m x 2m ala Ref.1, eq.(55)
    q, r = qr(H[:, -n:])
    H = q[:, n:].conj().T.dot(H[:, :2*m])
    J = q[:2*m, n:].conj().T.dot(J[:2*m, :2*m])

    # Decide on which output type is needed for QZ
    out_str = 'real' if r_or_c is float else 'complex'

    _, _, _, _, _, u = ordqz(H, J, sort='lhp', overwrite_a=True,
                             overwrite_b=True, check_finite=False,
                             output=out_str)

    # Get the relevant parts of the stable subspace basis
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # Solve via back-substituion after checking the condition of u00
    up, ul, uu = lu(u00)
    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    # Exploit the triangular structure
    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # Check the deviation from symmetry for lack of success
    # See proof of Thm.5 item 3 in [2]
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])

    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated Hamiltonian pencil has eigenvalues '
                          'too close to the imaginary axis')

    return (x + x.conj().T)/2


def solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True):
    r"""
    Solves the discrete-time algebraic Riccati equation (DARE).

    The DARE is defined as

    .. math::

          A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0

    The limitations for a solution to exist are :

        * All eigenvalues of :math:`A` outside the unit disc, should be
          controllable.

        * The associated symplectic pencil (See Notes), should have
          eigenvalues sufficiently away from the unit circle.

    Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the
    generalized version of DARE

    .. math::

          A^HXA - E^HXE - (A^HXB+S) (R+B^HXB)^{-1} (B^HXA+S^H) + Q = 0

    is solved. When omitted, ``e`` is assumed to be the identity and ``s``
    is assumed to be the zero matrix.

    Parameters
    ----------
    a : (M, M) array_like
        Square matrix
    b : (M, N) array_like
        Input
    q : (M, M) array_like
        Input
    r : (N, N) array_like
        Square matrix
    e : (M, M) array_like, optional
        Nonsingular square matrix
    s : (M, N) array_like, optional
        Input
    balanced : bool
        The boolean that indicates whether a balancing step is performed
        on the data. The default is set to True.

    Returns
    -------
    x : (M, M) ndarray
        Solution to the discrete algebraic Riccati equation.

    Raises
    ------
    LinAlgError
        For cases where the stable subspace of the pencil could not be
        isolated. See Notes section and the references for details.

    See Also
    --------
    solve_continuous_are : Solves the continuous algebraic Riccati equation

    Notes
    -----
    The equation is solved by forming the extended symplectic matrix pencil,
    as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

           [  A   0   B ]             [ E   0   B ]
           [ -Q  E^H -S ] - \lambda * [ 0  A^H  0 ]
           [ S^H  0   R ]             [ 0 -B^H  0 ]

    and using a QZ decomposition method.

    In this algorithm, the fail conditions are linked to the symmetry
    of the product :math:`U_2 U_1^{-1}` and condition number of
    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
    eigenvectors spanning the stable subspace with 2-m rows and partitioned
    into two m-row matrices. See [1]_ and [2]_ for more details.

    In order to improve the QZ decomposition accuracy, the pencil goes
    through a balancing step where the sum of absolute values of
    :math:`H` and :math:`J` rows/cols (after removing the diagonal entries)
    is balanced following the recipe given in [3]_. If the data has small
    numerical noise, balancing may amplify their effects and some clean up
    is required.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
       Riccati Equations.", SIAM Journal on Scientific and Statistical
       Computing, Vol.2(2), :doi:`10.1137/0902010`

    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
       Equations.", Massachusetts Institute of Technology. Laboratory for
       Information and Decision Systems. LIDS-R ; 859. Available online :
       http://hdl.handle.net/1721.1/1301

    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`

    Examples
    --------
    Given `a`, `b`, `q`, and `r` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> a = np.array([[0, 1], [0, -1]])
    >>> b = np.array([[1, 0], [2, 1]])
    >>> q = np.array([[-4, -4], [-4, 7]])
    >>> r = np.array([[9, 3], [3, 1]])
    >>> x = la.solve_discrete_are(a, b, q, r)
    >>> x
    array([[-4., -4.],
           [-4.,  7.]])
    >>> R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))
    >>> np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q)
    True

    """

    # Validate input arguments
    a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
                                                     a, b, q, r, e, s, 'dare')

    # Form the matrix pencil
    H = np.zeros((2*m+n, 2*m+n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = np.eye(m) if e is None else e.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, 2*m:] = r

    J = np.zeros_like(H, dtype=r_or_c)
    J[:m, :m] = np.eye(m) if e is None else e
    J[m:2*m, m:2*m] = a.conj().T
    J[2*m:, m:2*m] = -b.conj().T

    if balanced:
        # xGEBAL does not remove the diagonals before scaling. Also
        # to avoid destroying the Symplectic structure, we follow Ref.3
        M = np.abs(H) + np.abs(J)
        np.fill_diagonal(M, 0.)
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        # do we need to bother?
        if not np.allclose(sca, np.ones_like(sca)):
            # Now impose diag(D,inv(D)) from Benner where D is
            # square root of s_i/s_(n+i) for i=0,....
            sca = np.log2(sca)
            # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
            s = np.round((sca[m:2*m] - sca[:m])/2)
            sca = 2 ** np.r_[s, -s, sca[2*m:]]
            # Elementwise multiplication via broadcasting.
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale

    # Deflate the pencil by the R column ala Ref.1
    q_of_qr, _ = qr(H[:, -n:])
    H = q_of_qr[:, n:].conj().T.dot(H[:, :2*m])
    J = q_of_qr[:, n:].conj().T.dot(J[:, :2*m])

    # Decide on which output type is needed for QZ
    out_str = 'real' if r_or_c is float else 'complex'

    _, _, _, _, _, u = ordqz(H, J, sort='iuc',
                             overwrite_a=True,
                             overwrite_b=True,
                             check_finite=False,
                             output=out_str)

    # Get the relevant parts of the stable subspace basis
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # Solve via back-substituion after checking the condition of u00
    up, ul, uu = lu(u00)

    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    # Exploit the triangular structure
    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # Check the deviation from symmetry for lack of success
    # See proof of Thm.5 item 3 in [2]
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])

    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated symplectic pencil has eigenvalues '
                          'too close to the unit circle')

    return (x + x.conj().T)/2


def _are_validate_args(a, b, q, r, e, s, eq_type='care'):
    """
    A helper function to validate the arguments supplied to the
    Riccati equation solvers. Any discrepancy found in the input
    matrices leads to a ``ValueError`` exception.

    Essentially, it performs:

        - a check whether the input is free of NaN and Infs
        - a pass for the data through ``numpy.atleast_2d()``
        - squareness check of the relevant arrays
        - shape consistency check of the arrays
        - singularity check of the relevant arrays
        - symmetricity check of the relevant matrices
        - a check whether the regular or the generalized version is asked.

    This function is used by ``solve_continuous_are`` and
    ``solve_discrete_are``.

    Parameters
    ----------
    a, b, q, r, e, s : array_like
        Input data
    eq_type : str
        Accepted arguments are 'care' and 'dare'.

    Returns
    -------
    a, b, q, r, e, s : ndarray
        Regularized input data
    m, n : int
        shape of the problem
    r_or_c : type
        Data type of the problem, returns float or complex
    gen_or_not : bool
        Type of the equation, True for generalized and False for regular ARE.

    """

    if eq_type.lower() not in ("dare", "care"):
        raise ValueError("Equation type unknown. "
                         "Only 'care' and 'dare' is understood")

    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    b = np.atleast_2d(_asarray_validated(b, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r = np.atleast_2d(_asarray_validated(r, check_finite=True))

    # Get the correct data types otherwise NumPy complains
    # about pushing complex numbers into real arrays.
    r_or_c = complex if np.iscomplexobj(b) else float

    for ind, mat in enumerate((a, q, r)):
        if np.iscomplexobj(mat):
            r_or_c = complex

        if not np.equal(*mat.shape):
            raise ValueError(f"Matrix {'aqr'[ind]} should be square.")

    # Shape consistency checks
    m, n = b.shape
    if m != a.shape[0]:
        raise ValueError("Matrix a and b should have the same number of rows.")
    if m != q.shape[0]:
        raise ValueError("Matrix a and q should have the same shape.")
    if n != r.shape[0]:
        raise ValueError("Matrix b and r should have the same number of cols.")

    # Check if the data matrices q, r are (sufficiently) hermitian
    for ind, mat in enumerate((q, r)):
        if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1))*100:
            raise ValueError(f"Matrix {'qr'[ind]} should be symmetric/hermitian.")

    # Continuous time ARE should have a nonsingular r matrix.
    if eq_type == 'care':
        min_sv = svd(r, compute_uv=False)[-1]
        if min_sv == 0. or min_sv < np.spacing(1.)*norm(r, 1):
            raise ValueError('Matrix r is numerically singular.')

    # Check if the generalized case is required with omitted arguments
    # perform late shape checking etc.
    generalized_case = e is not None or s is not None

    if generalized_case:
        if e is not None:
            e = np.atleast_2d(_asarray_validated(e, check_finite=True))
            if not np.equal(*e.shape):
                raise ValueError("Matrix e should be square.")
            if m != e.shape[0]:
                raise ValueError("Matrix a and e should have the same shape.")
            # numpy.linalg.cond doesn't check for exact zeros and
            # emits a runtime warning. Hence the following manual check.
            min_sv = svd(e, compute_uv=False)[-1]
            if min_sv == 0. or min_sv < np.spacing(1.) * norm(e, 1):
                raise ValueError('Matrix e is numerically singular.')
            if np.iscomplexobj(e):
                r_or_c = complex
        if s is not None:
            s = np.atleast_2d(_asarray_validated(s, check_finite=True))
            if s.shape != b.shape:
                raise ValueError("Matrix b and s should have the same shape.")
            if np.iscomplexobj(s):
                r_or_c = complex

    return a, b, q, r, e, s, m, n, r_or_c, generalized_case
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm',
    'tanhm', 'logm', 'funm', 'signm', 'sqrtm',
    'expm_frechet', 'expm_cond', 'fractional_matrix_power',
    'khatri_rao', 'norm', 'solve', 'inv', 'svd', 'schur', 'rsf2csf'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="matfuncs",
                                   private_modules=["_matfuncs"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Matrix square root for general matrices and for upper triangular matrices.

This module exists to avoid cyclic imports.

"""
__all__ = ['sqrtm']

import numpy as np

from scipy._lib._util import _asarray_validated

# Local imports
from ._misc import norm
from .lapack import ztrsyl, dtrsyl
from ._decomp_schur import schur, rsf2csf
from ._basic import _ensure_dtype_cdsz



class SqrtmError(np.linalg.LinAlgError):
    pass


from ._matfuncs_sqrtm_triu import within_block_loop  # noqa: E402


def _sqrtm_triu(T, blocksize=64):
    """
    Matrix square root of an upper triangular matrix.

    This is a helper function for `sqrtm` and `logm`.

    Parameters
    ----------
    T : (N, N) array_like upper triangular
        Matrix whose square root to evaluate
    blocksize : int, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `T`

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    """
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag, initial=0.) >= 0

    # Cast to complex as necessary + ensure double precision
    if not keep_it_real:
        T = np.asarray(T, dtype=np.complex128, order="C")
        T_diag = np.asarray(T_diag, dtype=np.complex128)
    else:
        T = np.asarray(T, dtype=np.float64, order="C")
        T_diag = np.asarray(T_diag, dtype=np.float64)

    R = np.diag(np.sqrt(T_diag))

    # Compute the number of blocks to use; use at least one block.
    n, n = T.shape
    nblocks = max(n // blocksize, 1)

    # Compute the smaller of the two sizes of blocks that
    # we will actually use, and compute the number of large blocks.
    bsmall, nlarge = divmod(n, nblocks)
    blarge = bsmall + 1
    nsmall = nblocks - nlarge
    if nsmall * bsmall + nlarge * blarge != n:
        raise Exception('internal inconsistency')

    # Define the index range covered by each block.
    start_stop_pairs = []
    start = 0
    for count, size in ((nsmall, bsmall), (nlarge, blarge)):
        for i in range(count):
            start_stop_pairs.append((start, start + size))
            start += size

    # Within-block interactions (Cythonized)
    try:
        within_block_loop(R, T, start_stop_pairs, nblocks)
    except RuntimeError as e:
        raise SqrtmError(*e.args) from e

    # Between-block interactions (Cython would give no significant speedup)
    for j in range(nblocks):
        jstart, jstop = start_stop_pairs[j]
        for i in range(j-1, -1, -1):
            istart, istop = start_stop_pairs[i]
            S = T[istart:istop, jstart:jstop]
            if j - i > 1:
                S = S - R[istart:istop, istop:jstart].dot(R[istop:jstart,
                                                            jstart:jstop])

            # Invoke LAPACK.
            # For more details, see the solve_sylvester implementation
            # and the fortran dtrsyl and ztrsyl docs.
            Rii = R[istart:istop, istart:istop]
            Rjj = R[jstart:jstop, jstart:jstop]
            if keep_it_real:
                x, scale, info = dtrsyl(Rii, Rjj, S)
            else:
                x, scale, info = ztrsyl(Rii, Rjj, S)
            R[istart:istop, jstart:jstop] = x * scale

    # Return the matrix square root.
    return R


def sqrtm(A, disp=True, blocksize=64):
    """
    Matrix square root.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose square root to evaluate
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)
    blocksize : integer, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `A`. The dtype is float or complex.
        The precision (data size) is determined based on the precision of
        input `A`.

    errest : float
        (if disp == False)

        Frobenius norm of the estimated error, ||err||_F / ||A||_F

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import sqrtm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> r = sqrtm(a)
    >>> r
    array([[ 0.75592895,  1.13389342],
           [ 0.37796447,  1.88982237]])
    >>> r.dot(r)
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    A = _asarray_validated(A, check_finite=True, as_inexact=True)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    if blocksize < 1:
        raise ValueError("The blocksize should be at least 1.")
    A, = _ensure_dtype_cdsz(A)
    keep_it_real = np.isrealobj(A)
    if keep_it_real:
        T, Z = schur(A)
        d0 = np.diagonal(T)
        d1 = np.diagonal(T, -1)
        eps = np.finfo(T.dtype).eps
        needs_conversion = abs(d1) > eps * (abs(d0[1:]) + abs(d0[:-1]))
        if needs_conversion.any():
            T, Z = rsf2csf(T, Z)
    else:
        T, Z = schur(A, output='complex')
    failflag = False
    try:
        R = _sqrtm_triu(T, blocksize=blocksize)
        ZH = np.conjugate(Z).T
        X = Z.dot(R).dot(ZH)
        dtype = np.result_type(A.dtype, 1j if np.iscomplexobj(X) else 1)
        X = X.astype(dtype, copy=False)
    except SqrtmError:
        failflag = True
        X = np.empty_like(A)
        X.fill(np.nan)

    if disp:
        if failflag:
            print("Failed to find a square root.")
        return X
    else:
        try:
            arg2 = norm(X.dot(X) - A, 'fro')**2 / norm(A, 'fro')
        except ValueError:
            # NaNs in matrix
            arg2 = np.inf

        return X, arg2
<|endoftext|>#
# Author: Travis Oliphant, March 2002
#
import warnings
from itertools import product

import numpy as np
from numpy import (dot, diag, prod, logical_not, ravel, transpose,
                   conjugate, absolute, amax, sign, isfinite, triu)

# Local imports
from scipy.linalg import LinAlgError, bandwidth
from ._misc import norm
from ._basic import solve, inv
from ._decomp_svd import svd
from ._decomp_schur import schur, rsf2csf
from ._expm_frechet import expm_frechet, expm_cond
from ._matfuncs_sqrtm import sqrtm
from ._matfuncs_expm import pick_pade_structure, pade_UV_calc
from ._linalg_pythran import _funm_loops  # type: ignore[import-not-found]

__all__ = ['expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm', 'tanhm', 'logm',
           'funm', 'signm', 'sqrtm', 'fractional_matrix_power', 'expm_frechet',
           'expm_cond', 'khatri_rao']

eps = np.finfo('d').eps
feps = np.finfo('f').eps

_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}


###############################################################################
# Utility functions.


def _asarray_square(A):
    """
    Wraps asarray with the extra requirement that the input be a square matrix.

    The motivation is that the matfuncs module has real functions that have
    been lifted to square matrix functions.

    Parameters
    ----------
    A : array_like
        A square matrix.

    Returns
    -------
    out : ndarray
        An ndarray copy or view or other representation of A.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square array_like input')
    return A


def _maybe_real(A, B, tol=None):
    """
    Return either B or the real part of B, depending on properties of A and B.

    The motivation is that B has been computed as a complicated function of A,
    and B may be perturbed by negligible imaginary components.
    If A is real and B is complex with small imaginary components,
    then return a real copy of B.  The assumption in that case would be that
    the imaginary components of B are numerical artifacts.

    Parameters
    ----------
    A : ndarray
        Input array whose type is to be checked as real vs. complex.
    B : ndarray
        Array to be returned, possibly without its imaginary part.
    tol : float
        Absolute tolerance.

    Returns
    -------
    out : real or complex array
        Either the input array B or only the real part of the input array B.

    """
    # Note that booleans and integers compare as real.
    if np.isrealobj(A) and np.iscomplexobj(B):
        if tol is None:
            tol = {0: feps*1e3, 1: eps*1e6}[_array_precision[B.dtype.char]]
        if np.allclose(B.imag, 0.0, atol=tol):
            B = B.real
    return B


###############################################################################
# Matrix functions.


def fractional_matrix_power(A, t):
    """
    Compute the fractional power of a matrix.

    Proceeds according to the discussion in section (6) of [1]_.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose fractional power to evaluate.
    t : float
        Fractional power.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import fractional_matrix_power
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> b = fractional_matrix_power(a, 0.5)
    >>> b
    array([[ 0.75592895,  1.13389342],
           [ 0.37796447,  1.88982237]])
    >>> np.dot(b, b)      # Verify square root
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    # This fixes some issue with imports;
    # this function calls onenormest which is in scipy.sparse.
    A = _asarray_square(A)
    import scipy.linalg._matfuncs_inv_ssq
    return scipy.linalg._matfuncs_inv_ssq._fractional_matrix_power(A, t)


def logm(A, disp=True):
    """
    Compute matrix logarithm.

    The matrix logarithm is the inverse of
    expm: expm(logm(`A`)) == `A`

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose logarithm to evaluate
    disp : bool, optional
        Emit warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import logm, expm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> b = logm(a)
    >>> b
    array([[-1.02571087,  2.05142174],
           [ 0.68380725,  1.02571087]])
    >>> expm(b)         # Verify expm(logm(a)) returns a
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    A = np.asarray(A)  # squareness checked in `_logm`
    # Avoid circular import ... this is OK, right?
    import scipy.linalg._matfuncs_inv_ssq
    F = scipy.linalg._matfuncs_inv_ssq._logm(A)
    F = _maybe_real(A, F)
    errtol = 1000*eps
    # TODO use a better error approximation
    with np.errstate(divide='ignore', invalid='ignore'):
        errest = norm(expm(F)-A, 1) / np.asarray(norm(A, 1), dtype=A.dtype).real[()]
    if disp:
        if not isfinite(errest) or errest >= errtol:
            message = f"logm result may be inaccurate, approximate err = {errest}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return F
    else:
        return F, errest


def expm(A):
    """Compute the matrix exponential of an array.

    Parameters
    ----------
    A : ndarray
        Input with last two dimensions are square ``(..., n, n)``.

    Returns
    -------
    eA : ndarray
        The resulting matrix exponential with the same shape of ``A``

    Notes
    -----
    Implements the algorithm given in [1], which is essentially a Pade
    approximation with a variable order that is decided based on the array
    data.

    For input with size ``n``, the memory usage is in the worst case in the
    order of ``8*(n**2)``. If the input data is not of single and double
    precision of real and complex dtypes, it is copied to a new array.

    For cases ``n >= 400``, the exact 1-norm computation cost, breaks even with
    1-norm estimation and from that point on the estimation scheme given in
    [2] is used to decide on the approximation order.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham, (2009), "A New Scaling
           and Squaring Algorithm for the Matrix Exponential", SIAM J. Matrix
           Anal. Appl. 31(3):970-989, :doi:`10.1137/09074721X`

    .. [2] Nicholas J. Higham and Francoise Tisseur (2000), "A Block Algorithm
           for Matrix 1-Norm Estimation, with an Application to 1-Norm
           Pseudospectra." SIAM J. Matrix Anal. Appl. 21(4):1185-1201,
           :doi:`10.1137/S0895479899356080`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    Matrix version of the formula exp(0) = 1:

    >>> expm(np.zeros((3, 2, 2)))
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]]])

    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
    applied to a matrix:

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    a = np.asarray(A)
    if a.size == 1 and a.ndim < 2:
        return np.array([[np.exp(a.item())]])

    if a.ndim < 2:
        raise LinAlgError('The input array must be at least two-dimensional')
    if a.shape[-1] != a.shape[-2]:
        raise LinAlgError('Last 2 dimensions of the array must be square')

    # Empty array
    if min(*a.shape) == 0:
        dtype = expm(np.eye(2, dtype=a.dtype)).dtype
        return np.empty_like(a, dtype=dtype)

    # Scalar case
    if a.shape[-2:] == (1, 1):
        return np.exp(a)

    if not np.issubdtype(a.dtype, np.inexact):
        a = a.astype(np.float64)
    elif a.dtype == np.float16:
        a = a.astype(np.float32)

    # An explicit formula for 2x2 case exists (formula (2.2) in [1]). However, without
    # Kahan's method, numerical instabilities can occur (See gh-19584). Hence removed
    # here until we have a more stable implementation.

    n = a.shape[-1]
    eA = np.empty(a.shape, dtype=a.dtype)
    # working memory to hold intermediate arrays
    Am = np.empty((5, n, n), dtype=a.dtype)

    # Main loop to go through the slices of an ndarray and passing to expm
    for ind in product(*[range(x) for x in a.shape[:-2]]):
        aw = a[ind]

        lu = bandwidth(aw)
        if not any(lu):  # a is diagonal?
            eA[ind] = np.diag(np.exp(np.diag(aw)))
            continue

        # Generic/triangular case; copy the slice into scratch and send.
        # Am will be mutated by pick_pade_structure
        # If s != 0, scaled Am will be returned from pick_pade_structure.
        Am[0, :, :] = aw
        m, s = pick_pade_structure(Am)
        if (m < 0):
            raise MemoryError("scipy.linalg.expm could not allocate sufficient"
                              " memory while trying to compute the Pade "
                              f"structure (error code {m}).")
        info = pade_UV_calc(Am, m)
        if info != 0:
            if info <= -11:
                # We raise it from failed mallocs; negative LAPACK codes > -7
                raise MemoryError("scipy.linalg.expm could not allocate "
                              "sufficient memory while trying to compute the "
                              f"exponential (error code {info}).")
            else:
                # LAPACK wrong argument error or exact singularity.
                # Neither should happen.
                raise RuntimeError("scipy.linalg.expm got an internal LAPACK "
                                   "error during the exponential computation "
                                   f"(error code {info})")
        eAw = Am[0]

        if s != 0:  # squaring needed

            if (lu[1] == 0) or (lu[0] == 0):  # lower/upper triangular
                # This branch implements Code Fragment 2.1 of [1]

                diag_aw = np.diag(aw)
                # einsum returns a writable view
                np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2**(-s))
                # super/sub diagonal
                sd = np.diag(aw, k=-1 if lu[1] == 0 else 1)

                for i in range(s-1, -1, -1):
                    eAw = eAw @ eAw

                    # diagonal
                    np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2.**(-i))
                    exp_sd = _exp_sinch(diag_aw * (2.**(-i))) * (sd * 2**(-i))
                    if lu[1] == 0:  # lower
                        np.einsum('ii->i', eAw[1:, :-1])[:] = exp_sd
                    else:  # upper
                        np.einsum('ii->i', eAw[:-1, 1:])[:] = exp_sd

            else:  # generic
                for _ in range(s):
                    eAw = eAw @ eAw

        # Zero out the entries from np.empty in case of triangular input
        if (lu[0] == 0) or (lu[1] == 0):
            eA[ind] = np.triu(eAw) if lu[0] == 0 else np.tril(eAw)
        else:
            eA[ind] = eAw

    return eA


def _exp_sinch(x):
    # Higham's formula (10.42), might overflow, see GH-11839
    lexp_diff = np.diff(np.exp(x))
    l_diff = np.diff(x)
    mask_z = l_diff == 0.
    lexp_diff[~mask_z] /= l_diff[~mask_z]
    lexp_diff[mask_z] = np.exp(x[:-1][mask_z])
    return lexp_diff


def cosm(A):
    """
    Compute the matrix cosine.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array

    Returns
    -------
    cosm : (N, N) ndarray
        Matrix cosine of A

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
    applied to a matrix:

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    A = _asarray_square(A)
    if np.iscomplexobj(A):
        return 0.5*(expm(1j*A) + expm(-1j*A))
    else:
        return expm(1j*A).real


def sinm(A):
    """
    Compute the matrix sine.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array.

    Returns
    -------
    sinm : (N, N) ndarray
        Matrix sine of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
    applied to a matrix:

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    A = _asarray_square(A)
    if np.iscomplexobj(A):
        return -0.5j*(expm(1j*A) - expm(-1j*A))
    else:
        return expm(1j*A).imag


def tanm(A):
    """
    Compute the matrix tangent.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array.

    Returns
    -------
    tanm : (N, N) ndarray
        Matrix tangent of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanm, sinm, cosm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> t = tanm(a)
    >>> t
    array([[ -2.00876993,  -8.41880636],
           [ -2.80626879, -10.42757629]])

    Verify tanm(a) = sinm(a).dot(inv(cosm(a)))

    >>> s = sinm(a)
    >>> c = cosm(a)
    >>> s.dot(np.linalg.inv(c))
    array([[ -2.00876993,  -8.41880636],
           [ -2.80626879, -10.42757629]])

    """
    A = _asarray_square(A)
    return _maybe_real(A, solve(cosm(A), sinm(A)))


def coshm(A):
    """
    Compute the hyperbolic matrix cosine.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array.

    Returns
    -------
    coshm : (N, N) ndarray
        Hyperbolic matrix cosine of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanhm, sinhm, coshm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> c = coshm(a)
    >>> c
    array([[ 11.24592233,  38.76236492],
           [ 12.92078831,  50.00828725]])

    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

    >>> t = tanhm(a)
    >>> s = sinhm(a)
    >>> t - s.dot(np.linalg.inv(c))
    array([[  2.72004641e-15,   4.55191440e-15],
           [  0.00000000e+00,  -5.55111512e-16]])

    """
    A = _asarray_square(A)
    return _maybe_real(A, 0.5 * (expm(A) + expm(-A)))


def sinhm(A):
    """
    Compute the hyperbolic matrix sine.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array.

    Returns
    -------
    sinhm : (N, N) ndarray
        Hyperbolic matrix sine of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanhm, sinhm, coshm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> s = sinhm(a)
    >>> s
    array([[ 10.57300653,  39.28826594],
           [ 13.09608865,  49.86127247]])

    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

    >>> t = tanhm(a)
    >>> c = coshm(a)
    >>> t - s.dot(np.linalg.inv(c))
    array([[  2.72004641e-15,   4.55191440e-15],
           [  0.00000000e+00,  -5.55111512e-16]])

    """
    A = _asarray_square(A)
    return _maybe_real(A, 0.5 * (expm(A) - expm(-A)))


def tanhm(A):
    """
    Compute the hyperbolic matrix tangent.

    This routine uses expm to compute the matrix exponentials.

    Parameters
    ----------
    A : (N, N) array_like
        Input array

    Returns
    -------
    tanhm : (N, N) ndarray
        Hyperbolic matrix tangent of `A`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import tanhm, sinhm, coshm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> t = tanhm(a)
    >>> t
    array([[ 0.3428582 ,  0.51987926],
           [ 0.17329309,  0.86273746]])

    Verify tanhm(a) = sinhm(a).dot(inv(coshm(a)))

    >>> s = sinhm(a)
    >>> c = coshm(a)
    >>> t - s.dot(np.linalg.inv(c))
    array([[  2.72004641e-15,   4.55191440e-15],
           [  0.00000000e+00,  -5.55111512e-16]])

    """
    A = _asarray_square(A)
    return _maybe_real(A, solve(coshm(A), sinhm(A)))


def funm(A, func, disp=True):
    """
    Evaluate a matrix function specified by a callable.

    Returns the value of matrix-valued function ``f`` at `A`. The
    function ``f`` is an extension of the scalar-valued function `func`
    to matrices.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix at which to evaluate the function
    func : callable
        Callable object that evaluates a scalar function f.
        Must be vectorized (eg. using vectorize).
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    funm : (N, N) ndarray
        Value of the matrix function specified by func evaluated at `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    Notes
    -----
    This function implements the general algorithm based on Schur decomposition
    (Algorithm 9.1.1. in [1]_).

    If the input matrix is known to be diagonalizable, then relying on the
    eigendecomposition is likely to be faster. For example, if your matrix is
    Hermitian, you can do

    >>> from scipy.linalg import eigh
    >>> def funm_herm(a, func, check_finite=False):
    ...     w, v = eigh(a, check_finite=check_finite)
    ...     ## if you further know that your matrix is positive semidefinite,
    ...     ## you can optionally guard against precision errors by doing
    ...     # w = np.maximum(w, 0)
    ...     w = func(w)
    ...     return (v * w).dot(v.conj().T)

    References
    ----------
    .. [1] Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import funm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> funm(a, lambda x: x*x)
    array([[  4.,  15.],
           [  5.,  19.]])
    >>> a.dot(a)
    array([[  4.,  15.],
           [  5.,  19.]])

    """
    A = _asarray_square(A)
    # Perform Shur decomposition (lapack ?gees)
    T, Z = schur(A)
    T, Z = rsf2csf(T, Z)
    n, n = T.shape
    F = diag(func(diag(T)))  # apply function to diagonal elements
    F = F.astype(T.dtype.char)  # e.g., when F is real but T is complex

    minden = abs(T[0, 0])

    # implement Algorithm 11.1.1 from Golub and Van Loan
    #                 "matrix Computations."
    F, minden = _funm_loops(F, T, n, minden)

    F = dot(dot(Z, F), transpose(conjugate(Z)))
    F = _maybe_real(A, F)

    tol = {0: feps, 1: eps}[_array_precision[F.dtype.char]]
    if minden == 0.0:
        minden = tol
    err = min(1, max(tol, (tol/minden)*norm(triu(T, 1), 1)))
    if prod(ravel(logical_not(isfinite(F))), axis=0):
        err = np.inf
    if disp:
        if err > 1000*tol:
            print("funm result may be inaccurate, approximate err =", err)
        return F
    else:
        return F, err


def signm(A, disp=True):
    """
    Matrix sign function.

    Extension of the scalar sign(x) to matrices.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix at which to evaluate the sign function
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)

    Returns
    -------
    signm : (N, N) ndarray
        Value of the sign function at `A`
    errest : float
        (if disp == False)

        1-norm of the estimated error, ||err||_1 / ||A||_1

    Examples
    --------
    >>> from scipy.linalg import signm, eigvals
    >>> a = [[1,2,3], [1,2,1], [1,1,1]]
    >>> eigvals(a)
    array([ 4.12488542+0.j, -0.76155718+0.j,  0.63667176+0.j])
    >>> eigvals(signm(a))
    array([-1.+0.j,  1.+0.j,  1.+0.j])

    """
    A = _asarray_square(A)

    def rounded_sign(x):
        rx = np.real(x)
        if rx.dtype.char == 'f':
            c = 1e3*feps*amax(x)
        else:
            c = 1e3*eps*amax(x)
        return sign((absolute(rx) > c) * rx)
    result, errest = funm(A, rounded_sign, disp=0)
    errtol = {0: 1e3*feps, 1: 1e3*eps}[_array_precision[result.dtype.char]]
    if errest < errtol:
        return result

    # Handle signm of defective matrices:

    # See "E.D.Denman and J.Leyva-Ramos, Appl.Math.Comp.,
    # 8:237-250,1981" for how to improve the following (currently a
    # rather naive) iteration process:

    # a = result # sometimes iteration converges faster but where??

    # Shifting to avoid zero eigenvalues. How to ensure that shifting does
    # not change the spectrum too much?
    vals = svd(A, compute_uv=False)
    max_sv = np.amax(vals)
    # min_nonzero_sv = vals[(vals>max_sv*errtol).tolist().count(1)-1]
    # c = 0.5/min_nonzero_sv
    c = 0.5/max_sv
    S0 = A + c*np.identity(A.shape[0])
    prev_errest = errest
    for i in range(100):
        iS0 = inv(S0)
        S0 = 0.5*(S0 + iS0)
        Pp = 0.5*(dot(S0, S0)+S0)
        errest = norm(dot(Pp, Pp)-Pp, 1)
        if errest < errtol or prev_errest == errest:
            break
        prev_errest = errest
    if disp:
        if not isfinite(errest) or errest >= errtol:
            print("signm result may be inaccurate, approximate err =", errest)
        return S0
    else:
        return S0, errest


def khatri_rao(a, b):
    r"""
    Khatri-rao product

    A column-wise Kronecker product of two matrices

    Parameters
    ----------
    a : (n, k) array_like
        Input array
    b : (m, k) array_like
        Input array

    Returns
    -------
    c:  (n*m, k) ndarray
        Khatri-rao product of `a` and `b`.

    Notes
    -----
    The mathematical definition of the Khatri-Rao product is:

    .. math::

        (A_{ij}  \bigotimes B_{ij})_{ij}

    which is the Kronecker product of every column of A and B, e.g.::

        c = np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> b = np.array([[3, 4, 5], [6, 7, 8], [2, 3, 9]])
    >>> linalg.khatri_rao(a, b)
    array([[ 3,  8, 15],
           [ 6, 14, 24],
           [ 2,  6, 27],
           [12, 20, 30],
           [24, 35, 48],
           [ 8, 15, 54]])

    """
    a = np.asarray(a)
    b = np.asarray(b)

    if not (a.ndim == 2 and b.ndim == 2):
        raise ValueError("The both arrays should be 2-dimensional.")

    if not a.shape[1] == b.shape[1]:
        raise ValueError("The number of columns for both arrays "
                         "should be equal.")

    # accommodate empty arrays
    if a.size == 0 or b.size == 0:
        m = a.shape[0] * b.shape[0]
        n = a.shape[1]
        return np.empty_like(a, shape=(m, n))

    # c = np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T
    c = a[..., :, np.newaxis, :] * b[..., np.newaxis, :, :]
    return c.reshape((-1,) + c.shape[2:])
<|endoftext|>#
# Author: Pearu Peterson, March 2002
#
# w/ additions by Travis Oliphant, March 2002
#              and Jake Vanderplas, August 2012

import warnings
from warnings import warn
from itertools import product
import numpy as np
from numpy import atleast_1d, atleast_2d
from .lapack import get_lapack_funcs, _compute_lwork
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from ._decomp import _asarray_validated
from . import _decomp, _decomp_svd
from ._solve_toeplitz import levinson
from ._cythonized_array_utils import (find_det_from_lu, bandwidth, issymmetric,
                                      ishermitian)

__all__ = ['solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
           'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
           'pinv', 'pinvh', 'matrix_balance', 'matmul_toeplitz']


# The numpy facilities for type-casting checks are too slow for small sized
# arrays and eat away the time budget for the checkups. Here we set a
# precomputed dict container of the numpy.can_cast() table.

# It can be used to determine quickly what a dtype can be cast to LAPACK
# compatible types, i.e., 'float32, float64, complex64, complex128'.
# Then it can be checked via "casting_dict[arr.dtype.char]"
lapack_cast_dict = {x: ''.join([y for y in 'fdFD' if np.can_cast(x, y)])
                    for x in np.typecodes['All']}


# Linear equations
def _solve_check(n, info, lamch=None, rcond=None):
    """ Check arguments during the different steps of the solution phase """
    if info < 0:
        raise ValueError(f'LAPACK reported an illegal value in {-info}-th argument.')
    elif 0 < info:
        raise LinAlgError('Matrix is singular.')

    if lamch is None:
        return
    E = lamch('E')
    if rcond < E:
        warn(f'Ill-conditioned matrix (rcond={rcond:.6g}): '
             'result may not be accurate.',
             LinAlgWarning, stacklevel=3)


def _find_matrix_structure(a):
    n = a.shape[0]
    n_below, n_above = bandwidth(a)

    if n_below == n_above == 0:
        kind = 'diagonal'
    elif n_above == 0:
        kind = 'lower triangular'
    elif n_below == 0:
        kind = 'upper triangular'
    elif n_above <= 1 and n_below <= 1 and n > 3:
        kind = 'tridiagonal'
    elif np.issubdtype(a.dtype, np.complexfloating) and ishermitian(a):
        kind = 'hermitian'
    elif issymmetric(a):
        kind = 'symmetric'
    else:
        kind = 'general'

    return kind, n_below, n_above


def solve(a, b, lower=False, overwrite_a=False,
          overwrite_b=False, check_finite=True, assume_a=None,
          transposed=False):
    """
    Solves the linear equation set ``a @ x == b`` for the unknown ``x``
    for square `a` matrix.

    If the data matrix is known to be a particular type then supplying the
    corresponding string to ``assume_a`` key chooses the dedicated solver.
    The available options are

    ===================  ================================
     diagonal             'diagonal'
     tridiagonal          'tridiagonal'
     banded               'banded'
     upper triangular     'upper triangular'
     lower triangular     'lower triangular'
     symmetric            'symmetric' (or 'sym')
     hermitian            'hermitian' (or 'her')
     positive definite    'positive definite' (or 'pos')
     general              'general' (or 'gen')
    ===================  ================================

    Parameters
    ----------
    a : (N, N) array_like
        Square input data
    b : (N, NRHS) array_like
        Input data for the right hand side.
    lower : bool, default: False
        Ignored unless ``assume_a`` is one of ``'sym'``, ``'her'``, or ``'pos'``.
        If True, the calculation uses only the data in the lower triangle of `a`;
        entries above the diagonal are ignored. If False (default), the
        calculation uses only the data in the upper triangle of `a`; entries
        below the diagonal are ignored.
    overwrite_a : bool, default: False
        Allow overwriting data in `a` (may enhance performance).
    overwrite_b : bool, default: False
        Allow overwriting data in `b` (may enhance performance).
    check_finite : bool, default: True
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    assume_a : str, optional
        Valid entries are described above.
        If omitted or ``None``, checks are performed to identify structure so the
        appropriate solver can be called.
    transposed : bool, default: False
        If True, solve ``a.T @ x == b``. Raises `NotImplementedError`
        for complex `a`.

    Returns
    -------
    x : (N, NRHS) ndarray
        The solution array.

    Raises
    ------
    ValueError
        If size mismatches detected or input a is not square.
    LinAlgError
        If the matrix is singular.
    LinAlgWarning
        If an ill-conditioned input a is detected.
    NotImplementedError
        If transposed is True and input a is a complex matrix.

    Notes
    -----
    If the input b matrix is a 1-D array with N elements, when supplied
    together with an NxN input a, it is assumed as a valid column vector
    despite the apparent size mismatch. This is compatible with the
    numpy.dot() behavior and the returned result is still 1-D array.

    The general, symmetric, Hermitian and positive definite solutions are
    obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of
    LAPACK respectively.

    The datatype of the arrays define which solver is called regardless
    of the values. In other words, even when the complex array entries have
    precisely zero imaginary parts, the complex solver will be called based
    on the data type of the array.

    Examples
    --------
    Given `a` and `b`, solve for `x`:

    >>> import numpy as np
    >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    >>> b = np.array([2, 4, -1])
    >>> from scipy import linalg
    >>> x = linalg.solve(a, b)
    >>> x
    array([ 2., -2.,  9.])
    >>> np.dot(a, x) == b
    array([ True,  True,  True], dtype=bool)

    """
    # Flags for 1-D or N-D right-hand side
    b_is_1D = False

    # check finite after determining structure
    a1 = atleast_2d(_asarray_validated(a, check_finite=False))
    b1 = atleast_1d(_asarray_validated(b, check_finite=False))
    a1, b1 = _ensure_dtype_cdsz(a1, b1)
    n = a1.shape[0]

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input a needs to be a square matrix.')

    if n != b1.shape[0]:
        # Last chance to catch 1x1 scalar a and 1-D b arrays
        if not (n == 1 and b1.size != 0):
            raise ValueError('Input b has to have same number of rows as '
                             'input a')

    # accommodate empty arrays
    if b1.size == 0:
        dt = solve(np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)

    # regularize 1-D b arrays to 2D
    if b1.ndim == 1:
        if n == 1:
            b1 = b1[None, :]
        else:
            b1 = b1[:, None]
        b_is_1D = True

    if assume_a not in {None, 'diagonal', 'tridiagonal', 'banded', 'lower triangular',
                        'upper triangular', 'symmetric', 'hermitian',
                        'positive definite', 'general', 'sym', 'her', 'pos', 'gen'}:
        raise ValueError(f'{assume_a} is not a recognized matrix structure')

    # for a real matrix, describe it as "symmetric", not "hermitian"
    # (lapack doesn't know what to do with real hermitian matrices)
    if assume_a in {'hermitian', 'her'} and not np.iscomplexobj(a1):
        assume_a = 'symmetric'

    n_below, n_above = None, None
    if assume_a is None:
        assume_a, n_below, n_above = _find_matrix_structure(a1)

    # Get the correct lamch function.
    # The LAMCH functions only exists for S and D
    # So for complex values we have to convert to real/double.
    if a1.dtype.char in 'fF':  # single precision
        lamch = get_lapack_funcs('lamch', dtype='f')
    else:
        lamch = get_lapack_funcs('lamch', dtype='d')

    # Currently we do not have the other forms of the norm calculators
    #   lansy, lanpo, lanhe.
    # However, in any case they only reduce computations slightly...
    if assume_a == 'diagonal':
        _matrix_norm = _matrix_norm_diagonal
    elif assume_a == 'tridiagonal':
        _matrix_norm = _matrix_norm_tridiagonal
    elif assume_a in {'lower triangular', 'upper triangular'}:
        _matrix_norm = _matrix_norm_triangular(assume_a)
    else:
        _matrix_norm = _matrix_norm_general

    # Since the I-norm and 1-norm are the same for symmetric matrices
    # we can collect them all in this one call
    # Note however, that when issuing 'gen' and form!='none', then
    # the I-norm should be used
    if transposed:
        trans = 1
        norm = 'I'
        if np.iscomplexobj(a1):
            raise NotImplementedError('scipy.linalg.solve can currently '
                                      'not solve a^T x = b or a^H x = b '
                                      'for complex matrices.')
    else:
        trans = 0
        norm = '1'

    anorm = _matrix_norm(norm, a1, check_finite)

    info, rcond = 0, np.inf

    # Generalized case 'gesv'
    if assume_a in {'general', 'gen'}:
        gecon, getrf, getrs = get_lapack_funcs(('gecon', 'getrf', 'getrs'),
                                               (a1, b1))
        lu, ipvt, info = getrf(a1, overwrite_a=overwrite_a)
        _solve_check(n, info)
        x, info = getrs(lu, ipvt, b1,
                        trans=trans, overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = gecon(lu, anorm, norm=norm)
    # Hermitian case 'hesv'
    elif assume_a in {'hermitian', 'her'}:
        hecon, hesv, hesv_lw = get_lapack_funcs(('hecon', 'hesv',
                                                 'hesv_lwork'), (a1, b1))
        lwork = _compute_lwork(hesv_lw, n, lower)
        lu, ipvt, x, info = hesv(a1, b1, lwork=lwork,
                                 lower=lower,
                                 overwrite_a=overwrite_a,
                                 overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = hecon(lu, ipvt, anorm)
    # Symmetric case 'sysv'
    elif assume_a in {'symmetric', 'sym'}:
        sycon, sysv, sysv_lw = get_lapack_funcs(('sycon', 'sysv',
                                                 'sysv_lwork'), (a1, b1))
        lwork = _compute_lwork(sysv_lw, n, lower)
        lu, ipvt, x, info = sysv(a1, b1, lwork=lwork,
                                 lower=lower,
                                 overwrite_a=overwrite_a,
                                 overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = sycon(lu, ipvt, anorm)
    # Diagonal case
    elif assume_a == 'diagonal':
        diag_a = np.diag(a1)
        x = (b1.T / diag_a).T
        abs_diag_a = np.abs(diag_a)
        rcond = abs_diag_a.min() / abs_diag_a.max()
    # Tri-diagonal case
    elif assume_a == 'tridiagonal':
        a1 = a1.T if transposed else a1
        dl, d, du = np.diag(a1, -1), np.diag(a1, 0), np.diag(a1, 1)
        _gttrf, _gttrs, _gtcon = get_lapack_funcs(('gttrf', 'gttrs', 'gtcon'), (a1, b1))
        dl, d, du, du2, ipiv, info = _gttrf(dl, d, du)
        _solve_check(n, info)
        x, info = _gttrs(dl, d, du, du2, ipiv, b1, overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = _gtcon(dl, d, du, du2, ipiv, anorm)
    # Banded case
    elif assume_a == 'banded':
        a1, n_below, n_above = ((a1.T, n_above, n_below) if transposed
                                else (a1, n_below, n_above))
        n_below, n_above = bandwidth(a1) if n_below is None else (n_below, n_above)
        ab = _to_banded(n_below, n_above, a1)
        gbsv, = get_lapack_funcs(('gbsv',), (a1, b1))
        # Next two lines copied from `solve_banded`
        a2 = np.zeros((2*n_below + n_above + 1, ab.shape[1]), dtype=gbsv.dtype)
        a2[n_below:, :] = ab
        _, _, x, info = gbsv(n_below, n_above, a2, b1,
                             overwrite_ab=True, overwrite_b=overwrite_b)
        _solve_check(n, info)
        # TODO: wrap gbcon and use to get rcond
    # Triangular case
    elif assume_a in {'lower triangular', 'upper triangular'}:
        lower = assume_a == 'lower triangular'
        x, info = _solve_triangular(a1, b1, lower=lower, overwrite_b=overwrite_b,
                                    trans=transposed)
        _solve_check(n, info)
        _trcon = get_lapack_funcs(('trcon'), (a1, b1))
        rcond, info = _trcon(a1, uplo='L' if lower else 'U')
    # Positive definite case 'posv'
    else:
        pocon, posv = get_lapack_funcs(('pocon', 'posv'),
                                       (a1, b1))
        lu, x, info = posv(a1, b1, lower=lower,
                           overwrite_a=overwrite_a,
                           overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = pocon(lu, anorm)

    _solve_check(n, info, lamch, rcond)

    if b_is_1D:
        x = x.ravel()

    return x


def _matrix_norm_diagonal(_, a, check_finite):
    # Equivalent of dlange for diagonal matrix, assuming
    # norm is either 'I' or '1' (really just not the Frobenius norm)
    d = np.diag(a)
    d = np.asarray_chkfinite(d) if check_finite else d
    return np.abs(d).max()


def _matrix_norm_tridiagonal(norm, a, check_finite):
    # Equivalent of dlange for tridiagonal matrix, assuming
    # norm is either 'I' or '1'
    if norm == 'I':
        a = a.T
    # Context to avoid warning before error in cases like -inf + inf
    with np.errstate(invalid='ignore'):
        d = np.abs(np.diag(a))
        d[1:] += np.abs(np.diag(a, 1))
        d[:-1] += np.abs(np.diag(a, -1))
    d = np.asarray_chkfinite(d) if check_finite else d
    return d.max()


def _matrix_norm_triangular(structure):
    def fun(norm, a, check_finite):
        a = np.asarray_chkfinite(a) if check_finite else a
        lantr = get_lapack_funcs('lantr', (a,))
        return lantr(norm, a, 'L' if structure == 'lower triangular' else 'U' )
    return fun


def _matrix_norm_general(norm, a, check_finite):
    a = np.asarray_chkfinite(a) if check_finite else a
    lange = get_lapack_funcs('lange', (a,))
    return lange(norm, a)


def _to_banded(n_below, n_above, a):
    n = a.shape[0]
    rows = n_above + n_below + 1
    ab = np.zeros((rows, n), dtype=a.dtype)
    ab[n_above] = np.diag(a)
    for i in range(1, n_above + 1):
        ab[n_above - i, i:] = np.diag(a, i)
    for i in range(1, n_below + 1):
        ab[n_above + i, :-i] = np.diag(a, -i)
    return ab


def _ensure_dtype_cdsz(*arrays):
    # Ensure that the dtype of arrays is one of the standard types
    # compatible with LAPACK functions (single or double precision
    # real or complex).
    dtype = np.result_type(*arrays)
    if not np.issubdtype(dtype, np.inexact):
        return (array.astype(np.float64) for array in arrays)
    complex = np.issubdtype(dtype, np.complexfloating)
    if np.finfo(dtype).bits <= 32:
        dtype = np.complex64 if complex else np.float32
    elif np.finfo(dtype).bits >= 64:
        dtype = np.complex128 if complex else np.float64
    return (array.astype(dtype, copy=False) for array in arrays)


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=True):
    """
    Solve the equation ``a x = b`` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``a x = b``
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x  = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and
        will not be referenced.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``a x = b``.  Shape of return matches `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular

    Notes
    -----
    .. versionadded:: 0.9.0

    Examples
    --------
    Solve the lower triangular system a x = b, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> import numpy as np
    >>> from scipy.linalg import solve_triangular
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x)  # Check the result
    array([ 4.,  2.,  4.,  2.])

    """

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)

    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    if a1.shape[0] != b1.shape[0]:
        raise ValueError(f'shapes of a {a1.shape} and b {b1.shape} are incompatible')

    # accommodate empty arrays
    if b1.size == 0:
        dt_nonempty = solve_triangular(
            np.eye(2, dtype=a1.dtype), np.ones(2, dtype=b1.dtype)
        ).dtype
        return np.empty_like(b1, dtype=dt_nonempty)

    overwrite_b = overwrite_b or _datacopied(b1, b)

    x, _ = _solve_triangular(a1, b1, trans, lower, unit_diagonal, overwrite_b)
    return x


# solve_triangular without the input validation
def _solve_triangular(a1, b1, trans=0, lower=False, unit_diagonal=False,
                      overwrite_b=False):

    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
    trtrs, = get_lapack_funcs(('trtrs',), (a1, b1))
    if a1.flags.f_contiguous or trans == 2:
        x, info = trtrs(a1, b1, overwrite_b=overwrite_b, lower=lower,
                        trans=trans, unitdiag=unit_diagonal)
    else:
        # transposed system is solved since trtrs expects Fortran ordering
        x, info = trtrs(a1.T, b1, overwrite_b=overwrite_b, lower=not lower,
                        trans=not trans, unitdiag=unit_diagonal)

    if info == 0:
        return x, info
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" %
                          (info-1))
    raise ValueError('illegal value in %dth argument of internal trtrs' %
                     (-info))


def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False,
                 check_finite=True):
    """
    Solve the equation a x = b for x, assuming a is banded matrix.

    The matrix a is stored in `ab` using the matrix diagonal ordered form::

        ab[u + i - j, j] == a[i,j]

    Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::

        *    a01  a12  a23  a34  a45
        a00  a11  a22  a33  a44  a55
        a10  a21  a32  a43  a54   *
        a20  a31  a42  a53   *    *

    Parameters
    ----------
    (l, u) : (integer, integer)
        Number of non-zero lower and upper diagonals
    ab : (`l` + `u` + 1, M) array_like
        Banded matrix
    b : (M,) or (M, K) array_like
        Right-hand side
    overwrite_ab : bool, optional
        Discard data in `ab` (may enhance performance)
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system a x = b. Returned shape depends on the
        shape of `b`.

    Examples
    --------
    Solve the banded system a x = b, where::

            [5  2 -1  0  0]       [0]
            [1  4  2 -1  0]       [1]
        a = [0  1  3  2 -1]   b = [2]
            [0  0  1  2  2]       [2]
            [0  0  0  1  1]       [3]

    There is one nonzero diagonal below the main diagonal (l = 1), and
    two above (u = 2). The diagonal banded form of the matrix is::

             [*  * -1 -1 -1]
        ab = [*  2  2  2  2]
             [5  4  3  2  1]
             [1  1  1  1  *]

    >>> import numpy as np
    >>> from scipy.linalg import solve_banded
    >>> ab = np.array([[0,  0, -1, -1, -1],
    ...                [0,  2,  2,  2,  2],
    ...                [5,  4,  3,  2,  1],
    ...                [1,  1,  1,  1,  0]])
    >>> b = np.array([0, 1, 2, 2, 3])
    >>> x = solve_banded((1, 2), ab, b)
    >>> x
    array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])

    """

    a1 = _asarray_validated(ab, check_finite=check_finite, as_inexact=True)
    b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)

    # Validate shapes.
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")

    (nlower, nupper) = l_and_u
    if nlower + nupper + 1 != a1.shape[0]:
        raise ValueError("invalid values for the number of lower and upper "
                         "diagonals: l+u+1 (%d) does not equal ab.shape[0] "
                         "(%d)" % (nlower + nupper + 1, ab.shape[0]))

    # accommodate empty arrays
    if b1.size == 0:
        dt = solve(np.eye(1, dtype=a1.dtype), np.ones(1, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)

    overwrite_b = overwrite_b or _datacopied(b1, b)
    if a1.shape[-1] == 1:
        b2 = np.array(b1, copy=(not overwrite_b))
        # a1.shape[-1] == 1 -> original matrix is 1x1. Typically, the user
        # will pass u = l = 0 and `a1` will be 1x1. However, the rest of the
        # function works with unnecessary rows in `a1` as long as
        # `a1[u + i - j, j] == a[i,j]`. In the 1x1 case, we want i = j = 0,
        # so the diagonal is in row `u` of `a1`. See gh-8906.
        b2 /= a1[nupper, 0]
        return b2
    if nlower == nupper == 1:
        overwrite_ab = overwrite_ab or _datacopied(a1, ab)
        gtsv, = get_lapack_funcs(('gtsv',), (a1, b1))
        du = a1[0, 1:]
        d = a1[1, :]
        dl = a1[2, :-1]
        du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab,
                                   overwrite_ab, overwrite_b)
    else:
        gbsv, = get_lapack_funcs(('gbsv',), (a1, b1))
        a2 = np.zeros((2*nlower + nupper + 1, a1.shape[1]), dtype=gbsv.dtype)
        a2[nlower:, :] = a1
        lu, piv, x, info = gbsv(nlower, nupper, a2, b1, overwrite_ab=True,
                                overwrite_b=overwrite_b)
    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix")
    raise ValueError('illegal value in %d-th argument of internal '
                     'gbsv/gtsv' % -info)


def solveh_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False,
                  check_finite=True):
    """
    Solve equation a x = b. a is Hermitian positive-definite banded matrix.

    Uses Thomas' Algorithm, which is more efficient than standard LU
    factorization, but should only be used for Hermitian positive-definite
    matrices.

    The matrix ``a`` is stored in `ab` either in lower diagonal or upper
    diagonal ordered form:

        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

    Example of `ab` (shape of ``a`` is (6, 6), number of upper diagonals,
    ``u`` =2)::

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
    ab : (``u`` + 1, M) array_like
        Banded matrix
    b : (M,) or (M, K) array_like
        Right-hand side
    overwrite_ab : bool, optional
        Discard data in `ab` (may enhance performance)
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance)
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system ``a x = b``. Shape of return matches shape
        of `b`.

    Notes
    -----
    In the case of a non-positive definite matrix ``a``, the solver
    `solve_banded` may be used.

    Examples
    --------
    Solve the banded system ``A x = b``, where::

            [ 4  2 -1  0  0  0]       [1]
            [ 2  5  2 -1  0  0]       [2]
        A = [-1  2  6  2 -1  0]   b = [2]
            [ 0 -1  2  7  2 -1]       [3]
            [ 0  0 -1  2  8  2]       [3]
            [ 0  0  0 -1  2  9]       [3]

    >>> import numpy as np
    >>> from scipy.linalg import solveh_banded

    ``ab`` contains the main diagonal and the nonzero diagonals below the
    main diagonal. That is, we use the lower form:

    >>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
    ...                [ 2,  2,  2,  2, 2, 0],
    ...                [-1, -1, -1, -1, 0, 0]])
    >>> b = np.array([1, 2, 2, 3, 3, 3])
    >>> x = solveh_banded(ab, b, lower=True)
    >>> x
    array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
            0.34733894])


    Solve the Hermitian banded system ``H x = b``, where::

            [ 8   2-1j   0     0  ]        [ 1  ]
        H = [2+1j  5     1j    0  ]    b = [1+1j]
            [ 0   -1j    9   -2-1j]        [1-2j]
            [ 0    0   -2+1j   6  ]        [ 0  ]

    In this example, we put the upper diagonals in the array ``hb``:

    >>> hb = np.array([[0, 2-1j, 1j, -2-1j],
    ...                [8,  5,    9,   6  ]])
    >>> b = np.array([1, 1+1j, 1-2j, 0])
    >>> x = solveh_banded(hb, b)
    >>> x
    array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
            0.10077984-0.23035393j, -0.00479904-0.09358128j])

    """
    a1 = _asarray_validated(ab, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)

    # Validate shapes.
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")

    # accommodate empty arrays
    if b1.size == 0:
        dt = solve(np.eye(1, dtype=a1.dtype), np.ones(1, dtype=b1.dtype)).dtype
        return np.empty_like(b1, dtype=dt)

    overwrite_b = overwrite_b or _datacopied(b1, b)
    overwrite_ab = overwrite_ab or _datacopied(a1, ab)

    if a1.shape[0] == 2:
        ptsv, = get_lapack_funcs(('ptsv',), (a1, b1))
        if lower:
            d = a1[0, :].real
            e = a1[1, :-1]
        else:
            d = a1[1, :].real
            e = a1[0, 1:].conj()
        d, du, x, info = ptsv(d, e, b1, overwrite_ab, overwrite_ab,
                              overwrite_b)
    else:
        pbsv, = get_lapack_funcs(('pbsv',), (a1, b1))
        c, x, info = pbsv(a1, b1, lower=lower, overwrite_ab=overwrite_ab,
                          overwrite_b=overwrite_b)
    if info > 0:
        raise LinAlgError("%dth leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal '
                         'pbsv' % -info)
    return x


def solve_toeplitz(c_or_cr, b, check_finite=True):
    r"""Solve a Toeplitz system using Levinson Recursion

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    .. warning::

        Beginning in SciPy 1.17, multidimensional input will be treated as a batch,
        not ``ravel``\ ed. To preserve the existing behavior, ``ravel`` arguments
        before passing them to `solve_toeplitz`.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system ``T x = b``. Shape of return matches shape
        of `b`.

    See Also
    --------
    toeplitz : Toeplitz matrix

    Notes
    -----
    The solution is computed using Levinson-Durbin recursion, which is faster
    than generic least-squares methods, but can be less numerically stable.

    Examples
    --------
    Solve the Toeplitz system T x = b, where::

            [ 1 -1 -2 -3]       [1]
        T = [ 3  1 -1 -2]   b = [2]
            [ 6  3  1 -1]       [2]
            [10  6  3  1]       [5]

    To specify the Toeplitz matrix, only the first column and the first
    row are needed.

    >>> import numpy as np
    >>> c = np.array([1, 3, 6, 10])    # First column of T
    >>> r = np.array([1, -1, -2, -3])  # First row of T
    >>> b = np.array([1, 2, 2, 5])

    >>> from scipy.linalg import solve_toeplitz, toeplitz
    >>> x = solve_toeplitz((c, r), b)
    >>> x
    array([ 1.66666667, -1.        , -2.66666667,  2.33333333])

    Check the result by creating the full Toeplitz matrix and
    multiplying it by `x`.  We should get `b`.

    >>> T = toeplitz(c, r)
    >>> T.dot(x)
    array([ 1.,  2.,  2.,  5.])

    """
    # If numerical stability of this algorithm is a problem, a future
    # developer might consider implementing other O(N^2) Toeplitz solvers,
    # such as GKO (https://www.jstor.org/stable/2153371) or Bareiss.

    r, c, b, dtype, b_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, b, check_finite, keep_b_shape=True)

    # accommodate empty arrays
    if b.size == 0:
        return np.empty_like(b)

    # Form a 1-D array of values to be used in the matrix, containing a
    # reversed copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    if b is None:
        raise ValueError('illegal value, `b` is a required argument')

    if b.ndim == 1:
        x, _ = levinson(vals, np.ascontiguousarray(b))
    else:
        x = np.column_stack([levinson(vals, np.ascontiguousarray(b[:, i]))[0]
                             for i in range(b.shape[1])])
        x = x.reshape(*b_shape)

    return x


def _get_axis_len(aname, a, axis):
    ax = axis
    if ax < 0:
        ax += a.ndim
    if 0 <= ax < a.ndim:
        return a.shape[ax]
    raise ValueError(f"'{aname}axis' entry is out of bounds")


def solve_circulant(c, b, singular='raise', tol=None,
                    caxis=-1, baxis=0, outaxis=0):
    """Solve C x = b for x, where C is a circulant matrix.

    `C` is the circulant matrix associated with the vector `c`.

    The system is solved by doing division in Fourier space. The
    calculation is::

        x = ifft(fft(b) / fft(c))

    where `fft` and `ifft` are the fast Fourier transform and its inverse,
    respectively. For a large vector `c`, this is *much* faster than
    solving the system with the full circulant matrix.

    Parameters
    ----------
    c : array_like
        The coefficients of the circulant matrix.
    b : array_like
        Right-hand side matrix in ``a x = b``.
    singular : str, optional
        This argument controls how a near singular circulant matrix is
        handled.  If `singular` is "raise" and the circulant matrix is
        near singular, a `LinAlgError` is raised. If `singular` is
        "lstsq", the least squares solution is returned. Default is "raise".
    tol : float, optional
        If any eigenvalue of the circulant matrix has an absolute value
        that is less than or equal to `tol`, the matrix is considered to be
        near singular. If not given, `tol` is set to::

            tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps

        where `abs_eigs` is the array of absolute values of the eigenvalues
        of the circulant matrix.
    caxis : int
        When `c` has dimension greater than 1, it is viewed as a collection
        of circulant vectors. In this case, `caxis` is the axis of `c` that
        holds the vectors of circulant coefficients.
    baxis : int
        When `b` has dimension greater than 1, it is viewed as a collection
        of vectors. In this case, `baxis` is the axis of `b` that holds the
        right-hand side vectors.
    outaxis : int
        When `c` or `b` are multidimensional, the value returned by
        `solve_circulant` is multidimensional. In this case, `outaxis` is
        the axis of the result that holds the solution vectors.

    Returns
    -------
    x : ndarray
        Solution to the system ``C x = b``.

    Raises
    ------
    LinAlgError
        If the circulant matrix associated with `c` is near singular.

    See Also
    --------
    circulant : circulant matrix

    Notes
    -----
    For a 1-D vector `c` with length `m`, and an array `b`
    with shape ``(m, ...)``,

        solve_circulant(c, b)

    returns the same result as

        solve(circulant(c), b)

    where `solve` and `circulant` are from `scipy.linalg`.

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import solve_circulant, solve, circulant, lstsq

    >>> c = np.array([2, 2, 4])
    >>> b = np.array([1, 2, 3])
    >>> solve_circulant(c, b)
    array([ 0.75, -0.25,  0.25])

    Compare that result to solving the system with `scipy.linalg.solve`:

    >>> solve(circulant(c), b)
    array([ 0.75, -0.25,  0.25])

    A singular example:

    >>> c = np.array([1, 1, 0, 0])
    >>> b = np.array([1, 2, 3, 4])

    Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the
    least square solution, use the option ``singular='lstsq'``:

    >>> solve_circulant(c, b, singular='lstsq')
    array([ 0.25,  1.25,  2.25,  1.25])

    Compare to `scipy.linalg.lstsq`:

    >>> x, resid, rnk, s = lstsq(circulant(c), b)
    >>> x
    array([ 0.25,  1.25,  2.25,  1.25])

    A broadcasting example:

    Suppose we have the vectors of two circulant matrices stored in an array
    with shape (2, 5), and three `b` vectors stored in an array with shape
    (3, 5).  For example,

    >>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])
    >>> b = np.arange(15).reshape(-1, 5)

    We want to solve all combinations of circulant matrices and `b` vectors,
    with the result stored in an array with shape (2, 3, 5). When we
    disregard the axes of `c` and `b` that hold the vectors of coefficients,
    the shapes of the collections are (2,) and (3,), respectively, which are
    not compatible for broadcasting. To have a broadcast result with shape
    (2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has
    shape (2, 1, 5). The last dimension holds the coefficients of the
    circulant matrices, so when we call `solve_circulant`, we can use the
    default ``caxis=-1``. The coefficients of the `b` vectors are in the last
    dimension of the array `b`, so we use ``baxis=-1``. If we use the
    default `outaxis`, the result will have shape (5, 2, 3), so we'll use
    ``outaxis=-1`` to put the solution vectors in the last dimension.

    >>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)
    >>> x.shape
    (2, 3, 5)
    >>> np.set_printoptions(precision=3)  # For compact output of numbers.
    >>> x
    array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],
            [ 0.651,  0.989,  2.046,  0.627,  1.072],
            [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],
           [[ 0.401,  0.304,  0.694, -0.867,  0.377],
            [ 0.856,  0.758,  1.149, -0.412,  0.831],
            [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])

    Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):

    >>> solve_circulant(c[1], b[1, :])
    array([ 0.856,  0.758,  1.149, -0.412,  0.831])

    """
    c = np.atleast_1d(c)
    nc = _get_axis_len("c", c, caxis)
    b = np.atleast_1d(b)
    nb = _get_axis_len("b", b, baxis)
    if nc != nb:
        raise ValueError(f'Shapes of c {c.shape} and b {b.shape} are incompatible')

    # accommodate empty arrays
    if b.size == 0:
        dt = solve_circulant(np.arange(3, dtype=c.dtype),
                             np.ones(3, dtype=b.dtype)).dtype
        return np.empty_like(b, dtype=dt)

    fc = np.fft.fft(np.moveaxis(c, caxis, -1), axis=-1)
    abs_fc = np.abs(fc)
    if tol is None:
        # This is the same tolerance as used in np.linalg.matrix_rank.
        tol = abs_fc.max(axis=-1) * nc * np.finfo(np.float64).eps
        if tol.shape != ():
            tol.shape = tol.shape + (1,)
        else:
            tol = np.atleast_1d(tol)

    near_zeros = abs_fc <= tol
    is_near_singular = np.any(near_zeros)
    if is_near_singular:
        if singular == 'raise':
            raise LinAlgError("near singular circulant matrix.")
        else:
            # Replace the small values with 1 to avoid errors in the
            # division fb/fc below.
            fc[near_zeros] = 1

    fb = np.fft.fft(np.moveaxis(b, baxis, -1), axis=-1)

    q = fb / fc

    if is_near_singular:
        # `near_zeros` is a boolean array, same shape as `c`, that is
        # True where `fc` is (near) zero. `q` is the broadcasted result
        # of fb / fc, so to set the values of `q` to 0 where `fc` is near
        # zero, we use a mask that is the broadcast result of an array
        # of True values shaped like `b` with `near_zeros`.
        mask = np.ones_like(b, dtype=bool) & near_zeros
        q[mask] = 0

    x = np.fft.ifft(q, axis=-1)
    if not (np.iscomplexobj(c) or np.iscomplexobj(b)):
        x = x.real
    if outaxis != -1:
        x = np.moveaxis(x, -1, outaxis)
    return x


# matrix inversion
def inv(a, overwrite_a=False, check_finite=True):
    """
    Compute the inverse of a matrix.

    Parameters
    ----------
    a : array_like
        Square matrix to be inverted.
    overwrite_a : bool, optional
        Discard data in `a` (may improve performance). Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    ainv : ndarray
        Inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is singular.
    ValueError
        If `a` is not square, or not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> linalg.inv(a)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.dot(a, linalg.inv(a))
    array([[ 1.,  0.],
           [ 0.,  1.]])

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    # accommodate empty square matrices
    if a1.size == 0:
        dt = inv(np.eye(2, dtype=a1.dtype)).dtype
        return np.empty_like(a1, dtype=dt)

    overwrite_a = overwrite_a or _datacopied(a1, a)
    getrf, getri, getri_lwork = get_lapack_funcs(('getrf', 'getri',
                                                  'getri_lwork'),
                                                 (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info == 0:
        lwork = _compute_lwork(getri_lwork, a1.shape[0])

        # XXX: the following line fixes curious SEGFAULT when
        # benchmarking 500x500 matrix inverse. This seems to
        # be a bug in LAPACK ?getri routine because if lwork is
        # minimal (when using lwork[0] instead of lwork[1]) then
        # all tests pass. Further investigation is required if
        # more such SEGFAULTs occur.
        lwork = int(1.01 * lwork)
        inv_a, info = getri(lu, piv, lwork=lwork, overwrite_lu=1)
    if info > 0:
        raise LinAlgError("singular matrix")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'getrf|getri' % -info)
    return inv_a


# Determinant

def det(a, overwrite_a=False, check_finite=True):
    """
    Compute the determinant of a matrix

    The determinant is a scalar that is a function of the associated square
    matrix coefficients. The determinant value is zero for singular matrices.

    Parameters
    ----------
    a : (..., M, M) array_like
        Input array to compute determinants for.
    overwrite_a : bool, optional
        Allow overwriting data in a (may enhance performance).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    det : (...) float or complex
        Determinant of `a`. For stacked arrays, a scalar is returned for each
        (m, m) slice in the last two dimensions of the input. For example, an
        input of shape (p, q, m, m) will produce a result of shape (p, q). If
        all dimensions are 1 a scalar is returned regardless of ndim.

    Notes
    -----
    The determinant is computed by performing an LU factorization of the
    input with LAPACK routine 'getrf', and then calculating the product of
    diagonal entries of the U factor.

    Even if the input array is single precision (float32 or complex64), the
    result will be returned in double precision (float64 or complex128) to
    prevent overflows.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])  # A singular matrix
    >>> linalg.det(a)
    0.0
    >>> b = np.array([[0,2,3], [4,5,6], [7,8,9]])
    >>> linalg.det(b)
    3.0
    >>> # An array with the shape (3, 2, 2, 2)
    >>> c = np.array([[[[1., 2.], [3., 4.]],
    ...                [[5., 6.], [7., 8.]]],
    ...               [[[9., 10.], [11., 12.]],
    ...                [[13., 14.], [15., 16.]]],
    ...               [[[17., 18.], [19., 20.]],
    ...                [[21., 22.], [23., 24.]]]])
    >>> linalg.det(c)  # The resulting shape is (3, 2)
    array([[-2., -2.],
           [-2., -2.],
           [-2., -2.]])
    >>> linalg.det(c[0, 0])  # Confirm the (0, 0) slice, [[1, 2], [3, 4]]
    -2.0
    """
    # The goal is to end up with a writable contiguous array to pass to Cython

    # First we check and make arrays.
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    if a1.ndim < 2:
        raise ValueError('The input array must be at least two-dimensional.')
    if a1.shape[-1] != a1.shape[-2]:
        raise ValueError('Last 2 dimensions of the array must be square'
                         f' but received shape {a1.shape}.')

    # Also check if dtype is LAPACK compatible
    if a1.dtype.char not in 'fdFD':
        dtype_char = lapack_cast_dict[a1.dtype.char]
        if not dtype_char:  # No casting possible
            raise TypeError(f'The dtype "{a1.dtype.name}" cannot be cast '
                            'to float(32, 64) or complex(64, 128).')

        a1 = a1.astype(dtype_char[0])  # makes a copy, free to scratch
        overwrite_a = True

    # Empty array has determinant 1 because math.
    if min(*a1.shape) == 0:
        dtyp = np.float64 if a1.dtype.char not in 'FD' else np.complex128
        if a1.ndim == 2:
            return dtyp(1.0)
        else:
            return np.ones(shape=a1.shape[:-2], dtype=dtyp)

    # Scalar case
    if a1.shape[-2:] == (1, 1):
        a1 = a1[..., 0, 0]
        if a1.ndim == 0:
            a1 = a1[()]
        # Convert float32 to float64, and complex64 to complex128.
        if a1.dtype.char in 'dD':
            return a1
        return a1.astype('d') if a1.dtype.char == 'f' else a1.astype('D')

    # Then check overwrite permission
    if not _datacopied(a1, a):  # "a"  still alive through "a1"
        if not overwrite_a:
            # Data belongs to "a" so make a copy
            a1 = a1.copy(order='C')
        #  else: Do nothing we'll use "a" if possible
    # else:  a1 has its own data thus free to scratch

    # Then layout checks, might happen that overwrite is allowed but original
    # array was read-only or non-C-contiguous.
    if not (a1.flags['C_CONTIGUOUS'] and a1.flags['WRITEABLE']):
        a1 = a1.copy(order='C')

    if a1.ndim == 2:
        det = find_det_from_lu(a1)
        # Convert float, complex to NumPy scalars
        return (np.float64(det) if np.isrealobj(det) else np.complex128(det))

    # loop over the stacked array, and avoid overflows for single precision
    # Cf. np.linalg.det(np.diag([1e+38, 1e+38]).astype(np.float32))
    dtype_char = a1.dtype.char
    if dtype_char in 'fF':
        dtype_char = 'd' if dtype_char.islower() else 'D'

    det = np.empty(a1.shape[:-2], dtype=dtype_char)
    for ind in product(*[range(x) for x in a1.shape[:-2]]):
        det[ind] = find_det_from_lu(a1[ind])
    return det


# Linear Least Squares
def lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True, lapack_driver=None):
    """
    Compute least-squares solution to equation Ax = b.

    Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

    Parameters
    ----------
    a : (M, N) array_like
        Left-hand side array
    b : (M,) or (M, K) array_like
        Right hand side array
    cond : float, optional
        Cutoff for 'small' singular values; used to determine effective
        rank of a. Singular values smaller than
        ``cond * largest_singular_value`` are considered zero.
    overwrite_a : bool, optional
        Discard data in `a` (may enhance performance). Default is False.
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance). Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : str, optional
        Which LAPACK driver is used to solve the least-squares problem.
        Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
        (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
        faster on many problems.  ``'gelss'`` was used historically.  It is
        generally slow but uses less memory.

        .. versionadded:: 0.17.0

    Returns
    -------
    x : (N,) or (N, K) ndarray
        Least-squares solution.
    residues : (K,) ndarray or float
        Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
        ``rank(A) == n`` (returns a scalar if ``b`` is 1-D). Otherwise a
        (0,)-shaped array is returned.
    rank : int
        Effective rank of `a`.
    s : (min(M, N),) ndarray or None
        Singular values of `a`. The condition number of ``a`` is
        ``s[0] / s[-1]``.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    ValueError
        When parameters are not compatible.

    See Also
    --------
    scipy.optimize.nnls : linear least squares with non-negativity constraint

    Notes
    -----
    When ``'gelsy'`` is used as a driver, `residues` is set to a (0,)-shaped
    array and `s` is always ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lstsq
    >>> import matplotlib.pyplot as plt

    Suppose we have the following data:

    >>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    >>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

    We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
    to this data.  We first form the "design matrix" M, with a constant
    column of 1s and a column containing ``x**2``:

    >>> M = x[:, np.newaxis]**[0, 2]
    >>> M
    array([[  1.  ,   1.  ],
           [  1.  ,   6.25],
           [  1.  ,  12.25],
           [  1.  ,  16.  ],
           [  1.  ,  25.  ],
           [  1.  ,  49.  ],
           [  1.  ,  72.25]])

    We want to find the least-squares solution to ``M.dot(p) = y``,
    where ``p`` is a vector with length 2 that holds the parameters
    ``a`` and ``b``.

    >>> p, res, rnk, s = lstsq(M, y)
    >>> p
    array([ 0.20925829,  0.12013861])

    Plot the data and the fitted curve.

    >>> plt.plot(x, y, 'o', label='data')
    >>> xx = np.linspace(0, 9, 101)
    >>> yy = p[0] + p[1]*xx**2
    >>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.show()

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError('Input array a should be 2D')
    m, n = a1.shape
    if len(b1.shape) == 2:
        nrhs = b1.shape[1]
    else:
        nrhs = 1
    if m != b1.shape[0]:
        raise ValueError('Shape mismatch: a and b should have the same number'
                         f' of rows ({m} != {b1.shape[0]}).')
    if m == 0 or n == 0:  # Zero-sized problem, confuses LAPACK
        x = np.zeros((n,) + b1.shape[1:], dtype=np.common_type(a1, b1))
        if n == 0:
            residues = np.linalg.norm(b1, axis=0)**2
        else:
            residues = np.empty((0,))
        return x, residues, 0, np.empty((0,))

    driver = lapack_driver
    if driver is None:
        driver = lstsq.default_lapack_driver
    if driver not in ('gelsd', 'gelsy', 'gelss'):
        raise ValueError(f'LAPACK driver "{driver}" is not found')

    lapack_func, lapack_lwork = get_lapack_funcs((driver,
                                                 f'{driver}_lwork'),
                                                 (a1, b1))
    real_data = True if (lapack_func.dtype.kind == 'f') else False

    if m < n:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        if len(b1.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=lapack_func.dtype)
            b2[:m, :] = b1
        else:
            b2 = np.zeros(n, dtype=lapack_func.dtype)
            b2[:m] = b1
        b1 = b2

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    if cond is None:
        cond = np.finfo(lapack_func.dtype).eps

    if driver in ('gelss', 'gelsd'):
        if driver == 'gelss':
            lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
            v, x, s, rank, work, info = lapack_func(a1, b1, cond, lwork,
                                                    overwrite_a=overwrite_a,
                                                    overwrite_b=overwrite_b)

        elif driver == 'gelsd':
            if real_data:
                lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
                x, s, rank, info = lapack_func(a1, b1, lwork,
                                               iwork, cond, False, False)
            else:  # complex data
                lwork, rwork, iwork = _compute_lwork(lapack_lwork, m, n,
                                                     nrhs, cond)
                x, s, rank, info = lapack_func(a1, b1, lwork, rwork, iwork,
                                               cond, False, False)
        if info > 0:
            raise LinAlgError("SVD did not converge in Linear Least Squares")
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal %s'
                             % (-info, lapack_driver))
        resids = np.asarray([], dtype=x.dtype)
        if m > n:
            x1 = x[:n]
            if rank == n:
                resids = np.sum(np.abs(x[n:])**2, axis=0)
            x = x1
        return x, resids, rank, s

    elif driver == 'gelsy':
        lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
        jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
        v, x, j, rank, info = lapack_func(a1, b1, jptv, cond,
                                          lwork, False, False)
        if info < 0:
            raise ValueError("illegal value in %d-th argument of internal "
                             "gelsy" % -info)
        if m > n:
            x1 = x[:n]
            x = x1
        return x, np.array([], x.dtype), rank, None


lstsq.default_lapack_driver = 'gelsd'


def pinv(a, *, atol=None, rtol=None, return_rank=False, check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition ``U @ S @ V`` in the economy mode and picking
    up only the columns/rows that are associated with significant singular
    values.

    If ``s`` is the maximum singular value of ``a``, then the
    significance cut-off value is determined by ``atol + rtol * s``. Any
    singular value below this value is assumed insignificant.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    atol : float, optional
        Absolute threshold term, default value is 0.

        .. versionadded:: 1.7.0

    rtol : float, optional
        Relative threshold term, default value is ``max(M, N) * eps`` where
        ``eps`` is the machine precision value of the datatype of ``a``.

        .. versionadded:: 1.7.0

    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix. Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    pinvh : Moore-Penrose pseudoinverse of a hermitian matrix.

    Notes
    -----
    If ``A`` is invertible then the Moore-Penrose pseudoinverse is exactly
    the inverse of ``A`` [1]_. If ``A`` is not invertible then the
    Moore-Penrose pseudoinverse computes the ``x`` solution to ``Ax = b`` such
    that ``||Ax - b||`` is minimized [1]_.

    References
    ----------
    .. [1] Penrose, R. (1956). On best approximate solutions of linear matrix
           equations. Mathematical Proceedings of the Cambridge Philosophical
           Society, 52(1), 17-19. doi:10.1017/S0305004100030929

    Examples
    --------

    Given an ``m x n`` matrix ``A`` and an ``n x m`` matrix ``B`` the four
    Moore-Penrose conditions are:

    1. ``ABA = A`` (``B`` is a generalized inverse of ``A``),
    2. ``BAB = B`` (``A`` is a generalized inverse of ``B``),
    3. ``(AB)* = AB`` (``AB`` is hermitian),
    4. ``(BA)* = BA`` (``BA`` is hermitian) [1]_.

    Here, ``A*`` denotes the conjugate transpose. The Moore-Penrose
    pseudoinverse is a unique ``B`` that satisfies all four of these
    conditions and exists for any ``A``. Note that, unlike the standard
    matrix inverse, ``A`` does not have to be a square matrix or have
    linearly independent columns/rows.

    As an example, we can calculate the Moore-Penrose pseudoinverse of a
    random non-square matrix and verify it satisfies the four conditions.

    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> A = rng.standard_normal((9, 6))
    >>> B = linalg.pinv(A)
    >>> np.allclose(A @ B @ A, A)  # Condition 1
    True
    >>> np.allclose(B @ A @ B, B)  # Condition 2
    True
    >>> np.allclose((A @ B).conj().T, A @ B)  # Condition 3
    True
    >>> np.allclose((B @ A).conj().T, B @ A)  # Condition 4
    True

    """
    a = _asarray_validated(a, check_finite=check_finite)
    u, s, vh = _decomp_svd.svd(a, full_matrices=False, check_finite=False)
    t = u.dtype.char.lower()
    maxS = np.max(s, initial=0.)

    atol = 0. if atol is None else atol
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol

    if (atol < 0.) or (rtol < 0.):
        raise ValueError("atol and rtol values must be positive.")

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    if return_rank:
        return B, rank
    else:
        return B


def pinvh(a, atol=None, rtol=None, lower=True, return_rank=False,
          check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

    Calculate a generalized inverse of a complex Hermitian/real symmetric
    matrix using its eigenvalue decomposition and including all eigenvalues
    with 'large' absolute value.

    Parameters
    ----------
    a : (N, N) array_like
        Real symmetric or complex hermetian matrix to be pseudo-inverted

    atol : float, optional
        Absolute threshold term, default value is 0.

        .. versionadded:: 1.7.0

    rtol : float, optional
        Relative threshold term, default value is ``N * eps`` where
        ``eps`` is the machine precision value of the datatype of ``a``.

        .. versionadded:: 1.7.0

    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `a`. (Default: lower)
    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, N) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix.  Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If eigenvalue algorithm does not converge.

    See Also
    --------
    pinv : Moore-Penrose pseudoinverse of a matrix.

    Examples
    --------

    For a more detailed example see `pinv`.

    >>> import numpy as np
    >>> from scipy.linalg import pinvh
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((9, 6))
    >>> a = np.dot(a, a.T)
    >>> B = pinvh(a)
    >>> np.allclose(a, a @ B @ a)
    True
    >>> np.allclose(B, B @ a @ B)
    True

    """
    a = _asarray_validated(a, check_finite=check_finite)
    s, u = _decomp.eigh(a, lower=lower, check_finite=False, driver='ev')
    t = u.dtype.char.lower()
    maxS = np.max(np.abs(s), initial=0.)

    atol = 0. if atol is None else atol
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol

    if (atol < 0.) or (rtol < 0.):
        raise ValueError("atol and rtol values must be positive.")

    val = atol + maxS * rtol
    above_cutoff = (abs(s) > val)

    psigma_diag = 1.0 / s[above_cutoff]
    u = u[:, above_cutoff]

    B = (u * psigma_diag) @ u.conj().T

    if return_rank:
        return B, len(psigma_diag)
    else:
        return B


def matrix_balance(A, permute=True, scale=True, separate=False,
                   overwrite_a=False):
    """
    Compute a diagonal similarity transformation for row/column balancing.

    The balancing tries to equalize the row and column 1-norms by applying
    a similarity transformation such that the magnitude variation of the
    matrix entries is reflected to the scaling matrices.

    Moreover, if enabled, the matrix is first permuted to isolate the upper
    triangular parts of the matrix and, again if scaling is also enabled,
    only the remaining subblocks are subjected to scaling.

    The balanced matrix satisfies the following equality

    .. math::

                        B = T^{-1} A T

    The scaling coefficients are approximated to the nearest power of 2
    to avoid round-off errors.

    Parameters
    ----------
    A : (n, n) array_like
        Square data matrix for the balancing.
    permute : bool, optional
        The selector to define whether permutation of A is also performed
        prior to scaling.
    scale : bool, optional
        The selector to turn on and off the scaling. If False, the matrix
        will not be scaled.
    separate : bool, optional
        This switches from returning a full matrix of the transformation
        to a tuple of two separate 1-D permutation and scaling arrays.
    overwrite_a : bool, optional
        This is passed to xGEBAL directly. Essentially, overwrites the result
        to the data. It might increase the space efficiency. See LAPACK manual
        for details. This is False by default.

    Returns
    -------
    B : (n, n) ndarray
        Balanced matrix
    T : (n, n) ndarray
        A possibly permuted diagonal matrix whose nonzero entries are
        integer powers of 2 to avoid numerical truncation errors.
    scale, perm : (n,) ndarray
        If ``separate`` keyword is set to True then instead of the array
        ``T`` above, the scaling and the permutation vectors are given
        separately as a tuple without allocating the full array ``T``.

    Notes
    -----
    This algorithm is particularly useful for eigenvalue and matrix
    decompositions and in many cases it is already called by various
    LAPACK routines.

    The algorithm is based on the well-known technique of [1]_ and has
    been modified to account for special cases. See [2]_ for details
    which have been implemented since LAPACK v3.5.0. Before this version
    there are corner cases where balancing can actually worsen the
    conditioning. See [3]_ for such examples.

    The code is a wrapper around LAPACK's xGEBAL routine family for matrix
    balancing.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] B.N. Parlett and C. Reinsch, "Balancing a Matrix for
       Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,
       Vol.13(4), 1969, :doi:`10.1007/BF02165404`
    .. [2] R. James, J. Langou, B.R. Lowery, "On matrix balancing and
       eigenvector computation", 2014, :arxiv:`1401.5766`
    .. [3] D.S. Watkins. A case where balancing is harmful.
       Electron. Trans. Numer. Anal, Vol.23, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])

    >>> y, permscale = linalg.matrix_balance(x)
    >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
    array([ 3.66666667,  0.4995005 ,  0.91312162])

    >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
    array([ 1.2       ,  1.27041742,  0.92658316])  # may vary

    >>> permscale  # only powers of 2 (0.5 == 2^(-1))
    array([[  0.5,   0. ,  0. ],  # may vary
           [  0. ,   1. ,  0. ],
           [  0. ,   0. ,  1. ]])

    """

    A = np.atleast_2d(_asarray_validated(A, check_finite=True))

    if not np.equal(*A.shape):
        raise ValueError('The data matrix for balancing should be square.')

    # accommodate empty arrays
    if A.size == 0:
        b_n, t_n = matrix_balance(np.eye(2, dtype=A.dtype))
        B = np.empty_like(A, dtype=b_n.dtype)
        if separate:
            scaling = np.ones_like(A, shape=len(A))
            perm = np.arange(len(A))
            return B, (scaling, perm)
        return B, np.empty_like(A, dtype=t_n.dtype)

    gebal = get_lapack_funcs(('gebal'), (A,))
    B, lo, hi, ps, info = gebal(A, scale=scale, permute=permute,
                                overwrite_a=overwrite_a)

    if info < 0:
        raise ValueError('xGEBAL exited with the internal error '
                         f'"illegal value in argument number {-info}.". See '
                         'LAPACK documentation for the xGEBAL error codes.')

    # Separate the permutations from the scalings and then convert to int
    scaling = np.ones_like(ps, dtype=float)
    scaling[lo:hi+1] = ps[lo:hi+1]

    # gebal uses 1-indexing
    ps = ps.astype(int, copy=False) - 1
    n = A.shape[0]
    perm = np.arange(n)

    # LAPACK permutes with the ordering n --> hi, then 0--> lo
    if hi < n:
        for ind, x in enumerate(ps[hi+1:][::-1], 1):
            if n-ind == x:
                continue
            perm[[x, n-ind]] = perm[[n-ind, x]]

    if lo > 0:
        for ind, x in enumerate(ps[:lo]):
            if ind == x:
                continue
            perm[[x, ind]] = perm[[ind, x]]

    if separate:
        return B, (scaling, perm)

    # get the inverse permutation
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(n)

    return B, np.diag(scaling)[iperm, :]


def _validate_args_for_toeplitz_ops(c_or_cr, b, check_finite, keep_b_shape,
                                    enforce_square=True):
    """Validate arguments and format inputs for toeplitz functions

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.
    keep_b_shape : bool
        Whether to convert a (M,) dimensional b into a (M, 1) dimensional
        matrix.
    enforce_square : bool, optional
        If True (default), this verifies that the Toeplitz matrix is square.

    Returns
    -------
    r : array
        1d array corresponding to the first row of the Toeplitz matrix.
    c: array
        1d array corresponding to the first column of the Toeplitz matrix.
    b: array
        (M,), (M, 1) or (M, K) dimensional array, post validation,
        corresponding to ``b``.
    dtype: numpy datatype
        ``dtype`` stores the datatype of ``r``, ``c`` and ``b``. If any of
        ``r``, ``c`` or ``b`` are complex, ``dtype`` is ``np.complex128``,
        otherwise, it is ``np.float``.
    b_shape: tuple
        Shape of ``b`` after passing it through ``_asarray_validated``.

    """

    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        c = _asarray_validated(c, check_finite=check_finite)
        r = _asarray_validated(r, check_finite=check_finite)
    else:
        c = _asarray_validated(c_or_cr, check_finite=check_finite)
        r = c.conjugate()

    if c.ndim > 1 or r.ndim > 1:
        msg = ("Beginning in SciPy 1.17, multidimensional input will be treated as a "
               "batch, not `ravel`ed. To preserve the existing behavior and silence "
               "this warning, `ravel` arguments before passing them to "
               "`toeplitz`, `matmul_toeplitz`, and `solve_toeplitz`.")
        warnings.warn(msg, FutureWarning, stacklevel=2)
        c = c.ravel()
        r = r.ravel()

    if b is None:
        raise ValueError('`b` must be an array, not None.')

    b = _asarray_validated(b, check_finite=check_finite)
    b_shape = b.shape

    is_not_square = r.shape[0] != c.shape[0]
    if (enforce_square and is_not_square) or b.shape[0] != r.shape[0]:
        raise ValueError('Incompatible dimensions.')

    is_cmplx = np.iscomplexobj(r) or np.iscomplexobj(c) or np.iscomplexobj(b)
    dtype = np.complex128 if is_cmplx else np.float64
    r, c, b = (np.asarray(i, dtype=dtype) for i in (r, c, b))

    if b.ndim == 1 and not keep_b_shape:
        b = b.reshape(-1, 1)
    elif b.ndim != 1:
        b = b.reshape(b.shape[0], -1 if b.size > 0 else 0)

    return r, c, b, dtype, b_shape


def matmul_toeplitz(c_or_cr, x, check_finite=False, workers=None):
    r"""Efficient Toeplitz Matrix-Matrix Multiplication using FFT

    This function returns the matrix multiplication between a Toeplitz
    matrix and a dense matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    .. warning::

        Beginning in SciPy 1.17, multidimensional input will be treated as a batch,
        not ``ravel``\ ed. To preserve the existing behavior, ``ravel`` arguments
        before passing them to `matmul_toeplitz`.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``.
    x : (M,) or (M, K) array_like
        Matrix with which to multiply.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.
    workers : int, optional
        To pass to scipy.fft.fft and ifft. Maximum number of workers to use
        for parallel computation. If negative, the value wraps around from
        ``os.cpu_count()``. See scipy.fft.fft for more details.

    Returns
    -------
    T @ x : (M,) or (M, K) ndarray
        The result of the matrix multiplication ``T @ x``. Shape of return
        matches shape of `x`.

    See Also
    --------
    toeplitz : Toeplitz matrix
    solve_toeplitz : Solve a Toeplitz system using Levinson Recursion

    Notes
    -----
    The Toeplitz matrix is embedded in a circulant matrix and the FFT is used
    to efficiently calculate the matrix-matrix product.

    Because the computation is based on the FFT, integer inputs will
    result in floating point outputs.  This is unlike NumPy's `matmul`,
    which preserves the data type of the input.

    This is partly based on the implementation that can be found in [1]_,
    licensed under the MIT license. More information about the method can be
    found in reference [2]_. References [3]_ and [4]_ have more reference
    implementations in Python.

    .. versionadded:: 1.6.0

    References
    ----------
    .. [1] Jacob R Gardner, Geoff Pleiss, David Bindel, Kilian
       Q Weinberger, Andrew Gordon Wilson, "GPyTorch: Blackbox Matrix-Matrix
       Gaussian Process Inference with GPU Acceleration" with contributions
       from Max Balandat and Ruihan Wu. Available online:
       https://github.com/cornellius-gp/gpytorch

    .. [2] J. Demmel, P. Koev, and X. Li, "A Brief Survey of Direct Linear
       Solvers". In Z. Bai, J. Demmel, J. Dongarra, A. Ruhe, and H. van der
       Vorst, editors. Templates for the Solution of Algebraic Eigenvalue
       Problems: A Practical Guide. SIAM, Philadelphia, 2000. Available at:
       http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html

    .. [3] R. Scheibler, E. Bezzam, I. Dokmanic, Pyroomacoustics: A Python
       package for audio room simulations and array processing algorithms,
       Proc. IEEE ICASSP, Calgary, CA, 2018.
       https://github.com/LCAV/pyroomacoustics/blob/pypi-release/
       pyroomacoustics/adaptive/util.py

    .. [4] Marano S, Edwards B, Ferrari G and Fah D (2017), "Fitting
       Earthquake Spectra: Colored Noise and Incomplete Data", Bulletin of
       the Seismological Society of America., January, 2017. Vol. 107(1),
       pp. 276-291.

    Examples
    --------
    Multiply the Toeplitz matrix T with matrix x::

            [ 1 -1 -2 -3]       [1 10]
        T = [ 3  1 -1 -2]   x = [2 11]
            [ 6  3  1 -1]       [2 11]
            [10  6  3  1]       [5 19]

    To specify the Toeplitz matrix, only the first column and the first
    row are needed.

    >>> import numpy as np
    >>> c = np.array([1, 3, 6, 10])    # First column of T
    >>> r = np.array([1, -1, -2, -3])  # First row of T
    >>> x = np.array([[1, 10], [2, 11], [2, 11], [5, 19]])

    >>> from scipy.linalg import toeplitz, matmul_toeplitz
    >>> matmul_toeplitz((c, r), x)
    array([[-20., -80.],
           [ -7.,  -8.],
           [  9.,  85.],
           [ 33., 218.]])

    Check the result by creating the full Toeplitz matrix and
    multiplying it by ``x``.

    >>> toeplitz(c, r) @ x
    array([[-20, -80],
           [ -7,  -8],
           [  9,  85],
           [ 33, 218]])

    The full matrix is never formed explicitly, so this routine
    is suitable for very large Toeplitz matrices.

    >>> n = 1000000
    >>> matmul_toeplitz([1] + [0]*(n-1), np.ones(n))
    array([1., 1., 1., ..., 1., 1., 1.], shape=(1000000,))

    """

    from ..fft import fft, ifft, rfft, irfft

    r, c, x, dtype, x_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, x, check_finite, keep_b_shape=False, enforce_square=False)
    n, m = x.shape

    T_nrows = len(c)
    T_ncols = len(r)
    p = T_nrows + T_ncols - 1  # equivalent to len(embedded_col)
    return_shape = (T_nrows,) if len(x_shape) == 1 else (T_nrows, m)

    # accommodate empty arrays
    if x.size == 0:
        return np.empty_like(x, shape=return_shape)

    embedded_col = np.concatenate((c, r[-1:0:-1]))

    if np.iscomplexobj(embedded_col) or np.iscomplexobj(x):
        fft_mat = fft(embedded_col, axis=0, workers=workers).reshape(-1, 1)
        fft_x = fft(x, n=p, axis=0, workers=workers)

        mat_times_x = ifft(fft_mat*fft_x, axis=0,
                           workers=workers)[:T_nrows, :]
    else:
        # Real inputs; using rfft is faster
        fft_mat = rfft(embedded_col, axis=0, workers=workers).reshape(-1, 1)
        fft_x = rfft(x, n=p, axis=0, workers=workers)

        mat_times_x = irfft(fft_mat*fft_x, axis=0,
                            workers=workers, n=p)[:T_nrows, :]

    return mat_times_x.reshape(*return_shape)
<|endoftext|>from collections.abc import Iterable
import numpy as np

from scipy._lib._util import _asarray_validated
from scipy.linalg import block_diag, LinAlgError
from .lapack import _compute_lwork, get_lapack_funcs

__all__ = ['cossin']


def cossin(X, p=None, q=None, separate=False,
           swap_sign=False, compute_u=True, compute_vh=True):
    """
    Compute the cosine-sine (CS) decomposition of an orthogonal/unitary matrix.

    X is an ``(m, m)`` orthogonal/unitary matrix, partitioned as the following
    where upper left block has the shape of ``(p, q)``::

                                   ┌                   ┐
                                   │ I  0  0 │ 0  0  0 │
        ┌           ┐   ┌         ┐│ 0  C  0 │ 0 -S  0 │┌         ┐*
        │ X11 │ X12 │   │ U1 │    ││ 0  0  0 │ 0  0 -I ││ V1 │    │
        │ ────┼──── │ = │────┼────││─────────┼─────────││────┼────│
        │ X21 │ X22 │   │    │ U2 ││ 0  0  0 │ I  0  0 ││    │ V2 │
        └           ┘   └         ┘│ 0  S  0 │ 0  C  0 │└         ┘
                                   │ 0  0  I │ 0  0  0 │
                                   └                   ┘

    ``U1``, ``U2``, ``V1``, ``V2`` are square orthogonal/unitary matrices of
    dimensions ``(p,p)``, ``(m-p,m-p)``, ``(q,q)``, and ``(m-q,m-q)``
    respectively, and ``C`` and ``S`` are ``(r, r)`` nonnegative diagonal
    matrices satisfying ``C^2 + S^2 = I`` where ``r = min(p, m-p, q, m-q)``.

    Moreover, the rank of the identity matrices are ``min(p, q) - r``,
    ``min(p, m - q) - r``, ``min(m - p, q) - r``, and ``min(m - p, m - q) - r``
    respectively.

    X can be supplied either by itself and block specifications p, q or its
    subblocks in an iterable from which the shapes would be derived. See the
    examples below.

    Parameters
    ----------
    X : array_like, iterable
        complex unitary or real orthogonal matrix to be decomposed, or iterable
        of subblocks ``X11``, ``X12``, ``X21``, ``X22``, when ``p``, ``q`` are
        omitted.
    p : int, optional
        Number of rows of the upper left block ``X11``, used only when X is
        given as an array.
    q : int, optional
        Number of columns of the upper left block ``X11``, used only when X is
        given as an array.
    separate : bool, optional
        if ``True``, the low level components are returned instead of the
        matrix factors, i.e. ``(u1,u2)``, ``theta``, ``(v1h,v2h)`` instead of
        ``u``, ``cs``, ``vh``.
    swap_sign : bool, optional
        if ``True``, the ``-S``, ``-I`` block will be the bottom left,
        otherwise (by default) they will be in the upper right block.
    compute_u : bool, optional
        if ``False``, ``u`` won't be computed and an empty array is returned.
    compute_vh : bool, optional
        if ``False``, ``vh`` won't be computed and an empty array is returned.

    Returns
    -------
    u : ndarray
        When ``compute_u=True``, contains the block diagonal orthogonal/unitary
        matrix consisting of the blocks ``U1`` (``p`` x ``p``) and ``U2``
        (``m-p`` x ``m-p``) orthogonal/unitary matrices. If ``separate=True``,
        this contains the tuple of ``(U1, U2)``.
    cs : ndarray
        The cosine-sine factor with the structure described above.
         If ``separate=True``, this contains the ``theta`` array containing the
         angles in radians.
    vh : ndarray
        When ``compute_vh=True`, contains the block diagonal orthogonal/unitary
        matrix consisting of the blocks ``V1H`` (``q`` x ``q``) and ``V2H``
        (``m-q`` x ``m-q``) orthogonal/unitary matrices. If ``separate=True``,
        this contains the tuple of ``(V1H, V2H)``.

    References
    ----------
    .. [1] Brian D. Sutton. Computing the complete CS decomposition. Numer.
           Algorithms, 50(1):33-65, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cossin
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(4)
    >>> u, cs, vdh = cossin(x, p=2, q=2)
    >>> np.allclose(x, u @ cs @ vdh)
    True

    Same can be entered via subblocks without the need of ``p`` and ``q``. Also
    let's skip the computation of ``u``

    >>> ue, cs, vdh = cossin((x[:2, :2], x[:2, 2:], x[2:, :2], x[2:, 2:]),
    ...                      compute_u=False)
    >>> print(ue)
    []
    >>> np.allclose(x, u @ cs @ vdh)
    True

    """

    if p or q:
        p = 1 if p is None else int(p)
        q = 1 if q is None else int(q)
        X = _asarray_validated(X, check_finite=True)
        if not np.equal(*X.shape):
            raise ValueError("Cosine Sine decomposition only supports square"
                             f" matrices, got {X.shape}")
        m = X.shape[0]
        if p >= m or p <= 0:
            raise ValueError(f"invalid p={p}, 0<p<{X.shape[0]} must hold")
        if q >= m or q <= 0:
            raise ValueError(f"invalid q={q}, 0<q<{X.shape[0]} must hold")

        x11, x12, x21, x22 = X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:]
    elif not isinstance(X, Iterable):
        raise ValueError("When p and q are None, X must be an Iterable"
                         " containing the subblocks of X")
    else:
        if len(X) != 4:
            raise ValueError("When p and q are None, exactly four arrays"
                             f" should be in X, got {len(X)}")

        x11, x12, x21, x22 = (np.atleast_2d(x) for x in X)
        for name, block in zip(["x11", "x12", "x21", "x22"],
                               [x11, x12, x21, x22]):
            if block.shape[1] == 0:
                raise ValueError(f"{name} can't be empty")
        p, q = x11.shape
        mmp, mmq = x22.shape

        if x12.shape != (p, mmq):
            raise ValueError(f"Invalid x12 dimensions: desired {(p, mmq)}, "
                             f"got {x12.shape}")

        if x21.shape != (mmp, q):
            raise ValueError(f"Invalid x21 dimensions: desired {(mmp, q)}, "
                             f"got {x21.shape}")

        if p + mmp != q + mmq:
            raise ValueError("The subblocks have compatible sizes but "
                             "don't form a square array (instead they form a"
                              f" {p + mmp}x{q + mmq} array). This might be "
                              "due to missing p, q arguments.")

        m = p + mmp

    cplx = any([np.iscomplexobj(x) for x in [x11, x12, x21, x22]])
    driver = "uncsd" if cplx else "orcsd"
    csd, csd_lwork = get_lapack_funcs([driver, driver + "_lwork"],
                                      [x11, x12, x21, x22])
    lwork = _compute_lwork(csd_lwork, m=m, p=p, q=q)
    lwork_args = ({'lwork': lwork[0], 'lrwork': lwork[1]} if cplx else
                  {'lwork': lwork})
    *_, theta, u1, u2, v1h, v2h, info = csd(x11=x11, x12=x12, x21=x21, x22=x22,
                                            compute_u1=compute_u,
                                            compute_u2=compute_u,
                                            compute_v1t=compute_vh,
                                            compute_v2t=compute_vh,
                                            trans=False, signs=swap_sign,
                                            **lwork_args)

    method_name = csd.typecode + driver
    if info < 0:
        raise ValueError(f'illegal value in argument {-info} '
                         f'of internal {method_name}')
    if info > 0:
        raise LinAlgError(f"{method_name} did not converge: {info}")

    if separate:
        return (u1, u2), theta, (v1h, v2h)

    U = block_diag(u1, u2)
    VDH = block_diag(v1h, v2h)

    # Construct the middle factor CS
    c = np.diag(np.cos(theta))
    s = np.diag(np.sin(theta))
    r = min(p, q, m - p, m - q)
    n11 = min(p, q) - r
    n12 = min(p, m - q) - r
    n21 = min(m - p, q) - r
    n22 = min(m - p, m - q) - r
    Id = np.eye(np.max([n11, n12, n21, n22, r]), dtype=theta.dtype)
    CS = np.zeros((m, m), dtype=theta.dtype)

    CS[:n11, :n11] = Id[:n11, :n11]

    xs = n11 + r
    xe = n11 + r + n12
    ys = n11 + n21 + n22 + 2 * r
    ye = n11 + n21 + n22 + 2 * r + n12
    CS[xs: xe, ys:ye] = Id[:n12, :n12] if swap_sign else -Id[:n12, :n12]

    xs = p + n22 + r
    xe = p + n22 + r + + n21
    ys = n11 + r
    ye = n11 + r + n21
    CS[xs:xe, ys:ye] = -Id[:n21, :n21] if swap_sign else Id[:n21, :n21]

    CS[p:p + n22, q:q + n22] = Id[:n22, :n22]
    CS[n11:n11 + r, n11:n11 + r] = c
    CS[p + n22:p + n22 + r, n11 + r + n21 + n22:2 * r + n11 + n21 + n22] = c

    xs = n11
    xe = n11 + r
    ys = n11 + n21 + n22 + r
    ye = n11 + n21 + n22 + 2 * r
    CS[xs:xe, ys:ye] = s if swap_sign else -s

    CS[p + n22:p + n22 + r, n11:n11 + r] = -s if swap_sign else s

    return U, CS, VDH
<|endoftext|>"""LU decomposition functions."""

from warnings import warn

from numpy import asarray, asarray_chkfinite
import numpy as np
from itertools import product

# Local imports
from ._misc import _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs
from ._decomp_lu_cython import lu_dispatcher

lapack_cast_dict = {x: ''.join([y for y in 'fdFD' if np.can_cast(x, y)])
                    for x in np.typecodes['All']}

__all__ = ['lu', 'lu_solve', 'lu_factor']


def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a matrix.

    The decomposition is::

        A = P L U

    where P is a permutation matrix, L lower triangular with unit
    diagonal elements, and U upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose
    overwrite_a : bool, optional
        Whether to overwrite data in A (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : (M, N) ndarray
        Matrix containing U in its upper triangle, and L in its lower triangle.
        The unit diagonal elements of L are not stored.
    piv : (K,) ndarray
        Pivot indices representing the permutation matrix P:
        row i of matrix was interchanged with row piv[i].
        Of shape ``(K,)``, with ``K = min(M, N)``.

    See Also
    --------
    lu : gives lu factorization in more user-friendly format
    lu_solve : solve an equation system using the LU factorization of a matrix

    Notes
    -----
    This is a wrapper to the ``*GETRF`` routines from LAPACK. Unlike
    :func:`lu`, it outputs the L and U factors into a single array
    and returns pivot indices instead of a permutation matrix.

    While the underlying ``*GETRF`` routines return 1-based pivot indices, the
    ``piv`` array returned by ``lu_factor`` contains 0-based indices.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lu_factor
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> lu, piv = lu_factor(A)
    >>> piv
    array([2, 2, 3, 3], dtype=int32)

    Convert LAPACK's ``piv`` array to NumPy index and test the permutation

    >>> def pivot_to_permutation(piv):
    ...     perm = np.arange(len(piv))
    ...     for i in range(len(piv)):
    ...         perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
    ...     return perm
    ...
    >>> p_inv = pivot_to_permutation(piv)
    >>> p_inv
    array([2, 0, 3, 1])
    >>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
    >>> np.allclose(A[p_inv] - L @ U, np.zeros((4, 4)))
    True

    The P matrix in P L U is defined by the inverse permutation and
    can be recovered using argsort:

    >>> p = np.argsort(p_inv)
    >>> p
    array([1, 3, 0, 2])
    >>> np.allclose(A - L[p] @ U, np.zeros((4, 4)))
    True

    or alternatively:

    >>> P = np.eye(4)[p]
    >>> np.allclose(A - P @ L @ U, np.zeros((4, 4)))
    True
    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)

    # accommodate empty arrays
    if a1.size == 0:
        lu = np.empty_like(a1)
        piv = np.arange(0, dtype=np.int32)
        return lu, piv

    overwrite_a = overwrite_a or (_datacopied(a1, a))

    getrf, = get_lapack_funcs(('getrf',), (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('illegal value in %dth argument of '
                         'internal getrf (lu_factor)' % -info)
    if info > 0:
        warn("Diagonal number %d is exactly zero. Singular matrix." % info,
             LinAlgWarning, stacklevel=2)
    return lu, piv


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, a x = b, given the LU factorization of a

    Parameters
    ----------
    (lu, piv)
        Factorization of the coefficient matrix a, as given by lu_factor.
        In particular piv are 0-indexed pivot indices.
    b : array
        Right-hand side
    trans : {0, 1, 2}, optional
        Type of system to solve:

        =====  =========
        trans  system
        =====  =========
        0      a x   = b
        1      a^T x = b
        2      a^H x = b
        =====  =========
    overwrite_b : bool, optional
        Whether to overwrite data in b (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        Solution to the system

    See Also
    --------
    lu_factor : LU factorize a matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import lu_factor, lu_solve
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> b = np.array([1, 1, 1, 1])
    >>> lu, piv = lu_factor(A)
    >>> x = lu_solve((lu, piv), b)
    >>> np.allclose(A @ x - b, np.zeros((4,)))
    True

    """
    (lu, piv) = lu_and_piv
    if check_finite:
        b1 = asarray_chkfinite(b)
    else:
        b1 = asarray(b)

    overwrite_b = overwrite_b or _datacopied(b1, b)

    if lu.shape[0] != b1.shape[0]:
        raise ValueError(f"Shapes of lu {lu.shape} and b {b1.shape} are incompatible")

    # accommodate empty arrays
    if b1.size == 0:
        m = lu_solve((np.eye(2, dtype=lu.dtype), [0, 1]), np.ones(2, dtype=b.dtype))
        return np.empty_like(b1, dtype=m.dtype)

    getrs, = get_lapack_funcs(('getrs',), (lu, b1))
    x, info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
    if info == 0:
        return x
    raise ValueError('illegal value in %dth argument of internal gesv|posv'
                     % -info)


def lu(a, permute_l=False, overwrite_a=False, check_finite=True,
       p_indices=False):
    """
    Compute LU decomposition of a matrix with partial pivoting.

    The decomposition satisfies::

        A = P @ L @ U

    where ``P`` is a permutation matrix, ``L`` lower triangular with unit
    diagonal elements, and ``U`` upper triangular. If `permute_l` is set to
    ``True`` then ``L`` is returned already permuted and hence satisfying
    ``A = L @ U``.

    Parameters
    ----------
    a : (M, N) array_like
        Array to decompose
    permute_l : bool, optional
        Perform the multiplication P*L (Default: do not permute)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    p_indices : bool, optional
        If ``True`` the permutation information is returned as row indices.
        The default is ``False`` for backwards-compatibility reasons.

    Returns
    -------
    **(If `permute_l` is ``False``)**

    p : (..., M, M) ndarray
        Permutation arrays or vectors depending on `p_indices`
    l : (..., M, K) ndarray
        Lower triangular or trapezoidal array with unit diagonal.
        ``K = min(M, N)``
    u : (..., K, N) ndarray
        Upper triangular or trapezoidal array

    **(If `permute_l` is ``True``)**

    pl : (..., M, K) ndarray
        Permuted L matrix.
        ``K = min(M, N)``
    u : (..., K, N) ndarray
        Upper triangular or trapezoidal array

    Notes
    -----
    Permutation matrices are costly since they are nothing but row reorder of
    ``L`` and hence indices are strongly recommended to be used instead if the
    permutation is required. The relation in the 2D case then becomes simply
    ``A = L[P, :] @ U``. In higher dimensions, it is better to use `permute_l`
    to avoid complicated indexing tricks.

    In 2D case, if one has the indices however, for some reason, the
    permutation matrix is still needed then it can be constructed by
    ``np.eye(M)[P, :]``.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.linalg import lu
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> p, l, u = lu(A)
    >>> np.allclose(A, p @ l @ u)
    True
    >>> p  # Permutation matrix
    array([[0., 1., 0., 0.],  # Row index 1
           [0., 0., 0., 1.],  # Row index 3
           [1., 0., 0., 0.],  # Row index 0
           [0., 0., 1., 0.]]) # Row index 2
    >>> p, _, _ = lu(A, p_indices=True)
    >>> p
    array([1, 3, 0, 2], dtype=int32)  # as given by row indices above
    >>> np.allclose(A, l[p, :] @ u)
    True

    We can also use nd-arrays, for example, a demonstration with 4D array:

    >>> rng = np.random.default_rng()
    >>> A = rng.uniform(low=-4, high=4, size=[3, 2, 4, 8])
    >>> p, l, u = lu(A)
    >>> p.shape, l.shape, u.shape
    ((3, 2, 4, 4), (3, 2, 4, 4), (3, 2, 4, 8))
    >>> np.allclose(A, p @ l @ u)
    True
    >>> PL, U = lu(A, permute_l=True)
    >>> np.allclose(A, PL @ U)
    True

    """
    a1 = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    if a1.ndim < 2:
        raise ValueError('The input array must be at least two-dimensional.')

    # Also check if dtype is LAPACK compatible
    if a1.dtype.char not in 'fdFD':
        dtype_char = lapack_cast_dict[a1.dtype.char]
        if not dtype_char:  # No casting possible
            raise TypeError(f'The dtype {a1.dtype} cannot be cast '
                            'to float(32, 64) or complex(64, 128).')

        a1 = a1.astype(dtype_char[0])  # makes a copy, free to scratch
        overwrite_a = True

    *nd, m, n = a1.shape
    k = min(m, n)
    real_dchar = 'f' if a1.dtype.char in 'fF' else 'd'

    # Empty input
    if min(*a1.shape) == 0:
        if permute_l:
            PL = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return PL, U
        else:
            P = (np.empty([*nd, 0], dtype=np.int32) if p_indices else
                 np.empty([*nd, 0, 0], dtype=real_dchar))
            L = np.empty(shape=[*nd, m, k], dtype=a1.dtype)
            U = np.empty(shape=[*nd, k, n], dtype=a1.dtype)
            return P, L, U

    # Scalar case
    if a1.shape[-2:] == (1, 1):
        if permute_l:
            return np.ones_like(a1), (a1 if overwrite_a else a1.copy())
        else:
            P = (np.zeros(shape=[*nd, m], dtype=int) if p_indices
                 else np.ones_like(a1))
            return P, np.ones_like(a1), (a1 if overwrite_a else a1.copy())

    # Then check overwrite permission
    if not _datacopied(a1, a):  # "a"  still alive through "a1"
        if not overwrite_a:
            # Data belongs to "a" so make a copy
            a1 = a1.copy(order='C')
        #  else: Do nothing we'll use "a" if possible
    # else:  a1 has its own data thus free to scratch

    # Then layout checks, might happen that overwrite is allowed but original
    # array was read-only or non-contiguous.

    if not (a1.flags['C_CONTIGUOUS'] and a1.flags['WRITEABLE']):
        a1 = a1.copy(order='C')

    if not nd:  # 2D array

        p = np.empty(m, dtype=np.int32)
        u = np.zeros([k, k], dtype=a1.dtype)
        lu_dispatcher(a1, u, p, permute_l)
        P, L, U = (p, a1, u) if m > n else (p, u, a1)

    else:  # Stacked array

        # Prepare the contiguous data holders
        P = np.empty([*nd, m], dtype=np.int32)  # perm vecs

        if m > n:  # Tall arrays, U will be created
            U = np.zeros([*nd, k, k], dtype=a1.dtype)
            for ind in product(*[range(x) for x in a1.shape[:-2]]):
                lu_dispatcher(a1[ind], U[ind], P[ind], permute_l)
            L = a1

        else:  # Fat arrays, L will be created
            L = np.zeros([*nd, k, k], dtype=a1.dtype)
            for ind in product(*[range(x) for x in a1.shape[:-2]]):
                lu_dispatcher(a1[ind], L[ind], P[ind], permute_l)
            U = a1

    # Convert permutation vecs to permutation arrays
    # permute_l=False needed to enter here to avoid wasted efforts
    if (not p_indices) and (not permute_l):
        if nd:
            Pa = np.zeros([*nd, m, m], dtype=real_dchar)
            # An unreadable index hack - One-hot encoding for perm matrices
            nd_ix = np.ix_(*([np.arange(x) for x in nd]+[np.arange(m)]))
            Pa[(*nd_ix, P)] = 1
            P = Pa
        else:  # 2D case
            Pa = np.zeros([m, m], dtype=real_dchar)
            Pa[np.arange(m), P] = 1
            P = Pa

    return (L, U) if permute_l else (P, L, U)
<|endoftext|>import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided


__all__ = ['toeplitz', 'circulant', 'hankel',
           'hadamard', 'leslie', 'kron', 'block_diag', 'companion',
           'helmert', 'hilbert', 'invhilbert', 'pascal', 'invpascal', 'dft',
           'fiedler', 'fiedler_companion', 'convolution_matrix']


# -----------------------------------------------------------------------------
#  matrix construction functions
# -----------------------------------------------------------------------------


def toeplitz(c, r=None):
    r"""
    Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c : array_like
        First column of the matrix.
    r : array_like, optional
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.

        .. warning::

            Beginning in SciPy 1.17, multidimensional input will be treated as a batch,
            not ``ravel``\ ed. To preserve the existing behavior, ``ravel`` arguments
            before passing them to `toeplitz`.

    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    See Also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    solve_toeplitz : Solve a Toeplitz system.

    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0. The behavior in previous
    versions was undocumented and is no longer supported.

    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])

    """
    c = np.asarray(c)
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r)

    if c.ndim > 1 or r.ndim > 1:
        msg = ("Beginning in SciPy 1.17, multidimensional input will be treated as a "
               "batch, not `ravel`ed. To preserve the existing behavior and silence "
               "this warning, `ravel` arguments before passing them to `toeplitz`.")
        warnings.warn(msg, FutureWarning, stacklevel=2)

    c, r = c.ravel(), r.ravel()
    # Form a 1-D array containing a reversed c followed by r[1:] that could be
    # strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n)).copy()


def circulant(c):
    """
    Construct a circulant matrix.

    Parameters
    ----------
    c : (..., N,)  array_like
        The first column(s) of the matrix. Multidimensional arrays are treated as a
        batch: each slice along the last axis is the first column of an output matrix.

    Returns
    -------
    A : (..., N, N) ndarray
        A circulant matrix whose first column is given by `c`.  For batch input, each
        slice of shape ``(N, N)`` along the last two dimensions of the output
        corresponds with a slice of shape ``(N,)`` along the last dimension of the
        input.


    See Also
    --------
    toeplitz : Toeplitz matrix
    hankel : Hankel matrix
    solve_circulant : Solve a circulant system.

    Notes
    -----
    .. versionadded:: 0.8.0

    Examples
    --------
    >>> from scipy.linalg import circulant
    >>> circulant([1, 2, 3])
    array([[1, 3, 2],
           [2, 1, 3],
           [3, 2, 1]])

    >>> circulant([[1, 2, 3], [4, 5, 6]])
    array([[[1, 3, 2],
            [2, 1, 3],
            [3, 2, 1]],
           [[4, 6, 5],
            [5, 4, 6],
            [6, 5, 4]]])
    """
    c = np.atleast_1d(c)
    batch_shape, N = c.shape[:-1], c.shape[-1]
    # Need to use `prod(batch_shape)` instead of `-1` in case array has zero size
    c = c.reshape(math.prod(batch_shape), N) if batch_shape else c
    # Form an extended array that could be strided to give circulant version
    c_ext = np.concatenate((c[..., ::-1], c[..., :0:-1]), axis=-1).ravel()
    L = c.shape[-1]
    n = c_ext.strides[-1]
    if c.ndim == 1:
        A = as_strided(c_ext[L-1:], shape=(L, L), strides=(-n, n))
    else:
        m = c.shape[0]
        A = as_strided(c_ext[L-1:], shape=(m, L, L), strides=(n*(2*L-1), -n, n))
    return A.reshape(batch_shape + (N, N)).copy()


def hankel(c, r=None):
    """
    Construct a Hankel matrix.

    The Hankel matrix has constant anti-diagonals, with `c` as its
    first column and `r` as its last row. If the first element of `r`
    differs from the last element of `c`, the first element of `r` is
    replaced by the last element of `c` to ensure that anti-diagonals
    remain constant. If `r` is not given, then `r = zeros_like(c)` is
    assumed.

    Parameters
    ----------
    c : array_like
        First column of the matrix. Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like, optional
        Last row of the matrix. If None, ``r = zeros_like(c)`` is assumed.
        r[0] is ignored; the last row of the returned matrix is
        ``[c[-1], r[1:]]``. Whatever the actual shape of `r`, it will be
        converted to a 1-D array.

    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Hankel matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    See Also
    --------
    toeplitz : Toeplitz matrix
    circulant : circulant matrix

    Examples
    --------
    >>> from scipy.linalg import hankel
    >>> hankel([1, 17, 99])
    array([[ 1, 17, 99],
           [17, 99,  0],
           [99,  0,  0]])
    >>> hankel([1,2,3,4], [4,7,7,8,9])
    array([[1, 2, 3, 4, 7],
           [2, 3, 4, 7, 7],
           [3, 4, 7, 7, 8],
           [4, 7, 7, 8, 9]])

    """
    c = np.asarray(c).ravel()
    if r is None:
        r = np.zeros_like(c)
    else:
        r = np.asarray(r).ravel()
    # Form a 1-D array of values to be used in the matrix, containing `c`
    # followed by r[1:].
    vals = np.concatenate((c, r[1:]))
    # Stride on concatenated array to get hankel matrix
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return as_strided(vals, shape=out_shp, strides=(n, n)).copy()


def hadamard(n, dtype=int):
    """
    Construct an Hadamard matrix.

    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction. `n` must be a power of 2.

    Parameters
    ----------
    n : int
        The order of the matrix. `n` must be a power of 2.
    dtype : dtype, optional
        The data type of the array to be constructed.

    Returns
    -------
    H : (n, n) ndarray
        The Hadamard matrix.

    Notes
    -----
    .. versionadded:: 0.8.0

    Examples
    --------
    >>> from scipy.linalg import hadamard
    >>> hadamard(2, dtype=complex)
    array([[ 1.+0.j,  1.+0.j],
           [ 1.+0.j, -1.-0.j]])
    >>> hadamard(4)
    array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]])

    """

    # This function is a slightly modified version of the
    # function contributed by Ivo in ticket #675.

    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be "
                         "a power of 2")

    H = np.array([[1]], dtype=dtype)

    # Sylvester's construction
    for i in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H


def leslie(f, s):
    """
    Create a Leslie matrix.

    Given the length n array of fecundity coefficients `f` and the length
    n-1 array of survival coefficients `s`, return the associated Leslie
    matrix.

    Parameters
    ----------
    f : (..., N,) array_like
        The "fecundity" coefficients.
    s : (..., N-1,) array_like
        The "survival" coefficients. The length of each slice of `s` (along the last
        axis) must be one less than the length of `f`, and it must be at least 1.

    Returns
    -------
    L : (..., N, N) ndarray
        The array is zero except for the first row,
        which is `f`, and the first sub-diagonal, which is `s`.
        For 1-D input, the data-type of the array will be the data-type of
        ``f[0]+s[0]``.

    Notes
    -----
    .. versionadded:: 0.8.0

    The Leslie matrix is used to model discrete-time, age-structured
    population growth [1]_ [2]_. In a population with `n` age classes, two sets
    of parameters define a Leslie matrix: the `n` "fecundity coefficients",
    which give the number of offspring per-capita produced by each age
    class, and the `n` - 1 "survival coefficients", which give the
    per-capita survival rate of each age class.

    N-dimensional input are treated as a batches of coefficient arrays: each
    slice along the last axis of the input arrays is a 1-D coefficient array,
    and each slice along the last two dimensions of the output is the
    corresponding Leslie matrix.

    References
    ----------
    .. [1] P. H. Leslie, On the use of matrices in certain population
           mathematics, Biometrika, Vol. 33, No. 3, 183--212 (Nov. 1945)
    .. [2] P. H. Leslie, Some further notes on the use of matrices in
           population mathematics, Biometrika, Vol. 35, No. 3/4, 213--245
           (Dec. 1948)

    Examples
    --------
    >>> from scipy.linalg import leslie
    >>> leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
    array([[ 0.1,  2. ,  1. ,  0.1],
           [ 0.2,  0. ,  0. ,  0. ],
           [ 0. ,  0.8,  0. ,  0. ],
           [ 0. ,  0. ,  0.7,  0. ]])

    """
    f = np.atleast_1d(f)
    s = np.atleast_1d(s)

    if f.shape[-1] != s.shape[-1] + 1:
        raise ValueError("Incorrect lengths for f and s. The length of s along "
                         "the last axis must be one less than the length of f.")
    if s.shape[-1] == 0:
        raise ValueError("The length of s must be at least 1.")

    n = f.shape[-1]

    if f.ndim > 1 or s.ndim > 1:
        from scipy.stats._resampling import _vectorize_statistic
        _leslie_nd = _vectorize_statistic(leslie)
        return np.moveaxis(_leslie_nd(f, s, axis=-1), [0, 1], [-2, -1])

    tmp = f[0] + s[0]
    a = np.zeros((n, n), dtype=tmp.dtype)
    a[0] = f
    a[list(range(1, n)), list(range(0, n - 1))] = s
    return a


def kron(a, b):
    """
    Kronecker product.

    .. deprecated:: 1.15.0
        `kron` has been deprecated in favour of `numpy.kron` and will be
        removed in SciPy 1.17.0.

    The result is the block matrix::

        a[0,0]*b    a[0,1]*b  ... a[0,-1]*b
        a[1,0]*b    a[1,1]*b  ... a[1,-1]*b
        ...
        a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b

    Parameters
    ----------
    a : (M, N) ndarray
        Input array
    b : (P, Q) ndarray
        Input array

    Returns
    -------
    A : (M*P, N*Q) ndarray
        Kronecker product of `a` and `b`.

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.linalg import kron
    >>> kron(array([[1,2],[3,4]]), array([[1,1,1]]))
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])

    """
    msg = ("`kron` has been deprecated in favour of `numpy.kron` in SciPy"
           " 1.15.0 and will be removed in SciPy 1.17.0.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # accommodate empty arrays
    if a.size == 0 or b.size == 0:
        m = a.shape[0] * b.shape[0]
        n = a.shape[1] * b.shape[1]
        return np.empty_like(a, shape=(m, n))

    if not a.flags['CONTIGUOUS']:
        a = np.reshape(a, a.shape)
    if not b.flags['CONTIGUOUS']:
        b = np.reshape(b, b.shape)
    o = np.outer(a, b)
    o = o.reshape(a.shape + b.shape)
    return np.concatenate(np.concatenate(o, axis=1), axis=1)


def block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Parameters
    ----------
    A, B, C, ... : array_like, up to 2-D
        Input arrays.  A 1-D array or array_like sequence of length `n` is
        treated as a 2-D array with shape ``(1,n)``.

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal. `D` has the
        same dtype as `A`.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Empty sequences (i.e., array-likes of zero size) will not be ignored.
    Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import block_diag
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> P = np.zeros((2, 0), dtype='int32')
    >>> block_diag(A, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(A, P, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions "
                         f"have dimension greater than 2: {bad_args}")

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.result_type(*[arr.dtype for arr in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def companion(a):
    """
    Create a companion matrix.

    Create the companion matrix [1]_ associated with the polynomial whose
    coefficients are given in `a`.

    Parameters
    ----------
    a : (..., N) array_like
        1-D array of polynomial coefficients. The length of `a` must be
        at least two, and ``a[0]`` must not be zero.
        M-dimensional arrays are treated as a batch: each slice along the last
        axis is a 1-D array of polynomial coefficients.

    Returns
    -------
    c : (..., N-1, N-1) ndarray
        For 1-D input, the first row of `c` is ``-a[1:]/a[0]``, and the first
        sub-diagonal is all ones.  The data-type of the array is the same
        as the data-type of ``1.0*a[0]``.
        For batch input, each slice of shape ``(N-1, N-1)`` along the last two
        dimensions of the output corresponds with a slice of shape ``(N,)``
        along the last dimension of the input.

    Raises
    ------
    ValueError
        If any of the following are true: a) ``a.shape[-1] < 2``; b) ``a[..., 0] == 0``.

    Notes
    -----
    .. versionadded:: 0.8.0

    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    Examples
    --------
    >>> from scipy.linalg import companion
    >>> companion([1, -10, 31, -30])
    array([[ 10., -31.,  30.],
           [  1.,   0.,   0.],
           [  0.,   1.,   0.]])

    """
    a = np.atleast_1d(a)
    n = a.shape[-1]

    if n < 2:
        raise ValueError("The length of `a` along the last axis must be at least 2.")

    if np.any(a[..., 0] == 0):
        raise ValueError("The first coefficient(s) of `a` (i.e. elements "
                         "of `a[..., 0]`) must not be zero.")

    first_row = -a[..., 1:] / (1.0 * a[..., 0:1])
    c = np.zeros(a.shape[:-1] + (n - 1, n - 1), dtype=first_row.dtype)
    c[..., 0, :] = first_row
    c[..., np.arange(1, n - 1), np.arange(0, n - 2)] = 1
    return c


def helmert(n, full=False):
    """
    Create an Helmert matrix of order `n`.

    This has applications in statistics, compositional or simplicial analysis,
    and in Aitchison geometry.

    Parameters
    ----------
    n : int
        The size of the array to create.
    full : bool, optional
        If True the (n, n) ndarray will be returned.
        Otherwise the submatrix that does not include the first
        row will be returned.
        Default: False.

    Returns
    -------
    M : ndarray
        The Helmert matrix.
        The shape is (n, n) or (n-1, n) depending on the `full` argument.

    Examples
    --------
    >>> from scipy.linalg import helmert
    >>> helmert(5, full=True)
    array([[ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
           [ 0.70710678, -0.70710678,  0.        ,  0.        ,  0.        ],
           [ 0.40824829,  0.40824829, -0.81649658,  0.        ,  0.        ],
           [ 0.28867513,  0.28867513,  0.28867513, -0.8660254 ,  0.        ],
           [ 0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 , -0.89442719]])

    """
    H = np.tril(np.ones((n, n)), -1) - np.diag(np.arange(n))
    d = np.arange(n) * np.arange(1, n+1)
    H[0] = 1
    d[0] = n
    H_full = H / np.sqrt(d)[:, np.newaxis]
    if full:
        return H_full
    else:
        return H_full[1:]


def hilbert(n):
    """
    Create a Hilbert matrix of order `n`.

    Returns the `n` by `n` array with entries `h[i,j] = 1 / (i + j + 1)`.

    Parameters
    ----------
    n : int
        The size of the array to create.

    Returns
    -------
    h : (n, n) ndarray
        The Hilbert matrix.

    See Also
    --------
    invhilbert : Compute the inverse of a Hilbert matrix.

    Notes
    -----
    .. versionadded:: 0.10.0

    Examples
    --------
    >>> from scipy.linalg import hilbert
    >>> hilbert(3)
    array([[ 1.        ,  0.5       ,  0.33333333],
           [ 0.5       ,  0.33333333,  0.25      ],
           [ 0.33333333,  0.25      ,  0.2       ]])

    """
    values = 1.0 / (1.0 + np.arange(2 * n - 1))
    h = hankel(values[:n], r=values[n - 1:])
    return h


def invhilbert(n, exact=False):
    """
    Compute the inverse of the Hilbert matrix of order `n`.

    The entries in the inverse of a Hilbert matrix are integers. When `n`
    is greater than 14, some entries in the inverse exceed the upper limit
    of 64 bit integers. The `exact` argument provides two options for
    dealing with these large integers.

    Parameters
    ----------
    n : int
        The order of the Hilbert matrix.
    exact : bool, optional
        If False, the data type of the array that is returned is np.float64,
        and the array is an approximation of the inverse.
        If True, the array is the exact integer inverse array. To represent
        the exact inverse when n > 14, the returned array is an object array
        of long integers. For n <= 14, the exact inverse is returned as an
        array with data type np.int64.

    Returns
    -------
    invh : (n, n) ndarray
        The data type of the array is np.float64 if `exact` is False.
        If `exact` is True, the data type is either np.int64 (for n <= 14)
        or object (for n > 14). In the latter case, the objects in the
        array will be long integers.

    See Also
    --------
    hilbert : Create a Hilbert matrix.

    Notes
    -----
    .. versionadded:: 0.10.0

    Examples
    --------
    >>> from scipy.linalg import invhilbert
    >>> invhilbert(4)
    array([[   16.,  -120.,   240.,  -140.],
           [ -120.,  1200., -2700.,  1680.],
           [  240., -2700.,  6480., -4200.],
           [ -140.,  1680., -4200.,  2800.]])
    >>> invhilbert(4, exact=True)
    array([[   16,  -120,   240,  -140],
           [ -120,  1200, -2700,  1680],
           [  240, -2700,  6480, -4200],
           [ -140,  1680, -4200,  2800]], dtype=int64)
    >>> invhilbert(16)[7,7]
    4.2475099528537506e+19
    >>> invhilbert(16, exact=True)[7,7]
    42475099528537378560

    """
    from scipy.special import comb
    if exact:
        if n > 14:
            dtype = object
        else:
            dtype = np.int64
    else:
        dtype = np.float64
    invh = np.empty((n, n), dtype=dtype)
    for i in range(n):
        for j in range(0, i + 1):
            s = i + j
            invh[i, j] = ((-1) ** s * (s + 1) *
                          comb(n + i, n - j - 1, exact=exact) *
                          comb(n + j, n - i - 1, exact=exact) *
                          comb(s, i, exact=exact) ** 2)
            if i != j:
                invh[j, i] = invh[i, j]
    return invh


def pascal(n, kind='symmetric', exact=True):
    """
    Returns the n x n Pascal matrix.

    The Pascal matrix is a matrix containing the binomial coefficients as
    its elements.

    Parameters
    ----------
    n : int
        The size of the matrix to create; that is, the result is an n x n
        matrix.
    kind : str, optional
        Must be one of 'symmetric', 'lower', or 'upper'.
        Default is 'symmetric'.
    exact : bool, optional
        If `exact` is True, the result is either an array of type
        numpy.uint64 (if n < 35) or an object array of Python long integers.
        If `exact` is False, the coefficients in the matrix are computed using
        `scipy.special.comb` with ``exact=False``. The result will be a floating
        point array, and the values in the array will not be the exact
        coefficients, but this version is much faster than ``exact=True``.

    Returns
    -------
    p : (n, n) ndarray
        The Pascal matrix.

    See Also
    --------
    invpascal

    Notes
    -----
    See https://en.wikipedia.org/wiki/Pascal_matrix for more information
    about Pascal matrices.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.linalg import pascal
    >>> pascal(4)
    array([[ 1,  1,  1,  1],
           [ 1,  2,  3,  4],
           [ 1,  3,  6, 10],
           [ 1,  4, 10, 20]], dtype=uint64)
    >>> pascal(4, kind='lower')
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 2, 1, 0],
           [1, 3, 3, 1]], dtype=uint64)
    >>> pascal(50)[-1, -1]
    25477612258980856902730428600
    >>> from scipy.special import comb
    >>> comb(98, 49, exact=True)
    25477612258980856902730428600

    """

    from scipy.special import comb
    if kind not in ['symmetric', 'lower', 'upper']:
        raise ValueError("kind must be 'symmetric', 'lower', or 'upper'")

    if exact:
        if n >= 35:
            L_n = np.empty((n, n), dtype=object)
            L_n.fill(0)
        else:
            L_n = np.zeros((n, n), dtype=np.uint64)
        for i in range(n):
            for j in range(i + 1):
                L_n[i, j] = comb(i, j, exact=True)
    else:
        L_n = comb(*np.ogrid[:n, :n])

    if kind == 'lower':
        p = L_n
    elif kind == 'upper':
        p = L_n.T
    else:
        p = np.dot(L_n, L_n.T)

    return p


def invpascal(n, kind='symmetric', exact=True):
    """
    Returns the inverse of the n x n Pascal matrix.

    The Pascal matrix is a matrix containing the binomial coefficients as
    its elements.

    Parameters
    ----------
    n : int
        The size of the matrix to create; that is, the result is an n x n
        matrix.
    kind : str, optional
        Must be one of 'symmetric', 'lower', or 'upper'.
        Default is 'symmetric'.
    exact : bool, optional
        If `exact` is True, the result is either an array of type
        ``numpy.int64`` (if `n` <= 35) or an object array of Python integers.
        If `exact` is False, the coefficients in the matrix are computed using
        `scipy.special.comb` with `exact=False`. The result will be a floating
        point array, and for large `n`, the values in the array will not be the
        exact coefficients.

    Returns
    -------
    invp : (n, n) ndarray
        The inverse of the Pascal matrix.

    See Also
    --------
    pascal

    Notes
    -----

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] "Pascal matrix", https://en.wikipedia.org/wiki/Pascal_matrix
    .. [2] Cohen, A. M., "The inverse of a Pascal matrix", Mathematical
           Gazette, 59(408), pp. 111-112, 1975.

    Examples
    --------
    >>> from scipy.linalg import invpascal, pascal
    >>> invp = invpascal(5)
    >>> invp
    array([[  5, -10,  10,  -5,   1],
           [-10,  30, -35,  19,  -4],
           [ 10, -35,  46, -27,   6],
           [ -5,  19, -27,  17,  -4],
           [  1,  -4,   6,  -4,   1]])

    >>> p = pascal(5)
    >>> p.dot(invp)
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.]])

    An example of the use of `kind` and `exact`:

    >>> invpascal(5, kind='lower', exact=False)
    array([[ 1., -0.,  0., -0.,  0.],
           [-1.,  1., -0.,  0., -0.],
           [ 1., -2.,  1., -0.,  0.],
           [-1.,  3., -3.,  1., -0.],
           [ 1., -4.,  6., -4.,  1.]])

    """
    from scipy.special import comb

    if kind not in ['symmetric', 'lower', 'upper']:
        raise ValueError("'kind' must be 'symmetric', 'lower' or 'upper'.")

    if kind == 'symmetric':
        if exact:
            if n > 34:
                dt = object
            else:
                dt = np.int64
        else:
            dt = np.float64
        invp = np.empty((n, n), dtype=dt)
        for i in range(n):
            for j in range(0, i + 1):
                v = 0
                for k in range(n - i):
                    v += comb(i + k, k, exact=exact) * comb(i + k, i + k - j,
                                                            exact=exact)
                invp[i, j] = (-1)**(i - j) * v
                if i != j:
                    invp[j, i] = invp[i, j]
    else:
        # For the 'lower' and 'upper' cases, we computer the inverse by
        # changing the sign of every other diagonal of the pascal matrix.
        invp = pascal(n, kind=kind, exact=exact)
        if invp.dtype == np.uint64:
            # This cast from np.uint64 to int64 OK, because if `kind` is not
            # "symmetric", the values in invp are all much less than 2**63.
            invp = invp.view(np.int64)

        # The toeplitz matrix has alternating bands of 1 and -1.
        invp *= toeplitz((-1)**np.arange(n)).astype(invp.dtype)

    return invp


def dft(n, scale=None):
    """
    Discrete Fourier transform matrix.

    Create the matrix that computes the discrete Fourier transform of a
    sequence [1]_. The nth primitive root of unity used to generate the
    matrix is exp(-2*pi*i/n), where i = sqrt(-1).

    Parameters
    ----------
    n : int
        Size the matrix to create.
    scale : str, optional
        Must be None, 'sqrtn', or 'n'.
        If `scale` is 'sqrtn', the matrix is divided by `sqrt(n)`.
        If `scale` is 'n', the matrix is divided by `n`.
        If `scale` is None (the default), the matrix is not normalized, and the
        return value is simply the Vandermonde matrix of the roots of unity.

    Returns
    -------
    m : (n, n) ndarray
        The DFT matrix.

    Notes
    -----
    When `scale` is None, multiplying a vector by the matrix returned by
    `dft` is mathematically equivalent to (but much less efficient than)
    the calculation performed by `scipy.fft.fft`.

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] "DFT matrix", https://en.wikipedia.org/wiki/DFT_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import dft
    >>> np.set_printoptions(precision=2, suppress=True)  # for compact output
    >>> m = dft(5)
    >>> m
    array([[ 1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ],
           [ 1.  +0.j  ,  0.31-0.95j, -0.81-0.59j, -0.81+0.59j,  0.31+0.95j],
           [ 1.  +0.j  , -0.81-0.59j,  0.31+0.95j,  0.31-0.95j, -0.81+0.59j],
           [ 1.  +0.j  , -0.81+0.59j,  0.31-0.95j,  0.31+0.95j, -0.81-0.59j],
           [ 1.  +0.j  ,  0.31+0.95j, -0.81+0.59j, -0.81-0.59j,  0.31-0.95j]])
    >>> x = np.array([1, 2, 3, 0, 3])
    >>> m @ x  # Compute the DFT of x
    array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])

    Verify that ``m @ x`` is the same as ``fft(x)``.

    >>> from scipy.fft import fft
    >>> fft(x)     # Same result as m @ x
    array([ 9.  +0.j  ,  0.12-0.81j, -2.12+3.44j, -2.12-3.44j,  0.12+0.81j])
    """
    if scale not in [None, 'sqrtn', 'n']:
        raise ValueError("scale must be None, 'sqrtn', or 'n'; "
                         f"{scale!r} is not valid.")

    omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(-1, 1)
    m = omegas ** np.arange(n)
    if scale == 'sqrtn':
        m /= math.sqrt(n)
    elif scale == 'n':
        m /= n
    return m


def fiedler(a):
    """Returns a symmetric Fiedler matrix

    Given an sequence of numbers `a`, Fiedler matrices have the structure
    ``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative
    entries. A Fiedler matrix has a dominant positive eigenvalue and other
    eigenvalues are negative. Although not valid generally, for certain inputs,
    the inverse and the determinant can be derived explicitly as given in [1]_.

    Parameters
    ----------
    a : (..., n,) array_like
        Coefficient array. N-dimensional arrays are treated as a batch:
        each slice along the last axis is a 1-D coefficient array.

    Returns
    -------
    F : (..., n, n) ndarray
        Fiedler matrix. For batch input, each slice of shape ``(n, n)``
        along the last two dimensions of the output corresponds with a
        slice of shape ``(n,)`` along the last dimension of the input.

    See Also
    --------
    circulant, toeplitz

    Notes
    -----

    .. versionadded:: 1.3.0

    References
    ----------
    .. [1] J. Todd, "Basic Numerical Mathematics: Vol.2 : Numerical Algebra",
        1977, Birkhauser, :doi:`10.1007/978-3-0348-7286-7`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import det, inv, fiedler
    >>> a = [1, 4, 12, 45, 77]
    >>> n = len(a)
    >>> A = fiedler(a)
    >>> A
    array([[ 0,  3, 11, 44, 76],
           [ 3,  0,  8, 41, 73],
           [11,  8,  0, 33, 65],
           [44, 41, 33,  0, 32],
           [76, 73, 65, 32,  0]])

    The explicit formulas for determinant and inverse seem to hold only for
    monotonically increasing/decreasing arrays. Note the tridiagonal structure
    and the corners.

    >>> Ai = inv(A)
    >>> Ai[np.abs(Ai) < 1e-12] = 0.  # cleanup the numerical noise for display
    >>> Ai
    array([[-0.16008772,  0.16666667,  0.        ,  0.        ,  0.00657895],
           [ 0.16666667, -0.22916667,  0.0625    ,  0.        ,  0.        ],
           [ 0.        ,  0.0625    , -0.07765152,  0.01515152,  0.        ],
           [ 0.        ,  0.        ,  0.01515152, -0.03077652,  0.015625  ],
           [ 0.00657895,  0.        ,  0.        ,  0.015625  , -0.00904605]])
    >>> det(A)
    15409151.999999998
    >>> (-1)**(n-1) * 2**(n-2) * np.diff(a).prod() * (a[-1] - a[0])
    15409152

    """
    a = np.atleast_1d(a)

    if a.ndim > 1:
        return np.apply_along_axis(fiedler, -1, a)

    if a.size == 0:
        return np.array([], dtype=float)
    elif a.size == 1:
        return np.array([[0.]])
    else:
        return np.abs(a[:, None] - a)


def fiedler_companion(a):
    """ Returns a Fiedler companion matrix

    Given a polynomial coefficient array ``a``, this function forms a
    pentadiagonal matrix with a special structure whose eigenvalues coincides
    with the roots of ``a``.

    Parameters
    ----------
    a : (..., N) array_like
        1-D array of polynomial coefficients in descending order with a nonzero
        leading coefficient. For ``N < 2``, an empty array is returned.
        N-dimensional arrays are treated as a batch: each slice along the last
        axis is a 1-D array of polynomial coefficients.

    Returns
    -------
    c : (..., N-1, N-1) ndarray
        Resulting companion matrix. For batch input, each slice of shape
        ``(N-1, N-1)`` along the last two dimensions of the output corresponds
        with a slice of shape ``(N,)`` along the last dimension of the input.

    See Also
    --------
    companion

    Notes
    -----
    Similar to `companion`, each leading coefficient along the last axis of the
    input should be nonzero.
    If the leading coefficient is not 1, other coefficients are rescaled before
    the array generation. To avoid numerical issues, it is best to provide a
    monic polynomial.

    .. versionadded:: 1.3.0

    References
    ----------
    .. [1] M. Fiedler, " A note on companion matrices", Linear Algebra and its
        Applications, 2003, :doi:`10.1016/S0024-3795(03)00548-2`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import fiedler_companion, eigvals
    >>> p = np.poly(np.arange(1, 9, 2))  # [1., -16., 86., -176., 105.]
    >>> fc = fiedler_companion(p)
    >>> fc
    array([[  16.,  -86.,    1.,    0.],
           [   1.,    0.,    0.,    0.],
           [   0.,  176.,    0., -105.],
           [   0.,    1.,    0.,    0.]])
    >>> eigvals(fc)
    array([7.+0.j, 5.+0.j, 3.+0.j, 1.+0.j])

    """
    a = np.atleast_1d(a)

    if a.ndim > 1:
        return np.apply_along_axis(fiedler_companion, -1, a)

    if a.size <= 2:
        if a.size == 2:
            return np.array([[-(a/a[0])[-1]]])
        return np.array([], dtype=a.dtype)

    if a[0] == 0.:
        raise ValueError('Leading coefficient is zero.')

    a = a/a[0]
    n = a.size - 1
    c = np.zeros((n, n), dtype=a.dtype)
    # subdiagonals
    c[range(3, n, 2), range(1, n-2, 2)] = 1.
    c[range(2, n, 2), range(1, n-1, 2)] = -a[3::2]
    # superdiagonals
    c[range(0, n-2, 2), range(2, n, 2)] = 1.
    c[range(0, n-1, 2), range(1, n, 2)] = -a[2::2]
    c[[0, 1], 0] = [-a[1], 1]

    return c


def convolution_matrix(a, n, mode='full'):
    """
    Construct a convolution matrix.

    Constructs the Toeplitz matrix representing one-dimensional
    convolution [1]_.  See the notes below for details.

    Parameters
    ----------
    a : (..., m) array_like
        The 1-D array to convolve. N-dimensional arrays are treated as a
        batch: each slice along the last axis is a 1-D array to convolve.
    n : int
        The number of columns in the resulting matrix.  It gives the length
        of the input to be convolved with `a`.  This is analogous to the
        length of `v` in ``numpy.convolve(a, v)``.
    mode : str
        This is analogous to `mode` in ``numpy.convolve(v, a, mode)``.
        It must be one of ('full', 'valid', 'same').
        See below for how `mode` determines the shape of the result.

    Returns
    -------
    A : (..., k, n) ndarray
        The convolution matrix whose row count `k` depends on `mode`::

            =======  =========================
             mode    k
            =======  =========================
            'full'   m + n -1
            'same'   max(m, n)
            'valid'  max(m, n) - min(m, n) + 1
            =======  =========================

        For batch input, each slice of shape ``(k, n)`` along the last two
        dimensions of the output corresponds with a slice of shape ``(m,)``
        along the last dimension of the input.

    See Also
    --------
    toeplitz : Toeplitz matrix

    Notes
    -----
    The code::

        A = convolution_matrix(a, n, mode)

    creates a Toeplitz matrix `A` such that ``A @ v`` is equivalent to
    using ``convolve(a, v, mode)``.  The returned array always has `n`
    columns.  The number of rows depends on the specified `mode`, as
    explained above.

    In the default 'full' mode, the entries of `A` are given by::

        A[i, j] == (a[i-j] if (0 <= (i-j) < m) else 0)

    where ``m = len(a)``.  Suppose, for example, the input array is
    ``[x, y, z]``.  The convolution matrix has the form::

        [x, 0, 0, ..., 0, 0]
        [y, x, 0, ..., 0, 0]
        [z, y, x, ..., 0, 0]
        ...
        [0, 0, 0, ..., x, 0]
        [0, 0, 0, ..., y, x]
        [0, 0, 0, ..., z, y]
        [0, 0, 0, ..., 0, z]

    In 'valid' mode, the entries of `A` are given by::

        A[i, j] == (a[i-j+m-1] if (0 <= (i-j+m-1) < m) else 0)

    This corresponds to a matrix whose rows are the subset of those from
    the 'full' case where all the coefficients in `a` are contained in the
    row.  For input ``[x, y, z]``, this array looks like::

        [z, y, x, 0, 0, ..., 0, 0, 0]
        [0, z, y, x, 0, ..., 0, 0, 0]
        [0, 0, z, y, x, ..., 0, 0, 0]
        ...
        [0, 0, 0, 0, 0, ..., x, 0, 0]
        [0, 0, 0, 0, 0, ..., y, x, 0]
        [0, 0, 0, 0, 0, ..., z, y, x]

    In the 'same' mode, the entries of `A` are given by::

        d = (m - 1) // 2
        A[i, j] == (a[i-j+d] if (0 <= (i-j+d) < m) else 0)

    The typical application of the 'same' mode is when one has a signal of
    length `n` (with `n` greater than ``len(a)``), and the desired output
    is a filtered signal that is still of length `n`.

    For input ``[x, y, z]``, this array looks like::

        [y, x, 0, 0, ..., 0, 0, 0]
        [z, y, x, 0, ..., 0, 0, 0]
        [0, z, y, x, ..., 0, 0, 0]
        [0, 0, z, y, ..., 0, 0, 0]
        ...
        [0, 0, 0, 0, ..., y, x, 0]
        [0, 0, 0, 0, ..., z, y, x]
        [0, 0, 0, 0, ..., 0, z, y]

    .. versionadded:: 1.5.0

    References
    ----------
    .. [1] "Convolution", https://en.wikipedia.org/wiki/Convolution

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import convolution_matrix
    >>> A = convolution_matrix([-1, 4, -2], 5, mode='same')
    >>> A
    array([[ 4, -1,  0,  0,  0],
           [-2,  4, -1,  0,  0],
           [ 0, -2,  4, -1,  0],
           [ 0,  0, -2,  4, -1],
           [ 0,  0,  0, -2,  4]])

    Compare multiplication by `A` with the use of `numpy.convolve`.

    >>> x = np.array([1, 2, 0, -3, 0.5])
    >>> A @ x
    array([  2. ,   6. ,  -1. , -12.5,   8. ])

    Verify that ``A @ x`` produced the same result as applying the
    convolution function.

    >>> np.convolve([-1, 4, -2], x, mode='same')
    array([  2. ,   6. ,  -1. , -12.5,   8. ])

    For comparison to the case ``mode='same'`` shown above, here are the
    matrices produced by ``mode='full'`` and ``mode='valid'`` for the
    same coefficients and size.

    >>> convolution_matrix([-1, 4, -2], 5, mode='full')
    array([[-1,  0,  0,  0,  0],
           [ 4, -1,  0,  0,  0],
           [-2,  4, -1,  0,  0],
           [ 0, -2,  4, -1,  0],
           [ 0,  0, -2,  4, -1],
           [ 0,  0,  0, -2,  4],
           [ 0,  0,  0,  0, -2]])

    >>> convolution_matrix([-1, 4, -2], 5, mode='valid')
    array([[-2,  4, -1,  0,  0],
           [ 0, -2,  4, -1,  0],
           [ 0,  0, -2,  4, -1]])
    """
    if n <= 0:
        raise ValueError('n must be a positive integer.')

    a = np.asarray(a)

    if a.size == 0:
        raise ValueError('len(a) must be at least 1.')

    if mode not in ('full', 'valid', 'same'):
        raise ValueError(
            "'mode' argument must be one of ('full', 'valid', 'same')")

    if a.ndim > 1:
        return np.apply_along_axis(lambda a: convolution_matrix(a, n, mode), -1, a)

    # create zero padded versions of the array
    az = np.pad(a, (0, n-1), 'constant')
    raz = np.pad(a[::-1], (0, n-1), 'constant')

    if mode == 'same':
        trim = min(n, len(a)) - 1
        tb = trim//2
        te = trim - tb
        col0 = az[tb:len(az)-te]
        row0 = raz[-n-tb:len(raz)-tb]
    elif mode == 'valid':
        tb = min(n, len(a)) - 1
        te = tb
        col0 = az[tb:len(az)-te]
        row0 = raz[-n-tb:len(raz)-tb]
    else:  # 'full'
        col0 = az
        row0 = raz[-n:]
    return toeplitz(col0, row0)
<|endoftext|># SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Taneli Hukkinen
# Licensed to PSF under a Contributor Agreement.

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
import re

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any, Final

    from ._types import ParseFloat

_TIME_RE_STR: Final = r"""
([01][0-9]|2[0-3])             # hours
:([0-5][0-9])                  # minutes
(?:
    :([0-5][0-9])              # optional seconds
    (?:\.([0-9]{1,6})[0-9]*)?  # optional fractions of a second
)?
"""

RE_NUMBER: Final = re.compile(
    r"""
0
(?:
    x[0-9A-Fa-f](?:_?[0-9A-Fa-f])*   # hex
    |
    b[01](?:_?[01])*                 # bin
    |
    o[0-7](?:_?[0-7])*               # oct
)
|
[+-]?(?:0|[1-9](?:_?[0-9])*)         # dec, integer part
(?P<floatpart>
    (?:\.[0-9](?:_?[0-9])*)?         # optional fractional part
    (?:[eE][+-]?[0-9](?:_?[0-9])*)?  # optional exponent part
)
""",
    flags=re.VERBOSE,
)
RE_LOCALTIME: Final = re.compile(_TIME_RE_STR, flags=re.VERBOSE)
RE_DATETIME: Final = re.compile(
    rf"""
([0-9]{{4}})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])  # date, e.g. 1988-10-27
(?:
    [Tt ]
    {_TIME_RE_STR}
    (?:([Zz])|([+-])([01][0-9]|2[0-3]):([0-5][0-9]))?  # optional time offset
)?
""",
    flags=re.VERBOSE,
)


def match_to_datetime(match: re.Match[str]) -> datetime | date:
    """Convert a `RE_DATETIME` match to `datetime.datetime` or `datetime.date`.

    Raises ValueError if the match does not correspond to a valid date
    or datetime.
    """
    (
        year_str,
        month_str,
        day_str,
        hour_str,
        minute_str,
        sec_str,
        micros_str,
        zulu_time,
        offset_sign_str,
        offset_hour_str,
        offset_minute_str,
    ) = match.groups()
    year, month, day = int(year_str), int(month_str), int(day_str)
    if hour_str is None:
        return date(year, month, day)
    hour, minute = int(hour_str), int(minute_str)
    sec = int(sec_str) if sec_str else 0
    micros = int(micros_str.ljust(6, "0")) if micros_str else 0
    if offset_sign_str:
        tz: tzinfo | None = cached_tz(
            offset_hour_str, offset_minute_str, offset_sign_str
        )
    elif zulu_time:
        tz = timezone.utc
    else:  # local date-time
        tz = None
    return datetime(year, month, day, hour, minute, sec, micros, tzinfo=tz)


# No need to limit cache size. This is only ever called on input
# that matched RE_DATETIME, so there is an implicit bound of
# 24 (hours) * 60 (minutes) * 2 (offset direction) = 2880.
@lru_cache(maxsize=None)
def cached_tz(hour_str: str, minute_str: str, sign_str: str) -> timezone:
    sign = 1 if sign_str == "+" else -1
    return timezone(
        timedelta(
            hours=sign * int(hour_str),
            minutes=sign * int(minute_str),
        )
    )


def match_to_localtime(match: re.Match[str]) -> time:
    hour_str, minute_str, sec_str, micros_str = match.groups()
    sec = int(sec_str) if sec_str else 0
    micros = int(micros_str.ljust(6, "0")) if micros_str else 0
    return time(int(hour_str), int(minute_str), sec, micros)


def match_to_number(match: re.Match[str], parse_float: ParseFloat) -> Any:
    if match.group("floatpart"):
        return parse_float(match.group())
