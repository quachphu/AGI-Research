        traceA = traceest(A, m3=5) if is_linear_operator else _trace(A)
    mu = traceA / float(n)

    # Get the linspace samples, attempting to preserve the linspace defaults.
    linspace_kwargs = {'retstep': True}
    if num is not None:
        linspace_kwargs['num'] = num
    if endpoint is not None:
        linspace_kwargs['endpoint'] = endpoint
    samples, step = np.linspace(start, stop, **linspace_kwargs)

    # Convert the linspace output to the notation used by the publication.
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError('at least two time points are required')
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]

    # Define the output ndarray.
    # Use an ndim=3 shape, such that the last two indices
    # are the ones that may be involved in level 3 BLAS operations.
    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    t = t_q - t_0
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    ell = 2
    norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    # Compute the expm action up to the initial time point.
    action_t0 = _expm_multiply_simple_core(A, B, t_0, mu, m_star, s)
    if scipy.sparse.issparse(action_t0):
        action_t0 = action_t0.toarray()
    elif is_pydata_spmatrix(action_t0):
        action_t0 = action_t0.todense()
    X[0] = action_t0

    # Compute the expm action at the rest of the time points.
    if q <= s:
        if status_only:
            return 0
        else:
            return _expm_multiply_interval_core_0(A, X,
                    h, mu, q, norm_info, tol, ell,n0)
    elif not (q % s):
        if status_only:
            return 1
        else:
            return _expm_multiply_interval_core_1(A, X,
                    h, mu, m_star, s, q, tol)
    elif (q % s):
        if status_only:
            return 2
        else:
            return _expm_multiply_interval_core_2(A, X,
                    h, mu, m_star, s, q, tol)
    else:
        raise Exception('internal error')


def _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0):
    """
    A helper function, for the case q <= s.
    """

    # Compute the new values of m_star and s which should be applied
    # over intervals of size t/q
    if norm_info.onenorm() == 0:
        m_star, s = 0, 1
    else:
        norm_info.set_scale(1./q)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
        norm_info.set_scale(1)

    for k in range(q):
        X[k+1] = _expm_multiply_simple_core(A, X[k], h, mu, m_star, s)
    return X, 0


def _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s == 0.
    """
    d = q // s
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(s):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        for k in range(1, d+1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star+1):
                if p > high_p:
                    K[p] = h * A.dot(K[p-1]) / float(p)
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu) * F
    return X, 1


def _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s > 0.
    """
    d = q // s
    j = q // d
    r = q - d * j
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    K = np.empty(K_shape, dtype=X.dtype)
    for i in range(j + 1):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        for k in range(1, effective_d+1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star+1):
                if p == high_p + 1:
                    K[p] = h * A.dot(K[p-1]) / float(p)
                    high_p = p
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu) * F
    return X, 2
<|endoftext|>import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array

__all__ = ['LaplacianNd']
# Sakurai and Mikota classes are intended for tests and benchmarks
# and explicitly not included in the public API of this module.


class LaplacianNd(LinearOperator):
    """
    The grid Laplacian in ``N`` dimensions and its eigenvalues/eigenvectors.

    Construct Laplacian on a uniform rectangular grid in `N` dimensions
    and output its eigenvalues and eigenvectors.
    The Laplacian ``L`` is square, negative definite, real symmetric array
    with signed integer entries and zeros otherwise.

    Parameters
    ----------
    grid_shape : tuple
        A tuple of integers of length ``N`` (corresponding to the dimension of
        the Lapacian), where each entry gives the size of that dimension. The
        Laplacian matrix is square of the size ``np.prod(grid_shape)``.
    boundary_conditions : {'neumann', 'dirichlet', 'periodic'}, optional
        The type of the boundary conditions on the boundaries of the grid.
        Valid values are ``'dirichlet'`` or ``'neumann'``(default) or
        ``'periodic'``.
    dtype : dtype
        Numerical type of the array. Default is ``np.int8``.

    Methods
    -------
    toarray()
        Construct a dense array from Laplacian data
    tosparse()
        Construct a sparse array from Laplacian data
    eigenvalues(m=None)
        Construct a 1D array of `m` largest (smallest in absolute value)
        eigenvalues of the Laplacian matrix in ascending order.
    eigenvectors(m=None):
        Construct the array with columns made of `m` eigenvectors (``float``)
        of the ``Nd`` Laplacian corresponding to the `m` ordered eigenvalues.

    .. versionadded:: 1.12.0

    Notes
    -----
    Compared to the MATLAB/Octave implementation [1] of 1-, 2-, and 3-D
    Laplacian, this code allows the arbitrary N-D case and the matrix-free
    callable option, but is currently limited to pure Dirichlet, Neumann or
    Periodic boundary conditions only.

    The Laplacian matrix of a graph (`scipy.sparse.csgraph.laplacian`) of a
    rectangular grid corresponds to the negative Laplacian with the Neumann
    conditions, i.e., ``boundary_conditions = 'neumann'``.

    All eigenvalues and eigenvectors of the discrete Laplacian operator for
    an ``N``-dimensional  regular grid of shape `grid_shape` with the grid
    step size ``h=1`` are analytically known [2].

    References
    ----------
    .. [1] https://github.com/lobpcg/blopex/blob/master/blopex_\
tools/matlab/laplacian/laplacian.m
    .. [2] "Eigenvalues and eigenvectors of the second derivative", Wikipedia
           https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors_\
of_the_second_derivative

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LaplacianNd
    >>> from scipy.sparse import diags, csgraph
    >>> from scipy.linalg import eigvalsh

    The one-dimensional Laplacian demonstrated below for pure Neumann boundary
    conditions on a regular grid with ``n=6`` grid points is exactly the
    negative graph Laplacian for the undirected linear graph with ``n``
    vertices using the sparse adjacency matrix ``G`` represented by the
    famous tri-diagonal matrix:

    >>> n = 6
    >>> G = diags(np.ones(n - 1), 1, format='csr')
    >>> Lf = csgraph.laplacian(G, symmetrized=True, form='function')
    >>> grid_shape = (n, )
    >>> lap = LaplacianNd(grid_shape, boundary_conditions='neumann')
    >>> np.array_equal(lap.matmat(np.eye(n)), -Lf(np.eye(n)))
    True

    Since all matrix entries of the Laplacian are integers, ``'int8'`` is
    the default dtype for storing matrix representations.

    >>> lap.tosparse()
    <DIAgonal sparse array of dtype 'int8'
        with 16 stored elements (3 diagonals) and shape (6, 6)>
    >>> lap.toarray()
    array([[-1,  1,  0,  0,  0,  0],
           [ 1, -2,  1,  0,  0,  0],
           [ 0,  1, -2,  1,  0,  0],
           [ 0,  0,  1, -2,  1,  0],
           [ 0,  0,  0,  1, -2,  1],
           [ 0,  0,  0,  0,  1, -1]], dtype=int8)
    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())
    True
    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())
    True

    Any number of extreme eigenvalues and/or eigenvectors can be computed.
    
    >>> lap = LaplacianNd(grid_shape, boundary_conditions='periodic')
    >>> lap.eigenvalues()
    array([-4., -3., -3., -1., -1.,  0.])
    >>> lap.eigenvalues()[-2:]
    array([-1.,  0.])
    >>> lap.eigenvalues(2)
    array([-1.,  0.])
    >>> lap.eigenvectors(1)
    array([[0.40824829],
           [0.40824829],
           [0.40824829],
           [0.40824829],
           [0.40824829],
           [0.40824829]])
    >>> lap.eigenvectors(2)
    array([[ 0.5       ,  0.40824829],
           [ 0.        ,  0.40824829],
           [-0.5       ,  0.40824829],
           [-0.5       ,  0.40824829],
           [ 0.        ,  0.40824829],
           [ 0.5       ,  0.40824829]])
    >>> lap.eigenvectors()
    array([[ 0.40824829,  0.28867513,  0.28867513,  0.5       ,  0.5       ,
             0.40824829],
           [-0.40824829, -0.57735027, -0.57735027,  0.        ,  0.        ,
             0.40824829],
           [ 0.40824829,  0.28867513,  0.28867513, -0.5       , -0.5       ,
             0.40824829],
           [-0.40824829,  0.28867513,  0.28867513, -0.5       , -0.5       ,
             0.40824829],
           [ 0.40824829, -0.57735027, -0.57735027,  0.        ,  0.        ,
             0.40824829],
           [-0.40824829,  0.28867513,  0.28867513,  0.5       ,  0.5       ,
             0.40824829]])

    The two-dimensional Laplacian is illustrated on a regular grid with
    ``grid_shape = (2, 3)`` points in each dimension.

    >>> grid_shape = (2, 3)
    >>> n = np.prod(grid_shape)

    Numeration of grid points is as follows:

    >>> np.arange(n).reshape(grid_shape + (-1,))
    array([[[0],
            [1],
            [2]],
    <BLANKLINE>
           [[3],
            [4],
            [5]]])

    Each of the boundary conditions ``'dirichlet'``, ``'periodic'``, and
    ``'neumann'`` is illustrated separately; with ``'dirichlet'``

    >>> lap = LaplacianNd(grid_shape, boundary_conditions='dirichlet')
    >>> lap.tosparse()
    <Compressed Sparse Row sparse array of dtype 'int8'
        with 20 stored elements and shape (6, 6)>
    >>> lap.toarray()
    array([[-4,  1,  0,  1,  0,  0],
           [ 1, -4,  1,  0,  1,  0],
           [ 0,  1, -4,  0,  0,  1],
           [ 1,  0,  0, -4,  1,  0],
           [ 0,  1,  0,  1, -4,  1],
           [ 0,  0,  1,  0,  1, -4]], dtype=int8)
    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())
    True
    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())
    True
    >>> lap.eigenvalues()
    array([-6.41421356, -5.        , -4.41421356, -3.58578644, -3.        ,
           -1.58578644])
    >>> eigvals = eigvalsh(lap.toarray().astype(np.float64))
    >>> np.allclose(lap.eigenvalues(), eigvals)
    True
    >>> np.allclose(lap.toarray() @ lap.eigenvectors(),
    ...             lap.eigenvectors() @ np.diag(lap.eigenvalues()))
    True

    with ``'periodic'``

    >>> lap = LaplacianNd(grid_shape, boundary_conditions='periodic')
    >>> lap.tosparse()
    <Compressed Sparse Row sparse array of dtype 'int8'
        with 24 stored elements and shape (6, 6)>
    >>> lap.toarray()
        array([[-4,  1,  1,  2,  0,  0],
               [ 1, -4,  1,  0,  2,  0],
               [ 1,  1, -4,  0,  0,  2],
               [ 2,  0,  0, -4,  1,  1],
               [ 0,  2,  0,  1, -4,  1],
               [ 0,  0,  2,  1,  1, -4]], dtype=int8)
    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())
    True
    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())
    True
    >>> lap.eigenvalues()
    array([-7., -7., -4., -3., -3.,  0.])
    >>> eigvals = eigvalsh(lap.toarray().astype(np.float64))
    >>> np.allclose(lap.eigenvalues(), eigvals)
    True
    >>> np.allclose(lap.toarray() @ lap.eigenvectors(),
    ...             lap.eigenvectors() @ np.diag(lap.eigenvalues()))
    True

    and with ``'neumann'``

    >>> lap = LaplacianNd(grid_shape, boundary_conditions='neumann')
    >>> lap.tosparse()
    <Compressed Sparse Row sparse array of dtype 'int8'
        with 20 stored elements and shape (6, 6)>
    >>> lap.toarray()
    array([[-2,  1,  0,  1,  0,  0],
           [ 1, -3,  1,  0,  1,  0],
           [ 0,  1, -2,  0,  0,  1],
           [ 1,  0,  0, -2,  1,  0],
           [ 0,  1,  0,  1, -3,  1],
           [ 0,  0,  1,  0,  1, -2]], dtype=int8)
    >>> np.array_equal(lap.matmat(np.eye(n)), lap.toarray())
    True
    >>> np.array_equal(lap.tosparse().toarray(), lap.toarray())
    True
    >>> lap.eigenvalues()
    array([-5., -3., -3., -2., -1.,  0.])
    >>> eigvals = eigvalsh(lap.toarray().astype(np.float64))
    >>> np.allclose(lap.eigenvalues(), eigvals)
    True
    >>> np.allclose(lap.toarray() @ lap.eigenvectors(),
    ...             lap.eigenvectors() @ np.diag(lap.eigenvalues()))
    True

    """

    def __init__(self, grid_shape, *,
                 boundary_conditions='neumann',
                 dtype=np.int8):

        if boundary_conditions not in ('dirichlet', 'neumann', 'periodic'):
            raise ValueError(
                f"Unknown value {boundary_conditions!r} is given for "
                "'boundary_conditions' parameter. The valid options are "
                "'dirichlet', 'periodic', and 'neumann' (default)."
            )

        self.grid_shape = grid_shape
        self.boundary_conditions = boundary_conditions
        # LaplacianNd folds all dimensions in `grid_shape` into a single one
        N = np.prod(grid_shape)
        super().__init__(dtype=dtype, shape=(N, N))

    def _eigenvalue_ordering(self, m):
        """Compute `m` largest eigenvalues in each of the ``N`` directions,
        i.e., up to ``m * N`` total, order them and return `m` largest.
        """
        grid_shape = self.grid_shape
        if m is None:
            indices = np.indices(grid_shape)
            Leig = np.zeros(grid_shape)
        else:
            grid_shape_min = min(grid_shape,
                                 tuple(np.ones_like(grid_shape) * m))
            indices = np.indices(grid_shape_min)
            Leig = np.zeros(grid_shape_min)

        for j, n in zip(indices, grid_shape):
            if self.boundary_conditions == 'dirichlet':
                Leig += -4 * np.sin(np.pi * (j + 1) / (2 * (n + 1))) ** 2
            elif self.boundary_conditions == 'neumann':
                Leig += -4 * np.sin(np.pi * j / (2 * n)) ** 2
            else:  # boundary_conditions == 'periodic'
                Leig += -4 * np.sin(np.pi * np.floor((j + 1) / 2) / n) ** 2

        Leig_ravel = Leig.ravel()
        ind = np.argsort(Leig_ravel)
        eigenvalues = Leig_ravel[ind]
        if m is not None:
            eigenvalues = eigenvalues[-m:]
            ind = ind[-m:]

        return eigenvalues, ind

    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : float array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        eigenvalues, _ = self._eigenvalue_ordering(m)
        return eigenvalues

    def _ev1d(self, j, n):
        """Return 1 eigenvector in 1d with index `j`
        and number of grid points `n` where ``j < n``. 
        """
        if self.boundary_conditions == 'dirichlet':
            i = np.pi * (np.arange(n) + 1) / (n + 1)
            ev = np.sqrt(2. / (n + 1.)) * np.sin(i * (j + 1))
        elif self.boundary_conditions == 'neumann':
            i = np.pi * (np.arange(n) + 0.5) / n
            ev = np.sqrt((1. if j == 0 else 2.) / n) * np.cos(i * j)
        else:  # boundary_conditions == 'periodic'
            if j == 0:
                ev = np.sqrt(1. / n) * np.ones(n)
            elif j + 1 == n and n % 2 == 0:
                ev = np.sqrt(1. / n) * np.tile([1, -1], n//2)
            else:
                i = 2. * np.pi * (np.arange(n) + 0.5) / n
                ev = np.sqrt(2. / n) * np.cos(i * np.floor((j + 1) / 2))
        # make small values exact zeros correcting round-off errors
        # due to symmetry of eigenvectors the exact 0. is correct 
        ev[np.abs(ev) < np.finfo(np.float64).eps] = 0.
        return ev

    def _one_eve(self, k):
        """Return 1 eigenvector in Nd with multi-index `j`
        as a tensor product of the corresponding 1d eigenvectors. 
        """
        phi = [self._ev1d(j, n) for j, n in zip(k, self.grid_shape)]
        result = phi[0]
        for phi in phi[1:]:
            result = np.tensordot(result, phi, axes=0)
        return np.asarray(result).ravel()

    def eigenvectors(self, m=None):
        """Return the requested number of eigenvectors for ordered eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of eigenvectors to return. If not provided,
            then all eigenvectors will be returned.
            
        Returns
        -------
        eigenvectors : float array
            An array with columns made of the requested `m` or all eigenvectors.
            The columns are ordered according to the `m` ordered eigenvalues. 
        """
        _, ind = self._eigenvalue_ordering(m)
        if m is None:
            grid_shape_min = self.grid_shape
        else:
            grid_shape_min = min(self.grid_shape,
                                tuple(np.ones_like(self.grid_shape) * m))

        N_indices = np.unravel_index(ind, grid_shape_min)
        N_indices = [tuple(x) for x in zip(*N_indices)]
        eigenvectors_list = [self._one_eve(k) for k in N_indices]
        return np.column_stack(eigenvectors_list)

    def toarray(self):
        """
        Converts the Laplacian data to a dense array.

        Returns
        -------
        L : ndarray
            The shape is ``(N, N)`` where ``N = np.prod(grid_shape)``.

        """
        grid_shape = self.grid_shape
        n = np.prod(grid_shape)
        L = np.zeros([n, n], dtype=np.int8)
        # Scratch arrays
        L_i = np.empty_like(L)
        Ltemp = np.empty_like(L)

        for ind, dim in enumerate(grid_shape):
            # Start zeroing out L_i
            L_i[:] = 0
            # Allocate the top left corner with the kernel of L_i
            # Einsum returns writable view of arrays
            np.einsum("ii->i", L_i[:dim, :dim])[:] = -2
            np.einsum("ii->i", L_i[: dim - 1, 1:dim])[:] = 1
            np.einsum("ii->i", L_i[1:dim, : dim - 1])[:] = 1

            if self.boundary_conditions == 'neumann':
                L_i[0, 0] = -1
                L_i[dim - 1, dim - 1] = -1
            elif self.boundary_conditions == 'periodic':
                if dim > 1:
                    L_i[0, dim - 1] += 1
                    L_i[dim - 1, 0] += 1
                else:
                    L_i[0, 0] += 1

            # kron is too slow for large matrices hence the next two tricks
            # 1- kron(eye, mat) is block_diag(mat, mat, ...)
            # 2- kron(mat, eye) can be performed by 4d stride trick

            # 1-
            new_dim = dim
            # for block_diag we tile the top left portion on the diagonal
            if ind > 0:
                tiles = np.prod(grid_shape[:ind])
                for j in range(1, tiles):
                    L_i[j*dim:(j+1)*dim, j*dim:(j+1)*dim] = L_i[:dim, :dim]
                    new_dim += dim
            # 2-
            # we need the keep L_i, but reset the array
            Ltemp[:new_dim, :new_dim] = L_i[:new_dim, :new_dim]
            tiles = int(np.prod(grid_shape[ind+1:]))
            # Zero out the top left, the rest is already 0
            L_i[:new_dim, :new_dim] = 0
            idx = [x for x in range(tiles)]
            L_i.reshape(
                (new_dim, tiles,
                 new_dim, tiles)
                )[:, idx, :, idx] = Ltemp[:new_dim, :new_dim]

            L += L_i

        return L.astype(self.dtype)

    def tosparse(self):
        """
        Constructs a sparse array from the Laplacian data. The returned sparse
        array format is dependent on the selected boundary conditions.

        Returns
        -------
        L : scipy.sparse.sparray
            The shape is ``(N, N)`` where ``N = np.prod(grid_shape)``.

        """
        N = len(self.grid_shape)
        p = np.prod(self.grid_shape)
        L = dia_array((p, p), dtype=np.int8)

        for i in range(N):
            dim = self.grid_shape[i]
            data = np.ones([3, dim], dtype=np.int8)
            data[1, :] *= -2

            if self.boundary_conditions == 'neumann':
                data[1, 0] = -1
                data[1, -1] = -1

            L_i = dia_array((data, [-1, 0, 1]), shape=(dim, dim),
                            dtype=np.int8
                            )

            if self.boundary_conditions == 'periodic':
                t = dia_array((dim, dim), dtype=np.int8)
                t.setdiag([1], k=-dim+1)
                t.setdiag([1], k=dim-1)
                L_i += t

            for j in range(i):
                L_i = kron(eye(self.grid_shape[j], dtype=np.int8), L_i)
            for j in range(i + 1, N):
                L_i = kron(L_i, eye(self.grid_shape[j], dtype=np.int8))
            L += L_i
        return L.astype(self.dtype)

    def _matvec(self, x):
        grid_shape = self.grid_shape
        N = len(grid_shape)
        X = x.reshape(grid_shape + (-1,))
        Y = -2 * N * X
        for i in range(N):
            Y += np.roll(X, 1, axis=i)
            Y += np.roll(X, -1, axis=i)
            if self.boundary_conditions in ('neumann', 'dirichlet'):
                Y[(slice(None),)*i + (0,) + (slice(None),)*(N-i-1)
                  ] -= np.roll(X, 1, axis=i)[
                    (slice(None),) * i + (0,) + (slice(None),) * (N-i-1)
                ]
                Y[
                    (slice(None),) * i + (-1,) + (slice(None),) * (N-i-1)
                ] -= np.roll(X, -1, axis=i)[
                    (slice(None),) * i + (-1,) + (slice(None),) * (N-i-1)
                ]

                if self.boundary_conditions == 'neumann':
                    Y[
                        (slice(None),) * i + (0,) + (slice(None),) * (N-i-1)
                    ] += np.roll(X, 0, axis=i)[
                        (slice(None),) * i + (0,) + (slice(None),) * (N-i-1)
                    ]
                    Y[
                        (slice(None),) * i + (-1,) + (slice(None),) * (N-i-1)
                    ] += np.roll(X, 0, axis=i)[
                        (slice(None),) * i + (-1,) + (slice(None),) * (N-i-1)
                    ]

        return Y.reshape(-1, X.shape[-1])

    def _matmat(self, x):
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self


class Sakurai(LinearOperator):
    """
    Construct a Sakurai matrix in various formats and its eigenvalues.

    Constructs the "Sakurai" matrix motivated by reference [1]_:
    square real symmetric positive definite and 5-diagonal
    with the main diagonal ``[5, 6, 6, ..., 6, 6, 5], the ``+1`` and ``-1``
    diagonals filled with ``-4``, and the ``+2`` and ``-2`` diagonals
    made of ``1``. Its eigenvalues are analytically known to be
    ``16. * np.power(np.cos(0.5 * k * np.pi / (n + 1)), 4)``.
    The matrix gets ill-conditioned with its size growing.
    It is useful for testing and benchmarking sparse eigenvalue solvers
    especially those taking advantage of its banded 5-diagonal structure.
    See the notes below for details.

    Parameters
    ----------
    n : int
        The size of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.int8``.

    Methods
    -------
    toarray()
        Construct a dense array from Laplacian data
    tosparse()
        Construct a sparse array from Laplacian data
    tobanded()
        The Sakurai matrix in the format for banded symmetric matrices,
        i.e., (3, n) ndarray with 3 upper diagonals
        placing the main diagonal at the bottom.
    eigenvalues
        All eigenvalues of the Sakurai matrix ordered ascending.

    Notes
    -----
    Reference [1]_ introduces a generalized eigenproblem for the matrix pair
    `A` and `B` where `A` is the identity so we turn it into an eigenproblem
    just for the matrix `B` that this function outputs in various formats
    together with its eigenvalues.
    
    .. versionadded:: 1.12.0

    References
    ----------
    .. [1] T. Sakurai, H. Tadano, Y. Inadomi, and U. Nagashima,
       "A moment-based method for large-scale generalized
       eigenvalue problems",
       Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg._special_sparse_arrays import Sakurai
    >>> from scipy.linalg import eig_banded
    >>> n = 6
    >>> sak = Sakurai(n)

    Since all matrix entries are small integers, ``'int8'`` is
    the default dtype for storing matrix representations.

    >>> sak.toarray()
    array([[ 5, -4,  1,  0,  0,  0],
           [-4,  6, -4,  1,  0,  0],
           [ 1, -4,  6, -4,  1,  0],
           [ 0,  1, -4,  6, -4,  1],
           [ 0,  0,  1, -4,  6, -4],
           [ 0,  0,  0,  1, -4,  5]], dtype=int8)
    >>> sak.tobanded()
    array([[ 1,  1,  1,  1,  1,  1],
           [-4, -4, -4, -4, -4, -4],
           [ 5,  6,  6,  6,  6,  5]], dtype=int8)
    >>> sak.tosparse()
    <DIAgonal sparse array of dtype 'int8'
        with 24 stored elements (5 diagonals) and shape (6, 6)>
    >>> np.array_equal(sak.dot(np.eye(n)), sak.tosparse().toarray())
    True
    >>> sak.eigenvalues()
    array([0.03922866, 0.56703972, 2.41789479, 5.97822974,
           10.54287655, 14.45473055])
    >>> sak.eigenvalues(2)
    array([0.03922866, 0.56703972])

    The banded form can be used in scipy functions for banded matrices, e.g.,

    >>> e = eig_banded(sak.tobanded(), eigvals_only=True)
    >>> np.allclose(sak.eigenvalues, e, atol= n * n * n * np.finfo(float).eps)
    True

    """
    def __init__(self, n, dtype=np.int8):
        self.n = n
        self.dtype = dtype
        shape = (n, n)
        super().__init__(dtype, shape)

    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.float64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        if m is None:
            m = self.n
        k = np.arange(self.n + 1 -m, self.n + 1)
        return np.flip(16. * np.power(np.cos(0.5 * k * np.pi / (self.n + 1)), 4))

    def tobanded(self):
        """
        Construct the Sakurai matrix as a banded array.
        """
        d0 = np.r_[5, 6 * np.ones(self.n - 2, dtype=self.dtype), 5]
        d1 = -4 * np.ones(self.n, dtype=self.dtype)
        d2 = np.ones(self.n, dtype=self.dtype)
        return np.array([d2, d1, d0]).astype(self.dtype)

    def tosparse(self):
        """
        Construct the Sakurai matrix is a sparse format.
        """
        from scipy.sparse import spdiags
        d = self.tobanded()
        # the banded format has the main diagonal at the bottom
        # `spdiags` has no `dtype` parameter so inherits dtype from banded
        return spdiags([d[0], d[1], d[2], d[1], d[0]], [-2, -1, 0, 1, 2],
                       self.n, self.n)

    def toarray(self):
        return self.tosparse().toarray()
    
    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Sakurai matrix without constructing or storing the matrix itself
        using the knowledge of its entries and the 5-diagonal format.
        """
        x = x.reshape(self.n, -1)
        result_dtype = np.promote_types(x.dtype, self.dtype)
        sx = np.zeros_like(x, dtype=result_dtype)
        sx[0, :] = 5 * x[0, :] - 4 * x[1, :] + x[2, :]
        sx[-1, :] = 5 * x[-1, :] - 4 * x[-2, :] + x[-3, :]
        sx[1: -1, :] = (6 * x[1: -1, :] - 4 * (x[:-2, :] + x[2:, :])
                      + np.pad(x[:-3, :], ((1, 0), (0, 0)))
                      + np.pad(x[3:, :], ((0, 1), (0, 0))))
        return sx

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Sakurai matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """        
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self


class MikotaM(LinearOperator):
    """
    Construct a mass matrix in various formats of Mikota pair.

    The mass matrix `M` is square real diagonal
    positive definite with entries that are reciprocal to integers.

    Parameters
    ----------
    shape : tuple of int
        The shape of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.float64``.

    Methods
    -------
    toarray()
        Construct a dense array from Mikota data
    tosparse()
        Construct a sparse array from Mikota data
    tobanded()
        The format for banded symmetric matrices,
        i.e., (1, n) ndarray with the main diagonal.
    """
    def __init__(self, shape, dtype=np.float64):
        self.shape = shape
        self.dtype = dtype
        super().__init__(dtype, shape)

    def _diag(self):
        # The matrix is constructed from its diagonal 1 / [1, ..., N+1];
        # compute in a function to avoid duplicated code & storage footprint
        return (1. / np.arange(1, self.shape[0] + 1)).astype(self.dtype)

    def tobanded(self):
        return self._diag()

    def tosparse(self):
        from scipy.sparse import diags
        return diags([self._diag()], [0], shape=self.shape, dtype=self.dtype)

    def toarray(self):
        return np.diag(self._diag()).astype(self.dtype)

    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Mikota mass matrix without constructing or storing the matrix itself
        using the knowledge of its entries and the diagonal format.
        """
        x = x.reshape(self.shape[0], -1)
        return self._diag()[:, np.newaxis] * x

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Mikota mass matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """     
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self


class MikotaK(LinearOperator):
    """
    Construct a stiffness matrix in various formats of Mikota pair.

    The stiffness matrix `K` is square real tri-diagonal symmetric
    positive definite with integer entries. 

    Parameters
    ----------
    shape : tuple of int
        The shape of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.int32``.

    Methods
    -------
    toarray()
        Construct a dense array from Mikota data
    tosparse()
        Construct a sparse array from Mikota data
    tobanded()
        The format for banded symmetric matrices,
        i.e., (2, n) ndarray with 2 upper diagonals
        placing the main diagonal at the bottom.
    """
    def __init__(self, shape, dtype=np.int32):
        self.shape = shape
        self.dtype = dtype
        super().__init__(dtype, shape)
        # The matrix is constructed from its diagonals;
        # we precompute these to avoid duplicating the computation
        n = shape[0]
        self._diag0 = np.arange(2 * n - 1, 0, -2, dtype=self.dtype)
        self._diag1 = - np.arange(n - 1, 0, -1, dtype=self.dtype)

    def tobanded(self):
        return np.array([np.pad(self._diag1, (1, 0), 'constant'), self._diag0])

    def tosparse(self):
        from scipy.sparse import diags
        return diags([self._diag1, self._diag0, self._diag1], [-1, 0, 1],
                     shape=self.shape, dtype=self.dtype)

    def toarray(self):
        return self.tosparse().toarray()

    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Mikota stiffness matrix without constructing or storing the matrix
        itself using the knowledge of its entries and the 3-diagonal format.
        """
        x = x.reshape(self.shape[0], -1)
        result_dtype = np.promote_types(x.dtype, self.dtype)
        kx = np.zeros_like(x, dtype=result_dtype)
        d1 = self._diag1
        d0 = self._diag0
        kx[0, :] = d0[0] * x[0, :] + d1[0] * x[1, :]
        kx[-1, :] = d1[-1] * x[-2, :] + d0[-1] * x[-1, :]
        kx[1: -1, :] = (d1[:-1, None] * x[: -2, :]
                        + d0[1: -1, None] * x[1: -1, :]
                        + d1[1:, None] * x[2:, :])
        return kx

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Stiffness mass matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """  
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self


class MikotaPair:
    """
    Construct the Mikota pair of matrices in various formats and
    eigenvalues of the generalized eigenproblem with them.

    The Mikota pair of matrices [1, 2]_ models a vibration problem
    of a linear mass-spring system with the ends attached where
    the stiffness of the springs and the masses increase along
    the system length such that vibration frequencies are subsequent
    integers 1, 2, ..., `n` where `n` is the number of the masses. Thus,
    eigenvalues of the generalized eigenvalue problem for
    the matrix pair `K` and `M` where `K` is the system stiffness matrix
    and `M` is the system mass matrix are the squares of the integers,
    i.e., 1, 4, 9, ..., ``n * n``.

    The stiffness matrix `K` is square real tri-diagonal symmetric
    positive definite. The mass matrix `M` is diagonal with diagonal
    entries 1, 1/2, 1/3, ...., ``1/n``. Both matrices get
    ill-conditioned with `n` growing.

    Parameters
    ----------
    n : int
        The size of the matrices of the Mikota pair.
    dtype : dtype
        Numerical type of the array. Default is ``np.float64``.

    Attributes
    ----------
    eigenvalues : 1D ndarray, ``np.uint64``
        All eigenvalues of the Mikota pair ordered ascending.

    Methods
    -------
    MikotaK()
        A `LinearOperator` custom object for the stiffness matrix.
    MikotaM()
        A `LinearOperator` custom object for the mass matrix.
    
    .. versionadded:: 1.12.0

    References
    ----------
    .. [1] J. Mikota, "Frequency tuning of chain structure multibody oscillators
       to place the natural frequencies at omega1 and N-1 integer multiples
       omega2,..., omegaN", Z. Angew. Math. Mech. 81 (2001), S2, S201-S202.
       Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004).
    .. [2] Peter C. Muller and Metin Gurgoze,
       "Natural frequencies of a multi-degree-of-freedom vibration system",
       Proc. Appl. Math. Mech. 6, 319-320 (2006).
       http://dx.doi.org/10.1002/pamm.200610141.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
    >>> n = 6
    >>> mik = MikotaPair(n)
    >>> mik_k = mik.k
    >>> mik_m = mik.m
    >>> mik_k.toarray()
    array([[11., -5.,  0.,  0.,  0.,  0.],
           [-5.,  9., -4.,  0.,  0.,  0.],
           [ 0., -4.,  7., -3.,  0.,  0.],
           [ 0.,  0., -3.,  5., -2.,  0.],
           [ 0.,  0.,  0., -2.,  3., -1.],
           [ 0.,  0.,  0.,  0., -1.,  1.]])
    >>> mik_k.tobanded()
    array([[ 0., -5., -4., -3., -2., -1.],
           [11.,  9.,  7.,  5.,  3.,  1.]])
    >>> mik_m.tobanded()
    array([1.        , 0.5       , 0.33333333, 0.25      , 0.2       ,
        0.16666667])
    >>> mik_k.tosparse()
    <DIAgonal sparse array of dtype 'float64'
        with 20 stored elements (3 diagonals) and shape (6, 6)>
    >>> mik_m.tosparse()
    <DIAgonal sparse array of dtype 'float64'
        with 6 stored elements (1 diagonals) and shape (6, 6)>
    >>> np.array_equal(mik_k(np.eye(n)), mik_k.toarray())
    True
    >>> np.array_equal(mik_m(np.eye(n)), mik_m.toarray())
    True
    >>> mik.eigenvalues()
    array([ 1,  4,  9, 16, 25, 36])  
    >>> mik.eigenvalues(2)
    array([ 1,  4])

    """
    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.dtype = dtype
        self.shape = (n, n)
        self.m = MikotaM(self.shape, self.dtype)
        self.k = MikotaK(self.shape, self.dtype)

    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.uint64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        if m is None:
            m = self.n
        arange_plus1 = np.arange(1, m + 1, dtype=np.uint64)
        return arange_plus1 * arange_plus1
<|endoftext|>"""
Sparse linear algebra (:mod:`scipy.sparse.linalg`)
==================================================

.. currentmodule:: scipy.sparse.linalg

Abstract linear operators
-------------------------

.. autosummary::
   :toctree: generated/

   LinearOperator -- abstract representation of a linear operator
   aslinearoperator -- convert an object to an abstract linear operator

Matrix Operations
-----------------

.. autosummary::
   :toctree: generated/

   inv -- compute the sparse matrix inverse
   expm -- compute the sparse matrix exponential
   expm_multiply -- compute the product of a matrix exponential and a matrix
   matrix_power -- compute the matrix power by raising a matrix to an exponent

Matrix norms
------------

.. autosummary::
   :toctree: generated/

   norm -- Norm of a sparse matrix
   onenormest -- Estimate the 1-norm of a sparse matrix

Solving linear problems
-----------------------

Direct methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   spsolve -- Solve the sparse linear system Ax=b
   spsolve_triangular -- Solve sparse linear system Ax=b for a triangular A.
   is_sptriangular -- Check if sparse A is triangular.
   spbandwidth -- Find the bandwidth of a sparse matrix.
   factorized -- Pre-factorize matrix to a function solving a linear system
   MatrixRankWarning -- Warning on exactly singular matrices
   use_solver -- Select direct solver to use

Iterative methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   bicg -- Use BIConjugate Gradient iteration to solve Ax = b
   bicgstab -- Use BIConjugate Gradient STABilized iteration to solve Ax = b
   cg -- Use Conjugate Gradient iteration to solve Ax = b
   cgs -- Use Conjugate Gradient Squared iteration to solve Ax = b
   gmres -- Use Generalized Minimal RESidual iteration to solve Ax = b
   lgmres -- Solve a matrix equation using the LGMRES algorithm
   minres -- Use MINimum RESidual iteration to solve Ax = b
   qmr -- Use Quasi-Minimal Residual iteration to solve Ax = b
   gcrotmk -- Solve a matrix equation using the GCROT(m,k) algorithm
   tfqmr -- Use Transpose-Free Quasi-Minimal Residual iteration to solve Ax = b

Iterative methods for least-squares problems:

.. autosummary::
   :toctree: generated/

   lsqr -- Find the least-squares solution to a sparse linear equation system
   lsmr -- Find the least-squares solution to a sparse linear equation system

Matrix factorizations
---------------------

Eigenvalue problems:

.. autosummary::
   :toctree: generated/

   eigs -- Find k eigenvalues and eigenvectors of the square matrix A
   eigsh -- Find k eigenvalues and eigenvectors of a symmetric matrix
   lobpcg -- Solve symmetric partial eigenproblems with optional preconditioning

Singular values problems:

.. autosummary::
   :toctree: generated/

   svds -- Compute k singular values/vectors for a sparse matrix

The `svds` function supports the following solvers:

.. toctree::

    sparse.linalg.svds-arpack
    sparse.linalg.svds-lobpcg
    sparse.linalg.svds-propack

Complete or incomplete LU factorizations

.. autosummary::
   :toctree: generated/

   splu -- Compute a LU decomposition for a sparse matrix
   spilu -- Compute an incomplete LU decomposition for a sparse matrix
   SuperLU -- Object representing an LU factorization

Sparse arrays with structure
----------------------------

.. autosummary::
   :toctree: generated/

   LaplacianNd -- Laplacian on a uniform rectangular grid in ``N`` dimensions

Exceptions
----------

.. autosummary::
   :toctree: generated/

   ArpackNoConvergence
   ArpackError

"""

from ._isolve import *
from ._dsolve import *
from ._interface import *
from ._eigen import *
from ._matfuncs import *
from ._onenormest import *
from ._norm import *
from ._expm_multiply import *
from ._special_sparse_arrays import *

# Deprecated namespaces, to be removed in v2.0.0
from . import isolve, dsolve, interface, eigen, matfuncs

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = ["expm", "inv", "spsolve", "LinearOperator"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="matfuncs",
                                   private_modules=["_matfuncs"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'ArpackError', 'ArpackNoConvergence', 'ArpackError',
    'eigs', 'eigsh', 'lobpcg', 'svds', 'test'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="eigen",
                                   private_modules=["_eigen"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Sparse matrix functions
"""

#
# Authors: Travis Oliphant, March 2002
#          Anthony Scopatz, August 2012 (Sparse Updates)
#          Jake Vanderplas, August 2012 (Sparse Updates)
#

__all__ = ['expm', 'inv', 'matrix_power']

import numpy as np
from scipy.linalg._basic import solve, solve_triangular

from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike

import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye_array

from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm


UPPER_TRIANGULAR = 'upper_triangular'


def inv(A):
    """
    Compute the inverse of a sparse arrays

    Parameters
    ----------
    A : (M, M) sparse arrays
        square matrix to be inverted

    Returns
    -------
    Ainv : (M, M) sparse arrays
        inverse of `A`

    Notes
    -----
    This computes the sparse inverse of `A`. If the inverse of `A` is expected
    to be non-sparse, it will likely be faster to convert `A` to dense and use
    `scipy.linalg.inv`.

    Examples
    --------
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import inv
    >>> A = csc_array([[1., 0.], [1., 2.]])
    >>> Ainv = inv(A)
    >>> Ainv
    <Compressed Sparse Column sparse array of dtype 'float64'
        with 3 stored elements and shape (2, 2)>
    >>> A.dot(Ainv)
    <Compressed Sparse Column sparse array of dtype 'float64'
        with 2 stored elements and shape (2, 2)>
    >>> A.dot(Ainv).toarray()
    array([[ 1.,  0.],
           [ 0.,  1.]])

    .. versionadded:: 0.12.0

    """
    # Check input
    if not (issparse(A) or is_pydata_spmatrix(A)):
        raise TypeError('Input must be a sparse arrays')

    # Use sparse direct solver to solve "AX = I" accurately
    I = _ident_like(A)
    Ainv = spsolve(A, I)
    return Ainv


def _onenorm_matrix_power_nnm(A, p):
    """
    Compute the 1-norm of a non-negative integer power of a non-negative matrix.

    Parameters
    ----------
    A : a square ndarray or matrix or sparse arrays
        Input matrix with non-negative entries.
    p : non-negative integer
        The power to which the matrix is to be raised.

    Returns
    -------
    out : float
        The 1-norm of the matrix power p of A.

    """
    # Check input
    if int(p) != p or p < 0:
        raise ValueError('expected non-negative integer p')
    p = int(p)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # Explicitly make a column vector so that this works when A is a
    # numpy matrix (in addition to ndarray and sparse arrays).
    v = np.ones((A.shape[0], 1), dtype=float)
    M = A.T
    for i in range(p):
        v = M.dot(v)
    return np.max(v)


def _is_upper_triangular(A):
    # This function could possibly be of wider interest.
    if issparse(A):
        lower_part = scipy.sparse.tril(A, -1)
        # Check structural upper triangularity,
        # then coincidental upper triangularity if needed.
        return lower_part.nnz == 0 or lower_part.count_nonzero() == 0
    elif is_pydata_spmatrix(A):
        import sparse
        lower_part = sparse.tril(A, -1)
        return lower_part.nnz == 0
    else:
        return not np.tril(A, -1).any()


def _smart_matrix_product(A, B, alpha=None, structure=None):
    """
    A matrix product that knows about sparse and structured matrices.

    Parameters
    ----------
    A : 2d ndarray
        First matrix.
    B : 2d ndarray
        Second matrix.
    alpha : float
        The matrix product will be scaled by this constant.
    structure : str, optional
        A string describing the structure of both matrices `A` and `B`.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    M : 2d ndarray
        Matrix product of A and B.

    """
    if len(A.shape) != 2:
        raise ValueError('expected A to be a rectangular matrix')
    if len(B.shape) != 2:
        raise ValueError('expected B to be a rectangular matrix')
    f = None
    if structure == UPPER_TRIANGULAR:
        if (not issparse(A) and not issparse(B)
                and not is_pydata_spmatrix(A) and not is_pydata_spmatrix(B)):
            f, = scipy.linalg.get_blas_funcs(('trmm',), (A, B))
    if f is not None:
        if alpha is None:
            alpha = 1.
        out = f(alpha, A, B)
    else:
        if alpha is None:
            out = A.dot(B)
        else:
            out = alpha * A.dot(B)
    return out


class MatrixPowerOperator(LinearOperator):

    def __init__(self, A, p, structure=None):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0:
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self._structure = structure
        self.dtype = A.dtype
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x)
        return x

    def _rmatvec(self, x):
        A_T = self._A.T
        x = x.ravel()
        for i in range(self._p):
            x = A_T.dot(x)
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = _smart_matrix_product(self._A, X, structure=self._structure)
        return X

    @property
    def T(self):
        return MatrixPowerOperator(self._A.T, self._p)


class ProductOperator(LinearOperator):
    """
    For now, this is limited to products of multiple square matrices.
    """

    def __init__(self, *args, **kwargs):
        self._structure = kwargs.get('structure', None)
        for A in args:
            if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(
                        'For now, the ProductOperator implementation is '
                        'limited to the product of multiple square matrices.')
        if args:
            n = args[0].shape[0]
            for A in args:
                for d in A.shape:
                    if d != n:
                        raise ValueError(
                                'The square matrices of the ProductOperator '
                                'must all have the same shape.')
            self.shape = (n, n)
            self.ndim = len(self.shape)
        self.dtype = np.result_type(*[x.dtype for x in args])
        self._operator_sequence = args

    def _matvec(self, x):
        for A in reversed(self._operator_sequence):
            x = A.dot(x)
        return x

    def _rmatvec(self, x):
        x = x.ravel()
        for A in self._operator_sequence:
            x = A.T.dot(x)
        return x

    def _matmat(self, X):
        for A in reversed(self._operator_sequence):
            X = _smart_matrix_product(A, X, structure=self._structure)
        return X

    @property
    def T(self):
        T_args = [A.T for A in reversed(self._operator_sequence)]
        return ProductOperator(*T_args)


def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse arrays.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    return scipy.sparse.linalg.onenormest(
            MatrixPowerOperator(A, p, structure=structure))


def _onenormest_product(operator_seq,
        t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of the matrix product of the args.

    Parameters
    ----------
    operator_seq : linear operator sequence
        Matrices whose 1-norm of product is to be computed.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.
    structure : str, optional
        A string describing the structure of all operators.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse arrays.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    return scipy.sparse.linalg.onenormest(
            ProductOperator(*operator_seq, structure=structure))


class _ExpmPadeHelper:
    """
    Help lazily evaluate a matrix exponential.

    The idea is to not do more work than we need for high expm precision,
    so we lazily compute matrix powers and store or precompute
    other properties of the matrix.

    """

    def __init__(self, A, structure=None, use_exact_onenorm=False):
        """
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        structure : str, optional
            A string describing the structure of matrix `A`.
            Only `upper_triangular` is currently supported.
        use_exact_onenorm : bool, optional
            If True then only the exact one-norm of matrix powers and products
            will be used. Otherwise, the one-norm of powers and products
            may initially be estimated.
        """
        self.A = A
        self._A2 = None
        self._A4 = None
        self._A6 = None
        self._A8 = None
        self._A10 = None
        self._d4_exact = None
        self._d6_exact = None
        self._d8_exact = None
        self._d10_exact = None
        self._d4_approx = None
        self._d6_approx = None
        self._d8_approx = None
        self._d10_approx = None
        self.ident = _ident_like(A)
        self.structure = structure
        self.use_exact_onenorm = use_exact_onenorm

    @property
    def A2(self):
        if self._A2 is None:
            self._A2 = _smart_matrix_product(
                    self.A, self.A, structure=self.structure)
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            self._A4 = _smart_matrix_product(
                    self.A2, self.A2, structure=self.structure)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            self._A6 = _smart_matrix_product(
                    self.A4, self.A2, structure=self.structure)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            self._A8 = _smart_matrix_product(
                    self.A6, self.A2, structure=self.structure)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            self._A10 = _smart_matrix_product(
                    self.A4, self.A6, structure=self.structure)
        return self._A10

    @property
    def d4_tight(self):
        if self._d4_exact is None:
            self._d4_exact = _onenorm(self.A4)**(1/4.)
        return self._d4_exact

    @property
    def d6_tight(self):
        if self._d6_exact is None:
            self._d6_exact = _onenorm(self.A6)**(1/6.)
        return self._d6_exact

    @property
    def d8_tight(self):
        if self._d8_exact is None:
            self._d8_exact = _onenorm(self.A8)**(1/8.)
        return self._d8_exact

    @property
    def d10_tight(self):
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10)**(1/10.)
        return self._d10_exact

    @property
    def d4_loose(self):
        if self.use_exact_onenorm:
            return self.d4_tight
        if self._d4_exact is not None:
            return self._d4_exact
        else:
            if self._d4_approx is None:
                self._d4_approx = _onenormest_matrix_power(self.A2, 2,
                        structure=self.structure)**(1/4.)
            return self._d4_approx

    @property
    def d6_loose(self):
        if self.use_exact_onenorm:
            return self.d6_tight
        if self._d6_exact is not None:
            return self._d6_exact
        else:
            if self._d6_approx is None:
                self._d6_approx = _onenormest_matrix_power(self.A2, 3,
                        structure=self.structure)**(1/6.)
            return self._d6_approx

    @property
    def d8_loose(self):
        if self.use_exact_onenorm:
            return self.d8_tight
        if self._d8_exact is not None:
            return self._d8_exact
        else:
            if self._d8_approx is None:
                self._d8_approx = _onenormest_matrix_power(self.A4, 2,
                        structure=self.structure)**(1/8.)
            return self._d8_approx

    @property
    def d10_loose(self):
        if self.use_exact_onenorm:
            return self.d10_tight
        if self._d10_exact is not None:
            return self._d10_exact
        else:
            if self._d10_approx is None:
                self._d10_approx = _onenormest_product((self.A4, self.A6),
                        structure=self.structure)**(1/10.)
            return self._d10_approx

    def pade3(self):
        b = (120., 60., 12., 1.)
        U = _smart_matrix_product(self.A,
                b[3]*self.A2 + b[1]*self.ident,
                structure=self.structure)
        V = b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade5(self):
        b = (30240., 15120., 3360., 420., 30., 1.)
        U = _smart_matrix_product(self.A,
                b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
                structure=self.structure)
        V = b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade7(self):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = _smart_matrix_product(self.A,
                b[7]*self.A6 + b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
                structure=self.structure)
        V = b[6]*self.A6 + b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade9(self):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                2162160., 110880., 3960., 90., 1.)
        U = _smart_matrix_product(self.A,
                (b[9]*self.A8 + b[7]*self.A6 + b[5]*self.A4 +
                    b[3]*self.A2 + b[1]*self.ident),
                structure=self.structure)
        V = (b[8]*self.A8 + b[6]*self.A6 + b[4]*self.A4 +
                b[2]*self.A2 + b[0]*self.ident)
        return U, V

    def pade13_scaled(self, s):
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        B = self.A * 2**-s
        B2 = self.A2 * 2**(-2*s)
        B4 = self.A4 * 2**(-4*s)
        B6 = self.A6 * 2**(-6*s)
        U2 = _smart_matrix_product(B6,
                b[13]*B6 + b[11]*B4 + b[9]*B2,
                structure=self.structure)
        U = _smart_matrix_product(B,
                (U2 + b[7]*B6 + b[5]*B4 +
                    b[3]*B2 + b[1]*self.ident),
                structure=self.structure)
        V2 = _smart_matrix_product(B6,
                b[12]*B6 + b[10]*B4 + b[8]*B2,
                structure=self.structure)
        V = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident
        return U, V


def expm(A):
    """
    Compute the matrix exponential using Pade approximation.

    Parameters
    ----------
    A : (M,M) array_like or sparse array
        2D Array or Matrix (sparse or dense) to be exponentiated

    Returns
    -------
    expA : (M,M) ndarray
        Matrix exponential of `A`

    Notes
    -----
    This is algorithm (6.1) which is a simplification of algorithm (5.1).

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           "A New Scaling and Squaring Algorithm for the Matrix Exponential."
           SIAM Journal on Matrix Analysis and Applications.
           31 (3). pp. 970-989. ISSN 1095-7162

    Examples
    --------
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import expm
    >>> A = csc_array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> A.toarray()
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]], dtype=int64)
    >>> Aexp = expm(A)
    >>> Aexp
    <Compressed Sparse Column sparse array of dtype 'float64'
        with 3 stored elements and shape (3, 3)>
    >>> Aexp.toarray()
    array([[  2.71828183,   0.        ,   0.        ],
           [  0.        ,   7.3890561 ,   0.        ],
           [  0.        ,   0.        ,  20.08553692]])
    """
    return _expm(A, use_exact_onenorm='auto')


def _expm(A, use_exact_onenorm):
    # Core of expm, separated to allow testing exact and approximate
    # algorithms.

    # Avoid indiscriminate asarray() to allow sparse or other strange arrays.
    if isinstance(A, (list, tuple, np.matrix)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # gracefully handle size-0 input,
    # carefully handling sparse scenario
    if A.shape == (0, 0):
        out = np.zeros([0, 0], dtype=A.dtype)
        if issparse(A) or is_pydata_spmatrix(A):
            return A.__class__(out)
        return out

    # Trivial case
    if A.shape == (1, 1):
        out = [[np.exp(A[0, 0])]]

        # Avoid indiscriminate casting to ndarray to
        # allow for sparse or other strange arrays
        if issparse(A) or is_pydata_spmatrix(A):
            return A.__class__(out)

        return np.array(out)

    # Ensure input is of float type, to avoid integer overflows etc.
    if ((isinstance(A, np.ndarray) or issparse(A) or is_pydata_spmatrix(A))
            and not np.issubdtype(A.dtype, np.inexact)):
        A = A.astype(float)

    # Detect upper triangularity.
    structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None

    if use_exact_onenorm == "auto":
        # Hardcode a matrix order threshold for exact vs. estimated one-norms.
        use_exact_onenorm = A.shape[0] < 200

    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper(
            A, structure=structure, use_exact_onenorm=use_exact_onenorm)

    # Try Pade order 3.
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
        U, V = h.pade3()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade order 5.
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
        U, V = h.pade5()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade orders 7 and 9.
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
        U, V = h.pade7()
        return _solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
        U, V = h.pade9()
        return _solve_P_Q(U, V, structure=structure)

    # Use Pade order 13.
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25

    # Choose smallest s>=0 such that 2**(-s) eta_5 <= theta_13
    if eta_5 == 0:
        # Nilpotent special case
        s = 0
    else:
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2**-s * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V, structure=structure)
    if structure == UPPER_TRIANGULAR:
        # Invoke Code Fragment 2.1.
        X = _fragment_2_1(X, h.A, s)
    else:
        # X = r_13(A)^(2^s) by repeated squaring.
        for i in range(s):
            X = X.dot(X)
    return X


def _solve_P_Q(U, V, structure=None):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade numerator.
    V : ndarray
        Pade denominator.
    structure : str, optional
        A string describing the structure of both matrices `U` and `V`.
        Only `upper_triangular` is currently supported.

    Notes
    -----
    The `structure` argument is inspired by similar args
    for theano and cvxopt functions.

    """
    P = U + V
    Q = -U + V
    if issparse(U) or is_pydata_spmatrix(U):
        return spsolve(Q, P)
    elif structure is None:
        return solve(Q, P)
    elif structure == UPPER_TRIANGULAR:
        return solve_triangular(Q, P)
    else:
        raise ValueError('unsupported matrix structure: ' + str(structure))


def _exp_sinch(a, x):
    """
    Stably evaluate exp(a)*sinh(x)/x

    Notes
    -----
    The strategy of falling back to a sixth order Taylor expansion
    was suggested by the Spallation Neutron Source docs
    which was found on the internet by google search.
    http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html
    The details of the cutoff point and the Horner-like evaluation
    was picked without reference to anything in particular.

    Note that sinch is not currently implemented in scipy.special,
    whereas the "engineer's" definition of sinc is implemented.
    The implementation of sinc involves a scaling factor of pi
    that distinguishes it from the "mathematician's" version of sinc.

    """

    # If x is small then use sixth order Taylor expansion.
    # How small is small? I am using the point where the relative error
    # of the approximation is less than 1e-14.
    # If x is large then directly evaluate sinh(x) / x.
    if abs(x) < 0.0135:
        x2 = x*x
        return np.exp(a) * (1 + (x2/6.)*(1 + (x2/20.)*(1 + (x2/42.))))
    else:
        return (np.exp(a + x) - np.exp(a - x)) / (2*x)


def _eq_10_42(lam_1, lam_2, t_12):
    """
    Equation (10.42) of Functions of Matrices: Theory and Computation.

    Notes
    -----
    This is a helper function for _fragment_2_1 of expm_2009.
    Equation (10.42) is on page 251 in the section on Schur algorithms.
    In particular, section 10.4.3 explains the Schur-Parlett algorithm.
    expm([[lam_1, t_12], [0, lam_1])
    =
    [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],
    [0, exp(lam_2)]
    """

    # The plain formula t_12 * (exp(lam_2) - exp(lam_2)) / (lam_2 - lam_1)
    # apparently suffers from cancellation, according to Higham's textbook.
    # A nice implementation of sinch, defined as sinh(x)/x,
    # will apparently work around the cancellation.
    a = 0.5 * (lam_1 + lam_2)
    b = 0.5 * (lam_1 - lam_2)
    return t_12 * _exp_sinch(a, b)


def _fragment_2_1(X, T, s):
    """
    A helper function for expm_2009.

    Notes
    -----
    The argument X is modified in-place, but this modification is not the same
    as the returned value of the function.
    This function also takes pains to do things in ways that are compatible
    with sparse arrays, for example by avoiding fancy indexing
    and by using methods of the matrices whenever possible instead of
    using functions of the numpy or scipy libraries themselves.

    """
    # Form X = r_m(2^-s T)
    # Replace diag(X) by exp(2^-s diag(T)).
    n = X.shape[0]
    diag_T = np.ravel(T.diagonal().copy())

    # Replace diag(X) by exp(2^-s diag(T)).
    scale = 2 ** -s
    exp_diag = np.exp(scale * diag_T)
    for k in range(n):
        X[k, k] = exp_diag[k]

    for i in range(s-1, -1, -1):
        X = X.dot(X)

        # Replace diag(X) by exp(2^-i diag(T)).
        scale = 2 ** -i
        exp_diag = np.exp(scale * diag_T)
        for k in range(n):
            X[k, k] = exp_diag[k]

        # Replace (first) superdiagonal of X by explicit formula
        # for superdiagonal of exp(2^-i T) from Eq (10.42) of
        # the author's 2008 textbook
        # Functions of Matrices: Theory and Computation.
        for k in range(n-1):
            lam_1 = scale * diag_T[k]
            lam_2 = scale * diag_T[k+1]
            t_12 = scale * T[k, k+1]
            value = _eq_10_42(lam_1, lam_2, t_12)
            X[k, k+1] = value

    # Return the updated X matrix.
    return X


def _ell(A, m):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    A : linear operator
        A linear operator whose norm of power we care about.
    m : int
        The power of the linear operator

    Returns
    -------
    value : int
        A value related to a bound.

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
    # They are coefficients of terms of a generating function series expansion.
    c_i = {3: 100800.,
           5: 10059033600.,
           7: 4487938430976000.,
           9: 5914384781877411840000.,
           13: 113250775606021113483283660800000000.
           }
    abs_c_recip = c_i[m]

    # This is explained after Eq. (1.2) of the 2009 expm paper.
    # It is the "unit roundoff" of IEEE double precision arithmetic.
    u = 2**-53

    # Compute the one-norm of matrix power p of abs(A).
    A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), 2*m + 1)

    # Treat zero norm as a special case.
    if not A_abs_onenorm:
        return 0

    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    log2_alpha_div_u = np.log2(alpha/u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)

def matrix_power(A, power):
    """
    Raise a square matrix to the integer power, `power`.

    For non-negative integers, ``A**power`` is computed using repeated
    matrix multiplications. Negative integers are not supported. 

    Parameters
    ----------
    A : (M, M) square sparse array or matrix
        sparse array that will be raised to power `power`
    power : int
        Exponent used to raise sparse array `A`

    Returns
    -------
    A**power : (M, M) sparse array or matrix
        The output matrix will be the same shape as A, and will preserve
        the class of A, but the format of the output may be changed.
    
    Notes
    -----
    This uses a recursive implementation of the matrix power. For computing
    the matrix power using a reasonably large `power`, this may be less efficient
    than computing the product directly, using A @ A @ ... @ A.
    This is contingent upon the number of nonzero entries in the matrix. 

    .. versionadded:: 1.12.0

    Examples
    --------
    >>> from scipy import sparse
    >>> A = sparse.csc_array([[0,1,0],[1,0,1],[0,1,0]])
    >>> A.todense()
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
    >>> (A @ A).todense()
    array([[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]])
    >>> A2 = sparse.linalg.matrix_power(A, 2)
    >>> A2.todense()
    array([[1, 0, 1],
           [0, 2, 0],
           [1, 0, 1]])
    >>> A4 = sparse.linalg.matrix_power(A, 4)
    >>> A4.todense()
    array([[2, 0, 2],
           [0, 4, 0],
           [2, 0, 2]])

    """
    M, N = A.shape
    if M != N:
        raise TypeError('sparse matrix is not square')

    if isintlike(power):
        power = int(power)
        if power < 0:
            raise ValueError('exponent must be >= 0')

        if power == 0:
            return eye_array(M, dtype=A.dtype)

        if power == 1:
            return A.copy()

        tmp = matrix_power(A, power // 2)
        if power % 2:
            return A @ tmp @ tmp
        else:
            return tmp @ tmp
    else:
        raise ValueError("exponent must be an integer")
<|endoftext|>"""
Linear Solvers
==============

The default solver is SuperLU (included in the scipy distribution),
which can solve real or complex linear systems in both single and
double precisions.  It is automatically replaced by UMFPACK, if
available.  Note that UMFPACK works in double precision only, so
switch it off by::

    >>> from scipy.sparse.linalg import spsolve, use_solver
    >>> use_solver(useUmfpack=False)

to solve in the single precision. See also use_solver documentation.

Example session::

    >>> from scipy.sparse import csc_array, dia_array
    >>> from numpy import array
    >>>
    >>> print("Inverting a sparse linear system:")
    >>> print("The sparse matrix (constructed from diagonals):")
    >>> a = dia_array(([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1]), shape=(5, 5))
    >>> b = array([1, 2, 3, 4, 5])
    >>> print("Solve: single precision complex:")
    >>> use_solver( useUmfpack = False )
    >>> a = a.astype('F')
    >>> x = spsolve(a, b)
    >>> print(x)
    >>> print("Error: ", a@x-b)
    >>>
    >>> print("Solve: double precision complex:")
    >>> use_solver( useUmfpack = True )
    >>> a = a.astype('D')
    >>> x = spsolve(a, b)
    >>> print(x)
    >>> print("Error: ", a@x-b)
    >>>
    >>> print("Solve: double precision:")
    >>> a = a.astype('d')
    >>> x = spsolve(a, b)
    >>> print(x)
    >>> print("Error: ", a@x-b)
    >>>
    >>> print("Solve: single precision:")
    >>> use_solver( useUmfpack = False )
    >>> a = a.astype('f')
    >>> x = spsolve(a, b.astype('f'))
    >>> print(x)
    >>> print("Error: ", a@x-b)

"""

#import umfpack
#__doc__ = '\n\n'.join( (__doc__,  umfpack.__doc__) )
#del umfpack

from .linsolve import *
from ._superlu import SuperLU
from . import _add_newdocs
from . import linsolve

__all__ = [
    'MatrixRankWarning', 'SuperLU', 'factorized',
    'spilu', 'splu', 'spsolve', 'is_sptriangular',
    'spsolve_triangular', 'use_solver', 'spbandwidth',
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>from warnings import warn, catch_warnings, simplefilter

import numpy as np
from numpy import asarray
from scipy.sparse import (issparse, SparseEfficiencyWarning,
                          csr_array, csc_array, eye_array, diags_array)
from scipy.sparse._sputils import (is_pydata_spmatrix, convert_pydata_sparse_to_scipy,
                                   get_index_dtype, safely_cast_index_arrays)
from scipy.linalg import LinAlgError
import copy
import threading

from . import _superlu

noScikit = False
try:
    import scikits.umfpack as umfpack
except ImportError:
    noScikit = True

useUmfpack = threading.local()


__all__ = ['use_solver', 'spsolve', 'splu', 'spilu', 'factorized',
           'MatrixRankWarning', 'spsolve_triangular', 'is_sptriangular', 'spbandwidth']


class MatrixRankWarning(UserWarning):
    pass


def use_solver(**kwargs):
    """
    Select default sparse direct solver to be used.

    Parameters
    ----------
    useUmfpack : bool, optional
        Use UMFPACK [1]_, [2]_, [3]_, [4]_. over SuperLU. Has effect only
        if ``scikits.umfpack`` is installed. Default: True
    assumeSortedIndices : bool, optional
        Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
        Has effect only if useUmfpack is True and ``scikits.umfpack`` is
        installed. Default: False

    Notes
    -----
    The default sparse solver is UMFPACK when available
    (``scikits.umfpack`` is installed). This can be changed by passing
    useUmfpack = False, which then causes the always present SuperLU
    based solver to be used.

    UMFPACK requires a CSR/CSC matrix to have sorted column/row indices. If
    sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
    to gain some speed.

    References
    ----------
    .. [1] T. A. Davis, Algorithm 832:  UMFPACK - an unsymmetric-pattern
           multifrontal method with a column pre-ordering strategy, ACM
           Trans. on Mathematical Software, 30(2), 2004, pp. 196--199.
           https://dl.acm.org/doi/abs/10.1145/992200.992206

    .. [2] T. A. Davis, A column pre-ordering strategy for the
           unsymmetric-pattern multifrontal method, ACM Trans.
           on Mathematical Software, 30(2), 2004, pp. 165--195.
           https://dl.acm.org/doi/abs/10.1145/992200.992205

    .. [3] T. A. Davis and I. S. Duff, A combined unifrontal/multifrontal
           method for unsymmetric sparse matrices, ACM Trans. on
           Mathematical Software, 25(1), 1999, pp. 1--19.
           https://doi.org/10.1145/305658.287640

    .. [4] T. A. Davis and I. S. Duff, An unsymmetric-pattern multifrontal
           method for sparse LU factorization, SIAM J. Matrix Analysis and
           Computations, 18(1), 1997, pp. 140--158.
           https://doi.org/10.1137/S0895479894246905T.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import use_solver, spsolve
    >>> from scipy.sparse import csc_array
    >>> R = np.random.randn(5, 5)
    >>> A = csc_array(R)
    >>> b = np.random.randn(5)
    >>> use_solver(useUmfpack=False) # enforce superLU over UMFPACK
    >>> x = spsolve(A, b)
    >>> np.allclose(A.dot(x), b)
    True
    >>> use_solver(useUmfpack=True) # reset umfPack usage to default
    """
    global useUmfpack
    if 'useUmfpack' in kwargs:
        useUmfpack.u = kwargs['useUmfpack']
    if useUmfpack.u and 'assumeSortedIndices' in kwargs:
        umfpack.configure(assumeSortedIndices=kwargs['assumeSortedIndices'])

def _get_umf_family(A):
    """Get umfpack family string given the sparse matrix dtype."""
    _families = {
        (np.float64, np.int32): 'di',
        (np.complex128, np.int32): 'zi',
        (np.float64, np.int64): 'dl',
        (np.complex128, np.int64): 'zl'
    }

    # A.dtype.name can only be "float64" or
    # "complex128" in control flow
    f_type = getattr(np, A.dtype.name)
    # control flow may allow for more index
    # types to get through here
    i_type = getattr(np, A.indices.dtype.name)

    try:
        family = _families[(f_type, i_type)]

    except KeyError as e:
        msg = ('only float64 or complex128 matrices with int32 or int64 '
               f'indices are supported! (got: matrix: {f_type}, indices: {i_type})')
        raise ValueError(msg) from e

    # See gh-8278. Considered converting only if
    # A.shape[0]*A.shape[1] > np.iinfo(np.int32).max,
    # but that didn't always fix the issue.
    family = family[0] + "l"
    A_new = copy.copy(A)
    A_new.indptr = np.asarray(A.indptr, dtype=np.int64)
    A_new.indices = np.asarray(A.indices, dtype=np.int64)

    return family, A_new

def spsolve(A, b, permc_spec=None, use_umfpack=True):
    """Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    Parameters
    ----------
    A : ndarray or sparse array or matrix
        The square matrix A will be converted into CSC or CSR form
    b : ndarray or sparse array or matrix
        The matrix or vector representing the right hand side of the equation.
        If a vector, b.shape must be (n,) or (n, 1).
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering [1]_, [2]_.

    use_umfpack : bool, optional
        if True (default) then use UMFPACK for the solution [3]_, [4]_, [5]_,
        [6]_ . This is only referenced if b is a vector and
        ``scikits.umfpack`` is installed.

    Returns
    -------
    x : ndarray or sparse array or matrix
        the solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

    Notes
    -----
    For solving the matrix expression AX = B, this solver assumes the resulting
    matrix X is sparse, as is often the case for very sparse inputs.  If the
    resulting X is dense, the construction of this sparse result will be
    relatively expensive.  In that case, consider converting A to a dense
    matrix and using scipy.linalg.solve or its variants.

    References
    ----------
    .. [1] T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836:
           COLAMD, an approximate column minimum degree ordering algorithm,
           ACM Trans. on Mathematical Software, 30(3), 2004, pp. 377--380.
           :doi:`10.1145/1024074.1024080`

    .. [2] T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, A column approximate
           minimum degree ordering algorithm, ACM Trans. on Mathematical
           Software, 30(3), 2004, pp. 353--376. :doi:`10.1145/1024074.1024079`

    .. [3] T. A. Davis, Algorithm 832:  UMFPACK - an unsymmetric-pattern
           multifrontal method with a column pre-ordering strategy, ACM
           Trans. on Mathematical Software, 30(2), 2004, pp. 196--199.
           https://dl.acm.org/doi/abs/10.1145/992200.992206

    .. [4] T. A. Davis, A column pre-ordering strategy for the
           unsymmetric-pattern multifrontal method, ACM Trans.
           on Mathematical Software, 30(2), 2004, pp. 165--195.
           https://dl.acm.org/doi/abs/10.1145/992200.992205

    .. [5] T. A. Davis and I. S. Duff, A combined unifrontal/multifrontal
           method for unsymmetric sparse matrices, ACM Trans. on
           Mathematical Software, 25(1), 1999, pp. 1--19.
           https://doi.org/10.1145/305658.287640

    .. [6] T. A. Davis and I. S. Duff, An unsymmetric-pattern multifrontal
           method for sparse LU factorization, SIAM J. Matrix Analysis and
           Computations, 18(1), 1997, pp. 140--158.
           https://doi.org/10.1137/S0895479894246905T.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import spsolve
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> B = csc_array([[2, 0], [-1, 0], [2, 0]], dtype=float)
    >>> x = spsolve(A, B)
    >>> np.allclose(A.dot(x).toarray(), B.toarray())
    True
    """
    is_pydata_sparse = is_pydata_spmatrix(b)
    pydata_sparse_cls = b.__class__ if is_pydata_sparse else None
    A = convert_pydata_sparse_to_scipy(A)
    b = convert_pydata_sparse_to_scipy(b)

    if not (issparse(A) and A.format in ("csc", "csr")):
        A = csc_array(A)
        warn('spsolve requires A be CSC or CSR matrix format',
             SparseEfficiencyWarning, stacklevel=2)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = issparse(b)
    if not b_is_sparse:
        b = asarray(b)
    b_is_vector = ((b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1))

    # sum duplicates for non-canonical format
    A.sum_duplicates()
    A = A._asfptype()  # upcast to a floating point format
    result_dtype = np.promote_types(A.dtype, b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)

    # validate input shapes
    M, N = A.shape
    if (M != N):
        raise ValueError(f"matrix must be square (has shape {(M, N)})")

    if M != b.shape[0]:
        raise ValueError(f"matrix - rhs dimension mismatch ({A.shape} - {b.shape[0]})")

    if not hasattr(useUmfpack, 'u'):
        useUmfpack.u = not noScikit

    use_umfpack = use_umfpack and useUmfpack.u

    if b_is_vector and use_umfpack:
        if b_is_sparse:
            b_vec = b.toarray()
        else:
            b_vec = b
        b_vec = asarray(b_vec, dtype=A.dtype).ravel()

        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack.u = False")

        umf_family, A = _get_umf_family(A)
        umf = umfpack.UmfpackContext(umf_family)
        x = umf.linsolve(umfpack.UMFPACK_A, A, b_vec,
                         autoTranspose=True)
    else:
        if b_is_vector and b_is_sparse:
            b = b.toarray()
            b_is_sparse = False

        if not b_is_sparse:
            if A.format == "csc":
                flag = 1  # CSC format
            else:
                flag = 0  # CSR format

            indices = A.indices.astype(np.intc, copy=False)
            indptr = A.indptr.astype(np.intc, copy=False)
            options = dict(ColPerm=permc_spec)
            x, info = _superlu.gssv(N, A.nnz, A.data, indices, indptr,
                                    b, flag, options=options)
            if info != 0:
                warn("Matrix is exactly singular", MatrixRankWarning, stacklevel=2)
                x.fill(np.nan)
            if b_is_vector:
                x = x.ravel()
        else:
            # b is sparse
            Afactsolve = factorized(A)

            if not (b.format == "csc" or is_pydata_spmatrix(b)):
                warn('spsolve is more efficient when sparse b '
                     'is in the CSC matrix format',
                     SparseEfficiencyWarning, stacklevel=2)
                b = csc_array(b)

            # Create a sparse output matrix by repeatedly applying
            # the sparse factorization to solve columns of b.
            data_segs = []
            row_segs = []
            col_segs = []
            for j in range(b.shape[1]):
                bj = b[:, j].toarray().ravel()
                xj = Afactsolve(bj)
                w = np.flatnonzero(xj)
                segment_length = w.shape[0]
                row_segs.append(w)
                col_segs.append(np.full(segment_length, j, dtype=int))
                data_segs.append(np.asarray(xj[w], dtype=A.dtype))
            sparse_data = np.concatenate(data_segs)
            idx_dtype = get_index_dtype(maxval=max(b.shape))
            sparse_row = np.concatenate(row_segs, dtype=idx_dtype)
            sparse_col = np.concatenate(col_segs, dtype=idx_dtype)
            x = A.__class__((sparse_data, (sparse_row, sparse_col)),
                           shape=b.shape, dtype=A.dtype)

            if is_pydata_sparse:
                x = pydata_sparse_cls.from_scipy_sparse(x)

    return x


def splu(A, permc_spec=None, diag_pivot_thresh=None,
         relax=None, panel_size=None, options=None):
    """
    Compute the LU decomposition of a sparse, square matrix.

    Parameters
    ----------
    A : sparse array or matrix
        Sparse array to factorize. Most efficient when provided in CSC
        format. Other formats will be converted to CSC before factorization.
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering

    diag_pivot_thresh : float, optional
        Threshold used for a diagonal entry to be an acceptable pivot.
        See SuperLU user's guide for details [1]_
    relax : int, optional
        Expert option for customizing the degree of relaxing supernodes.
        See SuperLU user's guide for details [1]_
    panel_size : int, optional
        Expert option for customizing the panel size.
        See SuperLU user's guide for details [1]_
    options : dict, optional
        Dictionary containing additional expert options to SuperLU.
        See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
        for more details. For example, you can specify
        ``options=dict(Equil=False, IterRefine='SINGLE'))``
        to turn equilibration off and perform a single iterative refinement.

    Returns
    -------
    invA : scipy.sparse.linalg.SuperLU
        Object, which has a ``solve`` method.

    See also
    --------
    spilu : incomplete LU decomposition

    Notes
    -----
    This function uses the SuperLU library.

    References
    ----------
    .. [1] SuperLU https://portal.nersc.gov/project/sparse/superlu/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import splu
    >>> A = csc_array([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
    >>> B = splu(A)
    >>> x = np.array([1., 2., 3.], dtype=float)
    >>> B.solve(x)
    array([ 1. , -3. , -1.5])
    >>> A.dot(B.solve(x))
    array([ 1.,  2.,  3.])
    >>> B.solve(A.dot(x))
    array([ 1.,  2.,  3.])
    """

    if is_pydata_spmatrix(A):
        A_cls = type(A)
        def csc_construct_func(*a, cls=A_cls):
            return cls.from_scipy_sparse(csc_array(*a))
        A = A.to_scipy_sparse().tocsc()
    else:
        csc_construct_func = csc_array

    if not (issparse(A) and A.format == "csc"):
        A = csc_array(A)
        warn('splu converted its input to CSC format',
             SparseEfficiencyWarning, stacklevel=2)

    # sum duplicates for non-canonical format
    A.sum_duplicates()
    A = A._asfptype()  # upcast to a floating point format

    M, N = A.shape
    if (M != N):
        raise ValueError("can only factor square matrices")  # is this true?

    indices, indptr = safely_cast_index_arrays(A, np.intc, "SuperLU")

    _options = dict(DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
                    PanelSize=panel_size, Relax=relax)
    if options is not None:
        _options.update(options)

    # Ensure that no column permutations are applied
    if (_options["ColPerm"] == "NATURAL"):
        _options["SymmetricMode"] = True

    return _superlu.gstrf(N, A.nnz, A.data, indices, indptr,
                          csc_construct_func=csc_construct_func,
                          ilu=False, options=_options)


def spilu(A, drop_tol=None, fill_factor=None, drop_rule=None, permc_spec=None,
          diag_pivot_thresh=None, relax=None, panel_size=None, options=None):
    """
    Compute an incomplete LU decomposition for a sparse, square matrix.

    The resulting object is an approximation to the inverse of `A`.

    Parameters
    ----------
    A : (N, N) array_like
        Sparse array to factorize. Most efficient when provided in CSC format.
        Other formats will be converted to CSC before factorization.
    drop_tol : float, optional
        Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
        (default: 1e-4)
    fill_factor : float, optional
        Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
    drop_rule : str, optional
        Comma-separated string of drop rules to use.
        Available rules: ``basic``, ``prows``, ``column``, ``area``,
        ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

        See SuperLU documentation for details.

    Remaining other options
        Same as for `splu`

    Returns
    -------
    invA_approx : scipy.sparse.linalg.SuperLU
        Object, which has a ``solve`` method.

    See also
    --------
    splu : complete LU decomposition

    Notes
    -----
    To improve the better approximation to the inverse, you may need to
    increase `fill_factor` AND decrease `drop_tol`.

    This function uses the SuperLU library.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import spilu
    >>> A = csc_array([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
    >>> B = spilu(A)
    >>> x = np.array([1., 2., 3.], dtype=float)
    >>> B.solve(x)
    array([ 1. , -3. , -1.5])
    >>> A.dot(B.solve(x))
    array([ 1.,  2.,  3.])
    >>> B.solve(A.dot(x))
    array([ 1.,  2.,  3.])
    """

    if is_pydata_spmatrix(A):
        A_cls = type(A)
        def csc_construct_func(*a, cls=A_cls):
            return cls.from_scipy_sparse(csc_array(*a))
        A = A.to_scipy_sparse().tocsc()
    else:
        csc_construct_func = csc_array

    if not (issparse(A) and A.format == "csc"):
        A = csc_array(A)
        warn('spilu converted its input to CSC format',
             SparseEfficiencyWarning, stacklevel=2)

    # sum duplicates for non-canonical format
    A.sum_duplicates()
    A = A._asfptype()  # upcast to a floating point format

    M, N = A.shape
    if (M != N):
        raise ValueError("can only factor square matrices")  # is this true?

    indices, indptr = safely_cast_index_arrays(A, np.intc, "SuperLU")

    _options = dict(ILU_DropRule=drop_rule, ILU_DropTol=drop_tol,
                    ILU_FillFactor=fill_factor,
                    DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
                    PanelSize=panel_size, Relax=relax)
    if options is not None:
        _options.update(options)

    # Ensure that no column permutations are applied
    if (_options["ColPerm"] == "NATURAL"):
        _options["SymmetricMode"] = True

    return _superlu.gstrf(N, A.nnz, A.data, indices, indptr,
                          csc_construct_func=csc_construct_func,
                          ilu=True, options=_options)


def factorized(A):
    """
    Return a function for solving a sparse linear system, with A pre-factorized.

    Parameters
    ----------
    A : (N, N) array_like
        Input. A in CSC format is most efficient. A CSR format matrix will
        be converted to CSC before factorization.

    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import factorized
    >>> from scipy.sparse import csc_array
    >>> A = np.array([[ 3. ,  2. , -1. ],
    ...               [ 2. , -2. ,  4. ],
    ...               [-1. ,  0.5, -1. ]])
    >>> solve = factorized(csc_array(A)) # Makes LU decomposition.
    >>> rhs1 = np.array([1, -2, 0])
    >>> solve(rhs1) # Uses the LU factors.
    array([ 1., -2., -2.])

    """
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    if not hasattr(useUmfpack, 'u'):
        useUmfpack.u = not noScikit

    if useUmfpack.u:
        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')

        if not (issparse(A) and A.format == "csc"):
            A = csc_array(A)
            warn('splu converted its input to CSC format',
                 SparseEfficiencyWarning, stacklevel=2)

        A = A._asfptype()  # upcast to a floating point format

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack.u = False")

        umf_family, A = _get_umf_family(A)
        umf = umfpack.UmfpackContext(umf_family)

        # Make LU decomposition.
        umf.numeric(A)

        def solve(b):
            with np.errstate(divide="ignore", invalid="ignore"):
                # Ignoring warnings with numpy >= 1.23.0, see gh-16523
                result = umf.solve(umfpack.UMFPACK_A, A, b, autoTranspose=True)

            return result

        return solve
    else:
        return splu(A).solve


def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False,
                       unit_diagonal=False):
    """
    Solve the equation ``A x = b`` for `x`, assuming A is a triangular matrix.

    Parameters
    ----------
    A : (M, M) sparse array or matrix
        A sparse square triangular matrix. Should be in CSR or CSC format.
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``A x = b``
    lower : bool, optional
        Whether `A` is a lower or upper triangular matrix.
        Default is lower triangular matrix.
    overwrite_A : bool, optional
        Allow changing `A`.
        Enabling gives a performance gain. Default is False.
    overwrite_b : bool, optional
        Allow overwriting data in `b`.
        Enabling gives a performance gain. Default is False.
        If `overwrite_b` is True, it should be ensured that
        `b` has an appropriate dtype to be able to store the result.
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1.

        .. versionadded:: 1.4.0

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``A x = b``. Shape of return matches shape
        of `b`.

    Raises
    ------
    LinAlgError
        If `A` is singular or not triangular.
    ValueError
        If shape of `A` or shape of `b` do not match the requirements.

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import spsolve_triangular
    >>> A = csc_array([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
    >>> B = np.array([[2, 0], [-1, 0], [2, 0]], dtype=float)
    >>> x = spsolve_triangular(A, B)
    >>> np.allclose(A.dot(x), B)
    True
    """

    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    trans = "N"
    if issparse(A) and A.format == "csr":
        A = A.T
        trans = "T"
        lower = not lower

    if not (issparse(A) and A.format == "csc"):
        warn('CSC or CSR matrix format is required. Converting to CSC matrix.',
             SparseEfficiencyWarning, stacklevel=2)
        A = csc_array(A)
    elif not overwrite_A:
        A = A.copy()


    M, N = A.shape
    if M != N:
        raise ValueError(
            f'A must be a square matrix but its shape is {A.shape}.')

    if unit_diagonal:
        with catch_warnings():
            simplefilter('ignore', SparseEfficiencyWarning)
            A.setdiag(1)
    else:
        diag = A.diagonal()
        if np.any(diag == 0):
            raise LinAlgError(
                'A is singular: zero entry on diagonal.')
        invdiag = 1/diag
        if trans == "N":
            A = A @ diags_array(invdiag)
        else:
            A = (A.T @ diags_array(invdiag)).T

    # sum duplicates for non-canonical format
    A.sum_duplicates()

    b = np.asanyarray(b)

    if b.ndim not in [1, 2]:
        raise ValueError(
            f'b must have 1 or 2 dims but its shape is {b.shape}.')
    if M != b.shape[0]:
        raise ValueError(
            'The size of the dimensions of A must be equal to '
            'the size of the first dimension of b but the shape of A is '
            f'{A.shape} and the shape of b is {b.shape}.'
        )

    result_dtype = np.promote_types(np.promote_types(A.dtype, np.float32), b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)
    elif not overwrite_b:
        b = b.copy()

    if lower:
        L = A
        U = csc_array((N, N), dtype=result_dtype)
    else:
        L = eye_array(N, dtype=result_dtype, format='csc')
        U = A
        U.setdiag(0)

    x, info = _superlu.gstrs(trans,
                             N, L.nnz, L.data, L.indices, L.indptr,
                             N, U.nnz, U.data, U.indices, U.indptr,
                             b)
    if info:
        raise LinAlgError('A is singular.')

    if not unit_diagonal:
        invdiag = invdiag.reshape(-1, *([1] * (len(x.shape) - 1)))
        x = x * invdiag

    return x


def is_sptriangular(A):
    """Returns 2-tuple indicating lower/upper triangular structure for sparse ``A``

    Checks for triangular structure in ``A``. The result is summarized in
    two boolean values ``lower`` and ``upper`` to designate whether ``A`` is
    lower triangular or upper triangular respectively. Diagonal ``A`` will
    result in both being True. Non-triangular structure results in False for both.

    Only the sparse structure is used here. Values are not checked for zeros.

    This function will convert a copy of ``A`` to CSC format if it is not already
    CSR or CSC format. So it may be more efficient to convert it yourself if you
    have other uses for the CSR/CSC version.

    If ``A`` is not square, the portions outside the upper left square of the
    matrix do not affect its triangular structure. You probably want to work
    with the square portion of the matrix, though it is not requred here.

    Parameters
    ----------
    A : SciPy sparse array or matrix
        A sparse matrix preferrably in CSR or CSC format.

    Returns
    -------
    lower, upper : 2-tuple of bool

        .. versionadded:: 1.15.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array, eye_array
    >>> from scipy.sparse.linalg import is_sptriangular
    >>> A = csc_array([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
    >>> is_sptriangular(A)
    (True, False)
    >>> D = eye_array(3, format='csr')
    >>> is_sptriangular(D)
    (True, True)
    """
    if not (issparse(A) and A.format in ("csc", "csr", "coo", "dia", "dok", "lil")):
        warn('is_sptriangular needs sparse and not BSR format. Converting to CSR.',
             SparseEfficiencyWarning, stacklevel=2)
        A = csr_array(A)

    # bsr is better off converting to csr
    if A.format == "dia":
        return A.offsets.max() <= 0, A.offsets.min() >= 0
    elif A.format == "coo":
        rows, cols = A.coords
        return (cols <= rows).all(), (cols >= rows).all()
    elif A.format == "dok":
        return all(c <= r for r, c in A.keys()), all(c >= r for r, c in A.keys())
    elif A.format == "lil":
        lower = all(col <= row for row, cols in enumerate(A.rows) for col in cols)
        upper = all(col >= row for row, cols in enumerate(A.rows) for col in cols)
        return lower, upper
    # format in ("csc", "csr")
    indptr, indices = A.indptr, A.indices
    N = len(indptr) - 1

    lower, upper = True, True
    # check middle, 1st, last col (treat as CSC and switch at end if CSR)
    for col in [N // 2, 0, -1]:
        rows = indices[indptr[col]:indptr[col + 1]]
        upper = upper and (col >= rows).all()
        lower = lower and (col <= rows).all()
        if not upper and not lower:
            return False, False
    # check all cols
    cols = np.repeat(np.arange(N), np.diff(indptr))
    rows = indices
    upper = upper and (cols >= rows).all()
    lower = lower and (cols <= rows).all()
    if A.format == 'csr':
        return upper, lower
    return lower, upper


def spbandwidth(A):
    """Return the lower and upper bandwidth of a 2D numeric array.

    Computes the lower and upper limits on the bandwidth of the
    sparse 2D array ``A``. The result is summarized as a 2-tuple
    of positive integers ``(lo, hi)``. A zero denotes no sub/super
    diagonal entries on that side (tringular). The maximum value
    for ``lo``(``hi``) is one less than the number of rows(cols).

    Only the sparse structure is used here. Values are not checked for zeros.

    Parameters
    ----------
    A : SciPy sparse array or matrix
        A sparse matrix preferrably in CSR or CSC format.

    Returns
    -------
    below, above : 2-tuple of int
        The distance to the farthest non-zero diagonal below/above the
        main diagonal.

        .. versionadded:: 1.15.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import spbandwidth
    >>> from scipy.sparse import csc_array, eye_array
    >>> A = csc_array([[3, 0, 0], [1, -1, 0], [2, 0, 1]], dtype=float)
    >>> spbandwidth(A)
    (2, 0)
    >>> D = eye_array(3, format='csr')
    >>> spbandwidth(D)
    (0, 0)
    """
    if not (issparse(A) and A.format in ("csc", "csr", "coo", "dia", "dok")):
        warn('spbandwidth needs sparse format not LIL and BSR. Converting to CSR.',
             SparseEfficiencyWarning, stacklevel=2)
        A = csr_array(A)

    # bsr and lil are better off converting to csr
    if A.format == "dia":
        return max(0, -A.offsets.min().item()), max(0, A.offsets.max().item())
    if A.format in ("csc", "csr"):
        indptr, indices = A.indptr, A.indices
        N = len(indptr) - 1
        gap = np.repeat(np.arange(N), np.diff(indptr)) - indices
        if A.format == 'csr':
            gap = -gap
    elif A.format == "coo":
        gap = A.coords[1] - A.coords[0]
    elif A.format == "dok":
        gap = [(c - r) for r, c in A.keys()] + [0]
        return -min(gap), max(gap)
    return max(-np.min(gap).item(), 0), max(np.max(gap).item(), 0)
<|endoftext|>def _svds_arpack_doc(A, k=6, ncv=None, tol=0, which='LM', v0=None,
                     maxiter=None, return_singular_vectors=True,
                     solver='arpack', rng=None):
    """
    Partial singular value decomposition of a sparse matrix using ARPACK.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Matrix to decompose.
    k : int, optional
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= min(M, N) - 1``.
        Default is 6.
    ncv : int, optional
        The number of Lanczos vectors generated.
        The default is ``min(n, max(2*k + 1, 20))``.
        If specified, must satisfy ``k + 1 < ncv < min(M, N)``; ``ncv > 2*k``
        is recommended.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        The starting vector for iteration:
        an (approximate) left singular vector if ``N > M`` and a right singular
        vector otherwise. Must be of length ``min(M, N)``.
        Default: random
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed;
        default is ``min(M, N) * 10``.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: if ``M <= N``, compute only the left singular vectors and
          return ``None`` for the right singular vectors. Otherwise, compute
          all singular vectors.
        - ``"vh"``: if ``M > N``, compute only the right singular vectors and
          return ``None`` for the left singular vectors. Otherwise, compute
          all singular vectors.

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            This is the solver-specific documentation for ``solver='arpack'``.
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>` and
            :ref:`'propack' <sparse.linalg.svds-propack>`
            are also supported.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
        are currently supported; this parameter is reserved for future use.

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
    s : ndarray, shape=(k,)
        The singular values.
    vh : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.

    Notes
    -----
    This is a naive implementation using ARPACK as an eigensolver
    on ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on which one is more
    efficient.

    Examples
    --------
    Construct a matrix ``A`` from singular values and vectors.

    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from scipy.sparse import csc_array, diags_array
    >>> from scipy.sparse.linalg import svds
    >>> rng = np.random.default_rng()
    >>> orthogonal = csc_array(ortho_group.rvs(10, random_state=rng))
    >>> s = [0.0001, 0.001, 3, 4, 5]  # singular values
    >>> u = orthogonal[:, :5]         # left singular vectors
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
    >>> A = u @ diags_array(s) @ vT

    With only three singular values/vectors, the SVD approximates the original
    matrix.

    >>> u2, s2, vT2 = svds(A, k=3, solver='arpack')
    >>> A2 = u2 @ np.diag(s2) @ vT2
    >>> np.allclose(A2, A.toarray(), atol=1e-3)
    True

    With all five singular values/vectors, we can reproduce the original
    matrix.

    >>> u3, s3, vT3 = svds(A, k=5, solver='arpack')
    >>> A3 = u3 @ np.diag(s3) @ vT3
    >>> np.allclose(A3, A.toarray())
    True

    The singular values match the expected singular values, and the singular
    vectors are as expected up to a difference in sign.

    >>> (np.allclose(s3, s) and
    ...  np.allclose(np.abs(u3), np.abs(u.toarray())) and
    ...  np.allclose(np.abs(vT3), np.abs(vT.toarray())))
    True

    The singular vectors are also orthogonal.

    >>> (np.allclose(u3.T @ u3, np.eye(5)) and
    ...  np.allclose(vT3 @ vT3.T, np.eye(5)))
    True
    """
    pass


def _svds_lobpcg_doc(A, k=6, ncv=None, tol=0, which='LM', v0=None,
                     maxiter=None, return_singular_vectors=True,
                     solver='lobpcg', rng=None):
    """
    Partial singular value decomposition of a sparse matrix using LOBPCG.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Matrix to decompose.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= min(M, N) - 1``.
    ncv : int, optional
        Ignored.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        If `k` is 1, the starting vector for iteration:
        an (approximate) left singular vector if ``N > M`` and a right singular
        vector otherwise. Must be of length ``min(M, N)``.
        Ignored otherwise.
        Default: random
    maxiter : int, default: 20
        Maximum number of iterations.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: if ``M <= N``, compute only the left singular vectors and
          return ``None`` for the right singular vectors. Otherwise, compute
          all singular vectors.
        - ``"vh"``: if ``M > N``, compute only the right singular vectors and
          return ``None`` for the left singular vectors. Otherwise, compute
          all singular vectors.

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            This is the solver-specific documentation for ``solver='lobpcg'``.
            :ref:`'arpack' <sparse.linalg.svds-arpack>` and
            :ref:`'propack' <sparse.linalg.svds-propack>`
            are also supported.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
        are currently supported; this parameter is reserved for future use.

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
    s : ndarray, shape=(k,)
        The singular values.
    vh : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.

    Notes
    -----
    This is a naive implementation using LOBPCG as an eigensolver
    on ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on which one is more
    efficient.

    Examples
    --------
    Construct a matrix ``A`` from singular values and vectors.

    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from scipy.sparse import csc_array, diags_array
    >>> from scipy.sparse.linalg import svds
    >>> rng = np.random.default_rng()
    >>> orthogonal = csc_array(ortho_group.rvs(10, random_state=rng))
    >>> s = [0.0001, 0.001, 3, 4, 5]  # singular values
    >>> u = orthogonal[:, :5]         # left singular vectors
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
    >>> A = u @ diags_array(s) @ vT

    With only three singular values/vectors, the SVD approximates the original
    matrix.

    >>> u2, s2, vT2 = svds(A, k=3, solver='lobpcg')
    >>> A2 = u2 @ np.diag(s2) @ vT2
    >>> np.allclose(A2, A.toarray(), atol=1e-3)
    True

    With all five singular values/vectors, we can reproduce the original
    matrix.

    >>> u3, s3, vT3 = svds(A, k=5, solver='lobpcg')
    >>> A3 = u3 @ np.diag(s3) @ vT3
    >>> np.allclose(A3, A.toarray())
    True

    The singular values match the expected singular values, and the singular
    vectors are as expected up to a difference in sign.

    >>> (np.allclose(s3, s) and
    ...  np.allclose(np.abs(u3), np.abs(u.todense())) and
    ...  np.allclose(np.abs(vT3), np.abs(vT.todense())))
    True

    The singular vectors are also orthogonal.

    >>> (np.allclose(u3.T @ u3, np.eye(5)) and
    ...  np.allclose(vT3 @ vT3.T, np.eye(5)))
    True

    """
    pass


def _svds_propack_doc(A, k=6, ncv=None, tol=0, which='LM', v0=None,
                      maxiter=None, return_singular_vectors=True,
                      solver='propack', rng=None):
    """
    Partial singular value decomposition of a sparse matrix using PROPACK.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Matrix to decompose. If `A` is a ``LinearOperator``
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= min(M, N)``.
    ncv : int, optional
        Ignored.
    tol : float, optional
        The desired relative accuracy for computed singular values.
        Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values. Note that choosing
        ``which='SM'`` will force the ``irl`` option to be set ``True``.
    v0 : ndarray, optional
        Starting vector for iterations: must be of length ``A.shape[0]``.
        If not specified, PROPACK will generate a starting vector.
    maxiter : int, optional
        Maximum number of iterations / maximal dimension of the Krylov
        subspace. Default is ``10 * k``.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: compute only the left singular vectors; return ``None`` for
          the right singular vectors.
        - ``"vh"``: compute only the right singular vectors; return ``None``
          for the left singular vectors.

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            This is the solver-specific documentation for ``solver='propack'``.
            :ref:`'arpack' <sparse.linalg.svds-arpack>` and
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`
            are also supported.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
        are currently supported; this parameter is reserved for future use.

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
    s : ndarray, shape=(k,)
        The singular values.
    vh : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.

    Notes
    -----
    This is an interface to the Fortran library PROPACK [1]_.
    The current default is to run with IRL mode disabled unless seeking the
    smallest singular values/vectors (``which='SM'``).

    References
    ----------

    .. [1] Larsen, Rasmus Munk. "PROPACK-Software for large and sparse SVD
       calculations." Available online. URL
       http://sun.stanford.edu/~rmunk/PROPACK (2004): 2008-2009.

    Examples
    --------
    Construct a matrix ``A`` from singular values and vectors.

    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from scipy.sparse import csc_array, diags_array
    >>> from scipy.sparse.linalg import svds
    >>> rng = np.random.default_rng()
    >>> orthogonal = csc_array(ortho_group.rvs(10, random_state=rng))
    >>> s = [0.0001, 0.001, 3, 4, 5]  # singular values
    >>> u = orthogonal[:, :5]         # left singular vectors
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
    >>> A = u @ diags_array(s) @ vT

    With only three singular values/vectors, the SVD approximates the original
    matrix.

    >>> u2, s2, vT2 = svds(A, k=3, solver='propack')
    >>> A2 = u2 @ np.diag(s2) @ vT2
    >>> np.allclose(A2, A.todense(), atol=1e-3)
    True

    With all five singular values/vectors, we can reproduce the original
    matrix.

    >>> u3, s3, vT3 = svds(A, k=5, solver='propack')
    >>> A3 = u3 @ np.diag(s3) @ vT3
    >>> np.allclose(A3, A.todense())
    True

    The singular values match the expected singular values, and the singular
    vectors are as expected up to a difference in sign.

    >>> (np.allclose(s3, s) and
    ...  np.allclose(np.abs(u3), np.abs(u.toarray())) and
    ...  np.allclose(np.abs(vT3), np.abs(vT.toarray())))
    True

    The singular vectors are also orthogonal.

    >>> (np.allclose(u3.T @ u3, np.eye(5)) and
    ...  np.allclose(vT3 @ vT3.T, np.eye(5)))
    True

    """
    pass
<|endoftext|>"""
Sparse Eigenvalue Solvers
-------------------------

The submodules of sparse.linalg._eigen:
    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method

"""
from .arpack import *
from .lobpcg import *
from ._svds import svds

from . import arpack

__all__ = [
    'ArpackError', 'ArpackNoConvergence',
    'eigs', 'eigsh', 'lobpcg', 'svds'
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>import math
import numpy as np

from .arpack import _arpack  # type: ignore[attr-defined]
from . import eigsh

from scipy._lib._util import check_random_state, _transition_to_rng
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg  # type: ignore[no-redef]
from scipy.sparse.linalg._svdp import _svdp
from scipy.linalg import svd

arpack_int = _arpack.timing.nbx.dtype
__all__ = ['svds']


def _herm(x):
    return x.T.conj()


def _iv(A, k, ncv, tol, which, v0, maxiter,
        return_singular, solver, rng):

    # input validation/standardization for `solver`
    # out of order because it's needed for other parameters
    solver = str(solver).lower()
    solvers = {"arpack", "lobpcg", "propack"}
    if solver not in solvers:
        raise ValueError(f"solver must be one of {solvers}.")

    # input validation/standardization for `A`
    A = aslinearoperator(A)  # this takes care of some input validation
    if not np.issubdtype(A.dtype, np.number):
        message = "`A` must be of numeric data type."
        raise ValueError(message)
    if math.prod(A.shape) == 0:
        message = "`A` must not be empty."
        raise ValueError(message)

    # input validation/standardization for `k`
    kmax = min(A.shape) if solver == 'propack' else min(A.shape) - 1
    if int(k) != k or not (0 < k <= kmax):
        message = "`k` must be an integer satisfying `0 < k < min(A.shape)`."
        raise ValueError(message)
    k = int(k)

    # input validation/standardization for `ncv`
    if solver == "arpack" and ncv is not None:
        if int(ncv) != ncv or not (k < ncv < min(A.shape)):
            message = ("`ncv` must be an integer satisfying "
                       "`k < ncv < min(A.shape)`.")
            raise ValueError(message)
        ncv = int(ncv)

    # input validation/standardization for `tol`
    if tol < 0 or not np.isfinite(tol):
        message = "`tol` must be a non-negative floating point value."
        raise ValueError(message)
    tol = float(tol)

    # input validation/standardization for `which`
    which = str(which).upper()
    whichs = {'LM', 'SM'}
    if which not in whichs:
        raise ValueError(f"`which` must be in {whichs}.")

    # input validation/standardization for `v0`
    if v0 is not None:
        v0 = np.atleast_1d(v0)
        if not (np.issubdtype(v0.dtype, np.complexfloating)
                or np.issubdtype(v0.dtype, np.floating)):
            message = ("`v0` must be of floating or complex floating "
                       "data type.")
            raise ValueError(message)

        shape = (A.shape[0],) if solver == 'propack' else (min(A.shape),)
        if v0.shape != shape:
            message = f"`v0` must have shape {shape}."
            raise ValueError(message)

    # input validation/standardization for `maxiter`
    if maxiter is not None and (int(maxiter) != maxiter or maxiter <= 0):
        message = "`maxiter` must be a positive integer."
        raise ValueError(message)
    maxiter = int(maxiter) if maxiter is not None else maxiter

    # input validation/standardization for `return_singular_vectors`
    # not going to be flexible with this; too complicated for little gain
    rs_options = {True, False, "vh", "u"}
    if return_singular not in rs_options:
        raise ValueError(f"`return_singular_vectors` must be in {rs_options}.")

    rng = check_random_state(rng)

    return (A, k, ncv, tol, which, v0, maxiter,
            return_singular, solver, rng)


@_transition_to_rng("random_state", position_num=9)
def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True,
         solver='arpack', rng=None, options=None):
    """
    Partial singular value decomposition of a sparse matrix.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : ndarray, sparse matrix, or LinearOperator
        Matrix to decompose of a floating point numeric dtype.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= kmax``, where ``kmax=min(M, N)`` for
        ``solver='propack'`` and ``kmax=min(M, N) - 1`` otherwise.
    ncv : int, optional
        When ``solver='arpack'``, this is the number of Lanczos vectors
        generated. See :ref:`'arpack' <sparse.linalg.svds-arpack>` for details.
        When ``solver='lobpcg'`` or ``solver='propack'``, this parameter is
        ignored.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        The starting vector for iteration; see method-specific
        documentation (:ref:`'arpack' <sparse.linalg.svds-arpack>`,
        :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`), or
        :ref:`'propack' <sparse.linalg.svds-propack>` for details.
    maxiter : int, optional
        Maximum number of iterations; see method-specific
        documentation (:ref:`'arpack' <sparse.linalg.svds-arpack>`,
        :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`), or
        :ref:`'propack' <sparse.linalg.svds-propack>` for details.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: if ``M <= N``, compute only the left singular vectors and
          return ``None`` for the right singular vectors. Otherwise, compute
          all singular vectors.
        - ``"vh"``: if ``M > N``, compute only the right singular vectors and
          return ``None`` for the left singular vectors. Otherwise, compute
          all singular vectors.

        If ``solver='propack'``, the option is respected regardless of the
        matrix shape.

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            The solver used.
            :ref:`'arpack' <sparse.linalg.svds-arpack>`,
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`, and
            :ref:`'propack' <sparse.linalg.svds-propack>` are supported.
            Default: `'arpack'`.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
        are currently supported; this parameter is reserved for future use.

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
    s : ndarray, shape=(k,)
        The singular values.
    vh : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.

    Notes
    -----
    This is a naive implementation using ARPACK or LOBPCG as an eigensolver
    on the matrix ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on
    which one is smaller size, followed by the Rayleigh-Ritz method
    as postprocessing; see
    Using the normal matrix, in Rayleigh-Ritz method, (2022, Nov. 19),
    Wikipedia, https://w.wiki/4zms.

    Alternatively, the PROPACK solver can be called.

    Choices of the input matrix `A` numeric dtype may be limited.
    Only ``solver="lobpcg"`` supports all floating point dtypes
    real: 'np.float32', 'np.float64', 'np.longdouble' and
    complex: 'np.complex64', 'np.complex128', 'np.clongdouble'.
    The ``solver="arpack"`` supports only
    'np.float32', 'np.float64', and 'np.complex128'.

    Examples
    --------
    Construct a matrix `A` from singular values and vectors.

    >>> import numpy as np
    >>> from scipy import sparse, linalg, stats
    >>> from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator

    Construct a dense matrix `A` from singular values and vectors.

    >>> rng = np.random.default_rng(258265244568965474821194062361901728911)
    >>> orthogonal = stats.ortho_group.rvs(10, random_state=rng)
    >>> s = [1e-3, 1, 2, 3, 4]  # non-zero singular values
    >>> u = orthogonal[:, :5]         # left singular vectors
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
    >>> A = u @ np.diag(s) @ vT

    With only four singular values/vectors, the SVD approximates the original
    matrix.

    >>> u4, s4, vT4 = svds(A, k=4)
    >>> A4 = u4 @ np.diag(s4) @ vT4
    >>> np.allclose(A4, A, atol=1e-3)
    True

    With all five non-zero singular values/vectors, we can reproduce
    the original matrix more accurately.

    >>> u5, s5, vT5 = svds(A, k=5)
    >>> A5 = u5 @ np.diag(s5) @ vT5
    >>> np.allclose(A5, A)
    True

    The singular values match the expected singular values.

    >>> np.allclose(s5, s)
    True

    Since the singular values are not close to each other in this example,
    every singular vector matches as expected up to a difference in sign.

    >>> (np.allclose(np.abs(u5), np.abs(u)) and
    ...  np.allclose(np.abs(vT5), np.abs(vT)))
    True

    The singular vectors are also orthogonal.

    >>> (np.allclose(u5.T @ u5, np.eye(5)) and
    ...  np.allclose(vT5 @ vT5.T, np.eye(5)))
    True

    If there are (nearly) multiple singular values, the corresponding
    individual singular vectors may be unstable, but the whole invariant
    subspace containing all such singular vectors is computed accurately
    as can be measured by angles between subspaces via 'subspace_angles'.

    >>> rng = np.random.default_rng(178686584221410808734965903901790843963)
    >>> s = [1, 1 + 1e-6]  # non-zero singular values
    >>> u, _ = np.linalg.qr(rng.standard_normal((99, 2)))
    >>> v, _ = np.linalg.qr(rng.standard_normal((99, 2)))
    >>> vT = v.T
    >>> A = u @ np.diag(s) @ vT
    >>> A = A.astype(np.float32)
    >>> u2, s2, vT2 = svds(A, k=2, rng=rng)
    >>> np.allclose(s2, s)
    True

    The angles between the individual exact and computed singular vectors
    may not be so small. To check use:

    >>> (linalg.subspace_angles(u2[:, :1], u[:, :1]) +
    ...  linalg.subspace_angles(u2[:, 1:], u[:, 1:]))
    array([0.06562513])  # may vary
    >>> (linalg.subspace_angles(vT2[:1, :].T, vT[:1, :].T) +
    ...  linalg.subspace_angles(vT2[1:, :].T, vT[1:, :].T))
    array([0.06562507])  # may vary

    As opposed to the angles between the 2-dimensional invariant subspaces
    that these vectors span, which are small for rights singular vectors

    >>> linalg.subspace_angles(u2, u).sum() < 1e-6
    True

    as well as for left singular vectors.

    >>> linalg.subspace_angles(vT2.T, vT.T).sum() < 1e-6
    True

    The next example follows that of 'sklearn.decomposition.TruncatedSVD'.

    >>> rng = np.random.default_rng(0)
    >>> X_dense = rng.random(size=(100, 100))
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = sparse.csr_array(X_dense)
    >>> _, singular_values, _ = svds(X, k=5, rng=rng)
    >>> print(singular_values)
    [ 4.3221...  4.4043...  4.4907...  4.5858... 35.4549...]

    The function can be called without the transpose of the input matrix
    ever explicitly constructed.

    >>> rng = np.random.default_rng(102524723947864966825913730119128190974)
    >>> G = sparse.random_array((8, 9), density=0.5, rng=rng)
    >>> Glo = aslinearoperator(G)
    >>> _, singular_values_svds, _ = svds(Glo, k=5, rng=rng)
    >>> _, singular_values_svd, _ = linalg.svd(G.toarray())
    >>> np.allclose(singular_values_svds, singular_values_svd[-4::-1])
    True

    The most memory efficient scenario is where neither
    the original matrix, nor its transpose, is explicitly constructed.
    Our example computes the smallest singular values and vectors
    of 'LinearOperator' constructed from the numpy function 'np.diff' used
    column-wise to be consistent with 'LinearOperator' operating on columns.

    >>> diff0 = lambda a: np.diff(a, axis=0)

    Let us create the matrix from 'diff0' to be used for validation only.

    >>> n = 5  # The dimension of the space.
    >>> M_from_diff0 = diff0(np.eye(n))
    >>> print(M_from_diff0.astype(int))
    [[-1  1  0  0  0]
     [ 0 -1  1  0  0]
     [ 0  0 -1  1  0]
     [ 0  0  0 -1  1]]

    The matrix 'M_from_diff0' is bi-diagonal and could be alternatively
    created directly by

    >>> M = - np.eye(n - 1, n, dtype=int)
    >>> np.fill_diagonal(M[:,1:], 1)
    >>> np.allclose(M, M_from_diff0)
    True

    Its transpose

    >>> print(M.T)
    [[-1  0  0  0]
     [ 1 -1  0  0]
     [ 0  1 -1  0]
     [ 0  0  1 -1]
     [ 0  0  0  1]]

    can be viewed as the incidence matrix; see
    Incidence matrix, (2022, Nov. 19), Wikipedia, https://w.wiki/5YXU,
    of a linear graph with 5 vertices and 4 edges. The 5x5 normal matrix
    ``M.T @ M`` thus is

    >>> print(M.T @ M)
    [[ 1 -1  0  0  0]
     [-1  2 -1  0  0]
     [ 0 -1  2 -1  0]
     [ 0  0 -1  2 -1]
     [ 0  0  0 -1  1]]

    the graph Laplacian, while the actually used in 'svds' smaller size
    4x4 normal matrix ``M @ M.T``

    >>> print(M @ M.T)
    [[ 2 -1  0  0]
     [-1  2 -1  0]
     [ 0 -1  2 -1]
     [ 0  0 -1  2]]

    is the so-called edge-based Laplacian; see
    Symmetric Laplacian via the incidence matrix, in Laplacian matrix,
    (2022, Nov. 19), Wikipedia, https://w.wiki/5YXW.

    The 'LinearOperator' setup needs the options 'rmatvec' and 'rmatmat'
    of multiplication by the matrix transpose ``M.T``, but we want to be
    matrix-free to save memory, so knowing how ``M.T`` looks like, we
    manually construct the following function to be
    used in ``rmatmat=diff0t``.

    >>> def diff0t(a):
    ...     if a.ndim == 1:
    ...         a = a[:,np.newaxis]  # Turn 1D into 2D array
    ...     d = np.zeros((a.shape[0] + 1, a.shape[1]), dtype=a.dtype)
    ...     d[0, :] = - a[0, :]
    ...     d[1:-1, :] = a[0:-1, :] - a[1:, :]
    ...     d[-1, :] = a[-1, :]
    ...     return d

    We check that our function 'diff0t' for the matrix transpose is valid.

    >>> np.allclose(M.T, diff0t(np.eye(n-1)))
    True

    Now we setup our matrix-free 'LinearOperator' called 'diff0_func_aslo'
    and for validation the matrix-based 'diff0_matrix_aslo'.

    >>> def diff0_func_aslo_def(n):
    ...     return LinearOperator(matvec=diff0,
    ...                           matmat=diff0,
    ...                           rmatvec=diff0t,
    ...                           rmatmat=diff0t,
    ...                           shape=(n - 1, n))
    >>> diff0_func_aslo = diff0_func_aslo_def(n)
    >>> diff0_matrix_aslo = aslinearoperator(M_from_diff0)

    And validate both the matrix and its transpose in 'LinearOperator'.

    >>> np.allclose(diff0_func_aslo(np.eye(n)),
    ...             diff0_matrix_aslo(np.eye(n)))
    True
    >>> np.allclose(diff0_func_aslo.T(np.eye(n-1)),
    ...             diff0_matrix_aslo.T(np.eye(n-1)))
    True

    Having the 'LinearOperator' setup validated, we run the solver.

    >>> n = 100
    >>> diff0_func_aslo = diff0_func_aslo_def(n)
    >>> u, s, vT = svds(diff0_func_aslo, k=3, which='SM')

    The singular values squared and the singular vectors are known
    explicitly; see
    Pure Dirichlet boundary conditions, in
    Eigenvalues and eigenvectors of the second derivative,
    (2022, Nov. 19), Wikipedia, https://w.wiki/5YX6,
    since 'diff' corresponds to first
    derivative, and its smaller size n-1 x n-1 normal matrix
    ``M @ M.T`` represent the discrete second derivative with the Dirichlet
    boundary conditions. We use these analytic expressions for validation.

    >>> se = 2. * np.sin(np.pi * np.arange(1, 4) / (2. * n))
    >>> ue = np.sqrt(2 / n) * np.sin(np.pi * np.outer(np.arange(1, n),
    ...                              np.arange(1, 4)) / n)
    >>> np.allclose(s, se, atol=1e-3)
    True
    >>> np.allclose(np.abs(u), np.abs(ue), atol=1e-6)
    True

    """
    args = _iv(A, k, ncv, tol, which, v0, maxiter, return_singular_vectors,
               solver, rng)
    (A, k, ncv, tol, which, v0, maxiter,
     return_singular_vectors, solver, rng) = args

    largest = (which == 'LM')
    n, m = A.shape

    if n >= m:
        X_dot = A.matvec
        X_matmat = A.matmat
        XH_dot = A.rmatvec
        XH_mat = A.rmatmat
        transpose = False
    else:
        X_dot = A.rmatvec
        X_matmat = A.rmatmat
        XH_dot = A.matvec
        XH_mat = A.matmat
        transpose = True

        dtype = getattr(A, 'dtype', None)
        if dtype is None:
            dtype = A.dot(np.zeros([m, 1])).dtype

    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        return XH_mat(X_matmat(x))

    XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                          matmat=matmat_XH_X,
                          shape=(min(A.shape), min(A.shape)))

    # Get a low rank approximation of the implicitly defined gramian matrix.
    # This is not a stable way to approach the problem.
    if solver == 'lobpcg':

        if k == 1 and v0 is not None:
            X = np.reshape(v0, (-1, 1))
        else:
            X = rng.standard_normal(size=(min(A.shape), k))

        _, eigvec = lobpcg(XH_X, X, tol=tol ** 2, maxiter=maxiter,
                           largest=largest)

    elif solver == 'propack':
        jobu = return_singular_vectors in {True, 'u'}
        jobv = return_singular_vectors in {True, 'vh'}
        irl_mode = (which == 'SM')
        res = _svdp(A, k=k, tol=tol**2, which=which, maxiter=None,
                    compute_u=jobu, compute_v=jobv, irl_mode=irl_mode,
                    kmax=maxiter, v0=v0, rng=rng)

        u, s, vh, _ = res  # but we'll ignore bnd, the last output

        # PROPACK order appears to be largest first. `svds` output order is not
        # guaranteed, according to documentation, but for ARPACK and LOBPCG
        # they actually are ordered smallest to largest, so reverse for
        # consistency.
        s = s[::-1]
        u = u[:, ::-1]
        vh = vh[::-1]

        u = u if jobu else None
        vh = vh if jobv else None

        if return_singular_vectors:
            return u, s, vh
        else:
            return s

    elif solver == 'arpack' or solver is None:
        if v0 is None:
            v0 = rng.standard_normal(size=(min(A.shape),))
        _, eigvec = eigsh(XH_X, k=k, tol=tol ** 2, maxiter=maxiter,
                          ncv=ncv, which=which, v0=v0)
        # arpack do not guarantee exactly orthonormal eigenvectors
        # for clustered eigenvalues, especially in complex arithmetic
        eigvec, _ = np.linalg.qr(eigvec)

    # the eigenvectors eigvec must be orthonomal here; see gh-16712
    Av = X_matmat(eigvec)
    if not return_singular_vectors:
        s = svd(Av, compute_uv=False, overwrite_a=True)
        return s[::-1]

    # compute the left singular vectors of X and update the right ones
    # accordingly
    u, s, vh = svd(Av, full_matrices=False, overwrite_a=True)
    u = u[:, ::-1]
    s = s[::-1]
    vh = vh[::-1]

    jobu = return_singular_vectors in {True, 'u'}
    jobv = return_singular_vectors in {True, 'vh'}

    if transpose:
        u_tmp = eigvec @ _herm(vh) if jobu else None
        vh = _herm(u) if jobv else None
        u = u_tmp
    else:
        if not jobu:
            u = None
        vh = vh @ _herm(eigvec) if jobv else None

    return u, s, vh
<|endoftext|>"""
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).

References
----------
.. [1] A. V. Knyazev (2001),
       Toward the Optimal Preconditioned Eigensolver: Locally Optimal
       Block Preconditioned Conjugate Gradient Method.
       SIAM Journal on Scientific Computing 23, no. 2,
       pp. 517-541. :doi:`10.1137/S1064827500366124`

.. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),
       Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)
       in hypre and PETSc.  :arxiv:`0705.2626`

.. [3] A. V. Knyazev's C and MATLAB implementations:
       https://github.com/lobpcg/blopex
"""

import warnings
import numpy as np
from scipy.linalg import (inv, eigh, cho_factor, cho_solve,
                          cholesky, LinAlgError)
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse

__all__ = ["lobpcg"]


def _report_nonhermitian(M, name):
    """
    Report if `M` is not a Hermitian matrix given its type.
    """
    from scipy.linalg import norm

    md = M - M.T.conj()
    nmd = norm(md, 1)
    tol = 10 * np.finfo(M.dtype).eps
    tol = max(tol, tol * norm(M, 1))
    if nmd > tol:
        warnings.warn(
              f"Matrix {name} of the type {M.dtype} is not Hermitian: "
              f"condition: {nmd} < {tol} fails.",
              UserWarning, stacklevel=4
         )

def _as2d(ar):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
    if ar.ndim == 2:
        return ar
    else:  # Assume 1!
        aux = np.asarray(ar)
        aux.shape = (ar.shape[0], 1)
        return aux


def _makeMatMat(m):
    if m is None:
        return None
    elif callable(m):
        return lambda v: m(v)
    else:
        return lambda v: m @ v


def _matmul_inplace(x, y, verbosityLevel=0):
    """Perform 'np.matmul' in-place if possible.

    If some sufficient conditions for inplace matmul are met, do so.
    Otherwise try inplace update and fall back to overwrite if that fails.
    """
    if x.flags["CARRAY"] and x.shape[1] == y.shape[1] and x.dtype == y.dtype:
        # conditions where we can guarantee that inplace updates will work;
        # i.e. x is not a view/slice, x & y have compatible dtypes, and the
        # shape of the result of x @ y matches the shape of x.
        np.matmul(x, y, out=x)
    else:
        # ideally, we'd have an exhaustive list of conditions above when
        # inplace updates are possible; since we don't, we opportunistically
        # try if it works, and fall back to overwriting if necessary
        try:
            np.matmul(x, y, out=x)
        except Exception:
            if verbosityLevel:
                warnings.warn(
                    "Inplace update of x = x @ y failed, "
                    "x needs to be overwritten.",
                    UserWarning, stacklevel=3
                )
            x = x @ y
    return x


def _applyConstraints(blockVectorV, factYBY, blockVectorBY, blockVectorY):
    """Changes blockVectorV in-place."""
    YBV = blockVectorBY.T.conj() @ blockVectorV
    tmp = cho_solve(factYBY, YBV)
    blockVectorV -= blockVectorY @ tmp


def _b_orthonormalize(B, blockVectorV, blockVectorBV=None,
                      verbosityLevel=0):
    """in-place B-orthonormalize the given block vector using Cholesky."""
    if blockVectorBV is None:
        if B is None:
            blockVectorBV = blockVectorV
        else:
            try:
                blockVectorBV = B(blockVectorV)
            except Exception as e:
                if verbosityLevel:
                    warnings.warn(
                        f"Secondary MatMul call failed with error\n"
                        f"{e}\n",
                        UserWarning, stacklevel=3
                    )
                    return None, None, None
            if blockVectorBV.shape != blockVectorV.shape:
                raise ValueError(
                    f"The shape {blockVectorV.shape} "
                    f"of the orthogonalized matrix not preserved\n"
                    f"and changed to {blockVectorBV.shape} "
                    f"after multiplying by the secondary matrix.\n"
                )

    VBV = blockVectorV.T.conj() @ blockVectorBV
    try:
        # VBV is a Cholesky factor from now on...
        VBV = cholesky(VBV, overwrite_a=True)
        VBV = inv(VBV, overwrite_a=True)
        blockVectorV = _matmul_inplace(
            blockVectorV, VBV,
            verbosityLevel=verbosityLevel
        )
        if B is not None:
            blockVectorBV = _matmul_inplace(
                blockVectorBV, VBV,
                verbosityLevel=verbosityLevel
            )
        return blockVectorV, blockVectorBV, VBV
    except LinAlgError:
        if verbosityLevel:
            warnings.warn(
                "Cholesky has failed.",
                UserWarning, stacklevel=3
            )
        return None, None, None


def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    ii = np.argsort(_lambda)
    if largest:
        ii = ii[:-num - 1:-1]
    else:
        ii = ii[:num]

    return ii


def _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel):
    if verbosityLevel:
        _report_nonhermitian(gramA, "gramA")
        _report_nonhermitian(gramB, "gramB")


def lobpcg(
    A,
    X,
    B=None,
    M=None,
    Y=None,
    tol=None,
    maxiter=None,
    largest=True,
    verbosityLevel=0,
    retLambdaHistory=False,
    retResidualNormsHistory=False,
    restartControl=20,
):
    """Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).

    LOBPCG is a preconditioned eigensolver for large real symmetric and complex
    Hermitian definite generalized eigenproblems.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator, callable object}
        The Hermitian linear operator of the problem, usually given by a
        sparse matrix.  Often called the "stiffness matrix".
    X : ndarray, float32 or float64
        Initial approximation to the ``k`` eigenvectors (non-sparse).
        If `A` has ``shape=(n,n)`` then `X` must have ``shape=(n,k)``.
    B : {sparse matrix, ndarray, LinearOperator, callable object}
        Optional. By default ``B = None``, which is equivalent to identity.
        The right hand side operator in a generalized eigenproblem if present.
        Often called the "mass matrix". Must be Hermitian positive definite.
    M : {sparse matrix, ndarray, LinearOperator, callable object}
        Optional. By default ``M = None``, which is equivalent to identity.
        Preconditioner aiming to accelerate convergence.
    Y : ndarray, float32 or float64, default: None
        An ``n-by-sizeY`` ndarray of constraints with ``sizeY < n``.
        The iterations will be performed in the ``B``-orthogonal complement
        of the column-space of `Y`. `Y` must be full rank if present.
    tol : scalar, optional
        The default is ``tol=n*sqrt(eps)``.
        Solver tolerance for the stopping criterion.
    maxiter : int, default: 20
        Maximum number of iterations.
    largest : bool, default: True
        When True, solve for the largest eigenvalues, otherwise the smallest.
    verbosityLevel : int, optional
        By default ``verbosityLevel=0`` no output.
        Controls the solver standard/screen output.
    retLambdaHistory : bool, default: False
        Whether to return iterative eigenvalue history.
    retResidualNormsHistory : bool, default: False
        Whether to return iterative history of residual norms.
    restartControl : int, optional.
        Iterations restart if the residuals jump ``2**restartControl`` times
        compared to the smallest recorded in ``retResidualNormsHistory``.
        The default is ``restartControl=20``, making the restarts rare for
        backward compatibility.

    Returns
    -------
    lambda : ndarray of the shape ``(k, )``.
        Array of ``k`` approximate eigenvalues.
    v : ndarray of the same shape as ``X.shape``.
        An array of ``k`` approximate eigenvectors.
    lambdaHistory : ndarray, optional.
        The eigenvalue history, if `retLambdaHistory` is ``True``.
    ResidualNormsHistory : ndarray, optional.
        The history of residual norms, if `retResidualNormsHistory`
        is ``True``.

    Notes
    -----
    The iterative loop runs ``maxit=maxiter`` (20 if ``maxit=None``)
    iterations at most and finishes earlier if the tolerance is met.
    Breaking backward compatibility with the previous version, LOBPCG
    now returns the block of iterative vectors with the best accuracy rather
    than the last one iterated, as a cure for possible divergence.

    If ``X.dtype == np.float32`` and user-provided operations/multiplications
    by `A`, `B`, and `M` all preserve the ``np.float32`` data type,
    all the calculations and the output are in ``np.float32``.

    The size of the iteration history output equals to the number of the best
    (limited by `maxit`) iterations plus 3: initial, final, and postprocessing.

    If both `retLambdaHistory` and `retResidualNormsHistory` are ``True``,
    the return tuple has the following format
    ``(lambda, V, lambda history, residual norms history)``.

    In the following ``n`` denotes the matrix size and ``k`` the number
    of required eigenvalues (smallest or largest).

    The LOBPCG code internally solves eigenproblems of the size ``3k`` on every
    iteration by calling the dense eigensolver `eigh`, so if ``k`` is not
    small enough compared to ``n``, it makes no sense to call the LOBPCG code.
    Moreover, if one calls the LOBPCG algorithm for ``5k > n``, it would likely
    break internally, so the code calls the standard function `eigh` instead.
    It is not that ``n`` should be large for the LOBPCG to work, but rather the
    ratio ``n / k`` should be large. It you call LOBPCG with ``k=1``
    and ``n=10``, it works though ``n`` is small. The method is intended
    for extremely large ``n / k``.

    The convergence speed depends basically on three factors:

    1. Quality of the initial approximations `X` to the seeking eigenvectors.
       Randomly distributed around the origin vectors work well if no better
       choice is known.

    2. Relative separation of the desired eigenvalues from the rest
       of the eigenvalues. One can vary ``k`` to improve the separation.

    3. Proper preconditioning to shrink the spectral spread.
       For example, a rod vibration test problem (under tests
       directory) is ill-conditioned for large ``n``, so convergence will be
       slow, unless efficient preconditioning is used. For this specific
       problem, a good simple preconditioner function would be a linear solve
       for `A`, which is easy to code since `A` is tridiagonal.

    References
    ----------
    .. [1] A. V. Knyazev (2001),
           Toward the Optimal Preconditioned Eigensolver: Locally Optimal
           Block Preconditioned Conjugate Gradient Method.
           SIAM Journal on Scientific Computing 23, no. 2,
           pp. 517-541. :doi:`10.1137/S1064827500366124`

    .. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov
           (2007), Block Locally Optimal Preconditioned Eigenvalue Xolvers
           (BLOPEX) in hypre and PETSc. :arxiv:`0705.2626`

    .. [3] A. V. Knyazev's C and MATLAB implementations:
           https://github.com/lobpcg/blopex

    Examples
    --------
    Our first example is minimalistic - find the largest eigenvalue of
    a diagonal matrix by solving the non-generalized eigenvalue problem
    ``A x = lambda x`` without constraints or preconditioning.

    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from scipy.sparse.linalg import LinearOperator, aslinearoperator
    >>> from scipy.sparse.linalg import lobpcg

    The square matrix size is

    >>> n = 100

    and its diagonal entries are 1, ..., 100 defined by

    >>> vals = np.arange(1, n + 1).astype(np.int16)

    The first mandatory input parameter in this test is
    the sparse diagonal matrix `A`
    of the eigenvalue problem ``A x = lambda x`` to solve.

    >>> A = spdiags(vals, 0, n, n)
    >>> A = A.astype(np.int16)
    >>> A.toarray()
    array([[  1,   0,   0, ...,   0,   0,   0],
           [  0,   2,   0, ...,   0,   0,   0],
           [  0,   0,   3, ...,   0,   0,   0],
           ...,
           [  0,   0,   0, ...,  98,   0,   0],
           [  0,   0,   0, ...,   0,  99,   0],
           [  0,   0,   0, ...,   0,   0, 100]], shape=(100, 100), dtype=int16)

    The second mandatory input parameter `X` is a 2D array with the
    row dimension determining the number of requested eigenvalues.
    `X` is an initial guess for targeted eigenvectors.
    `X` must have linearly independent columns.
    If no initial approximations available, randomly oriented vectors
    commonly work best, e.g., with components normally distributed
    around zero or uniformly distributed on the interval [-1 1].
    Setting the initial approximations to dtype ``np.float32``
    forces all iterative values to dtype ``np.float32`` speeding up
    the run while still allowing accurate eigenvalue computations.

    >>> k = 1
    >>> rng = np.random.default_rng()
    >>> X = rng.normal(size=(n, k))
    >>> X = X.astype(np.float32)

    >>> eigenvalues, _ = lobpcg(A, X, maxiter=60)
    >>> eigenvalues
    array([100.], dtype=float32)

    `lobpcg` needs only access the matrix product with `A` rather
    then the matrix itself. Since the matrix `A` is diagonal in
    this example, one can write a function of the matrix product
    ``A @ X`` using the diagonal values ``vals`` only, e.g., by
    element-wise multiplication with broadcasting in the lambda-function

    >>> A_lambda = lambda X: vals[:, np.newaxis] * X

    or the regular function

    >>> def A_matmat(X):
    ...     return vals[:, np.newaxis] * X

    and use the handle to one of these callables as an input

    >>> eigenvalues, _ = lobpcg(A_lambda, X, maxiter=60)
    >>> eigenvalues
    array([100.], dtype=float32)
    >>> eigenvalues, _ = lobpcg(A_matmat, X, maxiter=60)
    >>> eigenvalues
    array([100.], dtype=float32)

    The traditional callable `LinearOperator` is no longer
    necessary but still supported as the input to `lobpcg`.
    Specifying ``matmat=A_matmat`` explicitly improves performance. 

    >>> A_lo = LinearOperator((n, n), matvec=A_matmat, matmat=A_matmat, dtype=np.int16)
    >>> eigenvalues, _ = lobpcg(A_lo, X, maxiter=80)
    >>> eigenvalues
    array([100.], dtype=float32)

    The least efficient callable option is `aslinearoperator`:

    >>> eigenvalues, _ = lobpcg(aslinearoperator(A), X, maxiter=80)
    >>> eigenvalues
    array([100.], dtype=float32)

    We now switch to computing the three smallest eigenvalues specifying

    >>> k = 3
    >>> X = np.random.default_rng().normal(size=(n, k))

    and ``largest=False`` parameter

    >>> eigenvalues, _ = lobpcg(A, X, largest=False, maxiter=90)
    >>> print(eigenvalues)  
    [1. 2. 3.]

    The next example illustrates computing 3 smallest eigenvalues of
    the same matrix `A` given by the function handle ``A_matmat`` but
    with constraints and preconditioning.

    Constraints - an optional input parameter is a 2D array comprising
    of column vectors that the eigenvectors must be orthogonal to

    >>> Y = np.eye(n, 3)

    The preconditioner acts as the inverse of `A` in this example, but
    in the reduced precision ``np.float32`` even though the initial `X`
    and thus all iterates and the output are in full ``np.float64``.

    >>> inv_vals = 1./vals
    >>> inv_vals = inv_vals.astype(np.float32)
    >>> M = lambda X: inv_vals[:, np.newaxis] * X

    Let us now solve the eigenvalue problem for the matrix `A` first
    without preconditioning requesting 80 iterations

    >>> eigenvalues, _ = lobpcg(A_matmat, X, Y=Y, largest=False, maxiter=80)
    >>> eigenvalues
    array([4., 5., 6.])
    >>> eigenvalues.dtype
    dtype('float64')

    With preconditioning we need only 20 iterations from the same `X`

    >>> eigenvalues, _ = lobpcg(A_matmat, X, Y=Y, M=M, largest=False, maxiter=20)
    >>> eigenvalues
    array([4., 5., 6.])

    Note that the vectors passed in `Y` are the eigenvectors of the 3
    smallest eigenvalues. The results returned above are orthogonal to those.

    The primary matrix `A` may be indefinite, e.g., after shifting
    ``vals`` by 50 from 1, ..., 100 to -49, ..., 50, we still can compute
    the 3 smallest or largest eigenvalues.

    >>> vals = vals - 50
    >>> X = rng.normal(size=(n, k))
    >>> eigenvalues, _ = lobpcg(A_matmat, X, largest=False, maxiter=99)
    >>> eigenvalues
    array([-49., -48., -47.])
    >>> eigenvalues, _ = lobpcg(A_matmat, X, largest=True, maxiter=99)
    >>> eigenvalues
    array([50., 49., 48.])

    """
    blockVectorX = X
    bestblockVectorX = blockVectorX
    blockVectorY = Y
    residualTolerance = tol
    if maxiter is None:
        maxiter = 20

    bestIterationNumber = maxiter

    sizeY = 0
    if blockVectorY is not None:
        if len(blockVectorY.shape) != 2:
            warnings.warn(
                f"Expected rank-2 array for argument Y, instead got "
                f"{len(blockVectorY.shape)}, "
                f"so ignore it and use no constraints.",
                UserWarning, stacklevel=2
            )
            blockVectorY = None
        else:
            sizeY = blockVectorY.shape[1]

    # Block size.
    if blockVectorX is None:
        raise ValueError("The mandatory initial matrix X cannot be None")
    if len(blockVectorX.shape) != 2:
        raise ValueError("expected rank-2 array for argument X")

    n, sizeX = blockVectorX.shape

    # Data type of iterates, determined by X, must be inexact
    if not np.issubdtype(blockVectorX.dtype, np.inexact):
        warnings.warn(
            f"Data type for argument X is {blockVectorX.dtype}, "
            f"which is not inexact, so casted to np.float32.",
            UserWarning, stacklevel=2
        )
        blockVectorX = np.asarray(blockVectorX, dtype=np.float32)

    if retLambdaHistory:
        lambdaHistory = np.zeros((maxiter + 3, sizeX),
                                 dtype=blockVectorX.dtype)
    if retResidualNormsHistory:
        residualNormsHistory = np.zeros((maxiter + 3, sizeX),
                                        dtype=blockVectorX.dtype)

    if verbosityLevel:
        aux = "Solving "
        if B is None:
            aux += "standard"
        else:
            aux += "generalized"
        aux += " eigenvalue problem with"
        if M is None:
            aux += "out"
        aux += " preconditioning\n\n"
        aux += "matrix size %d\n" % n
        aux += "block size %d\n\n" % sizeX
        if blockVectorY is None:
            aux += "No constraints\n\n"
        else:
            if sizeY > 1:
                aux += "%d constraints\n\n" % sizeY
            else:
                aux += "%d constraint\n\n" % sizeY
        print(aux)

    if (n - sizeY) < (5 * sizeX):
        warnings.warn(
            f"The problem size {n} minus the constraints size {sizeY} "
            f"is too small relative to the block size {sizeX}. "
            f"Using a dense eigensolver instead of LOBPCG iterations."
            f"No output of the history of the iterations.",
            UserWarning, stacklevel=2
        )

        sizeX = min(sizeX, n)

        if blockVectorY is not None:
            raise NotImplementedError(
                "The dense eigensolver does not support constraints."
            )

        # Define the closed range of indices of eigenvalues to return.
        if largest:
            eigvals = (n - sizeX, n - 1)
        else:
            eigvals = (0, sizeX - 1)

        try:
            if isinstance(A, LinearOperator):
                A = A(np.eye(n, dtype=int))
            elif callable(A):
                A = A(np.eye(n, dtype=int))
                if A.shape != (n, n):
                    raise ValueError(
                        f"The shape {A.shape} of the primary matrix\n"
                        f"defined by a callable object is wrong.\n"
                    )
            elif issparse(A):
                A = A.toarray()
            else:
                A = np.asarray(A)
        except Exception as e:
            raise Exception(
                f"Primary MatMul call failed with error\n"
                f"{e}\n")

        if B is not None:
            try:
                if isinstance(B, LinearOperator):
                    B = B(np.eye(n, dtype=int))
                elif callable(B):
                    B = B(np.eye(n, dtype=int))
                    if B.shape != (n, n):
                        raise ValueError(
                            f"The shape {B.shape} of the secondary matrix\n"
                            f"defined by a callable object is wrong.\n"
                        )
                elif issparse(B):
                    B = B.toarray()
                else:
                    B = np.asarray(B)
            except Exception as e:
                raise Exception(
                    f"Secondary MatMul call failed with error\n"
                    f"{e}\n")

        try:
            vals, vecs = eigh(A,
                              B,
                              subset_by_index=eigvals,
                              check_finite=False)
            if largest:
                # Reverse order to be compatible with eigs() in 'LM' mode.
                vals = vals[::-1]
                vecs = vecs[:, ::-1]

            return vals, vecs
        except Exception as e:
            raise Exception(
                f"Dense eigensolver failed with error\n"
                f"{e}\n"
            )

    if (residualTolerance is None) or (residualTolerance <= 0.0):
        residualTolerance = np.sqrt(np.finfo(blockVectorX.dtype).eps) * n

    A = _makeMatMat(A)
    B = _makeMatMat(B)
    M = _makeMatMat(M)

    # Apply constraints to X.
    if blockVectorY is not None:

        if B is not None:
            blockVectorBY = B(blockVectorY)
            if blockVectorBY.shape != blockVectorY.shape:
                raise ValueError(
                    f"The shape {blockVectorY.shape} "
                    f"of the constraint not preserved\n"
                    f"and changed to {blockVectorBY.shape} "
                    f"after multiplying by the secondary matrix.\n"
                )
        else:
            blockVectorBY = blockVectorY

        # gramYBY is a dense array.
        gramYBY = blockVectorY.T.conj() @ blockVectorBY
        try:
            # gramYBY is a Cholesky factor from now on...
            gramYBY = cho_factor(gramYBY, overwrite_a=True)
        except LinAlgError as e:
            raise ValueError("Linearly dependent constraints") from e

        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)

    ##
    # B-orthonormalize X.
    blockVectorX, blockVectorBX, _ = _b_orthonormalize(
        B, blockVectorX, verbosityLevel=verbosityLevel)
    if blockVectorX is None:
        raise ValueError("Linearly dependent initial approximations")

    ##
    # Compute the initial Ritz vectors: solve the eigenproblem.
    blockVectorAX = A(blockVectorX)
    if blockVectorAX.shape != blockVectorX.shape:
        raise ValueError(
            f"The shape {blockVectorX.shape} "
            f"of the initial approximations not preserved\n"
            f"and changed to {blockVectorAX.shape} "
            f"after multiplying by the primary matrix.\n"
        )

    gramXAX = blockVectorX.T.conj() @ blockVectorAX

    _lambda, eigBlockVector = eigh(gramXAX, check_finite=False)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    if retLambdaHistory:
        lambdaHistory[0, :] = _lambda

    eigBlockVector = np.asarray(eigBlockVector[:, ii])
    blockVectorX = _matmul_inplace(
        blockVectorX, eigBlockVector,
        verbosityLevel=verbosityLevel
    )
    blockVectorAX = _matmul_inplace(
        blockVectorAX, eigBlockVector,
        verbosityLevel=verbosityLevel
    )
    if B is not None:
        blockVectorBX = _matmul_inplace(
            blockVectorBX, eigBlockVector,
            verbosityLevel=verbosityLevel
        )

    ##
    # Active index set.
    activeMask = np.ones((sizeX,), dtype=bool)

    ##
    # Main iteration loop.

    blockVectorP = None  # set during iteration
    blockVectorAP = None
    blockVectorBP = None

    smallestResidualNorm = np.abs(np.finfo(blockVectorX.dtype).max)

    iterationNumber = -1
    restart = True
    forcedRestart = False
    explicitGramFlag = False
    while iterationNumber < maxiter:
        iterationNumber += 1

        if B is not None:
            aux = blockVectorBX * _lambda[np.newaxis, :]
        else:
            aux = blockVectorX * _lambda[np.newaxis, :]

        blockVectorR = blockVectorAX - aux

        aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
        residualNorms = np.sqrt(np.abs(aux))
        if retResidualNormsHistory:
            residualNormsHistory[iterationNumber, :] = residualNorms
        residualNorm = np.sum(np.abs(residualNorms)) / sizeX

        if residualNorm < smallestResidualNorm:
            smallestResidualNorm = residualNorm
            bestIterationNumber = iterationNumber
            bestblockVectorX = blockVectorX
        elif residualNorm > 2**restartControl * smallestResidualNorm:
            forcedRestart = True
            blockVectorAX = A(blockVectorX)
            if blockVectorAX.shape != blockVectorX.shape:
                raise ValueError(
                    f"The shape {blockVectorX.shape} "
                    f"of the restarted iterate not preserved\n"
                    f"and changed to {blockVectorAX.shape} "
                    f"after multiplying by the primary matrix.\n"
                )
            if B is not None:
                blockVectorBX = B(blockVectorX)
                if blockVectorBX.shape != blockVectorX.shape:
                    raise ValueError(
                        f"The shape {blockVectorX.shape} "
                        f"of the restarted iterate not preserved\n"
                        f"and changed to {blockVectorBX.shape} "
                        f"after multiplying by the secondary matrix.\n"
                    )

        ii = np.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii
        currentBlockSize = activeMask.sum()

        if verbosityLevel:
            print(f"iteration {iterationNumber}")
            print(f"current block size: {currentBlockSize}")
            print(f"eigenvalue(s):\n{_lambda}")
            print(f"residual norm(s):\n{residualNorms}")

        if currentBlockSize == 0:
            break

        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])

        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])

        if M is not None:
            # Apply preconditioner T to the active residuals.
            activeBlockVectorR = M(activeBlockVectorR)

        ##
        # Apply constraints to the preconditioned residuals.
        if blockVectorY is not None:
            _applyConstraints(activeBlockVectorR,
                              gramYBY,
                              blockVectorBY,
                              blockVectorY)

        ##
        # B-orthogonalize the preconditioned residuals to X.
        if B is not None:
            activeBlockVectorR = activeBlockVectorR - (
                blockVectorX @
                (blockVectorBX.T.conj() @ activeBlockVectorR)
            )
        else:
            activeBlockVectorR = activeBlockVectorR - (
                blockVectorX @
                (blockVectorX.T.conj() @ activeBlockVectorR)
            )

        ##
        # B-orthonormalize the preconditioned residuals.
        aux = _b_orthonormalize(
            B, activeBlockVectorR, verbosityLevel=verbosityLevel)
        activeBlockVectorR, activeBlockVectorBR, _ = aux

        if activeBlockVectorR is None:
            warnings.warn(
                f"Failed at iteration {iterationNumber} with accuracies "
                f"{residualNorms}\n not reaching the requested "
                f"tolerance {residualTolerance}.",
                UserWarning, stacklevel=2
            )
            break
        activeBlockVectorAR = A(activeBlockVectorR)

        if iterationNumber > 0:
            if B is not None:
                aux = _b_orthonormalize(
                    B, activeBlockVectorP, activeBlockVectorBP,
                    verbosityLevel=verbosityLevel
                )
                activeBlockVectorP, activeBlockVectorBP, invR = aux
            else:
                aux = _b_orthonormalize(B, activeBlockVectorP,
                                        verbosityLevel=verbosityLevel)
                activeBlockVectorP, _, invR = aux
            # Function _b_orthonormalize returns None if Cholesky fails
            if activeBlockVectorP is not None:
                activeBlockVectorAP = _matmul_inplace(
                    activeBlockVectorAP, invR,
                    verbosityLevel=verbosityLevel
                )
                restart = forcedRestart
            else:
                restart = True

        ##
        # Perform the Rayleigh Ritz Procedure:
        # Compute symmetric Gram matrices:

        if activeBlockVectorAR.dtype == "float32":
            myeps = 1
        else:
            myeps = np.sqrt(np.finfo(activeBlockVectorR.dtype).eps)

        if residualNorms.max() > myeps and not explicitGramFlag:
            explicitGramFlag = False
        else:
            # Once explicitGramFlag, forever explicitGramFlag.
            explicitGramFlag = True

        # Shared memory assignments to simplify the code
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP

        # Common submatrices:
        gramXAR = np.dot(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorAR)

        gramDtype = activeBlockVectorAR.dtype
        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = np.dot(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = np.dot(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorBR)
            gramXBR = np.dot(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = np.diag(_lambda).astype(gramDtype)
            gramXBX = np.eye(sizeX, dtype=gramDtype)
            gramRBR = np.eye(currentBlockSize, dtype=gramDtype)
            gramXBR = np.zeros((sizeX, currentBlockSize), dtype=gramDtype)

        if not restart:
            gramXAP = np.dot(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorAP)
            gramPAP = np.dot(activeBlockVectorP.T.conj(), activeBlockVectorAP)
            gramXBP = np.dot(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = np.dot(activeBlockVectorR.T.conj(), activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = np.dot(activeBlockVectorP.T.conj(),
                                 activeBlockVectorBP)
            else:
                gramPBP = np.eye(currentBlockSize, dtype=gramDtype)

            gramA = np.block(
                [
                    [gramXAX, gramXAR, gramXAP],
                    [gramXAR.T.conj(), gramRAR, gramRAP],
                    [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP],
                ]
            )
            gramB = np.block(
                [
                    [gramXBX, gramXBR, gramXBP],
                    [gramXBR.T.conj(), gramRBR, gramRBP],
                    [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP],
                ]
            )

            _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel)

            try:
                _lambda, eigBlockVector = eigh(gramA,
                                               gramB,
                                               check_finite=False)
            except LinAlgError as e:
                # raise ValueError("eigh failed in lobpcg iterations") from e
                if verbosityLevel:
                    warnings.warn(
                        f"eigh failed at iteration {iterationNumber} \n"
                        f"with error {e} causing a restart.\n",
                        UserWarning, stacklevel=2
                    )
                # try again after dropping the direction vectors P from RR
                restart = True

        if restart:
            gramA = np.block([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = np.block([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])

            _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel)

            try:
                _lambda, eigBlockVector = eigh(gramA,
                                               gramB,
                                               check_finite=False)
            except LinAlgError as e:
                # raise ValueError("eigh failed in lobpcg iterations") from e
                warnings.warn(
                    f"eigh failed at iteration {iterationNumber} with error\n"
                    f"{e}\n",
                    UserWarning, stacklevel=2
                )
                break

        ii = _get_indx(_lambda, sizeX, largest)
        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]
        if retLambdaHistory:
            lambdaHistory[iterationNumber + 1, :] = _lambda

        # Compute Ritz vectors.
        if B is not None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:
                                                 sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                pp += np.dot(activeBlockVectorP, eigBlockVectorP)

                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
                app += np.dot(activeBlockVectorAP, eigBlockVectorP)

                bpp = np.dot(activeBlockVectorBR, eigBlockVectorR)
                bpp += np.dot(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
                bpp = np.dot(activeBlockVectorBR, eigBlockVectorR)

            blockVectorX = np.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = np.dot(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = np.dot(blockVectorBX, eigBlockVectorX) + bpp

            blockVectorP, blockVectorAP, blockVectorBP = pp, app, bpp

        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:
                                                 sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]

                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                pp += np.dot(activeBlockVectorP, eigBlockVectorP)

                app = np.dot(activeBlockVectorAR, eigBlockVectorR)
                app += np.dot(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]

                pp = np.dot(activeBlockVectorR, eigBlockVectorR)
                app = np.dot(activeBlockVectorAR, eigBlockVectorR)

            blockVectorX = np.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = np.dot(blockVectorAX, eigBlockVectorX) + app

            blockVectorP, blockVectorAP = pp, app

    if B is not None:
        aux = blockVectorBX * _lambda[np.newaxis, :]
    else:
        aux = blockVectorX * _lambda[np.newaxis, :]

    blockVectorR = blockVectorAX - aux

    aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = np.sqrt(np.abs(aux))
    # Use old lambda in case of early loop exit.
    if retLambdaHistory:
        lambdaHistory[iterationNumber + 1, :] = _lambda
    if retResidualNormsHistory:
        residualNormsHistory[iterationNumber + 1, :] = residualNorms
    residualNorm = np.sum(np.abs(residualNorms)) / sizeX
    if residualNorm < smallestResidualNorm:
        smallestResidualNorm = residualNorm
        bestIterationNumber = iterationNumber + 1
        bestblockVectorX = blockVectorX

    if np.max(np.abs(residualNorms)) > residualTolerance:
        warnings.warn(
            f"Exited at iteration {iterationNumber} with accuracies \n"
            f"{residualNorms}\n"
            f"not reaching the requested tolerance {residualTolerance}.\n"
            f"Use iteration {bestIterationNumber} instead with accuracy \n"
            f"{smallestResidualNorm}.\n",
            UserWarning, stacklevel=2
        )

    if verbosityLevel:
        print(f"Final iterative eigenvalue(s):\n{_lambda}")
        print(f"Final iterative residual norm(s):\n{residualNorms}")

    blockVectorX = bestblockVectorX
    # Making eigenvectors "exactly" satisfy the blockVectorY constrains
    if blockVectorY is not None:
        _applyConstraints(blockVectorX,
                          gramYBY,
                          blockVectorBY,
                          blockVectorY)

    # Making eigenvectors "exactly" othonormalized by final "exact" RR
    blockVectorAX = A(blockVectorX)
    if blockVectorAX.shape != blockVectorX.shape:
        raise ValueError(
            f"The shape {blockVectorX.shape} "
            f"of the postprocessing iterate not preserved\n"
            f"and changed to {blockVectorAX.shape} "
            f"after multiplying by the primary matrix.\n"
        )
    gramXAX = np.dot(blockVectorX.T.conj(), blockVectorAX)

    blockVectorBX = blockVectorX
    if B is not None:
        blockVectorBX = B(blockVectorX)
        if blockVectorBX.shape != blockVectorX.shape:
            raise ValueError(
                f"The shape {blockVectorX.shape} "
                f"of the postprocessing iterate not preserved\n"
                f"and changed to {blockVectorBX.shape} "
                f"after multiplying by the secondary matrix.\n"
            )

    gramXBX = np.dot(blockVectorX.T.conj(), blockVectorBX)
    _handle_gramA_gramB_verbosity(gramXAX, gramXBX, verbosityLevel)
    gramXAX = (gramXAX + gramXAX.T.conj()) / 2
    gramXBX = (gramXBX + gramXBX.T.conj()) / 2
    try:
        _lambda, eigBlockVector = eigh(gramXAX,
                                       gramXBX,
                                       check_finite=False)
    except LinAlgError as e:
        raise ValueError("eigh has failed in lobpcg postprocessing") from e

    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    eigBlockVector = np.asarray(eigBlockVector[:, ii])

    blockVectorX = np.dot(blockVectorX, eigBlockVector)
    blockVectorAX = np.dot(blockVectorAX, eigBlockVector)

    if B is not None:
        blockVectorBX = np.dot(blockVectorBX, eigBlockVector)
        aux = blockVectorBX * _lambda[np.newaxis, :]
    else:
        aux = blockVectorX * _lambda[np.newaxis, :]

    blockVectorR = blockVectorAX - aux

    aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = np.sqrt(np.abs(aux))

    if retLambdaHistory:
        lambdaHistory[bestIterationNumber + 1, :] = _lambda
    if retResidualNormsHistory:
        residualNormsHistory[bestIterationNumber + 1, :] = residualNorms

    if retLambdaHistory:
        lambdaHistory = lambdaHistory[
            : bestIterationNumber + 2, :]
    if retResidualNormsHistory:
        residualNormsHistory = residualNormsHistory[
            : bestIterationNumber + 2, :]

    if np.max(np.abs(residualNorms)) > residualTolerance:
        warnings.warn(
            f"Exited postprocessing with accuracies \n"
            f"{residualNorms}\n"
            f"not reaching the requested tolerance {residualTolerance}.",
            UserWarning, stacklevel=2
        )

    if verbosityLevel:
        print(f"Final postprocessing eigenvalue(s):\n{_lambda}")
        print(f"Final residual norm(s):\n{residualNorms}")

    if retLambdaHistory:
        lambdaHistory = np.vsplit(lambdaHistory, np.shape(lambdaHistory)[0])
        lambdaHistory = [np.squeeze(i) for i in lambdaHistory]
    if retResidualNormsHistory:
        residualNormsHistory = np.vsplit(residualNormsHistory,
                                         np.shape(residualNormsHistory)[0])
        residualNormsHistory = [np.squeeze(i) for i in residualNormsHistory]

    if retLambdaHistory:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else:
            return _lambda, blockVectorX, lambdaHistory
    else:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, residualNormsHistory
        else:
            return _lambda, blockVectorX
<|endoftext|>"""
Find a few eigenvectors and eigenvalues of a matrix.


Uses ARPACK: https://github.com/opencollab/arpack-ng

"""
# Wrapper implementation notes
#
# ARPACK Entry Points
# -------------------
# The entry points to ARPACK are
# - (s,d)seupd : single and double precision symmetric matrix
# - (s,d,c,z)neupd: single,double,complex,double complex general matrix
# This wrapper puts the *neupd (general matrix) interfaces in eigs()
# and the *seupd (symmetric matrix) in eigsh().
# There is no specialized interface for complex Hermitian matrices.
# To find eigenvalues of a complex Hermitian matrix you
# may use eigsh(), but eigsh() will simply call eigs()
# and return the real part of the eigenvalues thus obtained.

# Number of eigenvalues returned and complex eigenvalues
# ------------------------------------------------------
# The ARPACK nonsymmetric real and double interface (s,d)naupd return
# eigenvalues and eigenvectors in real (float,double) arrays.
# Since the eigenvalues and eigenvectors are, in general, complex
# ARPACK puts the real and imaginary parts in consecutive entries
# in real-valued arrays.   This wrapper puts the real entries
# into complex data types and attempts to return the requested eigenvalues
# and eigenvectors.


# Solver modes
# ------------
# ARPACK and handle shifted and shift-inverse computations
# for eigenvalues by providing a shift (sigma) and a solver.

import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import (
    convert_pydata_sparse_to_scipy, isdense, is_pydata_spmatrix,
)
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock
from . import _arpack
arpack_int = _arpack.timing.nbx.dtype

__docformat__ = "restructuredtext en"

__all__ = ['eigs', 'eigsh', 'ArpackError', 'ArpackNoConvergence']


_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}
_ndigits = {'f': 5, 'd': 12, 'F': 5, 'D': 12}

DNAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found. IPARAM(5) "
       "returns the number of wanted converged Ritz values.",
    2: "No longer an informational error. Deprecated starting "
       "with release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the "
       "Implicitly restarted Arnoldi iteration. One possibility "
       "is to increase the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 2 and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: " WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from LAPACK eigenvalue calculation;",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible.",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated."
}

SNAUPD_ERRORS = DNAUPD_ERRORS

ZNAUPD_ERRORS = DNAUPD_ERRORS.copy()
ZNAUPD_ERRORS[-10] = "IPARAM(7) must be 1,2,3."

CNAUPD_ERRORS = ZNAUPD_ERRORS

DSAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found.",
    2: "No longer an informational error. Deprecated starting with "
       "release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the Implicitly "
       "restarted Arnoldi iteration. One possibility is to increase "
       "the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from trid. eigenvalue calculation; "
        "Informational error from LAPACK routine dsteqr .",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible. ",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated.",
}

SSAUPD_ERRORS = DSAUPD_ERRORS

DNEUPD_ERRORS = {
    0: "Normal exit.",
    1: "The Schur form computed by LAPACK routine dlahqr "
       "could not be reordered by LAPACK routine dtrsen. "
       "Re-enter subroutine dneupd  with IPARAM(5)NCV and "
       "increase the size of the arrays DR and DI to have "
       "dimension at least dimension NCV and allocate at least NCV "
       "columns for Z. NOTE: Not necessary if Z and V share "
       "the same space. Please notify the authors if this error"
       "occurs.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 2 and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: "Error return from calculation of a real Schur form. "
        "Informational error from LAPACK routine dlahqr .",
    -9: "Error return from calculation of eigenvectors. "
        "Informational error from LAPACK routine dtrevc.",
    -10: "IPARAM(7) must be 1,2,3,4.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "HOWMNY = 'S' not yet implemented",
    -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
    -14: "DNAUPD  did not find any eigenvalues to sufficient "
         "accuracy.",
    -15: "DNEUPD got a different count of the number of converged "
         "Ritz values than DNAUPD got.  This indicates the user "
         "probably made an error in passing data from DNAUPD to "
         "DNEUPD or that the data was modified before entering "
         "DNEUPD",
}

SNEUPD_ERRORS = DNEUPD_ERRORS.copy()
SNEUPD_ERRORS[1] = ("The Schur form computed by LAPACK routine slahqr "
                    "could not be reordered by LAPACK routine strsen . "
                    "Re-enter subroutine dneupd  with IPARAM(5)=NCV and "
                    "increase the size of the arrays DR and DI to have "
                    "dimension at least dimension NCV and allocate at least "
                    "NCV columns for Z. NOTE: Not necessary if Z and V share "
                    "the same space. Please notify the authors if this error "
                    "occurs.")
SNEUPD_ERRORS[-14] = ("SNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.")
SNEUPD_ERRORS[-15] = ("SNEUPD got a different count of the number of "
                      "converged Ritz values than SNAUPD got.  This indicates "
                      "the user probably made an error in passing data from "
                      "SNAUPD to SNEUPD or that the data was modified before "
                      "entering SNEUPD")

ZNEUPD_ERRORS = {0: "Normal exit.",
                 1: "The Schur form computed by LAPACK routine csheqr "
                    "could not be reordered by LAPACK routine ztrsen. "
                    "Re-enter subroutine zneupd with IPARAM(5)=NCV and "
                    "increase the size of the array D to have "
                    "dimension at least dimension NCV and allocate at least "
                    "NCV columns for Z. NOTE: Not necessary if Z and V share "
                    "the same space. Please notify the authors if this error "
                    "occurs.",
                 -1: "N must be positive.",
                 -2: "NEV must be positive.",
                 -3: "NCV-NEV >= 1 and less than or equal to N.",
                 -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
                 -6: "BMAT must be one of 'I' or 'G'.",
                 -7: "Length of private work WORKL array is not sufficient.",
                 -8: "Error return from LAPACK eigenvalue calculation. "
                     "This should never happened.",
                 -9: "Error return from calculation of eigenvectors. "
                     "Informational error from LAPACK routine ztrevc.",
                 -10: "IPARAM(7) must be 1,2,3",
                 -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
                 -12: "HOWMNY = 'S' not yet implemented",
                 -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
                 -14: "ZNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.",
                 -15: "ZNEUPD got a different count of the number of "
                      "converged Ritz values than ZNAUPD got.  This "
                      "indicates the user probably made an error in passing "
                      "data from ZNAUPD to ZNEUPD or that the data was "
                      "modified before entering ZNEUPD"
                 }

CNEUPD_ERRORS = ZNEUPD_ERRORS.copy()
CNEUPD_ERRORS[-14] = ("CNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.")
CNEUPD_ERRORS[-15] = ("CNEUPD got a different count of the number of "
                      "converged Ritz values than CNAUPD got.  This indicates "
                      "the user probably made an error in passing data from "
                      "CNAUPD to CNEUPD or that the data was modified before "
                      "entering CNEUPD")

DSEUPD_ERRORS = {
    0: "Normal exit.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: ("Error return from trid. eigenvalue calculation; "
         "Information error from LAPACK routine dsteqr."),
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "NEV and WHICH = 'BE' are incompatible.",
    -14: "DSAUPD  did not find any eigenvalues to sufficient accuracy.",
    -15: "HOWMNY must be one of 'A' or 'S' if RVEC = .true.",
    -16: "HOWMNY = 'S' not yet implemented",
    -17: ("DSEUPD  got a different count of the number of converged "
          "Ritz values than DSAUPD  got.  This indicates the user "
          "probably made an error in passing data from DSAUPD  to "
          "DSEUPD  or that the data was modified before entering  "
          "DSEUPD.")
}

SSEUPD_ERRORS = DSEUPD_ERRORS.copy()
SSEUPD_ERRORS[-14] = ("SSAUPD  did not find any eigenvalues "
                      "to sufficient accuracy.")
SSEUPD_ERRORS[-17] = ("SSEUPD  got a different count of the number of "
                      "converged "
                      "Ritz values than SSAUPD  got.  This indicates the user "
                      "probably made an error in passing data from SSAUPD  to "
                      "SSEUPD  or that the data was modified before entering  "
                      "SSEUPD.")

_SAUPD_ERRORS = {'d': DSAUPD_ERRORS,
                 's': SSAUPD_ERRORS}
_NAUPD_ERRORS = {'d': DNAUPD_ERRORS,
                 's': SNAUPD_ERRORS,
                 'z': ZNAUPD_ERRORS,
                 'c': CNAUPD_ERRORS}
_SEUPD_ERRORS = {'d': DSEUPD_ERRORS,
                 's': SSEUPD_ERRORS}
_NEUPD_ERRORS = {'d': DNEUPD_ERRORS,
                 's': SNEUPD_ERRORS,
                 'z': ZNEUPD_ERRORS,
                 'c': CNEUPD_ERRORS}

# accepted values of parameter WHICH in _SEUPD
_SEUPD_WHICH = ['LM', 'SM', 'LA', 'SA', 'BE']

# accepted values of parameter WHICH in _NAUPD
_NEUPD_WHICH = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']


class ArpackError(RuntimeError):
    """
    ARPACK error
    """

    def __init__(self, info, infodict=None):
        if infodict is None:
            infodict = _NAUPD_ERRORS

        msg = infodict.get(info, "Unknown error")
        super().__init__(f"ARPACK error {info}: {msg}")


class ArpackNoConvergence(ArpackError):
    """
    ARPACK iteration did not converge

    Attributes
    ----------
    eigenvalues : ndarray
        Partial result. Converged eigenvalues.
    eigenvectors : ndarray
        Partial result. Converged eigenvectors.

    """

    def __init__(self, msg, eigenvalues, eigenvectors):
        ArpackError.__init__(self, -1, {-1: msg})
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors


def choose_ncv(k):
    """
    Choose number of lanczos vectors based on target number
    of singular/eigen values and vectors to compute, k.
    """
    return max(2 * k + 1, 20)


class _ArpackParams:
    def __init__(self, n, k, tp, mode=1, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        if k <= 0:
            raise ValueError("k must be positive, k=%d" % k)

        if maxiter is None:
            maxiter = n * 10
        if maxiter <= 0:
            raise ValueError("maxiter must be positive, maxiter=%d" % maxiter)

        if tp not in 'fdFD':
            # Use `float64` libraries from integer dtypes.
            if np.can_cast(tp, 'd'):
                tp = 'd'
            else:
                raise ValueError("matrix type must be 'f', 'd', 'F', or 'D'")

        if v0 is not None:
            # ARPACK overwrites its initial resid,  make a copy
            self.resid = np.array(v0, copy=True)
            info = 1
        else:
            # ARPACK will use a random initial vector.
            self.resid = np.zeros(n, tp)
            info = 0

        if sigma is None:
            #sigma not used
            self.sigma = 0
        else:
            self.sigma = sigma

        if ncv is None:
            ncv = choose_ncv(k)
        ncv = min(ncv, n)

        self.v = np.zeros((n, ncv), tp)  # holds Ritz vectors
        self.iparam = np.zeros(11, arpack_int)

        # set solver mode and parameters
        ishfts = 1
        self.mode = mode
        self.iparam[0] = ishfts
        self.iparam[2] = maxiter
        self.iparam[3] = 1
        self.iparam[6] = mode

        self.n = n
        self.tol = tol
        self.k = k
        self.maxiter = maxiter
        self.ncv = ncv
        self.which = which
        self.tp = tp
        self.info = info

        self.converged = False
        self.ido = 0

    def _raise_no_convergence(self):
        msg = "No convergence (%d iterations, %d/%d eigenvectors converged)"
        k_ok = self.iparam[4]
        num_iter = self.iparam[2]
        try:
            ev, vec = self.extract(True)
        except ArpackError as err:
            msg = f"{msg} [{err}]"
            ev = np.zeros((0,))
            vec = np.zeros((self.n, 0))
            k_ok = 0
        raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)


class _SymmetricArpackParams(_ArpackParams):
    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None,
                 Minv_matvec=None, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        # The following modes are supported:
        #  mode = 1:
        #    Solve the standard eigenvalue problem:
        #      A*x = lambda*x :
        #       A - symmetric
