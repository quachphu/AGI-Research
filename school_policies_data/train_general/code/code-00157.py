        else:
            nrows, ncols = a.shape
            a_row, a_col = np.divmod(np.arange(nrows*ncols), ncols)
            row.append(a_row + r_idx)
            col.append(a_col + c_idx)
            data.append(a.ravel())
        r_idx += nrows
        c_idx += ncols
    idx_dtype = get_index_dtype(idx_arrays, maxval=max(r_idx, c_idx))
    row = np.concatenate(row, dtype=idx_dtype)
    col = np.concatenate(col, dtype=idx_dtype)
    data = np.concatenate(data)
    new_shape = (r_idx, c_idx)

    return container((data, (row, col)), shape=new_shape, dtype=dtype).asformat(format)


@_transition_to_rng("random_state")
def random_array(shape, *, density=0.01, format='coo', dtype=None,
                 rng=None, data_sampler=None):
    """Return a sparse array of uniformly random numbers in [0, 1)

    Returns a sparse array with the given shape and density
    where values are generated uniformly randomly in the range [0, 1).

    Parameters
    ----------
    shape : int or tuple of ints
        shape of the array
    density : real, optional (default: 0.01)
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional (default: 'coo')
        sparse matrix format.
    dtype : dtype, optional (default: np.float64)
        type of the returned matrix values.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        This random state will be used for sampling `indices` (the sparsity
        structure), and by default for the data values too (see `data_sampler`).
    data_sampler : callable, optional (default depends on dtype)
        Sampler of random data values with keyword arg `size`.
        This function should take a single keyword argument `size` specifying
        the length of its returned ndarray. It is used to generate the nonzero
        values in the matrix after the locations of those values are chosen.
        By default, uniform [0, 1) random values are used unless `dtype` is
        an integer (default uniform integers from that dtype) or
        complex (default uniform over the unit square in the complex plane).
        For these, the `rng` is used e.g. ``rng.uniform(size=size)``.

    Returns
    -------
    res : sparse array

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import numpy as np
    >>> import scipy as sp
    >>> rng = np.random.default_rng()

    Default sampling uniformly from [0, 1):

    >>> S = sp.sparse.random_array((3, 4), density=0.25, rng=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            rng=rng, data_sampler=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, rng=rng):
    ...     return rng.standard_normal(size) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, rng=rng,
    ...                            data_sampler=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, rng=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=rng) ** 2
    >>> S = sp.sparse.random_array((3, 4), density=0.25, rng=rng,
    ...                            data_sampler=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continuous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return rng.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X().rvs
    >>> S = sp.sparse.random_array((3, 4), density=0.25,
    ...                            rng=rng, data_sampler=Y)
    """
    data, ind = _random(shape, density, format, dtype, rng, data_sampler)

    # downcast, if safe, before calling coo_constructor
    idx_dtype = get_index_dtype(maxval=max(shape))
    ind = tuple(np.asarray(co, dtype=idx_dtype) for co in ind)
    return coo_array((data, ind), shape=shape).asformat(format)


def _random(shape, density=0.01, format=None, dtype=None,
            rng=None, data_sampler=None):
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")

    tot_prod = math.prod(shape)  # use `math` for when prod is >= 2**64

    # Number of non zero values
    size = int(round(density * tot_prod))

    rng = check_random_state(rng)

    if data_sampler is None:
        if np.issubdtype(dtype, np.integer):
            def data_sampler(size):
                return rng_integers(rng,
                                    np.iinfo(dtype).min,
                                    np.iinfo(dtype).max,
                                    size,
                                    dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            def data_sampler(size):
                return (rng.uniform(size=size) +
                        rng.uniform(size=size) * 1j)
        else:
            data_sampler = rng.uniform

    idx_dtype = get_index_dtype(maxval=max(shape))
    # rng.choice uses int64 if first arg is an int
    if tot_prod <= np.iinfo(np.int64).max:
        raveled_ind = rng.choice(tot_prod, size=size, replace=False)
        ind = np.unravel_index(raveled_ind, shape=shape, order='F')
        ind = tuple(np.asarray(co, idx_dtype) for co in ind)
    else:
        # for ravel indices bigger than dtype max, use sets to remove duplicates
        ndim = len(shape)
        seen = set()
        while len(seen) < size:
            dsize = size - len(seen)
            seen.update(map(tuple, rng_integers(rng, shape, size=(dsize, ndim))))
        ind = tuple(np.array(list(seen), dtype=idx_dtype).T)

    # size kwarg allows eg data_sampler=partial(np.random.poisson, lam=5)
    vals = data_sampler(size=size).astype(dtype, copy=False)
    return vals, ind


@_transition_to_rng("random_state", position_num=5)
def random(m, n, density=0.01, format='coo', dtype=None,
           rng=None, data_rvs=None):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage of the
        sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        This random state will be used for sampling the sparsity structure, but
        not necessarily for sampling the values of the structurally nonzero
        entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.

    Returns
    -------
    res : sparse matrix

    See Also
    --------
    random_array : constructs sparse arrays instead of sparse matrices

    Examples
    --------

    Passing a ``np.random.Generator`` instance for better performance:

    >>> import scipy as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> S = sp.sparse.random(3, 4, density=0.25, rng=rng)

    Providing a sampler for the values:

    >>> rvs = sp.stats.poisson(25, loc=10).rvs
    >>> S = sp.sparse.random(3, 4, density=0.25, rng=rng, data_rvs=rvs)
    >>> S.toarray()
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    Building a custom distribution.
    This example builds a squared normal from np.random:

    >>> def np_normal_squared(size=None, rng=rng):
    ...     return rng.standard_normal(size) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, rng=rng,
    ...                      data_rvs=np_normal_squared)

    Or we can build it from sp.stats style rvs functions:

    >>> def sp_stats_normal_squared(size=None, rng=rng):
    ...     std_normal = sp.stats.distributions.norm_gen().rvs
    ...     return std_normal(size=size, random_state=rng) ** 2
    >>> S = sp.sparse.random(3, 4, density=0.25, rng=rng,
    ...                      data_rvs=sp_stats_normal_squared)

    Or we can subclass sp.stats rv_continuous or rv_discrete:

    >>> class NormalSquared(sp.stats.rv_continuous):
    ...     def _rvs(self,  size=None, random_state=rng):
    ...         return rng.standard_normal(size) ** 2
    >>> X = NormalSquared()
    >>> Y = X()  # get a frozen version of the distribution
    >>> S = sp.sparse.random(3, 4, density=0.25, rng=rng, data_rvs=Y.rvs)
    """
    if n is None:
        n = m
    m, n = int(m), int(n)
    # make keyword syntax work for data_rvs e.g. data_rvs(size=7)
    if data_rvs is not None:
        def data_rvs_kw(size):
            return data_rvs(size)
    else:
        data_rvs_kw = None
    vals, ind = _random((m, n), density, format, dtype, rng, data_rvs_kw)
    return coo_matrix((vals, ind), shape=(m, n)).asformat(format)


@_transition_to_rng("random_state", position_num=5)
def rand(m, n, density=0.01, format="coo", dtype=None, rng=None):
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``random_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Only float types are supported for now.

    See Also
    --------
    random : Similar function allowing a custom random data sampler
    random_array : Similar to random() but returns a sparse array

    Examples
    --------
    >>> from scipy.sparse import rand
    >>> matrix = rand(3, 4, density=0.25, format="csr", rng=42)
    >>> matrix
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 3 stored elements and shape (3, 4)>
    >>> matrix.toarray()
    array([[0.05641158, 0.        , 0.        , 0.65088847],  # random
           [0.        , 0.        , 0.        , 0.14286682],
           [0.        , 0.        , 0.        , 0.        ]])

    """
    return random(m, n, density, format, dtype, rng)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'SparseEfficiencyWarning',
    'check_reshape_kwargs',
    'check_shape',
    'coo_matrix',
    'coo_matvec',
    'coo_tocsr',
    'coo_todense',
    'downcast_intp_index',
    'getdata',
    'getdtype',
    'isshape',
    'isspmatrix_coo',
    'operator',
    'spmatrix',
    'to_native',
    'upcast',
    'upcast_char',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="coo",
                                   private_modules=["_coo"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Base class for sparse matrices"""

import numpy as np

from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
                       get_sum_dtype, isdense, isscalarlike,
                       matrix, validateaxis, getdtype)

from ._matrix import spmatrix

__all__ = ['isspmatrix', 'issparse', 'sparray',
           'SparseWarning', 'SparseEfficiencyWarning']


class SparseWarning(Warning):
    pass


class SparseFormatWarning(SparseWarning):
    pass


class SparseEfficiencyWarning(SparseWarning):
    pass


# The formats that we might potentially understand.
_formats = {'csc': [0, "Compressed Sparse Column"],
            'csr': [1, "Compressed Sparse Row"],
            'dok': [2, "Dictionary Of Keys"],
            'lil': [3, "List of Lists"],
            'dod': [4, "Dictionary of Dictionaries"],
            'sss': [5, "Symmetric Sparse Skyline"],
            'coo': [6, "COOrdinate"],
            'lba': [7, "Linpack BAnded"],
            'egd': [8, "Ellpack-itpack Generalized Diagonal"],
            'dia': [9, "DIAgonal"],
            'bsr': [10, "Block Sparse Row"],
            'msr': [11, "Modified compressed Sparse Row"],
            'bsc': [12, "Block Sparse Column"],
            'msc': [13, "Modified compressed Sparse Column"],
            'ssk': [14, "Symmetric SKyline"],
            'nsk': [15, "Nonsymmetric SKyline"],
            'jad': [16, "JAgged Diagonal"],
            'uss': [17, "Unsymmetric Sparse Skyline"],
            'vbr': [18, "Variable Block Row"],
            'und': [19, "Undefined"]
            }


# These univariate ufuncs preserve zeros.
_ufuncs_with_fixed_point_at_zero = frozenset([
        np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
        np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad,
        np.rad2deg, np.floor, np.ceil, np.trunc, np.sqrt])


MAXPRINT = 50


class _spbase:
    """ This class provides a base class for all sparse arrays.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    """

    __array_priority__ = 10.1
    _format = 'und'  # undefined
    _allow_nd = (2,)

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def _shape_as_2d(self):
        s = self._shape
        return (1, s[-1]) if len(s) == 1 else s

    @property
    def _bsr_container(self):
        from ._bsr import bsr_array
        return bsr_array

    @property
    def _coo_container(self):
        from ._coo import coo_array
        return coo_array

    @property
    def _csc_container(self):
        from ._csc import csc_array
        return csc_array

    @property
    def _csr_container(self):
        from ._csr import csr_array
        return csr_array

    @property
    def _dia_container(self):
        from ._dia import dia_array
        return dia_array

    @property
    def _dok_container(self):
        from ._dok import dok_array
        return dok_array

    @property
    def _lil_container(self):
        from ._lil import lil_array
        return lil_array

    def __init__(self, arg1, *, maxprint=None):
        self._shape = None
        if self.__class__.__name__ == '_spbase':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        if isinstance(self, sparray) and np.isscalar(arg1):
            raise ValueError(
                "scipy sparse array classes do not support instantiation from a scalar"
            )
        self.maxprint = MAXPRINT if maxprint is None else maxprint

    @property
    def shape(self):
        return self._shape

    def reshape(self, *args, **kwargs):
        """reshape(self, shape, order='C', copy=False)

        Gives a new shape to a sparse array/matrix without changing its data.

        Parameters
        ----------
        shape : length-2 tuple of ints
            The new shape should be compatible with the original shape.
        order : {'C', 'F'}, optional
            Read the elements using this index order. 'C' means to read and
            write the elements using C-like index order; e.g., read entire first
            row, then second row, etc. 'F' means to read and write the elements
            using Fortran-like index order; e.g., read entire first column, then
            second column, etc.
        copy : bool, optional
            Indicates whether or not attributes of self should be copied
            whenever possible. The degree to which attributes are copied varies
            depending on the type of sparse array being used.

        Returns
        -------
        reshaped : sparse array/matrix
            A sparse array/matrix with the given `shape`, not necessarily of the same
            format as the current object.

        See Also
        --------
        numpy.reshape : NumPy's implementation of 'reshape' for ndarrays
        """
        # If the shape already matches, don't bother doing an actual reshape
        # Otherwise, the default is to convert to COO and use its reshape
        # Don't restrict ndim on this first call. That happens in constructor
        shape = check_shape(args, self.shape, allow_nd=range(1, 65))
        order, copy = check_reshape_kwargs(kwargs)
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        return self.tocoo(copy=copy).reshape(shape, order=order, copy=False)

    def resize(self, shape):
        """Resize the array/matrix in-place to dimensions given by ``shape``

        Any elements that lie within the new shape will remain at the same
        indices, while non-zero elements lying outside the new shape are
        removed.

        Parameters
        ----------
        shape : (int, int)
            number of rows and columns in the new array/matrix

        Notes
        -----
        The semantics are not identical to `numpy.ndarray.resize` or
        `numpy.resize`. Here, the same data will be maintained at each index
        before and after reshape, if that index is within the new bounds. In
        numpy, resizing maintains contiguity of the array, moving elements
        around in the logical array but not within a flattened representation.

        We give no guarantees about whether the underlying data attributes
        (arrays, etc.) will be modified in place or replaced with new objects.
        """
        # As an inplace operation, this requires implementation in each format.
        raise NotImplementedError(
            f'{type(self).__name__}.resize is not implemented')

    def astype(self, dtype, casting='unsafe', copy=True):
        """Cast the array/matrix elements to a specified type.

        Parameters
        ----------
        dtype : string or numpy dtype
            Typecode or data-type to which to cast the data.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.
            Defaults to 'unsafe' for backwards compatibility.
            'no' means the data types should not be cast at all.
            'equiv' means only byte-order changes are allowed.
            'safe' means only casts which can preserve values are allowed.
            'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
            'unsafe' means any data conversions may be done.
        copy : bool, optional
            If `copy` is `False`, the result might share some memory with this
            array/matrix. If `copy` is `True`, it is guaranteed that the result and
            this array/matrix do not share any memory.
        """

        dtype = getdtype(dtype)
        if self.dtype != dtype:
            return self.tocsr().astype(
                dtype, casting=casting, copy=copy).asformat(self.format)
        elif copy:
            return self.copy()
        else:
            return self

    @classmethod
    def _ascontainer(cls, X, **kwargs):
        if issubclass(cls, sparray):
            return np.asarray(X, **kwargs)
        else:
            return asmatrix(X, **kwargs)

    @classmethod
    def _container(cls, X, **kwargs):
        if issubclass(cls, sparray):
            return np.array(X, **kwargs)
        else:
            return matrix(X, **kwargs)

    def _asfptype(self):
        """Upcast array to a floating point format (if necessary)"""

        fp_types = ['f', 'd', 'F', 'D']

        if self.dtype.char in fp_types:
            return self
        else:
            for fp_type in fp_types:
                if self.dtype <= np.dtype(fp_type):
                    return self.astype(fp_type)

            raise TypeError(
                f'cannot upcast [{self.dtype.name}] to a floating point format'
            )

    def __iter__(self):
        for r in range(self.shape[0]):
            yield self[r]

    def _getmaxprint(self):
        """Maximum number of elements to display when printed."""
        return self.maxprint

    def count_nonzero(self, axis=None):
        """Number of non-zero entries, equivalent to

        np.count_nonzero(a.toarray(), axis=axis)

        Unlike the nnz property, which return the number of stored
        entries (the length of the data attribute), this method counts the
        actual number of non-zero entries in data.

        Duplicate entries are summed before counting.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Count nonzeros for the whole array, or along a specified axis.

            .. versionadded:: 1.15.0

        Returns
        -------
        numpy array
            A reduced array (no axis `axis`) holding the number of nonzero values
            for each of the indices of the nonaxis dimensions.

        Notes
        -----
        If you want to count nonzero and explicit zero stored values (e.g. nnz)
        along an axis, two fast idioms are provided by `numpy` functions for the
        common CSR, CSC, COO formats.

        For the major axis in CSR (rows) and CSC (cols) use `np.diff`:

            >>> import numpy as np
            >>> import scipy as sp
            >>> A = sp.sparse.csr_array([[4, 5, 0], [7, 0, 0]])
            >>> major_axis_stored_values = np.diff(A.indptr)  # -> np.array([2, 1])

        For the minor axis in CSR (cols) and CSC (rows) use `numpy.bincount` with
        minlength ``A.shape[1]`` for CSR and ``A.shape[0]`` for CSC:

            >>> csr_minor_stored_values = np.bincount(A.indices, minlength=A.shape[1])

        For COO, use the minor axis approach for either `axis`:

            >>> A = A.tocoo()
            >>> coo_axis0_stored_values = np.bincount(A.coords[0], minlength=A.shape[1])
            >>> coo_axis1_stored_values = np.bincount(A.coords[1], minlength=A.shape[0])

        Examples
        --------

            >>> A = sp.sparse.csr_array([[4, 5, 0], [7, 0, 0]])
            >>> A.count_nonzero(axis=0)
            array([2, 1, 0])
        """
        clsname = self.__class__.__name__
        raise NotImplementedError(f"count_nonzero not implemented for {clsname}.")

    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Report stored values for the whole array, or along a specified axis.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        clsname = self.__class__.__name__
        raise NotImplementedError(f"getnnz not implemented for {clsname}.")

    @property
    def nnz(self) -> int:
        """Number of stored values, including explicit zeros.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        return self._getnnz()

    @property
    def size(self) -> int:
        """Number of stored values.

        See also
        --------
        count_nonzero : Number of non-zero values.
        """
        return self._getnnz()

    @property
    def format(self) -> str:
        """Format string for matrix."""
        return self._format

    @property
    def T(self):
        """Transpose."""
        return self.transpose()

    @property
    def real(self):
        return self._real()

    @property
    def imag(self):
        return self._imag()

    def __repr__(self):
        _, format_name = _formats[self.format]
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        return (
            f"<{format_name} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements and shape {self.shape}>"
        )

    def __str__(self):
        maxprint = self._getmaxprint()

        A = self.tocoo()

        # helper function, outputs "(i,j)  v"
        def tostr(coords, data):
            pairs = zip(zip(*(c.tolist() for c in coords)), data)
            return '\n'.join(f'  {idx}\t{val}' for idx, val in pairs)

        out = repr(self)
        if self.nnz == 0:
            return out

        out += '\n  Coords\tValues\n'
        if self.nnz > maxprint:
            half = maxprint // 2
            out += tostr(tuple(c[:half] for c in A.coords), A.data[:half])
            out += "\n  :\t:\n"
            half = maxprint - half
            out += tostr(tuple(c[-half:] for c in A.coords), A.data[-half:])
        else:
            out += tostr(A.coords, A.data)

        return out

    def __bool__(self):  # Simple -- other ideas?
        if self.shape == (1, 1):
            return self.nnz != 0
        else:
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all().")
    __nonzero__ = __bool__

    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  But for some uses the number of
    # non-zeros is more important.  For now, raise an exception!
    def __len__(self):
        raise TypeError("sparse array length is ambiguous; use getnnz()"
                        " or shape[0]")

    def asformat(self, format, copy=False):
        """Return this array/matrix in the passed format.

        Parameters
        ----------
        format : {str, None}
            The desired sparse format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : This array/matrix in the passed format.
        """
        if format is None or format == self.format:
            if copy:
                return self.copy()
            else:
                return self
        else:
            try:
                convert_method = getattr(self, 'to' + format)
            except AttributeError as e:
                raise ValueError(f'Format {format} is unknown.') from e

            # Forward the copy kwarg, if it's accepted.
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    ###################################################################
    #  NOTE: All arithmetic operations use csr_matrix by default.
    # Therefore a new sparse array format just needs to define a
    # .tocsr() method to provide arithmetic support. Any of these
    # methods can be overridden for efficiency.
    ####################################################################

    def multiply(self, other):
        """Point-wise multiplication by another array/matrix."""
        if isscalarlike(other):
            return self._mul_scalar(other)
        return self.tocsr().multiply(other)

    def maximum(self, other):
        """Element-wise maximum between this and another array/matrix."""
        return self.tocsr().maximum(other)

    def minimum(self, other):
        """Element-wise minimum between this and another array/matrix."""
        return self.tocsr().minimum(other)

    def dot(self, other):
        """Ordinary dot product

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1], dtype=int64)

        """
        if np.isscalar(other):
            return self * other
        else:
            return self @ other

    def power(self, n, dtype=None):
        """Element-wise power."""
        return self.tocsr().power(n, dtype=dtype)

    def _broadcast_to(self, shape, copy=False):
        if self.shape == shape:
            return self.copy() if copy else self
        else:
            return self.tocsr()._broadcast_to(shape, copy)

    def __eq__(self, other):
        return self.tocsr().__eq__(other)

    def __ne__(self, other):
        return self.tocsr().__ne__(other)

    def __lt__(self, other):
        return self.tocsr().__lt__(other)

    def __gt__(self, other):
        return self.tocsr().__gt__(other)

    def __le__(self, other):
        return self.tocsr().__le__(other)

    def __ge__(self, other):
        return self.tocsr().__ge__(other)

    def __abs__(self):
        return abs(self.tocsr())

    def __round__(self, ndigits=0):
        return round(self.tocsr(), ndigits=ndigits)

    def _add_sparse(self, other):
        return self.tocsr()._add_sparse(other)

    def _add_dense(self, other):
        return self.tocoo()._add_dense(other)

    def _sub_sparse(self, other):
        return self.tocsr()._sub_sparse(other)

    def _sub_dense(self, other):
        return self.todense() - other

    def _rsub_dense(self, other):
        # note: this can't be replaced by other + (-self) for unsigned types
        return other - self.todense()

    def __add__(self, other):  # self + other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            # Now we would add this scalar to every element.
            raise NotImplementedError('adding a nonzero scalar to a '
                                      'sparse array is not supported')
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._add_sparse(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._add_dense(other)
        else:
            return NotImplemented

    def __radd__(self,other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):  # self - other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError('subtracting a nonzero scalar from a '
                                      'sparse array is not supported')
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._sub_sparse(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._sub_dense(other)
        else:
            return NotImplemented

    def __rsub__(self,other):  # other - self
        if isscalarlike(other):
            if other == 0:
                return -self.copy()
            raise NotImplementedError('subtracting a sparse array from a '
                                      'nonzero scalar is not supported')
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._rsub_dense(other)
        else:
            return NotImplemented

    def _matmul_dispatch(self, other):
        """np.array-like matmul & `np.matrix`-like mul, i.e. `dot` or `NotImplemented`

        interpret other and call one of the following
        self._mul_scalar()
        self._matmul_vector()
        self._matmul_multivector()
        self._matmul_sparse()
        """
        # This method has to be different from `__matmul__` because it is also
        # called by sparse matrix classes.

        # Currently matrix multiplication is only supported
        # for 2D arrays. Hence we unpacked and use only the
        # two last axes' lengths.
        M, N = self._shape_as_2d

        if other.__class__ is np.ndarray:
            # Fast path for the most common case
            if other.shape == (N,):
                return self._matmul_vector(other)
            elif other.shape == (N, 1):
                result = self._matmul_vector(other.ravel())
                if self.ndim == 1:
                    return result.reshape(1)
                return result.reshape(M, 1)
            elif other.ndim == 2 and other.shape[0] == N:
                return self._matmul_multivector(other)

        if isscalarlike(other):
            # scalar value
            return self._mul_scalar(other)

        err_prefix = "matmul: dimension mismatch with signature"
        if issparse(other):
            if N != other.shape[0]:
                raise ValueError(
                    f"{err_prefix} (n,k={N}),(k={other.shape[0]},m)->(n,m)"
                )
            return self._matmul_sparse(other)

        # If it's a list or whatever, treat it like an array
        other_a = np.asanyarray(other)

        if other_a.ndim == 0 and other_a.dtype == np.object_:
            # Not interpretable as an array; return NotImplemented so that
            # other's __rmatmul__ can kick in if that's implemented.
            return NotImplemented

        try:
            other.shape
        except AttributeError:
            other = other_a

        if other.ndim == 1 or other.ndim == 2 and other.shape[1] == 1:
            # dense row or column vector
            if other.shape[0] != N:
                raise ValueError(
                    f"{err_prefix} (n,k={N}),(k={other.shape[0]},1?)->(n,1?)"
                )

            result = self._matmul_vector(np.ravel(other))

            if isinstance(other, np.matrix):
                result = self._ascontainer(result)

            if other.ndim == 2 and other.shape[1] == 1:
                # If 'other' was an (nx1) column vector, reshape the result
                if self.ndim == 1:
                    result = result.reshape(1)
                else:
                    result = result.reshape(-1, 1)

            return result

        elif other.ndim == 2:
            ##
            # dense 2D array or matrix ("multivector")

            if other.shape[0] != N:
                raise ValueError(
                    f"{err_prefix} (n,k={N}),(k={other.shape[0]},m)->(n,m)"
                )

            result = self._matmul_multivector(np.asarray(other))

            if isinstance(other, np.matrix):
                result = self._ascontainer(result)

            return result

        else:
            raise ValueError('could not interpret dimensions')

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):  # other * self
        return self.multiply(other)

    # by default, use CSR for __mul__ handlers
    def _mul_scalar(self, other):
        return self.tocsr()._mul_scalar(other)

    def _matmul_vector(self, other):
        return self.tocsr()._matmul_vector(other)

    def _matmul_multivector(self, other):
        return self.tocsr()._matmul_multivector(other)

    def _matmul_sparse(self, other):
        return self.tocsr()._matmul_sparse(other)

    def _rmatmul_dispatch(self, other):
        if isscalarlike(other):
            return self._mul_scalar(other)
        else:
            # Don't use asarray unless we have to
            try:
                tr = other.transpose()
            except AttributeError:
                tr = np.asarray(other).transpose()
            ret = self.transpose()._matmul_dispatch(tr)
            if ret is NotImplemented:
                return NotImplemented
            return ret.transpose()

    #######################
    # matmul (@) operator #
    #######################

    def __matmul__(self, other):
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self._matmul_dispatch(other)

    def __rmatmul__(self, other):
        if isscalarlike(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self._rmatmul_dispatch(other)

    ####################
    # Other Arithmetic #
    ####################

    def _divide(self, other, true_divide=False, rdivide=False):
        if isscalarlike(other):
            if rdivide:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())

            if true_divide and np.can_cast(self.dtype, np.float64):
                return self.astype(np.float64)._mul_scalar(1./other)
            else:
                r = self._mul_scalar(1./other)

                scalar_dtype = np.asarray(other).dtype
                if (np.issubdtype(self.dtype, np.integer) and
                        np.issubdtype(scalar_dtype, np.integer)):
                    return r.astype(self.dtype)
                else:
                    return r

        elif isdense(other):
            if not rdivide:
                if true_divide:
                    recip = np.true_divide(1., other)
                else:
                    recip = np.divide(1., other)
                return self.multiply(recip)
            else:
                if true_divide:
                    return np.true_divide(other, self.todense())
                else:
                    return np.divide(other, self.todense())
        elif issparse(other):
            if rdivide:
                return other._divide(self, true_divide, rdivide=False)

            self_csr = self.tocsr()
            if true_divide and np.can_cast(self.dtype, np.float64):
                return self_csr.astype(np.float64)._divide_sparse(other)
            else:
                return self_csr._divide_sparse(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        return self._divide(other, true_divide=True)

    def __div__(self, other):
        # Always do true division
        return self._divide(other, true_divide=True)

    def __rtruediv__(self, other):
        # Implementing this as the inverse would be too magical -- bail out
        return NotImplemented

    def __rdiv__(self, other):
        # Implementing this as the inverse would be too magical -- bail out
        return NotImplemented

    def __neg__(self):
        return -self.tocsr()

    def __iadd__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        return NotImplemented

    def __pow__(self, *args, **kwargs):
        return self.power(*args, **kwargs)

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse array/matrix.

        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse array/matrix
            being used.

        Returns
        -------
        p : `self` with the dimensions reversed.

        Notes
        -----
        If `self` is a `csr_array` or a `csc_array`, then this will return a
        `csc_array` or a `csr_array`, respectively.

        See Also
        --------
        numpy.transpose : NumPy's implementation of 'transpose' for ndarrays
        """
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False)

    def conjugate(self, copy=True):
        """Element-wise complex conjugation.

        If the array/matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        """
        if np.issubdtype(self.dtype, np.complexfloating):
            return self.tocsr(copy=copy).conjugate(copy=False)
        elif copy:
            return self.copy()
        else:
            return self

    def conj(self, copy=True):
        return self.conjugate(copy=copy)

    conj.__doc__ = conjugate.__doc__

    def _real(self):
        return self.tocsr()._real()

    def _imag(self):
        return self.tocsr()._imag()

    def nonzero(self):
        """Nonzero indices of the array/matrix.

        Returns a tuple of arrays (row,col) containing the indices
        of the non-zero elements of the array.

        Examples
        --------
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A.nonzero()
        (array([0, 0, 1, 2, 2], dtype=int32), array([0, 1, 2, 0, 2], dtype=int32))

        """

        # convert to COOrdinate format
        A = self.tocoo()
        nz_mask = A.data != 0
        return tuple(idx[nz_mask] for idx in A.coords)

    def _getcol(self, j):
        """Returns a copy of column j of the array, as an (m x 1) sparse
        array (column vector).
        """
        if self.ndim == 1:
            raise ValueError("getcol not provided for 1d arrays. Use indexing A[j]")
        # Subclasses should override this method for efficiency.
        # Post-multiply by a (n x 1) column vector 'a' containing all zeros
        # except for a_j = 1
        N = self.shape[-1]
        if j < 0:
            j += N
        if j < 0 or j >= N:
            raise IndexError("index out of bounds")
        col_selector = self._csc_container(([1], [[j], [0]]),
                                           shape=(N, 1), dtype=self.dtype)
        result = self @ col_selector
        return result

    def _getrow(self, i):
        """Returns a copy of row i of the array, as a (1 x n) sparse
        array (row vector).
        """
        if self.ndim == 1:
            raise ValueError("getrow not meaningful for a 1d array")
        # Subclasses should override this method for efficiency.
        # Pre-multiply by a (1 x m) row vector 'a' containing all zeros
        # except for a_i = 1
        M = self.shape[0]
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError("index out of bounds")
        row_selector = self._csr_container(([1], [[0], [i]]),
                                           shape=(1, M), dtype=self.dtype)
        return row_selector @ self

    # The following dunder methods cannot be implemented.
    #
    # def __array__(self):
    #     # Sparse matrices rely on NumPy wrapping them in object arrays under
    #     # the hood to make unary ufuncs work on them. So we cannot raise
    #     # TypeError here - which would be handy to not give users object
    #     # arrays they probably don't want (they're looking for `.toarray()`).
    #     #
    #     # Conversion with `toarray()` would also break things because of the
    #     # behavior discussed above, plus we want to avoid densification by
    #     # accident because that can too easily blow up memory.
    #
    # def __array_ufunc__(self):
    #     # We cannot implement __array_ufunc__ due to mismatching semantics.
    #     # See gh-7707 and gh-7349 for details.
    #
    # def __array_function__(self):
    #     # We cannot implement __array_function__ due to mismatching semantics.
    #     # See gh-10362 for details.

    def todense(self, order=None, out=None):
        """
        Return a dense representation of this sparse array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The
            provided array must have the same shape and dtype as
            the sparse array on which you are calling the method.

        Returns
        -------
        arr : ndarray, 2-D
            An array with the same shape and containing the same
            data represented by the sparse array, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        """
        return self._ascontainer(self.toarray(order=order, out=out))

    def toarray(self, order=None, out=None):
        """
        Return a dense ndarray representation of this sparse array/matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multidimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The provided
            array must have the same shape and dtype as the sparse
            array/matrix on which you are calling the method. For most
            sparse types, `out` is required to be memory contiguous
            (either C or Fortran ordered).

        Returns
        -------
        arr : ndarray, 2-D
            An array with the same shape and containing the same
            data represented by the sparse array/matrix, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        """
        return self.tocoo(copy=False).toarray(order=order, out=out)

    # Any sparse array format deriving from _spbase must define one of
    # tocsr or tocoo. The other conversion methods may be implemented for
    # efficiency, but are not required.
    def tocsr(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Row format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant csr_array/matrix.
        """
        return self.tocoo(copy=copy).tocsr(copy=False)

    def todok(self, copy=False):
        """Convert this array/matrix to Dictionary Of Keys format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant dok_array/matrix.
        """
        return self.tocoo(copy=copy).todok(copy=False)

    def tocoo(self, copy=False):
        """Convert this array/matrix to COOrdinate format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant coo_array/matrix.
        """
        return self.tocsr(copy=False).tocoo(copy=copy)

    def tolil(self, copy=False):
        """Convert this array/matrix to List of Lists format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant lil_array/matrix.
        """
        return self.tocsr(copy=False).tolil(copy=copy)

    def todia(self, copy=False):
        """Convert this array/matrix to sparse DIAgonal format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant dia_array/matrix.
        """
        return self.tocoo(copy=copy).todia(copy=False)

    def tobsr(self, blocksize=None, copy=False):
        """Convert this array/matrix to Block Sparse Row format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant bsr_array/matrix.

        When blocksize=(R, C) is provided, it will be used for construction of
        the bsr_array/matrix.
        """
        return self.tocsr(copy=False).tobsr(blocksize=blocksize, copy=copy)

    def tocsc(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Column format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant csc_array/matrix.
        """
        return self.tocsr(copy=copy).tocsc(copy=False)

    def copy(self):
        """Returns a copy of this array/matrix.

        No data/indices will be shared between the returned value and current
        array/matrix.
        """
        return self.__class__(self, copy=True)

    def sum(self, axis=None, dtype=None, out=None):
        """
        Sum the array/matrix elements over a given axis.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the array/matrix elements, returning a scalar
            (i.e., `axis` = `None`).
        dtype : dtype, optional
            The type of the returned array/matrix and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.

        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices

        """
        validateaxis(axis)

        # Mimic numpy's casting.
        res_dtype = get_sum_dtype(self.dtype)

        if self.ndim == 1:
            if axis not in (None, -1, 0):
                raise ValueError("axis must be None, -1 or 0")
            res = self @ np.ones(self.shape, dtype=res_dtype)
            return res.sum(dtype=dtype, out=out)

        # We use multiplication by a matrix of ones to achieve this.
        # For some sparse array formats more efficient methods are
        # possible -- these should override this function.
        M, N = self.shape

        if axis is None:
            # sum over rows and columns
            return (
                self @ self._ascontainer(np.ones((N, 1), dtype=res_dtype))
            ).sum(dtype=dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            # sum over columns
            ret = self._ascontainer(
                np.ones((1, M), dtype=res_dtype)
            ) @ self
        else:
            # sum over rows
            ret = self @ self._ascontainer(
                np.ones((N, 1), dtype=res_dtype)
            )

        return ret.sum(axis=axis, dtype=dtype, out=out)

    def mean(self, axis=None, dtype=None, out=None):
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array/matrix elements. The average is taken
        over all elements in the array/matrix by default, otherwise over the
        specified axis. `float64` intermediate and return values are used
        for integer inputs.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the mean is computed. The default is to compute
            the mean of all elements in the array/matrix (i.e., `axis` = `None`).
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        m : np.matrix

        See Also
        --------
        numpy.matrix.mean : NumPy's implementation of 'mean' for matrices

        """
        validateaxis(axis)

        res_dtype = self.dtype.type
        integral = (np.issubdtype(self.dtype, np.integer) or
                    np.issubdtype(self.dtype, np.bool_))

        # output dtype
        if dtype is None:
            if integral:
                res_dtype = np.float64
        else:
            res_dtype = np.dtype(dtype).type

        # intermediate dtype for summation
        inter_dtype = np.float64 if integral else res_dtype
        inter_self = self.astype(inter_dtype)

        if self.ndim == 1:
            if axis not in (None, -1, 0):
                raise ValueError("axis must be None, -1 or 0")
            res = inter_self / self.shape[-1]
            return res.sum(dtype=res_dtype, out=out)

        if axis is None:
            return (inter_self / (self.shape[0] * self.shape[1]))\
                .sum(dtype=res_dtype, out=out)

        if axis < 0:
            axis += 2

        # axis = 0 or 1 now
        if axis == 0:
            return (inter_self * (1.0 / self.shape[0])).sum(
                axis=0, dtype=res_dtype, out=out)
        else:
            return (inter_self * (1.0 / self.shape[1])).sum(
                axis=1, dtype=res_dtype, out=out)

    def diagonal(self, k=0):
        """Returns the kth diagonal of the array/matrix.

        Parameters
        ----------
        k : int, optional
            Which diagonal to get, corresponding to elements a[i, i+k].
            Default: 0 (the main diagonal).

            .. versionadded:: 1.0

        See also
        --------
        numpy.diagonal : Equivalent numpy function.

        Examples
        --------
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A.diagonal()
        array([1, 0, 5])
        >>> A.diagonal(k=1)
        array([2, 3])
        """
        return self.tocsr().diagonal(k=k)

    def trace(self, offset=0):
        """Returns the sum along diagonals of the sparse array/matrix.

        Parameters
        ----------
        offset : int, optional
            Which diagonal to get, corresponding to elements a[i, i+offset].
            Default: 0 (the main diagonal).

        """
        return self.diagonal(k=offset).sum()

    def setdiag(self, values, k=0):
        """
        Set diagonal or off-diagonal elements of the array/matrix.

        Parameters
        ----------
        values : array_like
            New values of the diagonal elements.

            Values may have any length. If the diagonal is longer than values,
            then the remaining diagonal entries will not be set. If values are
            longer than the diagonal, then the remaining values are ignored.

            If a scalar value is given, all of the diagonal is set to it.

        k : int, optional
            Which off-diagonal to set, corresponding to elements a[i,i+k].
            Default: 0 (the main diagonal).

        """
        M, N = self.shape
        if (k > 0 and k >= N) or (k < 0 and -k >= M):
            raise ValueError("k exceeds array dimensions")
        self._setdiag(np.asarray(values), k)

    def _setdiag(self, values, k):
        """This part of the implementation gets overridden by the
        different formats.
        """
        M, N = self.shape
        if k < 0:
            if values.ndim == 0:
                # broadcast
                max_index = min(M+k, N)
                for i in range(max_index):
                    self[i - k, i] = values
            else:
                max_index = min(M+k, N, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i - k, i] = v
        else:
            if values.ndim == 0:
                # broadcast
                max_index = min(M, N-k)
                for i in range(max_index):
                    self[i, i + k] = values
            else:
                max_index = min(M, N-k, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i, i + k] = v

    def _process_toarray_args(self, order, out):
        if out is not None:
            if order is not None:
                raise ValueError('order cannot be specified if out '
                                 'is not None')
            if out.shape != self.shape or out.dtype != self.dtype:
                raise ValueError('out array must be same dtype and shape as '
                                 'sparse array')
            out[...] = 0.
            return out
        else:
            return np.zeros(self.shape, dtype=self.dtype, order=order)

    def _get_index_dtype(self, arrays=(), maxval=None, check_contents=False):
        """
        Determine index dtype for array.

        This wraps _sputils.get_index_dtype, providing compatibility for both
        array and matrix API sparse matrices. Matrix API sparse matrices would
        attempt to downcast the indices - which can be computationally
        expensive and undesirable for users. The array API changes this
        behaviour.

        See discussion: https://github.com/scipy/scipy/issues/16774

        The get_index_dtype import is due to implementation details of the test
        suite. It allows the decorator ``with_64bit_maxval_limit`` to mock a
        lower int32 max value for checks on the matrix API's downcasting
        behaviour.
        """
        from ._sputils import get_index_dtype

        # Don't check contents for array API
        return get_index_dtype(arrays,
                               maxval,
                               (check_contents and not isinstance(self, sparray)))


class sparray:
    """A namespace class to separate sparray from spmatrix"""


sparray.__doc__ = _spbase.__doc__


def issparse(x):
    """Is `x` of a sparse array or sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse array or sparse matrix

    Returns
    -------
    bool
        True if `x` is a sparse array or a sparse matrix, False otherwise

    Notes
    -----
    Use `isinstance(x, sp.sparse.sparray)` to check between an array or matrix.
    Use `a.format` to check the sparse format, e.g. `a.format == 'csr'`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array, csr_matrix, issparse
    >>> issparse(csr_matrix([[5]]))
    True
    >>> issparse(csr_array([[5]]))
    True
    >>> issparse(np.array([[5]]))
    False
    >>> issparse(5)
    False
    """
    return isinstance(x, _spbase)


def isspmatrix(x):
    """Is `x` of a sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse matrix

    Returns
    -------
    bool
        True if `x` is a sparse matrix, False otherwise

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array, csr_matrix, isspmatrix
    >>> isspmatrix(csr_matrix([[5]]))
    True
    >>> isspmatrix(csr_array([[5]]))
    False
    >>> isspmatrix(np.array([[5]]))
    False
    >>> isspmatrix(5)
    False
    """
    return isinstance(x, spmatrix)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'MAXPRINT',
    'SparseEfficiencyWarning',
    'SparseFormatWarning',
    'SparseWarning',
    'asmatrix',
    'check_reshape_kwargs',
    'check_shape',
    'get_sum_dtype',
    'isdense',
    'isscalarlike',
    'issparse',
    'isspmatrix',
    'spmatrix',
    'validateaxis',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="base",
                                   private_modules=["_base"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'isspmatrix_lil',
    'lil_array',
    'lil_matrix',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="lil",
                                   private_modules=["_lil"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'csc_matrix',
    'csc_tocsr',
    'expandptr',
    'isspmatrix_csc',
    'spmatrix',
    'upcast',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="csc",
                                   private_modules=["_csc"], all=__all__,
                                   attribute=name)
<|endoftext|>"""List of Lists sparse matrix class
"""

__docformat__ = "restructuredtext en"

__all__ = ['lil_array', 'lil_matrix', 'isspmatrix_lil']

from bisect import bisect_left

import numpy as np

from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from ._sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
                       check_shape, check_reshape_kwargs)
from . import _csparsetools


class _lil_base(_spbase, IndexMixin):
    _format = 'lil'

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _spbase.__init__(self, arg1, maxprint=maxprint)
        self.dtype = getdtype(dtype, arg1, default=float)

        # First get the shape
        if issparse(arg1):
            if arg1.format == "lil" and copy:
                A = arg1.copy()
            else:
                A = arg1.tolil()

            if dtype is not None:
                newdtype = getdtype(dtype)
                A = A.astype(newdtype, copy=False)

            self._shape = check_shape(A.shape)
            self.dtype = A.dtype
            self.rows = A.rows
            self.data = A.data
        elif isinstance(arg1,tuple):
            if isshape(arg1):
                if shape is not None:
                    raise ValueError('invalid use of shape parameter')
                M, N = arg1
                self._shape = check_shape((M, N))
                self.rows = np.empty((M,), dtype=object)
                self.data = np.empty((M,), dtype=object)
                for i in range(M):
                    self.rows[i] = []
                    self.data[i] = []
            else:
                raise TypeError('unrecognized lil_array constructor usage')
        else:
            # assume A is dense
            try:
                A = self._ascontainer(arg1)
            except TypeError as e:
                raise TypeError('unsupported matrix type') from e
            if isinstance(self, sparray) and A.ndim != 2:
                raise ValueError(f"LIL arrays don't support {A.ndim}D input. Use 2D")
            A = self._csr_container(A, dtype=dtype).tolil()

            self._shape = check_shape(A.shape)
            self.dtype = getdtype(A.dtype)
            self.rows = A.rows
            self.data = A.data

    def __iadd__(self,other):
        self[:,:] = self + other
        return self

    def __isub__(self,other):
        self[:,:] = self - other
        return self

    def __imul__(self,other):
        if isscalarlike(other):
            self[:,:] = self * other
            return self
        else:
            return NotImplemented

    def __itruediv__(self,other):
        if isscalarlike(other):
            self[:,:] = self / other
            return self
        else:
            return NotImplemented

    # Whenever the dimensions change, empty lists should be created for each
    # row

    def _getnnz(self, axis=None):
        if axis is None:
            return sum([len(rowvals) for rowvals in self.data])
        if axis < 0:
            axis += 2
        if axis == 0:
            out = np.zeros(self.shape[1], dtype=np.intp)
            for row in self.rows:
                out[row] += 1
            return out
        elif axis == 1:
            return np.array([len(rowvals) for rowvals in self.data], dtype=np.intp)
        else:
            raise ValueError('axis out of bounds')

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def count_nonzero(self, axis=None):
        if axis is None:
            return sum(np.count_nonzero(rowvals) for rowvals in self.data)

        if axis < 0:
            axis += 2
        if axis == 0:
            out = np.zeros(self.shape[1], dtype=np.intp)
            for row, data in zip(self.rows, self.data):
                mask = [c for c, d in zip(row, data) if d != 0]
                out[mask] += 1
            return out
        elif axis == 1:
            return np.array(
                [np.count_nonzero(rowvals) for rowvals in self.data], dtype=np.intp,
            )
        else:
            raise ValueError('axis out of bounds')

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def getrowview(self, i):
        """Returns a view of the 'i'th row (without copying).
        """
        new = self._lil_container((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i]
        new.data[0] = self.data[i]
        return new

    def getrow(self, i):
        """Returns a copy of the 'i'th row.
        """
        M, N = self.shape
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('row index out of bounds')
        new = self._lil_container((1, N), dtype=self.dtype)
        new.rows[0] = self.rows[i][:]
        new.data[0] = self.data[i][:]
        return new

    def __getitem__(self, key):
        # Fast path for simple (int, int) indexing.
        if (isinstance(key, tuple) and len(key) == 2 and
                isinstance(key[0], INT_TYPES) and
                isinstance(key[1], INT_TYPES)):
            # lil_get1 handles validation for us.
            return self._get_intXint(*key)
        # Everything else takes the normal path.
        return IndexMixin.__getitem__(self, key)

    def _asindices(self, idx, N):
        # LIL routines handle bounds-checking for us, so don't do it here.
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            raise IndexError('invalid index') from e
        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')
        return x

    def _get_intXint(self, row, col):
        v = _csparsetools.lil_get1(self.shape[0], self.shape[1], self.rows,
                                   self.data, row, col)
        return self.dtype.type(v)

    def _get_sliceXint(self, row, col):
        row = range(*row.indices(self.shape[0]))
        return self._get_row_ranges(row, slice(col, col+1))

    def _get_arrayXint(self, row, col):
        res = self._get_row_ranges(row.ravel(), slice(col, col+1))
        if row.ndim > 1:
            return res.reshape(row.shape)
        return res

    def _get_intXslice(self, row, col):
        return self._get_row_ranges((row,), col)

    def _get_sliceXslice(self, row, col):
        row = range(*row.indices(self.shape[0]))
        return self._get_row_ranges(row, col)

    def _get_arrayXslice(self, row, col):
        return self._get_row_ranges(row, col)

    def _get_intXarray(self, row, col):
        row = np.array(row, dtype=col.dtype, ndmin=1)
        return self._get_columnXarray(row, col)

    def _get_sliceXarray(self, row, col):
        row = np.arange(*row.indices(self.shape[0]))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        # outer indexing
        row, col = _broadcast_arrays(row[:,None], col)
        return self._get_arrayXarray(row, col)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        i, j = map(np.atleast_2d, _prepare_index_for_memoryview(row, col))
        new = self._lil_container(i.shape, dtype=self.dtype)
        _csparsetools.lil_fancy_get(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    new.rows, new.data,
                                    i, j)
        return new

    def _get_row_ranges(self, rows, col_slice):
        """
        Fast path for indexing in the case where column index is slice.

        This gains performance improvement over brute force by more
        efficient skipping of zeros, by accessing the elements
        column-wise in order.

        Parameters
        ----------
        rows : sequence or range
            Rows indexed. If range, must be within valid bounds.
        col_slice : slice
            Columns indexed

        """
        j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
        col_range = range(j_start, j_stop, j_stride)
        nj = len(col_range)
        new = self._lil_container((len(rows), nj), dtype=self.dtype)

        _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1],
                                         self.rows, self.data,
                                         new.rows, new.data,
                                         rows,
                                         j_start, j_stop, j_stride, nj)

        return new

    def _set_intXint(self, row, col, x):
        _csparsetools.lil_insert(self.shape[0], self.shape[1], self.rows,
                                 self.data, row, col, x)

    def _set_arrayXarray(self, row, col, x):
        i, j, x = map(np.atleast_2d, _prepare_index_for_memoryview(row, col, x))
        _csparsetools.lil_fancy_set(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # Fall back to densifying x
        x = np.asarray(x.toarray(), dtype=self.dtype)
        x, _ = _broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)

    def __setitem__(self, key, x):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            # Fast path for simple (int, int) indexing.
            if isinstance(row, INT_TYPES) and isinstance(col, INT_TYPES):
                x = self.dtype.type(x)
                if x.size > 1:
                    raise ValueError("Trying to assign a sequence to an item")
                return self._set_intXint(row, col, x)
            # Fast path for full-matrix sparse assignment.
            if (isinstance(row, slice) and isinstance(col, slice) and
                    row == slice(None) and col == slice(None) and
                    issparse(x) and x.shape == self.shape):
                x = self._lil_container(x, dtype=self.dtype)
                self.rows = x.rows
                self.data = x.data
                return
        # Everything else takes the normal path.
        IndexMixin.__setitem__(self, key, x)

    def _mul_scalar(self, other):
        if other == 0:
            # Multiply by zero: return the zero matrix
            new = self._lil_container(self.shape, dtype=self.dtype)
        else:
            res_dtype = upcast_scalar(self.dtype, other)

            new = self.copy()
            new = new.astype(res_dtype)
            # Multiply this scalar by every element.
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val*other for val in rowvals]
        return new

    def __truediv__(self, other):           # self / other
        if isscalarlike(other):
            new = self.copy()
            new.dtype = np.result_type(self, other)
            # Divide every element by this scalar
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val/other for val in rowvals]
            return new
        else:
            return self.tocsr() / other

    def copy(self):
        M, N = self.shape
        new = self._lil_container(self.shape, dtype=self.dtype)
        # This is ~14x faster than calling deepcopy() on rows and data.
        _csparsetools.lil_get_row_ranges(M, N, self.rows, self.data,
                                         new.rows, new.data, range(M),
                                         0, N, 1, N)
        return new

    copy.__doc__ = _spbase.copy.__doc__

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        new = self._lil_container(shape, dtype=self.dtype)

        if order == 'C':
            ncols = self.shape[1]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    new_r, new_c = np.unravel_index(i * ncols + j, shape)
                    new[new_r, new_c] = self[i, j]
        elif order == 'F':
            nrows = self.shape[0]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    new_r, new_c = np.unravel_index(i + j * nrows, shape, order)
                    new[new_r, new_c] = self[i, j]
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        return new

    reshape.__doc__ = _spbase.reshape.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        if new_M < M:
            self.rows = self.rows[:new_M]
            self.data = self.data[:new_M]
        elif new_M > M:
            self.rows = np.resize(self.rows, new_M)
            self.data = np.resize(self.data, new_M)
            for i in range(M, new_M):
                self.rows[i] = []
                self.data[i] = []

        if new_N < N:
            for row, data in zip(self.rows, self.data):
                trunc = bisect_left(row, new_N)
                del row[trunc:]
                del data[trunc:]

        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__

    def toarray(self, order=None, out=None):
        d = self._process_toarray_args(order, out)
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                d[i, j] = self.data[i][pos]
        return d

    toarray.__doc__ = _spbase.toarray.__doc__

    def transpose(self, axes=None, copy=False):
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False).tolil(copy=False)

    transpose.__doc__ = _spbase.transpose.__doc__

    def tolil(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tolil.__doc__ = _spbase.tolil.__doc__

    def tocsr(self, copy=False):
        M, N = self.shape
        if M == 0 or N == 0:
            return self._csr_container((M, N), dtype=self.dtype)

        # construct indptr array
        if M*N <= np.iinfo(np.int32).max:
            # fast path: it is known that 64-bit indexing will not be needed.
            idx_dtype = np.int32
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indptr[0] = 0
            _csparsetools.lil_get_lengths(self.rows, indptr[1:])
            np.cumsum(indptr, out=indptr)
            nnz = indptr[-1]
        else:
            idx_dtype = self._get_index_dtype(maxval=N)
            lengths = np.empty(M, dtype=idx_dtype)
            _csparsetools.lil_get_lengths(self.rows, lengths)
            nnz = lengths.sum(dtype=np.int64)
            idx_dtype = self._get_index_dtype(maxval=max(N, nnz))
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indptr[0] = 0
            np.cumsum(lengths, dtype=idx_dtype, out=indptr[1:])

        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=self.dtype)
        _csparsetools.lil_flatten_to_array(self.rows, indices)
        _csparsetools.lil_flatten_to_array(self.data, data)

        # init csr matrix
        return self._csr_container((data, indices, indptr), shape=self.shape)

    tocsr.__doc__ = _spbase.tocsr.__doc__


def _prepare_index_for_memoryview(i, j, x=None):
    """
    Convert index and data arrays to form suitable for passing to the
    Cython fancy getset routines.

    The conversions are necessary since to (i) ensure the integer
    index arrays are in one of the accepted types, and (ii) to ensure
    the arrays are writable so that Cython memoryview support doesn't
    choke on them.

    Parameters
    ----------
    i, j
        Index arrays
    x : optional
        Data arrays

    Returns
    -------
    i, j, x
        Re-formatted arrays (x is omitted, if input was None)

    """
    if i.dtype > j.dtype:
        j = j.astype(i.dtype)
    elif i.dtype < j.dtype:
        i = i.astype(j.dtype)

    if not i.flags.writeable or i.dtype not in (np.int32, np.int64):
        i = i.astype(np.intp)
    if not j.flags.writeable or j.dtype not in (np.int32, np.int64):
        j = j.astype(np.intp)

    if x is not None:
        if not x.flags.writeable:
            x = x.copy()
        return i, j, x
    else:
        return i, j


def isspmatrix_lil(x):
    """Is `x` of lil_matrix type?

    Parameters
    ----------
    x
        object to check for being a lil matrix

    Returns
    -------
    bool
        True if `x` is a lil matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import lil_array, lil_matrix, coo_matrix, isspmatrix_lil
    >>> isspmatrix_lil(lil_matrix([[5]]))
    True
    >>> isspmatrix_lil(lil_array([[5]]))
    False
    >>> isspmatrix_lil(coo_matrix([[5]]))
    False
    """
    return isinstance(x, lil_matrix)


# This namespace class separates array from matrix with isinstance
class lil_array(_lil_base, sparray):
    """
    Row-based LIst of Lists sparse array.

    This is a structure for constructing sparse arrays incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct the array efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_array(D)
            where D is a 2-D ndarray

        lil_array(S)
            with another sparse array or matrix S (equivalent to S.tolil())

        lil_array((M, N), [dtype])
            to construct an empty array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        LIL format data array of the array
    rows
        LIL format row index array of the array
    T

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the array sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse arrays
        - once an array has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large arrays

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.

    """


class lil_matrix(spmatrix, _lil_base):
    """
    Row-based LIst of Lists sparse matrix.

    This is a structure for constructing sparse matrices incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct the matrix efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_matrix(D)
            where D is a 2-D ndarray

        lil_matrix(S)
            with another sparse array or matrix S (equivalent to S.tolil())

        lil_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        LIL format data array of the matrix
    rows
        LIL format row index array of the matrix
    T

    Notes
    -----
    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse matrices
        - once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large matrices

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.

    """
<|endoftext|>r"""
Compressed sparse graph routines (:mod:`scipy.sparse.csgraph`)
==============================================================

.. currentmodule:: scipy.sparse.csgraph

Fast graph algorithms based on sparse matrix representations.

Contents
--------

.. autosummary::
   :toctree: generated/

   connected_components -- determine connected components of a graph
   laplacian -- compute the laplacian of a graph
   shortest_path -- compute the shortest path between points on a positive graph
   dijkstra -- use Dijkstra's algorithm for shortest path
   floyd_warshall -- use the Floyd-Warshall algorithm for shortest path
   bellman_ford -- use the Bellman-Ford algorithm for shortest path
   johnson -- use Johnson's algorithm for shortest path
   yen -- use Yen's algorithm for K-shortest paths between to nodes.
   breadth_first_order -- compute a breadth-first order of nodes
   depth_first_order -- compute a depth-first order of nodes
   breadth_first_tree -- construct the breadth-first tree from a given node
   depth_first_tree -- construct a depth-first tree from a given node
   minimum_spanning_tree -- construct the minimum spanning tree of a graph
   reverse_cuthill_mckee -- compute permutation for reverse Cuthill-McKee ordering
   maximum_flow -- solve the maximum flow problem for a graph
   maximum_bipartite_matching -- compute a maximum matching of a bipartite graph
   min_weight_full_bipartite_matching - compute a minimum weight full matching of a bipartite graph
   structural_rank -- compute the structural rank of a graph
   NegativeCycleError

.. autosummary::
   :toctree: generated/

   construct_dist_matrix
   csgraph_from_dense
   csgraph_from_masked
   csgraph_masked_from_dense
   csgraph_to_dense
   csgraph_to_masked
   reconstruct_path

Graph Representations
---------------------
This module uses graphs which are stored in a matrix format. A
graph with N nodes can be represented by an (N x N) adjacency matrix G.
If there is a connection from node i to node j, then G[i, j] = w, where
w is the weight of the connection. For nodes i and j which are
not connected, the value depends on the representation:

- for dense array representations, non-edges are represented by
  G[i, j] = 0, infinity, or NaN.

- for dense masked representations (of type np.ma.MaskedArray), non-edges
  are represented by masked values. This can be useful when graphs with
  zero-weight edges are desired.

- for sparse array representations, non-edges are represented by
  non-entries in the matrix. This sort of sparse representation also
  allows for edges with zero weights.

As a concrete example, imagine that you would like to represent the following
undirected graph::

              G

             (0)
            /   \
           1     2
          /       \
        (2)       (1)

This graph has three nodes, where node 0 and 1 are connected by an edge of
weight 2, and nodes 0 and 2 are connected by an edge of weight 1.
We can construct the dense, masked, and sparse representations as follows,
keeping in mind that an undirected graph is represented by a symmetric matrix::

    >>> import numpy as np
    >>> G_dense = np.array([[0, 2, 1],
    ...                     [2, 0, 0],
    ...                     [1, 0, 0]])
    >>> G_masked = np.ma.masked_values(G_dense, 0)
    >>> from scipy.sparse import csr_array
    >>> G_sparse = csr_array(G_dense)

This becomes more difficult when zero edges are significant. For example,
consider the situation when we slightly modify the above graph::

             G2

             (0)
            /   \
           0     2
          /       \
        (2)       (1)

This is identical to the previous graph, except nodes 0 and 2 are connected
by an edge of zero weight. In this case, the dense representation above
leads to ambiguities: how can non-edges be represented if zero is a meaningful
value? In this case, either a masked or sparse representation must be used
to eliminate the ambiguity::

    >>> import numpy as np
    >>> G2_data = np.array([[np.inf, 2,      0     ],
    ...                     [2,      np.inf, np.inf],
    ...                     [0,      np.inf, np.inf]])
    >>> G2_masked = np.ma.masked_invalid(G2_data)
    >>> from scipy.sparse.csgraph import csgraph_from_dense
    >>> # G2_sparse = csr_array(G2_data) would give the wrong result
    >>> G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
    >>> G2_sparse.data
    array([ 2.,  0.,  2.,  0.])

Here we have used a utility routine from the csgraph submodule in order to
convert the dense representation to a sparse representation which can be
understood by the algorithms in submodule. By viewing the data array, we
can see that the zero values are explicitly encoded in the graph.

Directed vs. undirected
^^^^^^^^^^^^^^^^^^^^^^^
Matrices may represent either directed or undirected graphs. This is
specified throughout the csgraph module by a boolean keyword. Graphs are
assumed to be directed by default. In a directed graph, traversal from node
i to node j can be accomplished over the edge G[i, j], but not the edge
G[j, i].  Consider the following dense graph::

    >>> import numpy as np
    >>> G_dense = np.array([[0, 1, 0],
    ...                     [2, 0, 3],
    ...                     [0, 4, 0]])

When ``directed=True`` we get the graph::

      ---1--> ---3-->
    (0)     (1)     (2)
      <--2--- <--4---

In a non-directed graph, traversal from node i to node j can be
accomplished over either G[i, j] or G[j, i].  If both edges are not null,
and the two have unequal weights, then the smaller of the two is used.

So for the same graph, when ``directed=False`` we get the graph::

    (0)--1--(1)--3--(2)

Note that a symmetric matrix will represent an undirected graph, regardless
of whether the 'directed' keyword is set to True or False. In this case,
using ``directed=True`` generally leads to more efficient computation.

The routines in this module accept as input either scipy.sparse representations
(csr, csc, or lil format), masked representations, or dense representations
with non-edges indicated by zeros, infinities, and NaN entries.
"""  # noqa: E501

__docformat__ = "restructuredtext en"

__all__ = ['connected_components',
           'laplacian',
           'shortest_path',
           'floyd_warshall',
           'dijkstra',
           'bellman_ford',
           'johnson',
           'yen',
           'breadth_first_order',
           'depth_first_order',
           'breadth_first_tree',
           'depth_first_tree',
           'minimum_spanning_tree',
           'reverse_cuthill_mckee',
           'maximum_flow',
           'maximum_bipartite_matching',
           'min_weight_full_bipartite_matching',
           'structural_rank',
           'construct_dist_matrix',
           'reconstruct_path',
           'csgraph_masked_from_dense',
           'csgraph_from_dense',
           'csgraph_from_masked',
           'csgraph_to_dense',
           'csgraph_to_masked',
           'NegativeCycleError']

from ._laplacian import laplacian
from ._shortest_path import (
    shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson, yen,
    NegativeCycleError
)
from ._traversal import (
    breadth_first_order, depth_first_order, breadth_first_tree,
    depth_first_tree, connected_components
)
from ._min_spanning_tree import minimum_spanning_tree
from ._flow import maximum_flow
from ._matching import (
    maximum_bipartite_matching, min_weight_full_bipartite_matching
)
from ._reordering import reverse_cuthill_mckee, structural_rank
from ._tools import (
    construct_dist_matrix, reconstruct_path, csgraph_from_dense,
    csgraph_to_dense, csgraph_masked_from_dense, csgraph_from_masked,
    csgraph_to_masked
)

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>import numpy as np
from scipy.sparse import issparse
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy
from scipy.sparse.csgraph._tools import (
    csgraph_to_dense, csgraph_from_dense,
    csgraph_masked_from_dense, csgraph_from_masked
)

DTYPE = np.float64


def validate_graph(csgraph, directed, dtype=DTYPE,
                   csr_output=True, dense_output=True,
                   copy_if_dense=False, copy_if_sparse=False,
                   null_value_in=0, null_value_out=np.inf,
                   infinity_null=True, nan_null=True):
    """Routine for validation and conversion of csgraph inputs"""
    if not (csr_output or dense_output):
        raise ValueError("Internal: dense or csr output must be true")

    accept_fv = [null_value_in]
    if infinity_null:
        accept_fv.append(np.inf)
    if nan_null:
        accept_fv.append(np.nan)
    csgraph = convert_pydata_sparse_to_scipy(csgraph, accept_fv=accept_fv)

    # if undirected and csc storage, then transposing in-place
    # is quicker than later converting to csr.
    if (not directed) and issparse(csgraph) and csgraph.format == "csc":
        csgraph = csgraph.T

    if issparse(csgraph):
        if csr_output:
            csgraph = csgraph.tocsr(copy=copy_if_sparse).astype(DTYPE, copy=False)
        else:
            csgraph = csgraph_to_dense(csgraph, null_value=null_value_out)
    elif np.ma.isMaskedArray(csgraph):
        if dense_output:
            mask = csgraph.mask
            csgraph = np.array(csgraph.data, dtype=DTYPE, copy=copy_if_dense)
            csgraph[mask] = null_value_out
        else:
            csgraph = csgraph_from_masked(csgraph)
    else:
        if dense_output:
            csgraph = csgraph_masked_from_dense(csgraph,
                                                copy=copy_if_dense,
                                                null_value=null_value_in,
                                                nan_null=nan_null,
                                                infinity_null=infinity_null)
            mask = csgraph.mask
            csgraph = np.asarray(csgraph.data, dtype=DTYPE)
            csgraph[mask] = null_value_out
        else:
            csgraph = csgraph_from_dense(csgraph, null_value=null_value_in,
                                         infinity_null=infinity_null,
                                         nan_null=nan_null)

    if csgraph.ndim != 2:
        raise ValueError("compressed-sparse graph must be 2-D")

    if csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError("compressed-sparse graph must be shape (N, N)")

    return csgraph
<|endoftext|>"""
Laplacian of a compressed-sparse graph
"""

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy, is_pydata_spmatrix


###############################################################################
# Graph laplacian
def laplacian(
    csgraph,
    normed=False,
    return_diag=False,
    use_out_degree=False,
    *,
    copy=True,
    form="array",
    dtype=None,
    symmetrized=False,
):
    """
    Return the Laplacian of a directed graph.

    Parameters
    ----------
    csgraph : array_like or sparse array or matrix, 2 dimensions
        compressed-sparse graph, with shape (N, N).
    normed : bool, optional
        If True, then compute symmetrically normalized Laplacian.
        Default: False.
    return_diag : bool, optional
        If True, then also return an array related to vertex degrees.
        Default: False.
    use_out_degree : bool, optional
        If True, then use out-degree instead of in-degree.
        This distinction matters only if the graph is asymmetric.
        Default: False.
    copy: bool, optional
        If False, then change `csgraph` in place if possible,
        avoiding doubling the memory use.
        Default: True, for backward compatibility.
    form: 'array', or 'function', or 'lo'
        Determines the format of the output Laplacian:

        * 'array' is a numpy array;
        * 'function' is a pointer to evaluating the Laplacian-vector
          or Laplacian-matrix product;
        * 'lo' results in the format of the `LinearOperator`.

        Choosing 'function' or 'lo' always avoids doubling
        the memory use, ignoring `copy` value.
        Default: 'array', for backward compatibility.
    dtype: None or one of numeric numpy dtypes, optional
        The dtype of the output. If ``dtype=None``, the dtype of the
        output matches the dtype of the input csgraph, except for
        the case ``normed=True`` and integer-like csgraph, where
        the output dtype is 'float' allowing accurate normalization,
        but dramatically increasing the memory use.
        Default: None, for backward compatibility.
    symmetrized: bool, optional
        If True, then the output Laplacian is symmetric/Hermitian.
        The symmetrization is done by ``csgraph + csgraph.T.conj``
        without dividing by 2 to preserve integer dtypes if possible
        prior to the construction of the Laplacian.
        The symmetrization will increase the memory footprint of
        sparse matrices unless the sparsity pattern is symmetric or
        `form` is 'function' or 'lo'.
        Default: False, for backward compatibility.

    Returns
    -------
    lap : ndarray, or sparse array or matrix, or `LinearOperator`
        The N x N Laplacian of csgraph. It will be a NumPy array (dense)
        if the input was dense, or a sparse array otherwise, or
        the format of a function or `LinearOperator` if
        `form` equals 'function' or 'lo', respectively.
    diag : ndarray, optional
        The length-N main diagonal of the Laplacian matrix.
        For the normalized Laplacian, this is the array of square roots
        of vertex degrees or 1 if the degree is zero.

    Notes
    -----
    The Laplacian matrix of a graph is sometimes referred to as the
    "Kirchhoff matrix" or just the "Laplacian", and is useful in many
    parts of spectral graph theory.
    In particular, the eigen-decomposition of the Laplacian can give
    insight into many properties of the graph, e.g.,
    is commonly used for spectral data embedding and clustering.

    The constructed Laplacian doubles the memory use if ``copy=True`` and
    ``form="array"`` which is the default.
    Choosing ``copy=False`` has no effect unless ``form="array"``
    or the matrix is sparse in the ``coo`` format, or dense array, except
    for the integer input with ``normed=True`` that forces the float output.

    Sparse input is reformatted into ``coo`` if ``form="array"``,
    which is the default.

    If the input adjacency matrix is not symmetric, the Laplacian is
    also non-symmetric unless ``symmetrized=True`` is used.

    Diagonal entries of the input adjacency matrix are ignored and
    replaced with zeros for the purpose of normalization where ``normed=True``.
    The normalization uses the inverse square roots of row-sums of the input
    adjacency matrix, and thus may fail if the row-sums contain
    negative or complex with a non-zero imaginary part values.

    The normalization is symmetric, making the normalized Laplacian also
    symmetric if the input csgraph was symmetric.

    References
    ----------
    .. [1] Laplacian matrix. https://en.wikipedia.org/wiki/Laplacian_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csgraph

    Our first illustration is the symmetric graph

    >>> G = np.arange(4) * np.arange(4)[:, np.newaxis]
    >>> G
    array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]])

    and its symmetric Laplacian matrix

    >>> csgraph.laplacian(G)
    array([[ 0,  0,  0,  0],
           [ 0,  5, -2, -3],
           [ 0, -2,  8, -6],
           [ 0, -3, -6,  9]])

    The non-symmetric graph

    >>> G = np.arange(9).reshape(3, 3)
    >>> G
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    has different row- and column sums, resulting in two varieties
    of the Laplacian matrix, using an in-degree, which is the default

    >>> L_in_degree = csgraph.laplacian(G)
    >>> L_in_degree
    array([[ 9, -1, -2],
           [-3,  8, -5],
           [-6, -7,  7]])

    or alternatively an out-degree

    >>> L_out_degree = csgraph.laplacian(G, use_out_degree=True)
    >>> L_out_degree
    array([[ 3, -1, -2],
           [-3,  8, -5],
           [-6, -7, 13]])

    Constructing a symmetric Laplacian matrix, one can add the two as

    >>> L_in_degree + L_out_degree.T
    array([[ 12,  -4,  -8],
            [ -4,  16, -12],
            [ -8, -12,  20]])

    or use the ``symmetrized=True`` option

    >>> csgraph.laplacian(G, symmetrized=True)
    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    that is equivalent to symmetrizing the original graph

    >>> csgraph.laplacian(G + G.T)
    array([[ 12,  -4,  -8],
           [ -4,  16, -12],
           [ -8, -12,  20]])

    The goal of normalization is to make the non-zero diagonal entries
    of the Laplacian matrix to be all unit, also scaling off-diagonal
    entries correspondingly. The normalization can be done manually, e.g.,

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True)
    >>> L
    array([[ 2, -1, -1],
           [-1,  2, -1],
           [-1, -1,  2]])
    >>> d
    array([2, 2, 2])
    >>> scaling = np.sqrt(d)
    >>> scaling
    array([1.41421356, 1.41421356, 1.41421356])
    >>> (1/scaling)*L*(1/scaling)
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    Or using ``normed=True`` option

    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    >>> L
    array([[ 1. , -0.5, -0.5],
           [-0.5,  1. , -0.5],
           [-0.5, -0.5,  1. ]])

    which now instead of the diagonal returns the scaling coefficients

    >>> d
    array([1.41421356, 1.41421356, 1.41421356])

    Zero scaling coefficients are substituted with 1s, where scaling
    has thus no effect, e.g.,

    >>> G = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    >>> G
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0]])
    >>> L, d = csgraph.laplacian(G, return_diag=True, normed=True)
    >>> L
    array([[ 0., -0., -0.],
           [-0.,  1., -1.],
           [-0., -1.,  1.]])
    >>> d
    array([1., 1., 1.])

    Only the symmetric normalization is implemented, resulting
    in a symmetric Laplacian matrix if and only if its graph is symmetric
    and has all non-negative degrees, like in the examples above.

    The output Laplacian matrix is by default a dense array or a sparse
    array or matrix inferring its class, shape, format, and dtype from
    the input graph matrix:

    >>> G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).astype(np.float32)
    >>> G
    array([[0., 1., 1.],
           [1., 0., 1.],
           [1., 1., 0.]], dtype=float32)
    >>> csgraph.laplacian(G)
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]], dtype=float32)

    but can alternatively be generated matrix-free as a LinearOperator:

    >>> L = csgraph.laplacian(G, form="lo")
    >>> L
    <3x3 _CustomLinearOperator with dtype=float32>
    >>> L(np.eye(3))
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]])

    or as a lambda-function:

    >>> L = csgraph.laplacian(G, form="function")
    >>> L
    <function _laplace.<locals>.<lambda> at 0x0000012AE6F5A598>
    >>> L(np.eye(3))
    array([[ 2., -1., -1.],
           [-1.,  2., -1.],
           [-1., -1.,  2.]])

    The Laplacian matrix is used for
    spectral data clustering and embedding
    as well as for spectral graph partitioning.
    Our final example illustrates the latter
    for a noisy directed linear graph.

    >>> from scipy.sparse import diags_array, random_array
    >>> from scipy.sparse.linalg import lobpcg

    Create a directed linear graph with ``N=35`` vertices
    using a sparse adjacency matrix ``G``:

    >>> N = 35
    >>> G = diags_array(np.ones(N - 1), offsets=1, format="csr")

    Fix a random seed ``rng`` and add a random sparse noise to the graph ``G``:

    >>> rng = np.random.default_rng()
    >>> G += 1e-2 * random_array((N, N), density=0.1, rng=rng)

    Set initial approximations for eigenvectors:

    >>> X = rng.random((N, 2))

    The constant vector of ones is always a trivial eigenvector
    of the non-normalized Laplacian to be filtered out:

    >>> Y = np.ones((N, 1))

    Alternating (1) the sign of the graph weights allows determining
    labels for spectral max- and min- cuts in a single loop.
    Since the graph is undirected, the option ``symmetrized=True``
    must be used in the construction of the Laplacian.
    The option ``normed=True`` cannot be used in (2) for the negative weights
    here as the symmetric normalization evaluates square roots.
    The option ``form="lo"`` in (2) is matrix-free, i.e., guarantees
    a fixed memory footprint and read-only access to the graph.
    Calling the eigenvalue solver ``lobpcg`` (3) computes the Fiedler vector
    that determines the labels as the signs of its components in (5).
    Since the sign in an eigenvector is not deterministic and can flip,
    we fix the sign of the first component to be always +1 in (4).

    >>> for cut in ["max", "min"]:
    ...     G = -G  # 1.
    ...     L = csgraph.laplacian(G, symmetrized=True, form="lo")  # 2.
    ...     _, eves = lobpcg(L, X, Y=Y, largest=False, tol=1e-2)  # 3.
    ...     eves *= np.sign(eves[0, 0])  # 4.
    ...     print(cut + "-cut labels:\\n", 1 * (eves[:, 0]>0))  # 5.
    max-cut labels:
    [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]
    min-cut labels:
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    As anticipated for a (slightly noisy) linear graph,
    the max-cut strips all the edges of the graph coloring all
    odd vertices into one color and all even vertices into another one,
    while the balanced min-cut partitions the graph
    in the middle by deleting a single edge.
    Both determined partitions are optimal.
    """
    is_pydata_sparse = is_pydata_spmatrix(csgraph)
    if is_pydata_sparse:
        pydata_sparse_cls = csgraph.__class__
        csgraph = convert_pydata_sparse_to_scipy(csgraph)
    if csgraph.ndim != 2 or csgraph.shape[0] != csgraph.shape[1]:
        raise ValueError('csgraph must be a square matrix or array')

    if normed and (
        np.issubdtype(csgraph.dtype, np.signedinteger)
        or np.issubdtype(csgraph.dtype, np.uint)
    ):
        csgraph = csgraph.astype(np.float64)

    if form == "array":
        create_lap = (
            _laplacian_sparse if issparse(csgraph) else _laplacian_dense
        )
    else:
        create_lap = (
            _laplacian_sparse_flo
            if issparse(csgraph)
            else _laplacian_dense_flo
        )

    degree_axis = 1 if use_out_degree else 0

    lap, d = create_lap(
        csgraph,
        normed=normed,
        axis=degree_axis,
        copy=copy,
        form=form,
        dtype=dtype,
        symmetrized=symmetrized,
    )
    if is_pydata_sparse:
        lap = pydata_sparse_cls.from_scipy_sparse(lap)
    if return_diag:
        return lap, d
    return lap


def _setdiag_dense(m, d):
    step = len(d) + 1
    m.flat[::step] = d


def _laplace(m, d):
    return lambda v: v * d[:, np.newaxis] - m @ v


def _laplace_normed(m, d, nd):
    laplace = _laplace(m, d)
    return lambda v: nd[:, np.newaxis] * laplace(v * nd[:, np.newaxis])


def _laplace_sym(m, d):
    return (
        lambda v: v * d[:, np.newaxis]
        - m @ v
        - np.transpose(np.conjugate(np.transpose(np.conjugate(v)) @ m))
    )


def _laplace_normed_sym(m, d, nd):
    laplace_sym = _laplace_sym(m, d)
    return lambda v: nd[:, np.newaxis] * laplace_sym(v * nd[:, np.newaxis])


def _linearoperator(mv, shape, dtype):
    return LinearOperator(matvec=mv, matmat=mv, shape=shape, dtype=dtype)


def _laplacian_sparse_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    # The keyword argument `copy` is unused and has no effect here.
    del copy

    if dtype is None:
        dtype = graph.dtype

    graph_sum = np.asarray(graph.sum(axis=axis)).ravel()
    graph_diagonal = graph.diagonal()
    diag = graph_sum - graph_diagonal
    if symmetrized:
        graph_sum += np.asarray(graph.sum(axis=1 - axis)).ravel()
        diag = graph_sum - graph_diagonal - graph_diagonal

    if normed:
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(graph, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(graph, graph_sum, 1.0 / w)
        if form == "function":
            return md, w.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, w.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")
    else:
        if symmetrized:
            md = _laplace_sym(graph, graph_sum)
        else:
            md = _laplace(graph, graph_sum)
        if form == "function":
            return md, diag.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, diag.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")


def _laplacian_sparse(graph, normed, axis, copy, form, dtype, symmetrized):
    # The keyword argument `form` is unused and has no effect here.
    del form

    if dtype is None:
        dtype = graph.dtype

    needs_copy = False
    if graph.format in ('lil', 'dok'):
        m = graph.tocoo()
    else:
        m = graph
        if copy:
            needs_copy = True

    if symmetrized:
        m += m.T.conj()

    w = np.asarray(m.sum(axis=axis)).ravel() - m.diagonal()
    if normed:
        m = m.tocoo(copy=needs_copy)
        isolated_node_mask = (w == 0)
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m.data /= w[m.row]
        m.data /= w[m.col]
        m.data *= -1
        m.setdiag(1 - isolated_node_mask)
    else:
        if m.format == 'dia':
            m = m.copy()
        else:
            m = m.tocoo(copy=needs_copy)
        m.data *= -1
        m.setdiag(w)

    return m.astype(dtype, copy=False), w.astype(dtype)


def _laplacian_dense_flo(graph, normed, axis, copy, form, dtype, symmetrized):

    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    if dtype is None:
        dtype = m.dtype

    graph_sum = m.sum(axis=axis)
    graph_diagonal = m.diagonal()
    diag = graph_sum - graph_diagonal
    if symmetrized:
        graph_sum += m.sum(axis=1 - axis)
        diag = graph_sum - graph_diagonal - graph_diagonal

    if normed:
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(m, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(m, graph_sum, 1.0 / w)
        if form == "function":
            return md, w.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, w.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")
    else:
        if symmetrized:
            md = _laplace_sym(m, graph_sum)
        else:
            md = _laplace(m, graph_sum)
        if form == "function":
            return md, diag.astype(dtype, copy=False)
        elif form == "lo":
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return m, diag.astype(dtype, copy=False)
        else:
            raise ValueError(f"Invalid form: {form!r}")


def _laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized):

    if form != "array":
        raise ValueError(f'{form!r} must be "array"')

    if dtype is None:
        dtype = graph.dtype

    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    if dtype is None:
        dtype = m.dtype

    if symmetrized:
        m += m.T.conj()
    np.fill_diagonal(m, 0)
    w = m.sum(axis=axis)
    if normed:
        isolated_node_mask = (w == 0)
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m /= w
        m /= w[:, np.newaxis]
        m *= -1
        _setdiag_dense(m, 1 - isolated_node_mask)
    else:
        m *= -1
        _setdiag_dense(m, w)

    return m.astype(dtype, copy=False), w.astype(dtype, copy=False)
<|endoftext|>"""Sparse block 1-norm estimator.
"""

import numpy as np
from scipy.sparse.linalg import aslinearoperator


__all__ = ['onenormest']


def onenormest(A, t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Compute a lower bound of the 1-norm of a sparse array.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can be transposed and that can
        produce matrix products.
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
        An underestimate of the 1-norm of the sparse array.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    Notes
    -----
    This is algorithm 2.4 of [1].

    In [2] it is described as follows.
    "This algorithm typically requires the evaluation of
    about 4t matrix-vector products and almost invariably
    produces a norm estimate (which is, in fact, a lower
    bound on the norm) correct to within a factor 3."

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Nicholas J. Higham and Francoise Tisseur (2000),
           "A Block Algorithm for Matrix 1-Norm Estimation,
           with an Application to 1-Norm Pseudospectra."
           SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2009),
           "A new scaling and squaring algorithm for the matrix exponential."
           SIAM J. Matrix Anal. Appl. Vol. 31, No. 3, pp. 970-989.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import onenormest
    >>> A = csc_array([[1., 0., 0.], [5., 8., 2.], [0., -1., 0.]], dtype=float)
    >>> A.toarray()
    array([[ 1.,  0.,  0.],
           [ 5.,  8.,  2.],
           [ 0., -1.,  0.]])
    >>> onenormest(A)
    9.0
    >>> np.linalg.norm(A.toarray(), ord=1)
    9.0
    """

    # Check the input.
    A = aslinearoperator(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected the operator to act like a square matrix')

    # If the operator size is small compared to t,
    # then it is easier to compute the exact norm.
    # Otherwise estimate the norm.
    n = A.shape[1]
    if t >= n:
        A_explicit = np.asarray(aslinearoperator(A).matmat(np.identity(n)))
        if A_explicit.shape != (n, n):
            raise Exception('internal error: ',
                    'unexpected shape ' + str(A_explicit.shape))
        col_abs_sums = abs(A_explicit).sum(axis=0)
        if col_abs_sums.shape != (n, ):
            raise Exception('internal error: ',
                    'unexpected shape ' + str(col_abs_sums.shape))
        argmax_j = np.argmax(col_abs_sums)
        v = elementary_vector(n, argmax_j)
        w = A_explicit[:, argmax_j]
        est = col_abs_sums[argmax_j]
    else:
        est, v, w, nmults, nresamples = _onenormest_core(A, A.H, t, itmax)

    # Report the norm estimate along with some certificates of the estimate.
    if compute_v or compute_w:
        result = (est,)
        if compute_v:
            result += (v,)
        if compute_w:
            result += (w,)
        return result
    else:
        return est


def _blocked_elementwise(func):
    """
    Decorator for an elementwise function, to apply it blockwise along
    first dimension, to avoid excessive memory usage in temporaries.
    """
    block_size = 2**20

    def wrapper(x):
        if x.shape[0] < block_size:
            return func(x)
        else:
            y0 = func(x[:block_size])
            y = np.zeros((x.shape[0],) + y0.shape[1:], dtype=y0.dtype)
            y[:block_size] = y0
            del y0
            for j in range(block_size, x.shape[0], block_size):
                y[j:j+block_size] = func(x[j:j+block_size])
            return y
    return wrapper


@_blocked_elementwise
def sign_round_up(X):
    """
    This should do the right thing for both real and complex matrices.

    From Higham and Tisseur:
    "Everything in this section remains valid for complex matrices
    provided that sign(A) is redefined as the matrix (aij / |aij|)
    (and sign(0) = 1) transposes are replaced by conjugate transposes."

    """
    Y = X.copy()
    Y[Y == 0] = 1
    Y /= np.abs(Y)
    return Y


@_blocked_elementwise
def _max_abs_axis1(X):
    return np.max(np.abs(X), axis=1)


def _sum_abs_axis0(X):
    block_size = 2**20
    r = None
    for j in range(0, X.shape[0], block_size):
        y = np.sum(np.abs(X[j:j+block_size]), axis=0)
        if r is None:
            r = y
        else:
            r += y
    return r


def elementary_vector(n, i):
    v = np.zeros(n, dtype=float)
    v[i] = 1
    return v


def vectors_are_parallel(v, w):
    # Columns are considered parallel when they are equal or negative.
    # Entries are required to be in {-1, 1},
    # which guarantees that the magnitudes of the vectors are identical.
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError('expected conformant vectors with entries in {-1,1}')
    n = v.shape[0]
    return np.dot(v, w) == n


def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    for v in X.T:
        if not any(vectors_are_parallel(v, w) for w in Y.T):
            return False
    return True


def column_needs_resampling(i, X, Y=None):
    # column i of X needs resampling if either
    # it is parallel to a previous column of X or
    # it is parallel to a column of Y
    n, t = X.shape
    v = X[:, i]
    if any(vectors_are_parallel(v, X[:, j]) for j in range(i)):
        return True
    if Y is not None:
        if any(vectors_are_parallel(v, w) for w in Y.T):
            return True
    return False


def resample_column(i, X):
    X[:, i] = np.random.randint(0, 2, size=X.shape[0])*2 - 1


def less_than_or_close(a, b):
    return np.allclose(a, b) or (a < b)


def _algorithm_2_2(A, AT, t):
    """
    This is Algorithm 2.2.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.

    Returns
    -------
    g : sequence
        A non-negative decreasing vector
        such that g[j] is a lower bound for the 1-norm
        of the column of A of jth largest 1-norm.
        The first entry of this vector is therefore a lower bound
        on the 1-norm of the linear operator A.
        This sequence has length t.
    ind : sequence
        The ith entry of ind is the index of the column A whose 1-norm
        is given by g[i].
        This sequence of indices has length t, and its entries are
        chosen from range(n), possibly with repetition,
        where n is the order of the operator A.

    Notes
    -----
    This algorithm is mainly for testing.
    It uses the 'ind' array in a way that is similar to
    its usage in algorithm 2.4. This algorithm 2.2 may be easier to test,
    so it gives a chance of uncovering bugs related to indexing
    which could have propagated less noticeably to algorithm 2.4.

    """
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    n = A_linear_operator.shape[0]

    # Initialize the X block with columns of unit 1-norm.
    X = np.ones((n, t))
    if t > 1:
        X[:, 1:] = np.random.randint(0, 2, size=(n, t-1))*2 - 1
    X /= float(n)

    # Iteratively improve the lower bounds.
    # Track extra things, to assert invariants for debugging.
    g_prev = None
    h_prev = None
    k = 1
    ind = range(t)
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        g = _sum_abs_axis0(Y)
        best_j = np.argmax(g)
        g.sort()
        g = g[::-1]
        S = sign_round_up(Y)
        Z = np.asarray(AT_linear_operator.matmat(S))
        h = _max_abs_axis1(Z)

        # If this algorithm runs for fewer than two iterations,
        # then its return values do not have the properties indicated
        # in the description of the algorithm.
        # In particular, the entries of g are not 1-norms of any
        # column of A until the second iteration.
        # Therefore we will require the algorithm to run for at least
        # two iterations, even though this requirement is not stated
        # in the description of the algorithm.
        if k >= 2:
            if less_than_or_close(max(h), np.dot(Z[:, best_j], X[:, best_j])):
                break
        ind = np.argsort(h)[::-1][:t]
        h = h[ind]
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])

        # Check invariant (2.2).
        if k >= 2:
            if not less_than_or_close(g_prev[0], h_prev[0]):
                raise Exception('invariant (2.2) is violated')
            if not less_than_or_close(h_prev[0], g[0]):
                raise Exception('invariant (2.2) is violated')

        # Check invariant (2.3).
        if k >= 3:
            for j in range(t):
                if not less_than_or_close(g[j], g_prev[j]):
                    raise Exception('invariant (2.3) is violated')

        # Update for the next iteration.
        g_prev = g
        h_prev = h
        k += 1

    # Return the lower bounds and the corresponding column indices.
    return g, ind


def _onenormest_core(A, AT, t, itmax):
    """
    Compute a lower bound of the 1-norm of a sparse array.

    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
    itmax : int, optional
        Use at most this many iterations.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse array.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.
    nmults : int, optional
        The number of matrix products that were computed.
    nresamples : int, optional
        The number of times a parallel column was observed,
        necessitating a re-randomization of the column.

    Notes
    -----
    This is algorithm 2.4.

    """
    # This function is a more or less direct translation
    # of Algorithm 2.4 from the Higham and Tisseur (2000) paper.
    A_linear_operator = aslinearoperator(A)
    AT_linear_operator = aslinearoperator(AT)
    if itmax < 2:
        raise ValueError('at least two iterations are required')
    if t < 1:
        raise ValueError('at least one column is required')
    n = A.shape[0]
    if t >= n:
        raise ValueError('t should be smaller than the order of A')
    # Track the number of big*small matrix multiplications
    # and the number of resamplings.
    nmults = 0
    nresamples = 0
    # "We now explain our choice of starting matrix.  We take the first
    # column of X to be the vector of 1s [...] This has the advantage that
    # for a matrix with nonnegative elements the algorithm converges
    # with an exact estimate on the second iteration, and such matrices
    # arise in applications [...]"
    X = np.ones((n, t), dtype=float)
    # "The remaining columns are chosen as rand{-1,1},
    # with a check for and correction of parallel columns,
    # exactly as for S in the body of the algorithm."
    if t > 1:
        for i in range(1, t):
            # These are technically initial samples, not resamples,
            # so the resampling count is not incremented.
            resample_column(i, X)
        for i in range(t):
            while column_needs_resampling(i, X):
                resample_column(i, X)
                nresamples += 1
    # "Choose starting matrix X with columns of unit 1-norm."
    X /= float(n)
    # "indices of used unit vectors e_j"
    ind_hist = np.zeros(0, dtype=np.intp)
    est_old = 0
    S = np.zeros((n, t), dtype=float)
    k = 1
    ind = None
    while True:
        Y = np.asarray(A_linear_operator.matmat(X))
        nmults += 1
        mags = _sum_abs_axis0(Y)
        est = np.max(mags)
        best_j = np.argmax(mags)
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = ind[best_j]
            w = Y[:, best_j]
        # (1)
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > itmax:
            break
        S = sign_round_up(Y)
        del Y
        # (2)
        if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
            break
        if t > 1:
            # "Ensure that no column of S is parallel to another column of S
            # or to a column of S_old by replacing columns of S by rand{-1,1}."
            for i in range(t):
                while column_needs_resampling(i, S, S_old):
                    resample_column(i, S)
                    nresamples += 1
        del S_old
        # (3)
        Z = np.asarray(AT_linear_operator.matmat(S))
        nmults += 1
        h = _max_abs_axis1(Z)
        del Z
        # (4)
        if k >= 2 and max(h) == h[ind_best]:
            break
        # "Sort h so that h_first >= ... >= h_last
        # and re-order ind correspondingly."
        #
        # Later on, we will need at most t+len(ind_hist) largest
        # entries, so drop the rest
        ind = np.argsort(h)[::-1][:t+len(ind_hist)].copy()
        del h
        if t > 1:
            # (5)
            # Break if the most promising t vectors have been visited already.
            if np.isin(ind[:t], ind_hist).all():
                break
            # Put the most promising unvisited vectors at the front of the list
            # and put the visited vectors at the end of the list.
            # Preserve the order of the indices induced by the ordering of h.
            seen = np.isin(ind, ind_hist)
            ind = np.concatenate((ind[~seen], ind[seen]))
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])

        new_ind = ind[:t][~np.isin(ind[:t], ind_hist)]
        ind_hist = np.concatenate((ind_hist, new_ind))
        k += 1
    v = elementary_vector(n, ind_best)
    return est, v, w, nmults, nresamples
<|endoftext|>"""Abstract linear algebra library.

This module defines a class hierarchy that implements a kind of "lazy"
matrix representation, called the ``LinearOperator``. It can be used to do
linear algebra with extremely large sparse or structured matrices, without
representing those explicitly in memory. Such matrices can be added,
multiplied, transposed, etc.

As a motivating example, suppose you want have a matrix where almost all of
the elements have the value one. The standard sparse matrix representation
skips the storage of zeros, but not ones. By contrast, a LinearOperator is
able to represent such matrices efficiently. First, we need a compact way to
represent an all-ones matrix::

    >>> import numpy as np
    >>> from scipy.sparse.linalg._interface import LinearOperator
    >>> class Ones(LinearOperator):
    ...     def __init__(self, shape):
    ...         super().__init__(dtype=None, shape=shape)
    ...     def _matvec(self, x):
    ...         return np.repeat(x.sum(), self.shape[0])

Instances of this class emulate ``np.ones(shape)``, but using a constant
amount of storage, independent of ``shape``. The ``_matvec`` method specifies
how this linear operator multiplies with (operates on) a vector. We can now
add this operator to a sparse matrix that stores only offsets from one::

    >>> from scipy.sparse.linalg._interface import aslinearoperator
    >>> from scipy.sparse import csr_array
    >>> offsets = csr_array([[1, 0, 2], [0, -1, 0], [0, 0, 3]])
    >>> A = aslinearoperator(offsets) + Ones(offsets.shape)
    >>> A.dot([1, 2, 3])
    array([13,  4, 15])

The result is the same as that given by its dense, explicitly-stored
counterpart::

    >>> (np.ones(A.shape, A.dtype) + offsets.toarray()).dot([1, 2, 3])
    array([13,  4, 15])

Several algorithms in the ``scipy.sparse`` library are able to operate on
``LinearOperator`` instances.
"""

import warnings

import numpy as np

from scipy.sparse import issparse
from scipy.sparse._sputils import isshape, isintlike, asmatrix, is_pydata_spmatrix

__all__ = ['LinearOperator', 'aslinearoperator']


class LinearOperator:
    """Common interface for performing matrix vector products

    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A@x=b.
    Such solvers only require the computation of matrix vector
    products, A@v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.

    To construct a concrete LinearOperator, either pass appropriate
    callables to the constructor of this class, or subclass it.

    A subclass must implement either one of the methods ``_matvec``
    and ``_matmat``, and the attributes/properties ``shape`` (pair of
    integers) and ``dtype`` (may be None). It may call the ``__init__``
    on this class to have these attributes validated. Implementing
    ``_matvec`` automatically implements ``_matmat`` (using a naive
    algorithm) and vice-versa.

    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint``
    to implement the Hermitian adjoint (conjugate transpose). As with
    ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or
    ``_adjoint`` implements the other automatically. Implementing
    ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for
    backwards compatibility.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns returns A @ v.
    rmatvec : callable f(v)
        Returns A^H @ v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A @ V, where V is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the matrix.
    rmatmat : callable f(V)
        Returns A^H @ V, where V is a dense matrix with dimensions (M, K).

    Attributes
    ----------
    args : tuple
        For linear operators describing products etc. of other linear
        operators, the operands of the binary operation.
    ndim : int
        Number of dimensions (this is always 2)

    See Also
    --------
    aslinearoperator : Construct LinearOperators

    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.

    It is highly recommended to explicitly specify the `dtype`, otherwise
    it is determined automatically at the cost of a single matvec application
    on `int8` zero vector using the promoted `dtype` of the output.
    Python `int` could be difficult to automatically cast to numpy integers
    in the definition of the `matvec` so the determination may be inaccurate.
    It is assumed that `matmat`, `rmatvec`, and `rmatmat` would result in
    the same dtype of the output given an `int8` input as `matvec`.

    LinearOperator instances can also be multiplied, added with each
    other and exponentiated, all lazily: the result of these operations
    is always a new, composite LinearOperator, that defers linear
    operations to the original operators and combines the results.

    More details regarding how to subclass a LinearOperator and several
    examples of concrete LinearOperator instances can be found in the
    external project `PyLops <https://pylops.readthedocs.io>`_.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=int8>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A @ np.ones(2)
    array([ 2.,  3.])

    """

    ndim = 2
    # Necessary for right matmul with numpy arrays.
    __array_ufunc__ = None

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # Operate as _CustomLinearOperator factory.
            return super().__new__(_CustomLinearOperator)
        else:
            obj = super().__new__(cls)

            if (type(obj)._matvec == LinearOperator._matvec
                    and type(obj)._matmat == LinearOperator._matmat):
                warnings.warn("LinearOperator subclass should implement"
                              " at least one of _matvec and _matmat.",
                              category=RuntimeWarning, stacklevel=2)

            return obj

    def __init__(self, dtype, shape):
        """Initialize this LinearOperator.

        To be called by subclasses. ``dtype`` may be None; ``shape`` should
        be convertible to a length-2 tuple.
        """
        if dtype is not None:
            dtype = np.dtype(dtype)

        shape = tuple(shape)
        if not isshape(shape):
            raise ValueError(f"invalid shape {shape!r} (must be 2-d)")

        self.dtype = dtype
        self.shape = shape

    def _init_dtype(self):
        """Determine the dtype by executing `matvec` on an `int8` test vector.

        In `np.promote_types` hierarchy, the type `int8` is the smallest,
        so we call `matvec` on `int8` and use the promoted dtype of the output
        to set the default `dtype` of the `LinearOperator`.
        We assume that `matmat`, `rmatvec`, and `rmatmat` would result in
        the same dtype of the output given an `int8` input as `matvec`.

        Called from subclasses at the end of the __init__ routine.
        """
        if self.dtype is None:
            v = np.zeros(self.shape[-1], dtype=np.int8)
            try:
                matvec_v = np.asarray(self.matvec(v))
            except OverflowError:
                # Python large `int` promoted to `np.int64`or `np.int32`
                self.dtype = np.dtype(int)
            else:
                self.dtype = matvec_v.dtype

    def _matmat(self, X):
        """Default matrix-matrix multiplication handler.

        Falls back on the user-defined _matvec method, so defining that will
        define matrix multiplication (though in a very suboptimal way).
        """

        return np.hstack([self.matvec(col.reshape(-1,1)) for col in X.T])

    def _matvec(self, x):
        """Default matrix-vector multiplication handler.

        If self is a linear operator of shape (M, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (M,) or (M, 1) ndarray.

        This default implementation falls back on _matmat, so defining that
        will define matrix-vector multiplication as well.
        """
        return self.matmat(x.reshape(-1, 1))

    def matvec(self, x):
        """Matrix-vector multiplication.

        Performs the operation y=A@x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.

        """

        x = np.asanyarray(x)

        M,N = self.shape

        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if isinstance(x, np.matrix):
            y = asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M,1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')

        return y

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H @ x where A is an MxN linear
        operator and x is a column vector or 1-d array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (M,) or (M,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (N,) or (N,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden
        _rmatvec method to ensure that y has the correct shape and type.

        """

        x = np.asanyarray(x)

        M,N = self.shape

        if x.shape != (M,) and x.shape != (M,1):
            raise ValueError('dimension mismatch')

        y = self._rmatvec(x)

        if isinstance(x, np.matrix):
            y = asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N,1)
        else:
            raise ValueError('invalid shape returned by user-defined rmatvec()')

        return y

    def _rmatvec(self, x):
        """Default implementation of _rmatvec; defers to adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            # _adjoint not overridden, prevent infinite recursion
            if (hasattr(self, "_rmatmat")
                    and type(self)._rmatmat != LinearOperator._rmatmat):
                # Try to use _rmatmat as a fallback
                return self._rmatmat(x.reshape(-1, 1)).reshape(-1)
            raise NotImplementedError
        else:
            return self.H.matvec(x)

    def matmat(self, X):
        """Matrix-matrix multiplication.

        Performs the operation y=A@X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : {matrix, ndarray}
            An array with shape (N,K).

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden
        _matmat method to ensure that y has the correct type.

        """
        if not (issparse(X) or is_pydata_spmatrix(X)):
            X = np.asanyarray(X)

        if X.ndim != 2:
            raise ValueError(f'expected 2-d ndarray or matrix, not {X.ndim}-d')

        if X.shape[0] != self.shape[1]:
            raise ValueError(f'dimension mismatch: {self.shape}, {X.shape}')

        try:
            Y = self._matmat(X)
        except Exception as e:
            if issparse(X) or is_pydata_spmatrix(X):
                raise TypeError(
                    "Unable to multiply a LinearOperator with a sparse matrix."
                    " Wrap the matrix in aslinearoperator first."
                ) from e
            raise

        if isinstance(Y, np.matrix):
            Y = asmatrix(Y)

        return Y

    def rmatmat(self, X):
        """Adjoint matrix-matrix multiplication.

        Performs the operation y = A^H @ x where A is an MxN linear
        operator and x is a column vector or 1-d array, or 2-d array.
        The default implementation defers to the adjoint.

        Parameters
        ----------
        X : {matrix, ndarray}
            A matrix or 2D array.

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or 2D array depending on the type of the input.

        Notes
        -----
        This rmatmat wraps the user-specified rmatmat routine.

        """
        if not (issparse(X) or is_pydata_spmatrix(X)):
            X = np.asanyarray(X)

        if X.ndim != 2:
            raise ValueError('expected 2-d ndarray or matrix, not %d-d'
                             % X.ndim)

        if X.shape[0] != self.shape[0]:
            raise ValueError(f'dimension mismatch: {self.shape}, {X.shape}')

        try:
            Y = self._rmatmat(X)
        except Exception as e:
            if issparse(X) or is_pydata_spmatrix(X):
                raise TypeError(
                    "Unable to multiply a LinearOperator with a sparse matrix."
                    " Wrap the matrix in aslinearoperator() first."
                ) from e
            raise

        if isinstance(Y, np.matrix):
            Y = asmatrix(Y)
        return Y

    def _rmatmat(self, X):
        """Default implementation of _rmatmat defers to rmatvec or adjoint."""
        if type(self)._adjoint == LinearOperator._adjoint:
            return np.hstack([self.rmatvec(col.reshape(-1, 1)) for col in X.T])
        else:
            return self.H.matmat(X)

    def __call__(self, x):
        return self@x

    def __mul__(self, x):
        return self.dot(x)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise ValueError("Can only divide a linear operator by a scalar.")

        return _ScaledLinearOperator(self, 1.0/other)

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.

        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if not issparse(x) and not is_pydata_spmatrix(x):
                # Sparse matrices shouldn't be converted to numpy arrays.
                x = np.asarray(x)

            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError(f'expected 1-d or 2-d array or matrix, got {x!r}')

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return self._rdot(x)

    def _rdot(self, x):
        """Matrix-matrix or matrix-vector multiplication from the right.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        xA : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x from the right.

        Notes
        -----
        This is copied from dot to implement right multiplication.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(x, self)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if not issparse(x) and not is_pydata_spmatrix(x):
                # Sparse matrices shouldn't be converted to numpy arrays.
                x = np.asarray(x)

            # We use transpose instead of rmatvec/rmatmat to avoid
            # unnecessary complex conjugation if possible.
            if x.ndim == 1 or x.ndim == 2 and x.shape[0] == 1:
                return self.T.matvec(x.T).T
            elif x.ndim == 2:
                return self.T.matmat(x.T).T
            else:
                raise ValueError(f'expected 1-d or 2-d array or matrix, got {x!r}')

    def __pow__(self, p):
        if np.isscalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M,N = self.shape
        if self.dtype is None:
            dt = 'unspecified dtype'
        else:
            dt = 'dtype=' + str(self.dtype)

        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def adjoint(self):
        """Hermitian adjoint.

        Returns the Hermitian adjoint of self, aka the Hermitian
        conjugate or Hermitian transpose. For a complex matrix, the
        Hermitian adjoint is equal to the conjugate transpose.

        Can be abbreviated self.H instead of self.adjoint().

        Returns
        -------
        A_H : LinearOperator
            Hermitian adjoint of self.
        """
        return self._adjoint()

    H = property(adjoint)

    def transpose(self):
        """Transpose this linear operator.

        Returns a LinearOperator that represents the transpose of this one.
        Can be abbreviated self.T instead of self.transpose().
        """
        return self._transpose()

    T = property(transpose)

    def _adjoint(self):
        """Default implementation of _adjoint; defers to rmatvec."""
        return _AdjointLinearOperator(self)

    def _transpose(self):
        """ Default implementation of _transpose; defers to rmatvec + conj"""
        return _TransposedLinearOperator(self)


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 dtype=None, rmatmat=None):
        super().__init__(dtype, shape)

        self.args = ()

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec
        self.__rmatmat_impl = rmatmat
        self.__matmat_impl = matmat

        self._init_dtype()

    def _matmat(self, X):
        if self.__matmat_impl is not None:
            return self.__matmat_impl(X)
        else:
            return super()._matmat(X)

    def _matvec(self, x):
        return self.__matvec_impl(x)

    def _rmatvec(self, x):
        func = self.__rmatvec_impl
        if func is None:
            raise NotImplementedError("rmatvec is not defined")
        return self.__rmatvec_impl(x)

    def _rmatmat(self, X):
        if self.__rmatmat_impl is not None:
            return self.__rmatmat_impl(X)
        else:
            return super()._rmatmat(X)

    def _adjoint(self):
        return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
                                     matvec=self.__rmatvec_impl,
                                     rmatvec=self.__matvec_impl,
                                     matmat=self.__rmatmat_impl,
                                     rmatmat=self.__matmat_impl,
                                     dtype=self.dtype)


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super().__init__(dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        return self.A._rmatvec(x)

    def _rmatvec(self, x):
        return self.A._matvec(x)

    def _matmat(self, x):
        return self.A._rmatmat(x)

    def _rmatmat(self, x):
        return self.A._matmat(x)

class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super().__init__(dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        # NB. np.conj works also on sparse matrices
        return np.conj(self.A._rmatvec(np.conj(x)))

    def _rmatvec(self, x):
        return np.conj(self.A._matvec(np.conj(x)))

    def _matmat(self, x):
        # NB. np.conj works also on sparse matrices
        return np.conj(self.A._rmatmat(np.conj(x)))

    def _rmatmat(self, x):
        return np.conj(self.A._matmat(np.conj(x)))

def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, 'dtype'):
            dtypes.append(obj.dtype)
    return np.result_type(*dtypes)


class _SumLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape != B.shape:
            raise ValueError(f'cannot add {A} and {B}: shape mismatch')
        self.args = (A, B)
        super().__init__(_get_dtype([A, B]), A.shape)

    def _matvec(self, x):
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    def _rmatvec(self, x):
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    def _rmatmat(self, x):
        return self.args[0].rmatmat(x) + self.args[1].rmatmat(x)

    def _matmat(self, x):
        return self.args[0].matmat(x) + self.args[1].matmat(x)

    def _adjoint(self):
        A, B = self.args
        return A.H + B.H


class _ProductLinearOperator(LinearOperator):
    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or \
                not isinstance(B, LinearOperator):
            raise ValueError('both operands have to be a LinearOperator')
        if A.shape[1] != B.shape[0]:
            raise ValueError(f'cannot multiply {A} and {B}: shape mismatch')
        super().__init__(_get_dtype([A, B]),
                                                     (A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x):
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x):
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _rmatmat(self, x):
        return self.args[1].rmatmat(self.args[0].rmatmat(x))

    def _matmat(self, x):
        return self.args[0].matmat(self.args[1].matmat(x))

    def _adjoint(self):
        A, B = self.args
        return B.H @ A.H


class _ScaledLinearOperator(LinearOperator):
    def __init__(self, A, alpha):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if not np.isscalar(alpha):
            raise ValueError('scalar expected as alpha')
        if isinstance(A, _ScaledLinearOperator):
            A, alpha_original = A.args
            # Avoid in-place multiplication so that we don't accidentally mutate
            # the original prefactor.
            alpha = alpha * alpha_original

        dtype = _get_dtype([A], [type(alpha)])
        super().__init__(dtype, A.shape)
        self.args = (A, alpha)
        # Note: args[1] is alpha (a scalar), so use `*` below, not `@`

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return np.conj(self.args[1]) * self.args[0].rmatvec(x)

    def _rmatmat(self, x):
        return np.conj(self.args[1]) * self.args[0].rmatmat(x)

    def _matmat(self, x):
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self):
        A, alpha = self.args
        return A.H * np.conj(alpha)


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A, p):
        if not isinstance(A, LinearOperator):
            raise ValueError('LinearOperator expected as A')
        if A.shape[0] != A.shape[1]:
            raise ValueError(f'square LinearOperator expected, got {A!r}')
        if not isintlike(p) or p < 0:
            raise ValueError('non-negative integer expected as p')

        super().__init__(_get_dtype([A]), A.shape)
        self.args = (A, p)

    def _power(self, fun, x):
        res = np.array(x, copy=True)
        for i in range(self.args[1]):
            res = fun(res)
        return res

    def _matvec(self, x):
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x):
        return self._power(self.args[0].rmatvec, x)

    def _rmatmat(self, x):
        return self._power(self.args[0].rmatmat, x)

    def _matmat(self, x):
        return self._power(self.args[0].matmat, x)

    def _adjoint(self):
        A, p = self.args
        return A.H ** p


class MatrixLinearOperator(LinearOperator):
    def __init__(self, A):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixOperator(self.A)
        return self.__adj


class _AdjointMatrixOperator(MatrixLinearOperator):
    def __init__(self, adjoint_array):
        self.A = adjoint_array.T.conj()
        self.args = (adjoint_array,)
        self.shape = adjoint_array.shape[1], adjoint_array.shape[0]

    @property
    def dtype(self):
        return self.args[0].dtype

    def _adjoint(self):
        return MatrixLinearOperator(self.args[0])


class IdentityOperator(LinearOperator):
    def __init__(self, shape, dtype=None):
        super().__init__(dtype, shape)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _rmatmat(self, x):
        return x

    def _matmat(self, x):
        return x

    def _adjoint(self):
        return self


def aslinearoperator(A):
    """Return A as a LinearOperator.

    'A' may be any of the following types:
     - ndarray
     - matrix
     - sparse array (e.g. csr_array, lil_array, etc.)
     - LinearOperator
     - An object with .shape and .matvec attributes

    See the LinearOperator documentation for additional information.

    Notes
    -----
    If 'A' has no .dtype attribute, the data type is determined by calling
    :func:`LinearOperator.matvec()` - set the .dtype attribute to prevent this
    call upon the linear operator creation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import aslinearoperator
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> aslinearoperator(M)
    <2x3 MatrixLinearOperator with dtype=int32>
    """
    if isinstance(A, LinearOperator):
        return A

    elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
        if A.ndim > 2:
            raise ValueError('array must have ndim <= 2')
        A = np.atleast_2d(np.asarray(A))
        return MatrixLinearOperator(A)

    elif issparse(A) or is_pydata_spmatrix(A):
        return MatrixLinearOperator(A)

    else:
        if hasattr(A, 'shape') and hasattr(A, 'matvec'):
            rmatvec = None
            rmatmat = None
            dtype = None

            if hasattr(A, 'rmatvec'):
                rmatvec = A.rmatvec
            if hasattr(A, 'rmatmat'):
                rmatmat = A.rmatmat
            if hasattr(A, 'dtype'):
                dtype = A.dtype
            return LinearOperator(A.shape, A.matvec, rmatvec=rmatvec,
                                  rmatmat=rmatmat, dtype=dtype)

        else:
            raise TypeError('type not understood')
<|endoftext|>"""
Python wrapper for PROPACK
--------------------------

PROPACK is a collection of Fortran routines for iterative computation
of partial SVDs of large matrices or linear operators.

Based on BSD licensed pypropack project:
  http://github.com/jakevdp/pypropack
  Author: Jake Vanderplas <vanderplas@astro.washington.edu>

PROPACK source is BSD licensed, and available at
  http://soi.stanford.edu/~rmunk/PROPACK/
"""

__all__ = ['_svdp']

import numpy as np

from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import LinAlgError

from ._propack import _spropack  # type: ignore[attr-defined]
from ._propack import _dpropack  # type: ignore[attr-defined]
from ._propack import _cpropack  # type: ignore[attr-defined]
from ._propack import _zpropack  # type: ignore[attr-defined]


_lansvd_dict = {
    'f': _spropack.slansvd,
    'd': _dpropack.dlansvd,
    'F': _cpropack.clansvd,
    'D': _zpropack.zlansvd,
}


_lansvd_irl_dict = {
    'f': _spropack.slansvd_irl,
    'd': _dpropack.dlansvd_irl,
    'F': _cpropack.clansvd_irl,
    'D': _zpropack.zlansvd_irl,
}

_which_converter = {
    'LM': 'L',
    'SM': 'S',
}


class _AProd:
    """
    Wrapper class for linear operator

    The call signature of the __call__ method matches the callback of
    the PROPACK routines.
    """
    def __init__(self, A):
        try:
            self.A = aslinearoperator(A)
        except TypeError:
            self.A = aslinearoperator(np.asarray(A))

    def __call__(self, transa, m, n, x, y, sparm, iparm):
        if transa == 'n':
            y[:] = self.A.matvec(x)
        else:
            y[:] = self.A.rmatvec(x)

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        try:
            return self.A.dtype
        except AttributeError:
            return self.A.matvec(np.zeros(self.A.shape[1])).dtype


def _svdp(A, k, which='LM', irl_mode=True, kmax=None,
          compute_u=True, compute_v=True, v0=None, full_output=False, tol=0,
          delta=None, eta=None, anorm=0, cgs=False, elr=True,
          min_relgap=0.002, shifts=None, maxiter=None, rng=None):
    """
    Compute the singular value decomposition of a linear operator using PROPACK

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        Operator for which SVD will be computed.  If `A` is a LinearOperator
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int
        Number of singular values/vectors to compute
    which : {"LM", "SM"}
        Which singular triplets to compute:
        - 'LM': compute triplets corresponding to the `k` largest singular
                values
        - 'SM': compute triplets corresponding to the `k` smallest singular
                values
        `which='SM'` requires `irl_mode=True`.  Computes largest singular
        values by default.
    irl_mode : bool, optional
        If `True`, then compute SVD using IRL (implicitly restarted Lanczos)
        mode.  Default is `True`.
    kmax : int, optional
        Maximal number of iterations / maximal dimension of the Krylov
        subspace. Default is ``10 * k``.
    compute_u : bool, optional
        If `True` (default) then compute left singular vectors, `u`.
    compute_v : bool, optional
        If `True` (default) then compute right singular vectors, `v`.
    tol : float, optional
        The desired relative accuracy for computed singular values.
        If not specified, it will be set based on machine precision.
    v0 : array_like, optional
        Starting vector for iterations: must be of length ``A.shape[0]``.
        If not specified, PROPACK will generate a starting vector.
    full_output : bool, optional
        If `True`, then return sigma_bound.  Default is `False`.
    delta : float, optional
        Level of orthogonality to maintain between Lanczos vectors.
        Default is set based on machine precision.
    eta : float, optional
        Orthogonality cutoff.  During reorthogonalization, vectors with
        component larger than `eta` along the Lanczos vector will be purged.
        Default is set based on machine precision.
    anorm : float, optional
        Estimate of ``||A||``.  Default is ``0``.
    cgs : bool, optional
        If `True`, reorthogonalization is done using classical Gram-Schmidt.
        If `False` (default), it is done using modified Gram-Schmidt.
    elr : bool, optional
        If `True` (default), then extended local orthogonality is enforced
        when obtaining singular vectors.
    min_relgap : float, optional
        The smallest relative gap allowed between any shift in IRL mode.
        Default is ``0.001``.  Accessed only if ``irl_mode=True``.
    shifts : int, optional
        Number of shifts per restart in IRL mode.  Default is determined
        to satisfy ``k <= min(kmax-shifts, m, n)``.  Must be
        >= 0, but choosing 0 might lead to performance degradation.
        Accessed only if ``irl_mode=True``.
    maxiter : int, optional
        Maximum number of restarts in IRL mode.  Default is ``1000``.
        Accessed only if ``irl_mode=True``.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    u : ndarray
        The `k` largest (``which="LM"``) or smallest (``which="SM"``) left
        singular vectors, ``shape == (A.shape[0], 3)``, returned only if
        ``compute_u=True``.
    sigma : ndarray
        The top `k` singular values, ``shape == (k,)``
    vt : ndarray
        The `k` largest (``which="LM"``) or smallest (``which="SM"``) right
        singular vectors, ``shape == (3, A.shape[1])``, returned only if
        ``compute_v=True``.
    sigma_bound : ndarray
        the error bounds on the singular values sigma, returned only if
        ``full_output=True``.

    """
    if rng is None:
        raise ValueError("`rng` must be a normalized numpy.random.Generator instance")

    which = which.upper()
    if which not in {'LM', 'SM'}:
        raise ValueError("`which` must be either 'LM' or 'SM'")
    if not irl_mode and which == 'SM':
        raise ValueError("`which`='SM' requires irl_mode=True")

    aprod = _AProd(A)
    typ = aprod.dtype.char

    try:
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]
    except KeyError:
        # work with non-supported types using native system precision
        if np.iscomplexobj(np.empty(0, dtype=typ)):
            typ = np.dtype(complex).char
        else:
            typ = np.dtype(float).char
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]

    m, n = aprod.shape
    if (k < 1) or (k > min(m, n)):
        raise ValueError("k must be positive and not greater than m or n")

    if kmax is None:
        kmax = 10*k
    if maxiter is None:
        maxiter = 1000

    # guard against unnecessarily large kmax
    kmax = min(m + 1, n + 1, kmax)
    if kmax < k:
        raise ValueError(
            "kmax must be greater than or equal to k, "
            f"but kmax ({kmax}) < k ({k})")

    # convert python args to fortran args
    jobu = 'y' if compute_u else 'n'
    jobv = 'y' if compute_v else 'n'

    # these will be the output arrays
    u = np.zeros((m, kmax + 1), order='F', dtype=typ)
    v = np.zeros((n, kmax), order='F', dtype=typ)

    # Specify the starting vector.  if v0 is all zero, PROPACK will generate
    # a random starting vector: the random seed cannot be controlled in that
    # case, so we'll instead use numpy to generate a random vector
    if v0 is None:
        u[:, 0] = rng.uniform(size=m)
        if np.iscomplexobj(np.empty(0, dtype=typ)):  # complex type
            u[:, 0] += 1j * rng.uniform(size=m)
    else:
        try:
            u[:, 0] = v0
        except ValueError:
            raise ValueError(f"v0 must be of length {m}")

    # process options for the fit
    if delta is None:
        delta = np.sqrt(np.finfo(typ).eps)
    if eta is None:
        eta = np.finfo(typ).eps ** 0.75

    if irl_mode:
        doption = np.array((delta, eta, anorm, min_relgap), dtype=typ.lower())

        # validate or find default shifts
        if shifts is None:
            shifts = kmax - k
        if k > min(kmax - shifts, m, n):
            raise ValueError('shifts must satisfy '
                             'k <= min(kmax-shifts, m, n)!')
        elif shifts < 0:
            raise ValueError('shifts must be >= 0!')

    else:
        doption = np.array((delta, eta, anorm), dtype=typ.lower())

    ioption = np.array((int(bool(cgs)), int(bool(elr))), dtype='i')

    # If computing `u` or `v` (left and right singular vectors,
    # respectively), `blocksize` controls how large a fraction of the
    # work is done via fast BLAS level 3 operations.  A larger blocksize
    # may lead to faster computation at the expense of greater memory
    # consumption.  `blocksize` must be ``>= 1``.  Choosing blocksize
    # of 16, but docs don't specify; it's almost surely a
    # power of 2.
    blocksize = 16

    # Determine lwork & liwork:
    # the required lengths are specified in the PROPACK documentation
    if compute_u or compute_v:
        lwork = m + n + 9*kmax + 5*kmax*kmax + 4 + max(
            3*kmax*kmax + 4*kmax + 4,
            blocksize*max(m, n))
        liwork = 8*kmax
    else:
        lwork = m + n + 9*kmax + 2*kmax*kmax + 4 + max(m + n, 4*kmax + 4)
        liwork = 2*kmax + 1
    work = np.empty(lwork, dtype=typ.lower())
    iwork = np.empty(liwork, dtype=np.int32)

    # dummy arguments: these are passed to aprod, and not used in this wrapper
    dparm = np.empty(1, dtype=typ.lower())
    iparm = np.empty(1, dtype=np.int32)

    if typ.isupper():
        # PROPACK documentation is unclear on the required length of zwork.
        # Use the same length Julia's wrapper uses
        # see https://github.com/JuliaSmoothOptimizers/PROPACK.jl/
        zwork = np.empty(m + n + 32*m, dtype=typ)
        works = work, zwork, iwork
    else:
        works = work, iwork

    if irl_mode:
        u, sigma, bnd, v, info = lansvd_irl(_which_converter[which], jobu,
                                            jobv, m, n, shifts, k, maxiter,
                                            aprod, u, v, tol, *works, doption,
                                            ioption, dparm, iparm)
    else:
        u, sigma, bnd, v, info = lansvd(jobu, jobv, m, n, k, aprod, u, v, tol,
                                        *works, doption, ioption, dparm, iparm)

    if info > 0:
        raise LinAlgError(
            f"An invariant subspace of dimension {info} was found.")
    elif info < 0:
        raise LinAlgError(
            f"k={k} singular triplets did not converge within "
            f"kmax={kmax} iterations")

    # info == 0: The K largest (or smallest) singular triplets were computed
    # successfully!

    return u[:, :k], sigma, v[:, :k].conj().T, bnd
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'MatrixRankWarning', 'SuperLU', 'factorized',
    'spilu', 'splu', 'spsolve',
    'spsolve_triangular', 'use_solver', 'test'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="dsolve",
                                   private_modules=["_dsolve"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'LinearOperator', 'aslinearoperator',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="interface",
                                   private_modules=["_interface"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'bicg', 'bicgstab', 'cg', 'cgs', 'gcrotmk', 'gmres',
    'lgmres', 'lsmr', 'lsqr',
    'minres', 'qmr', 'tfqmr', 'test'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse.linalg", module="isolve",
                                   private_modules=["_isolve"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Sparse matrix norms.

"""
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy
import scipy.sparse as sp

from numpy import sqrt, abs

__all__ = ['norm']


def _sparse_frobenius_norm(x):
    data = sp._sputils._todata(x)
    return np.linalg.norm(data)


def norm(x, ord=None, axis=None):
    """
    Norm of a sparse matrix

    This function is able to return one of seven different matrix norms,
    depending on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : a sparse array
        Input sparse array.
    ord : {non-zero int, inf, -inf, 'fro'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.

    Returns
    -------
    n : float or ndarray

    Notes
    -----
    Some of the ord are not implemented because some associated functions like,
    _multi_svd_norm, are not yet available for sparse array.

    This docstring is modified based on numpy.linalg.norm.
    https://github.com/numpy/numpy/blob/main/numpy/linalg/linalg.py

    The following norms can be calculated:

    =====  ============================
    ord    norm for sparse arrays
    =====  ============================
    None   Frobenius norm
    'fro'  Frobenius norm
    inf    max(sum(abs(x), axis=1))
    -inf   min(sum(abs(x), axis=1))
    0      abs(x).sum(axis=axis)
    1      max(sum(abs(x), axis=0))
    -1     min(sum(abs(x), axis=0))
    2      Spectral norm (the largest singular value)
    -2     Not implemented
    other  Not implemented
    =====  ============================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from scipy.sparse import csr_array, diags_array
    >>> import numpy as np
    >>> from scipy.sparse.linalg import norm
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1, 0, 1],
           [ 2, 3, 4]])

    >>> b = csr_array(b)
    >>> norm(b)
    7.745966692414834
    >>> norm(b, 'fro')
    7.745966692414834
    >>> norm(b, np.inf)
    9
    >>> norm(b, -np.inf)
    2
    >>> norm(b, 1)
    7
    >>> norm(b, -1)
    6

    The matrix 2-norm or the spectral norm is the largest singular
    value, computed approximately and with limitations.

    >>> b = diags_array([-1, 1], [0, 1], shape=(9, 10))
    >>> norm(b, 2)
    1.9753...
    """
    x = convert_pydata_sparse_to_scipy(x, target_format="csr")
    if not issparse(x):
        raise TypeError("input is not sparse. use numpy.linalg.norm")

    # Check the default case first and handle it immediately.
    if axis is None and ord in (None, 'fro', 'f'):
        return _sparse_frobenius_norm(x)

    # Some norms require functions that are not implemented for all types.
    x = x.tocsr()

    if axis is None:
        axis = tuple(range(x.ndim))
    elif not isinstance(axis, tuple):
        msg = "'axis' must be None, an integer or a tuple of integers"
        try:
            int_axis = int(axis)
        except TypeError as e:
            raise TypeError(msg) from e
        if axis != int_axis:
            raise TypeError(msg)
        axis = (int_axis,)

    nd = x.ndim
    if len(axis) == 2:
        row_axis, col_axis = axis
        if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
            message = f'Invalid axis {axis!r} for an array with shape {x.shape!r}'
            raise ValueError(message)
        if row_axis % nd == col_axis % nd:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            # Only solver="lobpcg" supports all numpy dtypes
            _, s, _ = svds(x, k=1, solver="lobpcg")
            return s[0]
        elif ord == -2:
            raise NotImplementedError
            #return _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            return abs(x).sum(axis=row_axis).max()
        elif ord == np.inf:
            return abs(x).sum(axis=col_axis).max()
        elif ord == -1:
            return abs(x).sum(axis=row_axis).min()
        elif ord == -np.inf:
            return abs(x).sum(axis=col_axis).min()
        elif ord in (None, 'f', 'fro'):
            # The axis order does not matter for this norm.
            return _sparse_frobenius_norm(x)
        else:
            raise ValueError("Invalid norm order for matrices.")
    elif len(axis) == 1:
        a, = axis
        if not (-nd <= a < nd):
            message = f'Invalid axis {axis!r} for an array with shape {x.shape!r}'
            raise ValueError(message)
        if ord == np.inf:
            M = abs(x).max(axis=a)
        elif ord == -np.inf:
            M = abs(x).min(axis=a)
        elif ord == 0:
            # Zero norm
            M = (x != 0).sum(axis=a)
        elif ord == 1:
            # special case for speedup
            M = abs(x).sum(axis=a)
        elif ord in (2, None):
            M = sqrt(abs(x).power(2).sum(axis=a))
        else:
            try:
                ord + 1
            except TypeError as e:
                raise ValueError('Invalid norm order for vectors.') from e
            M = np.power(abs(x).power(ord).sum(axis=a), 1 / ord)
        if hasattr(M, 'toarray'):
            return M.toarray().ravel()
        elif hasattr(M, 'A'):
            return M.A.ravel()
        else:
            return M.ravel()
    else:
        raise ValueError("Improper number of dimensions to norm.")
<|endoftext|>"""Compute the action of the matrix exponential."""
from warnings import warn

import numpy as np

import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest

__all__ = ['expm_multiply']


def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=1).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=1))
    else:
        return np.linalg.norm(A, np.inf)


def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=0).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=0))
    else:
        return np.linalg.norm(A, 1)


def _trace(A):
    # A compatibility function which should eventually disappear.
    if is_pydata_spmatrix(A):
        return A.to_scipy_sparse().trace()
    else:
        return A.trace()


def traceest(A, m3, seed=None):
    """Estimate `np.trace(A)` using `3*m3` matrix-vector products.

    The result is not deterministic.

    Parameters
    ----------
    A : LinearOperator
        Linear operator whose trace will be estimated. Has to be square.
    m3 : int
        Number of matrix-vector products divided by 3 used to estimate the
        trace.
    seed : optional
        Seed for `numpy.random.default_rng`.
        Can be provided to obtain deterministic results.

    Returns
    -------
    trace : LinearOperator.dtype
        Estimate of the trace

    Notes
    -----
    This is the Hutch++ algorithm given in [1]_.

    References
    ----------
    .. [1] Meyer, Raphael A., Cameron Musco, Christopher Musco, and David P.
       Woodruff. "Hutch++: Optimal Stochastic Trace Estimation." In Symposium
       on Simplicity in Algorithms (SOSA), pp. 142-155. Society for Industrial
       and Applied Mathematics, 2021
       https://doi.org/10.1137/1.9781611976496.16

    """
    rng = np.random.default_rng(seed)
    if len(A.shape) != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Expected A to be like a square matrix.")
    n = A.shape[-1]
    S = rng.choice([-1.0, +1.0], [n, m3])
    Q, _ = qr(A.matmat(S), overwrite_a=True, mode='economic')
    trQAQ = np.trace(Q.conj().T @ A.matmat(Q))
    G = rng.choice([-1, +1], [n, m3])
    right = G - Q@(Q.conj().T @ G)
    trGAG = np.trace(right.conj().T @ A.matmat(right))
    return trQAQ + trGAG/m3


def _ident_like(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.issparse(A):
        # Creates a sparse matrix in dia format
        out = scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
        if scipy.sparse.issparse(A):
            return out.asformat(A.format)
        return scipy.sparse.dia_array(out).asformat(A.format)
    elif is_pydata_spmatrix(A):
        import sparse
        return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        return IdentityOperator(A.shape, dtype=A.dtype)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)


def expm_multiply(A, B, start=None, stop=None, num=None,
                  endpoint=None, traceA=None):
    """
    Compute the action of the matrix exponential of A on B.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray, sparse array
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable.
        For linear operators, `traceA` should be provided to ensure performance
        as the estimation is not guaranteed to be reliable for all cases.

        .. versionadded:: 1.9.0

    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.

    Warns
    -----
    UserWarning
        If `A` is a linear operator and ``traceA=None`` (default).

    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.

    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import expm, expm_multiply
    >>> A = csc_array([[1, 0], [0, 1]])
    >>> A.toarray()
    array([[1, 0],
           [0, 1]], dtype=int64)
    >>> B = np.array([np.exp(-1.), np.exp(-2.)])
    >>> B
    array([ 0.36787944,  0.13533528])
    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
    array([[ 1.        ,  0.36787944],
           [ 1.64872127,  0.60653066],
           [ 2.71828183,  1.        ]])
    >>> expm(A).dot(B)                  # Verify 1st timestep
    array([ 1.        ,  0.36787944])
    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
    array([ 1.64872127,  0.60653066])
    >>> expm(2*A).dot(B)                # Verify 3rd timestep
    array([ 2.71828183,  1.        ])
    """
    if all(arg is None for arg in (start, stop, num, endpoint)):
        X = _expm_multiply_simple(A, B, traceA=traceA)
    else:
        X, status = _expm_multiply_interval(A, B, start, stop, num,
                                            endpoint, traceA=traceA)
    return X


def _expm_multiply_simple(A, B, t=1.0, traceA=None, balance=False):
    """
    Compute the action of the matrix exponential at a single time point.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray
        :math:`e^{t A} B`

    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).

    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError(f'shapes of matrices A {A.shape} and B {B.shape}'
                         ' are incompatible')
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    if traceA is None:
        if is_linear_operator:
            warn("Trace of LinearOperator not available, it will be estimated."
                 " Provide `traceA` to ensure performance.", stacklevel=3)
        # m3=1 is bit arbitrary choice, a more accurate trace (larger m3) might
        # speed up exponential calculation, but trace estimation is more costly
        traceA = traceest(A, m3=1) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)


def _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None, balance=False):
    """
    A helper function.
    """
    if balance:
        raise NotImplementedError
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = B
    eta = np.exp(t*mu / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            B = coeff * A.dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        B = F
    return F


# This table helps to compute bounds.
# They seem to have been difficult to calculate, involving symbolic
# manipulation of equations, followed by numerical root finding.
_theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }


def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
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
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    #XXX Eventually turn this into an API function in the  _onenormest module,
    #XXX and remove its underscore,
    #XXX but wait until expm_multiply goes into scipy.
    from scipy.sparse.linalg._onenormest import onenormest
    return onenormest(aslinearoperator(A) ** p)

class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """

    def __init__(self, A, A_1_norm=None, ell=2, scale=1):
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        A_1_norm : float, optional
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        self._A = A
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}
        self._scale = scale

    def set_scale(self,scale):
        """
        Set the scale parameter.
        """
        self._scale = scale

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        if self._A_1_norm is None:
            self._A_1_norm = _exact_1_norm(self._A)
        return self._scale*self._A_1_norm

    def d(self, p):
        """
        Lazily estimate :math:`d_p(A) ~= || A^p ||^(1/p)`
        where :math:`||.||` is the 1-norm.
        """
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, self._ell)
            self._d[p] = est ** (1.0 / p)
        return self._scale*self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p+1))

def _compute_cost_div_m(m, p, norm_info):
    """
    A helper function for computing bounds.

    This is equation (3.10).
    It measures cost in terms of the number of required matrix products.

    Parameters
    ----------
    m : int
        A valid key of _theta.
    p : int
        A matrix power.
    norm_info : LazyOperatorNormInfo
        Information about 1-norms of related operators.

    Returns
    -------
    cost_div_m : int
        Required number of matrix products divided by m.

    """
    return int(np.ceil(norm_info.alpha(p) / _theta[m]))


def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    best_m : int
        Related to bounds for error control.
    best_s : int
        Amount of scaling.

    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).

    """
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
        for m, theta in _theta.items():
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return best_m, best_s


def _condition_3_13(A_1_norm, n0, m_max, ell):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    A_1_norm : float
        The precomputed 1-norm of A.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    value : bool
        Indicates whether or not the condition has been met.

    Notes
    -----
    This is condition (3.13) in Al-Mohy and Higham (2011).

    """

    # This is the rhs of equation (3.12).
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)

    # Evaluate the condition (3.13).
    b = _theta[m_max] / float(n0 * m_max)
    return A_1_norm <= a * b


def _expm_multiply_interval(A, B, start=None, stop=None, num=None,
                            endpoint=None, traceA=None, balance=False,
                            status_only=False):
    """
    Compute the action of the matrix exponential at multiple time points.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable
    endpoint : bool, optional
        If True, `stop` is the last time point. Otherwise, it is not included.
    balance : bool
        Indicates whether or not to apply balancing.
    status_only : bool
        A flag that is set to True for some debugging and testing operations.

    Returns
    -------
    F : ndarray
        :math:`e^{t_k A} B`
    status : int
        An integer status for testing and debugging.

    Notes
    -----
    This is algorithm (5.2) in Al-Mohy and Higham (2011).

    There seems to be a typo, where line 15 of the algorithm should be
    moved to line 6.5 (between lines 6 and 7).

    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError(f'shapes of matrices A {A.shape} and B {B.shape}'
                         ' are incompatible')
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    if traceA is None:
        if is_linear_operator:
            warn("Trace of LinearOperator not available, it will be estimated."
                 " Provide `traceA` to ensure performance.", stacklevel=3)
        # m3=5 is bit arbitrary choice, a more accurate trace (larger m3) might
        # speed up exponential calculation, but trace estimation is also costly
        # an educated guess would need to consider the number of time points
