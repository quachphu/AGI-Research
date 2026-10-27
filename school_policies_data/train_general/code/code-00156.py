            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    size
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    - Allows for efficient O(1) access of individual elements.
    - Duplicates are not allowed.
    - Can be efficiently converted to a coo_array once constructed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import dok_array
    >>> S = dok_array((5, 5), dtype=np.float32)
    >>> for i in range(5):
    ...     for j in range(5):
    ...         S[i, j] = i + j    # Update element

    """


class dok_matrix(spmatrix, _dok_base):
    """
    Dictionary Of Keys based sparse matrix.

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        dok_matrix(D)
            where D is a 2-D ndarray

        dok_matrix(S)
            with another sparse array or matrix S (equivalent to S.todok())

        dok_matrix((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    size
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    - Allows for efficient O(1) access of individual elements.
    - Duplicates are not allowed.
    - Can be efficiently converted to a coo_matrix once constructed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import dok_matrix
    >>> S = dok_matrix((5, 5), dtype=np.float32)
    >>> for i in range(5):
    ...     for j in range(5):
    ...         S[i, j] = i + j    # Update element

    """

    def set_shape(self, shape):
        new_matrix = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_matrix.__dict__

    def get_shape(self):
        """Get shape of a sparse matrix."""
        return self._shape

    shape = property(fget=get_shape, fset=set_shape)

    def __reversed__(self):
        return self._dict.__reversed__()

    def __or__(self, other):
        if isinstance(other, _dok_base):
            return self._dict | other._dict
        return self._dict | other

    def __ror__(self, other):
        if isinstance(other, _dok_base):
            return self._dict | other._dict
        return self._dict | other

    def __ior__(self, other):
        if isinstance(other, _dok_base):
            self._dict |= other._dict
        else:
            self._dict |= other
        return self
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'isscalarlike',
    'name',
    'npfunc',
    'validateaxis',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="data",
                                   private_modules=["_data"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Functions to extract parts of sparse matrices
"""

__docformat__ = "restructuredtext en"

__all__ = ['find', 'tril', 'triu']


from ._coo import coo_matrix, coo_array
from ._base import sparray


def find(A):
    """Return the indices and values of the nonzero elements of a matrix

    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose nonzero elements are desired.

    Returns
    -------
    (I,J,V) : tuple of arrays
        I,J, and V contain the row indices, column indices, and values
        of the nonzero entries.


    Examples
    --------
    >>> from scipy.sparse import csr_array, find
    >>> A = csr_array([[7.0, 8.0, 0],[0, 0, 9.0]])
    >>> find(A)
    (array([0, 0, 1], dtype=int32),
     array([0, 1, 2], dtype=int32),
     array([ 7.,  8.,  9.]))

    """

    A = coo_array(A, copy=True)
    A.sum_duplicates()
    # remove explicit zeros
    nz_mask = A.data != 0
    return A.row[nz_mask], A.col[nz_mask], A.data[nz_mask]


def tril(A, k=0, format=None):
    """Return the lower triangular portion of a sparse array or matrix

    Returns the elements on or below the k-th diagonal of A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose lower trianglar portion is desired.
    k : integer : optional
        The top-most diagonal of the lower triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse matrix
        Lower triangular portion of A in sparse format.

    See Also
    --------
    triu : upper triangle in sparse format

    Examples
    --------
    >>> from scipy.sparse import csr_array, tril
    >>> A = csr_array([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...               dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]], dtype=int32)
    >>> tril(A).toarray()
    array([[1, 0, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 0, 0]], dtype=int32)
    >>> tril(A).nnz
    4
    >>> tril(A, k=1).toarray()
    array([[1, 2, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 9, 0]], dtype=int32)
    >>> tril(A, k=-1).toarray()
    array([[0, 0, 0, 0, 0],
           [4, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=int32)
    >>> tril(A, format='csc')
    <Compressed Sparse Column sparse array of dtype 'int32'
        with 4 stored elements and shape (3, 5)>

    """
    coo_sparse = coo_array if isinstance(A, sparray) else coo_matrix

    # convert to COOrdinate format where things are easy
    A = coo_sparse(A, copy=False)
    mask = A.row + k >= A.col

    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    new_coo = coo_sparse((data, (row, col)), shape=A.shape, dtype=A.dtype)
    return new_coo.asformat(format)


def triu(A, k=0, format=None):
    """Return the upper triangular portion of a sparse array or matrix

    Returns the elements on or above the k-th diagonal of A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse array or matrix
        Matrix whose upper trianglar portion is desired.
    k : integer : optional
        The bottom-most diagonal of the upper triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse array or matrix
        Upper triangular portion of A in sparse format.
        Sparse array if A is a sparse array, otherwise matrix.

    See Also
    --------
    tril : lower triangle in sparse format

    Examples
    --------
    >>> from scipy.sparse import csr_array, triu
    >>> A = csr_array([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...                dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]], dtype=int32)
    >>> triu(A).toarray()
    array([[1, 2, 0, 0, 3],
           [0, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]], dtype=int32)
    >>> triu(A).nnz
    8
    >>> triu(A, k=1).toarray()
    array([[0, 2, 0, 0, 3],
           [0, 0, 0, 6, 7],
           [0, 0, 0, 9, 0]], dtype=int32)
    >>> triu(A, k=-1).toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]], dtype=int32)
    >>> triu(A, format='csc')
    <Compressed Sparse Column sparse array of dtype 'int32'
        with 8 stored elements and shape (3, 5)>

    """
    coo_sparse = coo_array if isinstance(A, sparray) else coo_matrix

    # convert to COOrdinate format where things are easy
    A = coo_sparse(A, copy=False)
    mask = A.row + k <= A.col

    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    new_coo = coo_sparse((data, (row, col)), shape=A.shape, dtype=A.dtype)
    return new_coo.asformat(format)
<|endoftext|>"""Base class for sparse matrice with a .data attribute

    subclasses must provide a _with_data() method that
    creates a new matrix with the same sparsity pattern
    as self but with a different data array

"""

import math
import numpy as np

from ._base import _spbase, sparray, _ufuncs_with_fixed_point_at_zero
from ._sputils import isscalarlike, validateaxis

__all__ = []


# TODO implement all relevant operations
# use .data.__methods__() instead of /=, *=, etc.
class _data_matrix(_spbase):
    def __init__(self, arg1, *, maxprint=None):
        _spbase.__init__(self, arg1, maxprint=maxprint)

    @property
    def dtype(self):
        return self.data.dtype

    @dtype.setter
    def dtype(self, newtype):
        self.data.dtype = newtype

    def _deduped_data(self):
        if hasattr(self, 'sum_duplicates'):
            self.sum_duplicates()
        return self.data

    def __abs__(self):
        return self._with_data(abs(self._deduped_data()))

    def __round__(self, ndigits=0):
        return self._with_data(np.around(self._deduped_data(), decimals=ndigits))

    def _real(self):
        return self._with_data(self.data.real)

    def _imag(self):
        return self._with_data(self.data.imag)

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('negating a boolean sparse array is not '
                                      'supported')
        return self._with_data(-self.data)

    def __imul__(self, other):  # self *= other
        if isscalarlike(other):
            self.data *= other
            return self
        return NotImplemented

    def __itruediv__(self, other):  # self /= other
        if isscalarlike(other):
            recip = 1.0 / other
            self.data *= recip
            return self
        else:
            return NotImplemented

    def astype(self, dtype, casting='unsafe', copy=True):
        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            matrix = self._with_data(
                self.data.astype(dtype, casting=casting, copy=True),
                copy=True
            )
            return matrix._with_data(matrix._deduped_data(), copy=False)
        elif copy:
            return self.copy()
        else:
            return self

    astype.__doc__ = _spbase.astype.__doc__

    def conjugate(self, copy=True):
        if np.issubdtype(self.dtype, np.complexfloating):
            return self._with_data(self.data.conjugate(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    conjugate.__doc__ = _spbase.conjugate.__doc__

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    copy.__doc__ = _spbase.copy.__doc__

    def power(self, n, dtype=None):
        """
        This function performs element-wise power.

        Parameters
        ----------
        n : scalar
            n is a non-zero scalar (nonzero avoids dense ones creation)
            If zero power is desired, special case it to use `np.ones`

        dtype : If dtype is not specified, the current dtype will be preserved.

        Raises
        ------
        NotImplementedError : if n is a zero scalar
            If zero power is desired, special case it to use
            ``np.ones(A.shape, dtype=A.dtype)``
        """
        if not isscalarlike(n):
            raise NotImplementedError("input is not scalar")
        if not n:
            raise NotImplementedError(
                "zero power is not supported as it would densify the matrix.\n"
                "Use `np.ones(A.shape, dtype=A.dtype)` for this case."
            )

        data = self._deduped_data()
        if dtype is not None:
            data = data.astype(dtype)
        return self._with_data(data ** n)

    ###########################
    # Multiplication handlers #
    ###########################

    def _mul_scalar(self, other):
        return self._with_data(self.data * other)


# Add the numpy unary ufuncs for which func(0) = 0 to _data_matrix.
for npfunc in _ufuncs_with_fixed_point_at_zero:
    name = npfunc.__name__

    def _create_method(op):
        def method(self):
            result = op(self._deduped_data())
            return self._with_data(result, copy=True)

        method.__doc__ = (f"Element-wise {name}.\n\n"
                          f"See `numpy.{name}` for more information.")
        method.__name__ = name

        return method

    setattr(_data_matrix, name, _create_method(npfunc))


def _find_missing_index(ind, n):
    for k, a in enumerate(ind):
        if k != a:
            return k

    k += 1
    if k < n:
        return k
    else:
        return -1


class _minmax_mixin:
    """Mixin for min and max methods.

    These are not implemented for dia_matrix, hence the separate class.
    """

    def _min_or_max_axis(self, axis, min_or_max, explicit):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]
        idx_dtype = self._get_index_dtype(maxval=M)

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        major_index, value = mat._minor_reduce(min_or_max)
        if not explicit:
            not_full = np.diff(mat.indptr)[major_index] < N
            value[not_full] = min_or_max(value[not_full], 0)

        mask = value != 0
        major_index = np.compress(mask, major_index).astype(idx_dtype, copy=False)
        value = np.compress(mask, value)

        if isinstance(self, sparray):
            coords = (major_index,)
            shape = (M,)
            return self._coo_container((value, coords), shape=shape, dtype=self.dtype)

        if axis == 0:
            return self._coo_container(
                (value, (np.zeros(len(value), dtype=idx_dtype), major_index)),
                dtype=self.dtype, shape=(1, M)
            )
        else:
            return self._coo_container(
                (value, (major_index, np.zeros(len(value), dtype=idx_dtype))),
                dtype=self.dtype, shape=(M, 1)
            )

    def _min_or_max(self, axis, out, min_or_max, explicit):
        if out is not None:
            raise ValueError("Sparse arrays do not support an 'out' parameter.")

        validateaxis(axis)
        if self.ndim == 1:
            if axis not in (None, 0, -1):
                raise ValueError("axis out of range")
            axis = None  # avoid calling special axis case. no impact on 1d

        if axis is None:
            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")

            zero = self.dtype.type(0)
            if self.nnz == 0:
                return zero
            m = min_or_max.reduce(self._deduped_data().ravel())
            if self.nnz != math.prod(self.shape) and not explicit:
                m = min_or_max(zero, m)
            return m

        if axis < 0:
            axis += 2

        if (axis == 0) or (axis == 1):
            return self._min_or_max_axis(axis, min_or_max, explicit)
        else:
            raise ValueError("axis out of range")

    def _arg_min_or_max_axis(self, axis, argmin_or_argmax, compare, explicit):
        if self.shape[axis] == 0:
            raise ValueError("Cannot apply the operation along a zero-sized dimension.")

        if axis < 0:
            axis += 2

        zero = self.dtype.type(0)

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        ret_size, line_size = mat._swap(mat.shape)
        ret = np.zeros(ret_size, dtype=int)

        nz_lines, = np.nonzero(np.diff(mat.indptr))
        for i in nz_lines:
            p, q = mat.indptr[i:i + 2]
            data = mat.data[p:q]
            indices = mat.indices[p:q]
            extreme_index = argmin_or_argmax(data)
            extreme_value = data[extreme_index]
            if explicit:
                if q - p > 0:
                    ret[i] = indices[extreme_index]
            else:
                if compare(extreme_value, zero) or q - p == line_size:
                    ret[i] = indices[extreme_index]
                else:
                    zero_ind = _find_missing_index(indices, line_size)
                    if extreme_value == zero:
                        ret[i] = min(extreme_index, zero_ind)
                    else:
                        ret[i] = zero_ind

        if isinstance(self, sparray):
            return ret

        if axis == 1:
            ret = ret.reshape(-1, 1)

        return self._ascontainer(ret)

    def _arg_min_or_max(self, axis, out, argmin_or_argmax, compare, explicit):
        if out is not None:
            raise ValueError("Sparse types do not support an 'out' parameter.")

        validateaxis(axis)

        if self.ndim == 1:
            if axis not in (None, 0, -1):
                raise ValueError("axis out of range")
            axis = None  # avoid calling special axis case. no impact on 1d

        if axis is not None:
            return self._arg_min_or_max_axis(axis, argmin_or_argmax, compare, explicit)

        if 0 in self.shape:
            raise ValueError("Cannot apply the operation to an empty matrix.")

        if self.nnz == 0:
            if explicit:
                raise ValueError("Cannot apply the operation to zero matrix "
                                 "when explicit=True.")
            return 0

        zero = self.dtype.type(0)
        mat = self.tocoo()
        # Convert to canonical form: no duplicates, sorted indices.
        mat.sum_duplicates()
        extreme_index = argmin_or_argmax(mat.data)
        if explicit:
            return extreme_index
        extreme_value = mat.data[extreme_index]
        num_col = mat.shape[-1]

        # If the min value is less than zero, or max is greater than zero,
        # then we do not need to worry about implicit zeros.
        if compare(extreme_value, zero):
            # cast to Python int to avoid overflow and RuntimeError
            return int(mat.row[extreme_index]) * num_col + int(mat.col[extreme_index])

        # Cheap test for the rare case where we have no implicit zeros.
        size = math.prod(self.shape)
        if size == mat.nnz:
            return int(mat.row[extreme_index]) * num_col + int(mat.col[extreme_index])

        # At this stage, any implicit zero could be the min or max value.
        # After sum_duplicates(), the `row` and `col` arrays are guaranteed to
        # be sorted in C-order, which means the linearized indices are sorted.
        linear_indices = mat.row * num_col + mat.col
        first_implicit_zero_index = _find_missing_index(linear_indices, size)
        if extreme_value == zero:
            return min(first_implicit_zero_index, extreme_index)
        return first_implicit_zero_index

    def max(self, axis=None, out=None, *, explicit=False):
        """Return the maximum of the array/matrix or maximum along an axis.

        By default, all elements are taken into account, not just the non-zero ones.
        But with `explicit` set, only the stored elements are considered.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the maximum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        explicit : {False, True} optional (default: False)
            When set to True, only the stored elements will be considered.
            If a row/column is empty, the sparse.coo_array returned
            has no stored element (i.e. an implicit zero) for that row/column.

            .. versionadded:: 1.15.0

        Returns
        -------
        amax : coo_array or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_array of dimension
            ``a.ndim - 1``.

        See Also
        --------
        min : The minimum value of a sparse array/matrix along a given axis.
        numpy.max : NumPy's implementation of 'max'

        """
        return self._min_or_max(axis, out, np.maximum, explicit)

    def min(self, axis=None, out=None, *, explicit=False):
        """Return the minimum of the array/matrix or maximum along an axis.

        By default, all elements are taken into account, not just the non-zero ones.
        But with `explicit` set, only the stored elements are considered.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the minimum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        explicit : {False, True} optional (default: False)
            When set to True, only the stored elements will be considered.
            If a row/column is empty, the sparse.coo_array returned
            has no stored element (i.e. an implicit zero) for that row/column.

            .. versionadded:: 1.15.0

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_array of dimension
            ``a.ndim - 1``.

        See Also
        --------
        max : The maximum value of a sparse array/matrix along a given axis.
        numpy.min : NumPy's implementation of 'min'

        """
        return self._min_or_max(axis, out, np.minimum, explicit)

    def nanmax(self, axis=None, out=None, *, explicit=False):
        """Return the maximum, ignoring any Nans, along an axis.

        Return the maximum, ignoring any Nans, of the array/matrix along an axis.
        By default this takes all elements into account, but with `explicit` set,
        only stored elements are considered.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the maximum is computed. The default is to
            compute the maximum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        explicit : {False, True} optional (default: False)
            When set to True, only the stored elements will be considered.
            If a row/column is empty, the sparse.coo_array returned
            has no stored element (i.e. an implicit zero) for that row/column.

            .. versionadded:: 1.15.0

        Returns
        -------
        amax : coo_array or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_array of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmin : The minimum value of a sparse array/matrix along a given axis,
                 ignoring NaNs.
        max : The maximum value of a sparse array/matrix along a given axis,
              propagating NaNs.
        numpy.nanmax : NumPy's implementation of 'nanmax'.

        """
        return self._min_or_max(axis, out, np.fmax, explicit)

    def nanmin(self, axis=None, out=None, *, explicit=False):
        """Return the minimum, ignoring any Nans, along an axis.

        Return the minimum, ignoring any Nans, of the array/matrix along an axis.
        By default this takes all elements into account, but with `explicit` set,
        only stored elements are considered.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the minimum is computed. The default is to
            compute the minimum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        explicit : {False, True} optional (default: False)
            When set to True, only the stored elements will be considered.
            If a row/column is empty, the sparse.coo_array returned
            has no stored element (i.e. an implicit zero) for that row/column.

            .. versionadded:: 1.15.0

        Returns
        -------
        amin : coo_array or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_array of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmax : The maximum value of a sparse array/matrix along a given axis,
                 ignoring NaNs.
        min : The minimum value of a sparse array/matrix along a given axis,
              propagating NaNs.
        numpy.nanmin : NumPy's implementation of 'nanmin'.

        """
        return self._min_or_max(axis, out, np.fmin, explicit)

    def argmax(self, axis=None, out=None, *, explicit=False):
        """Return indices of maximum elements along an axis.

        By default, implicit zero elements are taken into account. If there are
        several minimum values, the index of the first occurrence is returned.
        If `explicit` is set, only explicitly stored elements will be considered.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmax is computed. If None (default), index
            of the maximum element in the flatten data is returned.

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        explicit : {False, True} optional (default: False)
            When set to True, only explicitly stored elements will be considered.
            If axis is not None and a row/column has no stored elements, argmax
            is undefined, so the index ``0`` is returned for that row/column.

            .. versionadded:: 1.15.0

        Returns
        -------
        ind : numpy.matrix or int
            Indices of maximum elements. If matrix, its size along `axis` is 1.
        """
        return self._arg_min_or_max(axis, out, np.argmax, np.greater, explicit)

    def argmin(self, axis=None, out=None, *, explicit=False):
        """Return indices of minimum elements along an axis.

        By default, implicit zero elements are taken into account. If there are
        several minimum values, the index of the first occurrence is returned.
        If `explicit` is set, only explicitly stored elements will be considered.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmin is computed. If None (default), index
            of the minimum element in the flatten data is returned.

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        explicit : {False, True} optional (default: False)
            When set to True, only explicitly stored elements will be considered.
            If axis is not None and a row/column has no stored elements, argmin
            is undefined, so the index ``0`` is returned for that row/column.

            .. versionadded:: 1.15.0

        Returns
        -------
         ind : numpy.matrix or int
            Indices of minimum elements. If matrix, its size along `axis` is 1.
        """
        return self._arg_min_or_max(axis, out, np.argmin, np.less, explicit)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'check_shape',
    'dia_matrix',
    'dia_matvec',
    'get_sum_dtype',
    'getdtype',
    'isshape',
    'isspmatrix_dia',
    'spmatrix',
    'upcast_char',
    'validateaxis',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="dia",
                                   private_modules=["_dia"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Base class for sparse matrix formats using compressed storage."""
__all__ = []

from warnings import warn
import itertools
import operator

import numpy as np
from scipy._lib._util import _prune_array, copy_if_needed

from ._base import _spbase, issparse, sparray, SparseEfficiencyWarning
from ._data import _data_matrix, _minmax_mixin
from . import _sparsetools
from ._sparsetools import (get_csr_submatrix, csr_sample_offsets, csr_todense,
                           csr_sample_values, csr_row_index, csr_row_slice,
                           csr_column_index1, csr_column_index2)
from ._index import IndexMixin
from ._sputils import (upcast, upcast_char, to_native, isdense, isshape,
                       getdtype, isscalarlike, isintlike, downcast_intp_index,
                       get_sum_dtype, check_shape, get_index_dtype, broadcast_shapes,
                       is_pydata_spmatrix)


class _cs_matrix(_data_matrix, _minmax_mixin, IndexMixin):
    """
    base array/matrix class for compressed row- and column-oriented arrays/matrices
    """

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _data_matrix.__init__(self, arg1, maxprint=maxprint)

        if issparse(arg1):
            if arg1.format == self.format and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.asformat(self.format)
            self.indptr, self.indices, self.data, self._shape = (
                arg1.indptr, arg1.indices, arg1.data, arg1._shape
            )

        elif isinstance(arg1, tuple):
            if isshape(arg1, allow_nd=self._allow_nd):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self._shape = check_shape(arg1, allow_nd=self._allow_nd)
                M, N = self._swap(self._shape_as_2d)
                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                idx_dtype = self._get_index_dtype(maxval=max(self.shape))
                self.data = np.zeros(0, getdtype(dtype, default=float))
                self.indices = np.zeros(0, idx_dtype)
                self.indptr = np.zeros(M + 1, dtype=idx_dtype)
            else:
                if len(arg1) == 2:
                    # (data, ij) format
                    coo = self._coo_container(arg1, shape=shape, dtype=dtype)
                    arrays = coo._coo_to_compressed(self._swap)
                    self.indptr, self.indices, self.data, self._shape = arrays
                    self.sum_duplicates()
                elif len(arg1) == 3:
                    # (data, indices, indptr) format
                    (data, indices, indptr) = arg1

                    # Select index dtype large enough to pass array and
                    # scalar parameters to sparsetools
                    maxval = None
                    if shape is not None and 0 not in shape:
                        maxval = max(shape)
                    idx_dtype = self._get_index_dtype((indices, indptr),
                                                maxval=maxval,
                                                check_contents=True)

                    if not copy:
                        copy = copy_if_needed
                    self.indices = np.array(indices, copy=copy, dtype=idx_dtype)
                    self.indptr = np.array(indptr, copy=copy, dtype=idx_dtype)
                    self.data = np.array(data, copy=copy, dtype=dtype)
                else:
                    raise ValueError(f"unrecognized {self.__class__.__name__} "
                                     f"constructor input: {arg1}")

        else:
            # must be dense
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise ValueError(f"unrecognized {self.__class__.__name__} "
                                 f"constructor input: {arg1}") from e
            if isinstance(self, sparray) and arg1.ndim != 2 and self.format == "csc":
                raise ValueError(f"CSC arrays don't support {arg1.ndim}D input. Use 2D")
            if arg1.ndim > 2:
                raise ValueError(f"CSR arrays don't yet support {arg1.ndim}D.")

            coo = self._coo_container(arg1, dtype=dtype)
            arrays = coo._coo_to_compressed(self._swap)
            self.indptr, self.indices, self.data, self._shape = arrays

        # Read matrix dimensions given, if any
        if shape is not None:
            self._shape = check_shape(shape, allow_nd=self._allow_nd)
        elif self.shape is None:
            # shape not already set, try to infer dimensions
            try:
                M = len(self.indptr) - 1
                N = self.indices.max() + 1
            except Exception as e:
                raise ValueError('unable to infer matrix dimensions') from e

            self._shape = check_shape(self._swap((M, N)), allow_nd=self._allow_nd)

        if dtype is not None:
            newdtype = getdtype(dtype)
            self.data = self.data.astype(newdtype, copy=False)

        self.check_format(full_check=False)

    def _getnnz(self, axis=None):
        if axis is None:
            return int(self.indptr[-1])
        elif self.ndim == 1:
            if axis in (0, -1):
                return int(self.indptr[-1])
            raise ValueError('axis out of bounds')
        else:
            if axis < 0:
                axis += 2
            axis, _ = self._swap((axis, 1 - axis))
            _, N = self._swap(self.shape)
            if axis == 0:
                return np.bincount(downcast_intp_index(self.indices), minlength=N)
            elif axis == 1:
                return np.diff(self.indptr)
            raise ValueError('axis out of bounds')

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def count_nonzero(self, axis=None):
        self.sum_duplicates()
        if axis is None:
            return np.count_nonzero(self.data)

        if self.ndim == 1:
            if axis not in (0, -1):
                raise ValueError('axis out of bounds')
            return np.count_nonzero(self.data)

        if axis < 0:
            axis += 2
        axis, _ = self._swap((axis, 1 - axis))
        if axis == 0:
            _, N = self._swap(self.shape)
            mask = self.data != 0
            idx = self.indices if mask.all() else self.indices[mask]
            return np.bincount(downcast_intp_index(idx), minlength=N)
        elif axis == 1:
            if self.data.all():
                return np.diff(self.indptr)
            pairs = itertools.pairwise(self.indptr)
            return np.array([np.count_nonzero(self.data[i:j]) for i, j in pairs])
        else:
            raise ValueError('axis out of bounds')

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def check_format(self, full_check=True):
        """Check whether the array/matrix respects the CSR or CSC format.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, run rigorous check, scanning arrays for valid values.
            Note that activating those check might copy arrays for casting,
            modifying indices and index pointers' inplace.
            If `False`, run basic checks on attributes. O(1) operations.
            Default is `True`.
        """
        # index arrays should have integer data types
        if self.indptr.dtype.kind != 'i':
            warn(f"indptr array has non-integer dtype ({self.indptr.dtype.name})",
                 stacklevel=3)
        if self.indices.dtype.kind != 'i':
            warn(f"indices array has non-integer dtype ({self.indices.dtype.name})",
                 stacklevel=3)

        # check array shapes
        for x in [self.data.ndim, self.indices.ndim, self.indptr.ndim]:
            if x != 1:
                raise ValueError('data, indices, and indptr should be 1-D')

        # check index pointer. Use _swap to determine proper bounds
        M, N = self._swap(self._shape_as_2d)

        if (len(self.indptr) != M + 1):
            raise ValueError(f"index pointer size {len(self.indptr)} should be {M + 1}")
        if (self.indptr[0] != 0):
            raise ValueError("index pointer should start with 0")

        # check index and data arrays
        if (len(self.indices) != len(self.data)):
            raise ValueError("indices and data should have the same size")
        if (self.indptr[-1] > len(self.indices)):
            raise ValueError("Last value of index pointer should be less than "
                             "the size of index and data arrays")

        self.prune()

        if full_check:
            # check format validity (more expensive)
            if self.nnz > 0:
                if self.indices.max() >= N:
                    raise ValueError(f"indices must be < {N}")
                if self.indices.min() < 0:
                    raise ValueError("indices must be >= 0")
                if np.diff(self.indptr).min() < 0:
                    raise ValueError("indptr must be a non-decreasing sequence")

            idx_dtype = self._get_index_dtype((self.indptr, self.indices))
            self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
            self.indices = np.asarray(self.indices, dtype=idx_dtype)
            self.data = to_native(self.data)

        # if not self.has_sorted_indices():
        #    warn('Indices were not in sorted order.  Sorting indices.')
        #    self.sort_indices()
        #    assert(self.has_sorted_indices())
        # TODO check for duplicates?

    #######################
    # Boolean comparisons #
    #######################

    def _scalar_binopt(self, other, op):
        """Scalar version of self._binopt, for cases in which no new nonzeros
        are added. Produces a new sparse array in canonical form.
        """
        self.sum_duplicates()
        res = self._with_data(op(self.data, other), copy=True)
        res.eliminate_zeros()
        return res

    def __eq__(self, other):
        # Scalar other.
        if isscalarlike(other):
            if np.isnan(other):
                return self.__class__(self.shape, dtype=np.bool_)

            if other == 0:
                warn("Comparing a sparse matrix with 0 using == is inefficient"
                     ", try using != instead.", SparseEfficiencyWarning,
                     stacklevel=3)
                all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
                inv = self._scalar_binopt(other, operator.ne)
                return all_true - inv
            else:
                return self._scalar_binopt(other, operator.eq)
        # Dense other.
        elif isdense(other):
            return self.todense() == other
        # Pydata sparse other.
        elif is_pydata_spmatrix(other):
            return NotImplemented
        # Sparse other.
        elif issparse(other):
            warn("Comparing sparse matrices using == is inefficient, try using"
                 " != instead.", SparseEfficiencyWarning, stacklevel=3)
            # TODO sparse broadcasting
            if self.shape != other.shape:
                return False
            elif self.format != other.format:
                other = other.asformat(self.format)
            res = self._binopt(other, '_ne_')
            all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
            return all_true - res
        else:
            return NotImplemented

    def __ne__(self, other):
        # Scalar other.
        if isscalarlike(other):
            if np.isnan(other):
                warn("Comparing a sparse matrix with nan using != is"
                     " inefficient", SparseEfficiencyWarning, stacklevel=3)
                all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
                return all_true
            elif other != 0:
                warn("Comparing a sparse matrix with a nonzero scalar using !="
                     " is inefficient, try using == instead.",
                     SparseEfficiencyWarning, stacklevel=3)
                all_true = self.__class__(np.ones(self.shape), dtype=np.bool_)
                inv = self._scalar_binopt(other, operator.eq)
                return all_true - inv
            else:
                return self._scalar_binopt(other, operator.ne)
        # Dense other.
        elif isdense(other):
            return self.todense() != other
        # Pydata sparse other.
        elif is_pydata_spmatrix(other):
            return NotImplemented
        # Sparse other.
        elif issparse(other):
            # TODO sparse broadcasting
            if self.shape != other.shape:
                return True
            elif self.format != other.format:
                other = other.asformat(self.format)
            return self._binopt(other, '_ne_')
        else:
            return NotImplemented

    def _inequality(self, other, op, op_name, bad_scalar_msg):
        # Scalar other.
        if isscalarlike(other):
            if 0 == other and op_name in ('_le_', '_ge_'):
                raise NotImplementedError(" >= and <= don't work with 0.")
            elif op(0, other):
                warn(bad_scalar_msg, SparseEfficiencyWarning, stacklevel=3)
                other_arr = np.empty(self.shape, dtype=np.result_type(other))
                other_arr.fill(other)
                other_arr = self.__class__(other_arr)
                return self._binopt(other_arr, op_name)
            else:
                return self._scalar_binopt(other, op)
        # Dense other.
        elif isdense(other):
            return op(self.todense(), other)
        # Sparse other.
        elif issparse(other):
            # TODO sparse broadcasting
            if self.shape != other.shape:
                raise ValueError("inconsistent shapes")
            elif self.format != other.format:
                other = other.asformat(self.format)
            if op_name not in ('_ge_', '_le_'):
                return self._binopt(other, op_name)

            warn("Comparing sparse matrices using >= and <= is inefficient, "
                 "using <, >, or !=, instead.",
                 SparseEfficiencyWarning, stacklevel=3)
            all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
            res = self._binopt(other, '_gt_' if op_name == '_le_' else '_lt_')
            return all_true - res
        else:
            return NotImplemented

    def __lt__(self, other):
        return self._inequality(other, operator.lt, '_lt_',
                                "Comparing a sparse matrix with a scalar "
                                "greater than zero using < is inefficient, "
                                "try using >= instead.")

    def __gt__(self, other):
        return self._inequality(other, operator.gt, '_gt_',
                                "Comparing a sparse matrix with a scalar "
                                "less than zero using > is inefficient, "
                                "try using <= instead.")

    def __le__(self, other):
        return self._inequality(other, operator.le, '_le_',
                                "Comparing a sparse matrix with a scalar "
                                "greater than zero using <= is inefficient, "
                                "try using > instead.")

    def __ge__(self, other):
        return self._inequality(other, operator.ge, '_ge_',
                                "Comparing a sparse matrix with a scalar "
                                "less than zero using >= is inefficient, "
                                "try using < instead.")

    #################################
    # Arithmetic operator overrides #
    #################################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        order = self._swap('CF')[0]
        result = np.array(other, dtype=dtype, order=order, copy=True)
        y = result if result.flags.c_contiguous else result.T
        M, N = self._swap(self._shape_as_2d)
        csr_todense(M, N, self.indptr, self.indices, self.data, y)
        return self._container(result, copy=False)

    def _add_sparse(self, other):
        return self._binopt(other, '_plus_')

    def _sub_sparse(self, other):
        return self._binopt(other, '_minus_')

    def multiply(self, other):
        """Point-wise multiplication by array/matrix, vector, or scalar."""
        # Scalar multiplication.
        if isscalarlike(other):
            return self._mul_scalar(other)
        # Sparse matrix or vector.
        if issparse(other):
            if self.shape == other.shape:
                other = self.__class__(other)
                return self._binopt(other, '_elmul_')
            # Single element.
            if other.shape == (1, 1):
                result = self._mul_scalar(other.toarray()[0, 0])
                if self.ndim == 1:
                    return result.reshape((1, self.shape[0]))
                return result
            if other.shape == (1,):
                return self._mul_scalar(other.toarray()[0])
            if self.shape in ((1,), (1, 1)):
                return other._mul_scalar(self.data.sum())

            # broadcast. treat 1d like a row
            sM, sN = self._shape_as_2d
            oM, oN = other._shape_as_2d
            # A row times a column.
            if sM == 1 and oN == 1:
                return other._matmul_sparse(self.reshape(sM, sN).tocsc())
            if sN == 1 and oM == 1:
                return self._matmul_sparse(other.reshape(oM, oN).tocsc())

            is_array = isinstance(self, sparray)
            # Other is a row.
            if oM == 1 and sN == oN:
                new_other = _make_diagonal_csr(other.toarray().ravel(), is_array)
                result = self._matmul_sparse(new_other)
                return result if self.ndim == 2 else result.reshape((1, oN))
            # self is a row.
            if sM == 1 and sN == oN:
                copy = _make_diagonal_csr(self.toarray().ravel(), is_array)
                return other._matmul_sparse(copy)

            # Other is a column.
            if oN == 1 and sM == oM:
                new_other = _make_diagonal_csr(other.toarray().ravel(), is_array)
                return new_other._matmul_sparse(self)
            # self is a column.
            if sN == 1 and sM == oM:
                new_self = _make_diagonal_csr(self.toarray().ravel(), is_array)
                return new_self._matmul_sparse(other)
            raise ValueError("inconsistent shapes")

        # Assume other is a dense matrix/array, which produces a single-item
        # object array if other isn't convertible to ndarray.
        other = np.asanyarray(other)

        if other.ndim > 2:
            return np.multiply(self.toarray(), other)
        # Single element / wrapped object.
        if other.size == 1:
            if other.dtype == np.object_:
                # 'other' not convertible to ndarray.
                return NotImplemented
            bshape = broadcast_shapes(self.shape, other.shape)
            return self._mul_scalar(other.flat[0]).reshape(bshape)
        # Fast case for trivial sparse matrix.
        if self.shape in ((1,), (1, 1)):
            bshape = broadcast_shapes(self.shape, other.shape)
            return np.multiply(self.data.sum(), other).reshape(bshape)

        ret = self.tocoo()
        # Matching shapes.
        if self.shape == other.shape:
            data = np.multiply(ret.data, other[ret.coords])
            ret.data = data.view(np.ndarray).ravel()
            return ret

        # convert other to 2d
        other2d = np.atleast_2d(other)
        # Sparse row vector times...
        if self.shape[0] == 1 or self.ndim == 1:
            if other2d.shape[1] == 1:  # Dense column vector.
                data = np.multiply(ret.data, other2d)
            elif other2d.shape[1] == self.shape[-1]:  # Dense 2d matrix.
                data = np.multiply(ret.data, other2d[:, ret.col])
            else:
                raise ValueError("inconsistent shapes")
            idx_dtype = self._get_index_dtype(ret.col,
                                              maxval=ret.nnz * other2d.shape[0])
            row = np.repeat(np.arange(other2d.shape[0], dtype=idx_dtype), ret.nnz)
            col = np.tile(ret.col.astype(idx_dtype, copy=False), other2d.shape[0])
            return self._coo_container(
                (data.view(np.ndarray).ravel(), (row, col)),
                shape=(other2d.shape[0], self.shape[-1]),
                copy=False
            )
        # Sparse column vector times...
        if self.shape[1] == 1:
            if other2d.shape[0] == 1:  # Dense row vector.
                data = np.multiply(ret.data[:, None], other2d)
            elif other2d.shape[0] == self.shape[0]:  # Dense 2d array.
                data = np.multiply(ret.data[:, None], other2d[ret.row])
            else:
                raise ValueError("inconsistent shapes")
            idx_dtype = self._get_index_dtype(ret.row,
                                              maxval=ret.nnz * other2d.shape[1])
            row = np.repeat(ret.row.astype(idx_dtype, copy=False), other2d.shape[1])
            col = np.tile(np.arange(other2d.shape[1], dtype=idx_dtype), ret.nnz)
            return self._coo_container(
                (data.view(np.ndarray).ravel(), (row, col)),
                shape=(self.shape[0], other2d.shape[1]),
                copy=False
            )
        # Sparse matrix times dense row vector.
        if other2d.shape[0] == 1 and self.shape[1] == other2d.shape[1]:
            data = np.multiply(ret.data, other2d[:, ret.col].ravel())
        # Sparse matrix times dense column vector.
        elif other2d.shape[1] == 1 and self.shape[0] == other2d.shape[0]:
            data = np.multiply(ret.data, other2d[ret.row].ravel())
        else:
            raise ValueError("inconsistent shapes")
        ret.data = data.view(np.ndarray).ravel()
        return ret

    ###########################
    # Multiplication handlers #
    ###########################

    def _matmul_vector(self, other):
        M, N = self._shape_as_2d

        # output array
        result = np.zeros(M, dtype=upcast_char(self.dtype.char, other.dtype.char))

        # csr_matvec or csc_matvec
        fn = getattr(_sparsetools, self.format + '_matvec')
        fn(M, N, self.indptr, self.indices, self.data, other, result)

        return result[0] if self.ndim == 1 else result

    def _matmul_multivector(self, other):
        M, N = self._shape_as_2d
        n_vecs = other.shape[-1]  # number of column vectors

        result = np.zeros((M, n_vecs),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))

        # csr_matvecs or csc_matvecs
        fn = getattr(_sparsetools, self.format + '_matvecs')
        fn(M, N, n_vecs, self.indptr, self.indices, self.data,
           other.ravel(), result.ravel())

        if self.ndim == 1:
            return result.reshape((n_vecs,))
        return result

    def _matmul_sparse(self, other):
        M, K1 = self._shape_as_2d
        # if other is 1d, treat as a **column**
        o_ndim = other.ndim
        if o_ndim == 1:
            # convert 1d array to a 2d column when on the right of @
            other = other.reshape((1, other.shape[0])).T  # Note: converts to CSC
        K2, N = other._shape if other.ndim == 2 else (other.shape[0], 1)

        # find new_shape: (M, N), (M,), (N,) or ()
        new_shape = ()
        if self.ndim == 2:
            new_shape += (M,)
        if o_ndim == 2:
            new_shape += (N,)
        faux_shape = (M if self.ndim == 2 else 1, N if o_ndim == 2 else 1)

        major_dim = self._swap((M, N))[0]
        other = self.__class__(other)  # convert to this format

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices))

        fn = getattr(_sparsetools, self.format + '_matmat_maxnnz')
        nnz = fn(M, N,
                 np.asarray(self.indptr, dtype=idx_dtype),
                 np.asarray(self.indices, dtype=idx_dtype),
                 np.asarray(other.indptr, dtype=idx_dtype),
                 np.asarray(other.indices, dtype=idx_dtype))
        if nnz == 0:
            if new_shape == ():
                return np.array(0, dtype=upcast(self.dtype, other.dtype))
            return self.__class__(new_shape, dtype=upcast(self.dtype, other.dtype))

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=nnz)

        indptr = np.empty(major_dim + 1, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

        fn = getattr(_sparsetools, self.format + '_matmat')
        fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        if new_shape == ():
            return np.array(data[0])
        res = self.__class__((data, indices, indptr), shape=faux_shape)
        if faux_shape != new_shape:
            if res.format != 'csr':
                res = res.tocsr()
            res = res.reshape(new_shape)
        return res

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        fn = getattr(_sparsetools, self.format + "_diagonal")
        y = np.empty(min(rows + min(k, 0), cols - max(k, 0)),
                     dtype=upcast(self.dtype))
        fn(k, self.shape[0], self.shape[1], self.indptr, self.indices,
           self.data, y)
        return y

    diagonal.__doc__ = _spbase.diagonal.__doc__

    #####################
    # Other binary ops  #
    #####################

    def _maximum_minimum(self, other, npop, op_name, dense_check):
        if isscalarlike(other):
            if dense_check(other):
                warn("Taking maximum (minimum) with > 0 (< 0) number results"
                     " to a dense matrix.", SparseEfficiencyWarning,
                     stacklevel=3)
                other_arr = np.empty(self.shape, dtype=np.asarray(other).dtype)
                other_arr.fill(other)
                other_arr = self.__class__(other_arr)
                return self._binopt(other_arr, op_name)
            else:
                self.sum_duplicates()
                new_data = npop(self.data, np.asarray(other))
                mat = self.__class__((new_data, self.indices, self.indptr),
                                     dtype=new_data.dtype, shape=self.shape)
                return mat
        elif isdense(other):
            return npop(self.todense(), other)
        elif issparse(other):
            return self._binopt(other, op_name)
        else:
            raise ValueError("Operands not compatible.")

    def maximum(self, other):
        return self._maximum_minimum(other, np.maximum,
                                     '_maximum_', lambda x: np.asarray(x) > 0)

    maximum.__doc__ = _spbase.maximum.__doc__

    def minimum(self, other):
        return self._maximum_minimum(other, np.minimum,
                                     '_minimum_', lambda x: np.asarray(x) < 0)

    minimum.__doc__ = _spbase.minimum.__doc__

    #####################
    # Reduce operations #
    #####################

    def sum(self, axis=None, dtype=None, out=None):
        """Sum the array/matrix over the given axis.  If the axis is None, sum
        over both rows and columns, returning a scalar.
        """
        # The _spbase base class already does axis=0 and axis=1 efficiently
        # so we only do the case axis=None here
        if (self.ndim == 2 and not hasattr(self, 'blocksize') and
                axis in self._swap(((1, -1), (0, -2)))[0]):
            # faster than multiplication for large minor axis in CSC/CSR
            res_dtype = get_sum_dtype(self.dtype)
            ret = np.zeros(len(self.indptr) - 1, dtype=res_dtype)

            major_index, value = self._minor_reduce(np.add)
            ret[major_index] = value
            ret = self._ascontainer(ret)
            if axis % 2 == 1:
                ret = ret.T

            return ret.sum(axis=(), dtype=dtype, out=out)
        else:
            # _spbase handles the situations when axis is in {None, -2, -1, 0, 1}
            return _spbase.sum(self, axis=axis, dtype=dtype, out=out)

    sum.__doc__ = _spbase.sum.__doc__

    def _minor_reduce(self, ufunc, data=None):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.

        Warning: this does not call sum_duplicates()

        Returns
        -------
        major_index : array of ints
            Major indices where nonzero

        value : array of self.dtype
            Reduce result for nonzeros in each major_index
        """
        if data is None:
            data = self.data
        major_index = np.flatnonzero(np.diff(self.indptr))
        value = ufunc.reduceat(data,
                               downcast_intp_index(self.indptr[major_index]))
        return major_index, value

    #######################
    # Getting and Setting #
    #######################

    def _get_intXint(self, row, col):
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data,
            major, major + 1, minor, minor + 1)
        return data.sum(dtype=self.dtype)

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap((row, col))
        if major.step in (1, None) and minor.step in (1, None):
            return self._get_submatrix(major, minor, copy=True)
        return self._major_slice(major)._minor_slice(minor)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        idx_dtype = self.indices.dtype
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        major = np.asarray(major, dtype=idx_dtype)
        minor = np.asarray(minor, dtype=idx_dtype)

        val = np.empty(major.size, dtype=self.dtype)
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          major.size, major.ravel(), minor.ravel(), val)
        if major.ndim == 1:
            return self._ascontainer(val)
        return self.__class__(val.reshape(major.shape))

    def _get_columnXarray(self, row, col):
        # outer indexing
        major, minor = self._swap((row, col))
        return self._major_index_fancy(major)._minor_index_fancy(minor)

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        idx_dtype = self._get_index_dtype((self.indptr, self.indices))
        indices = np.asarray(idx, dtype=idx_dtype).ravel()

        N = self._swap(self._shape_as_2d)[1]
        M = len(indices)
        new_shape = self._swap((M, N)) if self.ndim == 2 else (M,)
        if M == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        row_nnz = (self.indptr[indices + 1] - self.indptr[indices]).astype(idx_dtype)
        res_indptr = np.zeros(M + 1, dtype=idx_dtype)
        np.cumsum(row_nnz, out=res_indptr[1:])

        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_row_index(
            M,
            indices,
            self.indptr.astype(idx_dtype, copy=False),
            self.indices.astype(idx_dtype, copy=False),
            self.data,
            res_indices,
            res_data
        )

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self._shape_as_2d)
        start, stop, step = idx.indices(M)
        M = len(range(start, stop, step))
        new_shape = self._swap((M, N)) if self.ndim == 2 else (M,)
        if M == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        # Work out what slices are needed for `row_nnz`
        # start,stop can be -1, only if step is negative
        start0, stop0 = start, stop
        if stop == -1 and start >= 0:
            stop0 = None
        start1, stop1 = start + 1, stop + 1

        row_nnz = self.indptr[start1:stop1:step] - \
            self.indptr[start0:stop0:step]
        idx_dtype = self.indices.dtype
        res_indptr = np.zeros(M+1, dtype=idx_dtype)
        np.cumsum(row_nnz, out=res_indptr[1:])

        if step == 1:
            all_idx = slice(self.indptr[start], self.indptr[stop])
            res_indices = np.array(self.indices[all_idx], copy=copy)
            res_data = np.array(self.data[all_idx], copy=copy)
        else:
            nnz = res_indptr[-1]
            res_indices = np.empty(nnz, dtype=idx_dtype)
            res_data = np.empty(nnz, dtype=self.dtype)
            csr_row_slice(start, stop, step, self.indptr, self.indices,
                          self.data, res_indices, res_data)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        idx_dtype = self._get_index_dtype((self.indices, self.indptr))
        indices = self.indices.astype(idx_dtype, copy=False)
        indptr = self.indptr.astype(idx_dtype, copy=False)

        idx = np.asarray(idx, dtype=idx_dtype).ravel()

        M, N = self._swap(self._shape_as_2d)
        k = len(idx)
        new_shape = self._swap((M, k)) if self.ndim == 2 else (k,)
        if k == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        # pass 1: count idx entries and compute new indptr
        col_offsets = np.zeros(N, dtype=idx_dtype)
        res_indptr = np.empty_like(self.indptr, dtype=idx_dtype)
        csr_column_index1(
            k,
            idx,
            M,
            N,
            indptr,
            indices,
            col_offsets,
            res_indptr,
        )

        # pass 2: copy indices/data for selected idxs
        col_order = np.argsort(idx).astype(idx_dtype, copy=False)
        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_column_index2(col_order, col_offsets, len(self.indices),
                          indices, self.data, res_indices, res_data)
        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._swap(self._shape_as_2d)
        start, stop, step = idx.indices(N)
        N = len(range(start, stop, step))
        if N == 0:
            return self.__class__(self._swap((M, N)), dtype=self.dtype)
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        # TODO: don't fall back to fancy indexing here
        return self._minor_index_fancy(np.arange(start, stop, step))

    def _get_submatrix(self, major=None, minor=None, copy=False):
        """Return a submatrix of this matrix.

        major, minor: None, int, or slice with step 1
        """
        M, N = self._swap(self._shape_as_2d)
        i0, i1 = _process_slice(major, M)
        j0, j1 = _process_slice(minor, N)

        if i0 == 0 and j0 == 0 and i1 == M and j1 == N:
            return self.copy() if copy else self

        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)

        shape = self._swap((i1 - i0, j1 - j0))
        if self.ndim == 1:
            shape = (shape[1],)
        return self.__class__((data, indices, indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _set_intXint(self, row, col, x):
        i, j = self._swap((row, col))
        self._set_many(i, j, x)

    def _set_arrayXarray(self, row, col, x):
        i, j = self._swap((row, col))
        self._set_many(i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # clear entries that will be overwritten
        self._zero_many(*self._swap((row, col)))

        M, N = row.shape  # matches col.shape
        broadcast_row = M != 1 and x.shape[0] == 1
        broadcast_col = N != 1 and x.shape[1] == 1
        r, c = x.row, x.col

        x = np.asarray(x.data, dtype=self.dtype)
        if x.size == 0:
            return

        if broadcast_row:
            r = np.repeat(np.arange(M), len(r))
            c = np.tile(c, M)
            x = np.tile(x, M)
        if broadcast_col:
            r = np.repeat(r, N)
            c = np.tile(np.arange(N), len(c))
            x = np.repeat(x, N)
        # only assign entries in the new sparsity structure
        i, j = self._swap((row[r, c], col[r, c]))
        self._set_many(i, j, x)

    def _setdiag(self, values, k):
        if 0 in self.shape:
            return
        if self.ndim == 1:
            raise NotImplementedError('diagonals cant be set in 1d arrays')

        M, N = self.shape
        broadcast = (values.ndim == 0)

        if k < 0:
            if broadcast:
                max_index = min(M + k, N)
            else:
                max_index = min(M + k, N, len(values))
            i = np.arange(-k, max_index - k, dtype=self.indices.dtype)
            j = np.arange(max_index, dtype=self.indices.dtype)

        else:
            if broadcast:
                max_index = min(M, N - k)
            else:
                max_index = min(M, N - k, len(values))
            i = np.arange(max_index, dtype=self.indices.dtype)
            j = np.arange(k, k + max_index, dtype=self.indices.dtype)

        if not broadcast:
            values = values[:len(i)]

        x = np.atleast_1d(np.asarray(values, dtype=self.dtype)).ravel()
        if x.squeeze().shape != i.squeeze().shape:
            x = np.broadcast_to(x, i.shape)
        if x.size == 0:
            return

        M, N = self._swap((M, N))
        i, j = self._swap((i, j))
        n_samples = x.size
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # rinse and repeat
            self.sum_duplicates()
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)
        if -1 not in offsets:
            # only affects existing non-zero cells
            self.data[offsets] = x
            return

        mask = (offsets >= 0)
        # Boundary between csc and convert to coo
        # The value 0.001 is justified in gh-19962#issuecomment-1920499678
        if self.nnz - mask.sum() < self.nnz * 0.001:
            # replace existing entries
            self.data[offsets[mask]] = x[mask]
            # create new entries
            mask = ~mask
            i = i[mask]
            j = j[mask]
            self._insert_many(i, j, x[mask])
        else:
            # convert to coo for _set_diag
            coo = self.tocoo()
            coo._setdiag(values, k)
            arrays = coo._coo_to_compressed(self._swap)
            self.indptr, self.indices, self.data, _ = arrays

    def _prepare_indices(self, i, j):
        M, N = self._swap(self._shape_as_2d)

        def check_bounds(indices, bound):
            idx = indices.max()
            if idx >= bound:
                raise IndexError(f'index ({idx}) out of range (>= {bound})')
            idx = indices.min()
            if idx < -bound:
                raise IndexError(f'index ({idx}) out of range (< -{bound})')

        i = np.atleast_1d(np.asarray(i, dtype=self.indices.dtype)).ravel()
        j = np.atleast_1d(np.asarray(j, dtype=self.indices.dtype)).ravel()
        check_bounds(i, M)
        check_bounds(j, N)
        return i, j, M, N

    def _set_many(self, i, j, x):
        """Sets value at each (i, j) to x

        Here (i,j) index major and minor respectively, and must not contain
        duplicate entries.
        """
        i, j, M, N = self._prepare_indices(i, j)
        x = np.atleast_1d(np.asarray(x, dtype=self.dtype)).ravel()

        n_samples = x.size
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # rinse and repeat
            self.sum_duplicates()
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)

        if -1 not in offsets:
            # only affects existing non-zero cells
            self.data[offsets] = x
            return

        else:
            warn(f"Changing the sparsity structure of a {self.__class__.__name__} is"
                 " expensive. lil and dok are more efficient.",
                 SparseEfficiencyWarning, stacklevel=3)
            # replace where possible
            mask = offsets > -1
            self.data[offsets[mask]] = x[mask]
            # only insertions remain
            mask = ~mask
            i = i[mask]
            i[i < 0] += M
            j = j[mask]
            j[j < 0] += N
            self._insert_many(i, j, x[mask])

    def _zero_many(self, i, j):
        """Sets value at each (i, j) to zero, preserving sparsity structure.

        Here (i,j) index major and minor respectively.
        """
        i, j, M, N = self._prepare_indices(i, j)

        n_samples = len(i)
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # rinse and repeat
            self.sum_duplicates()
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)

        # only assign zeros to the existing sparsity structure
        self.data[offsets[offsets > -1]] = 0

    def _insert_many(self, i, j, x):
        """Inserts new nonzero at each (i, j) with value x

        Here (i,j) index major and minor respectively.
        i, j and x must be non-empty, 1d arrays.
        Inserts each major group (e.g. all entries per row) at a time.
        Maintains has_sorted_indices property.
        Modifies i, j, x in place.
        """
        order = np.argsort(i, kind='mergesort')  # stable for duplicates
        i = i.take(order, mode='clip')
        j = j.take(order, mode='clip')
        x = x.take(order, mode='clip')

        do_sort = self.has_sorted_indices

        # Update index data type
        idx_dtype = self._get_index_dtype((self.indices, self.indptr),
                                    maxval=(self.indptr[-1] + x.size))
        self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
        self.indices = np.asarray(self.indices, dtype=idx_dtype)
        i = np.asarray(i, dtype=idx_dtype)
        j = np.asarray(j, dtype=idx_dtype)

        # Collate old and new in chunks by major index
        indices_parts = []
        data_parts = []
        ui, ui_indptr = np.unique(i, return_index=True)
        ui_indptr = np.append(ui_indptr, len(j))
        new_nnzs = np.diff(ui_indptr)
        prev = 0
        for c, (ii, js, je) in enumerate(zip(ui, ui_indptr, ui_indptr[1:])):
            # old entries
            start = self.indptr[prev]
            stop = self.indptr[ii]
            indices_parts.append(self.indices[start:stop])
            data_parts.append(self.data[start:stop])

            # handle duplicate j: keep last setting
            uj, uj_indptr = np.unique(j[js:je][::-1], return_index=True)
            if len(uj) == je - js:
                indices_parts.append(j[js:je])
                data_parts.append(x[js:je])
            else:
                indices_parts.append(j[js:je][::-1][uj_indptr])
                data_parts.append(x[js:je][::-1][uj_indptr])
                new_nnzs[c] = len(uj)

            prev = ii

        # remaining old entries
        start = self.indptr[ii]
        indices_parts.append(self.indices[start:])
        data_parts.append(self.data[start:])

        # update attributes
        self.indices = np.concatenate(indices_parts)
        self.data = np.concatenate(data_parts)
        nnzs = np.empty(self.indptr.shape, dtype=idx_dtype)
        nnzs[0] = idx_dtype(0)
        indptr_diff = np.diff(self.indptr)
        indptr_diff[ui] += new_nnzs
        nnzs[1:] = indptr_diff
        self.indptr = np.cumsum(nnzs, out=nnzs)

        if do_sort:
            # TODO: only sort where necessary
            self.has_sorted_indices = False
            self.sort_indices()

        self.check_format(full_check=False)

    ######################
    # Conversion methods #
    ######################

    def tocoo(self, copy=True):
        if self.ndim == 1:
            csr = self.tocsr()
            return self._coo_container((csr.data, (csr.indices,)), csr.shape, copy=copy)
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        _sparsetools.expandptr(major_dim, self.indptr, major_indices)
        coords = self._swap((major_indices, minor_indices))

        return self._coo_container(
            (self.data, coords), self.shape, copy=copy, dtype=self.dtype
        )

    tocoo.__doc__ = _spbase.tocoo.__doc__

    def toarray(self, order=None, out=None):
        if out is None and order is None:
            order = self._swap('cf')[0]
        out = self._process_toarray_args(order, out)
        if not (out.flags.c_contiguous or out.flags.f_contiguous):
            raise ValueError('Output array must be C or F contiguous')
        # align ideal order with output array order
        if out.flags.c_contiguous:
            x = self.tocsr()
            y = out
        else:
            x = self.tocsc()
            y = out.T
        M, N = x._swap(x._shape_as_2d)
        csr_todense(M, N, x.indptr, x.indices, x.data, y)
        return out

    toarray.__doc__ = _spbase.toarray.__doc__

    ##############################################################
    # methods that examine or modify the internal data structure #
    ##############################################################

    def eliminate_zeros(self):
        """Remove zero entries from the array/matrix

        This is an *in place* operation.
        """
        M, N = self._swap(self._shape_as_2d)
        _sparsetools.csr_eliminate_zeros(M, N, self.indptr, self.indices, self.data)
        self.prune()  # nnz may have changed

    @property
    def has_canonical_format(self) -> bool:
        """Whether the array/matrix has sorted indices and no duplicates

        Returns
            - True: if the above applies
            - False: otherwise

        has_canonical_format implies has_sorted_indices, so if the latter flag
        is False, so will the former be; if the former is found True, the
        latter flag is also set.
        """
        # first check to see if result was cached
        if not getattr(self, '_has_sorted_indices', True):
            # not sorted => not canonical
            self._has_canonical_format = False
        elif not hasattr(self, '_has_canonical_format'):
            self.has_canonical_format = bool(
                _sparsetools.csr_has_canonical_format(
                    len(self.indptr) - 1, self.indptr, self.indices)
                )
        return self._has_canonical_format

    @has_canonical_format.setter
    def has_canonical_format(self, val: bool):
        self._has_canonical_format = bool(val)
        if val:
            self.has_sorted_indices = True

    def sum_duplicates(self):
        """Eliminate duplicate entries by adding them together

        This is an *in place* operation.
        """
        if self.has_canonical_format:
            return
        self.sort_indices()

        M, N = self._swap(self._shape_as_2d)
        _sparsetools.csr_sum_duplicates(M, N, self.indptr, self.indices, self.data)

        self.prune()  # nnz may have changed
        self.has_canonical_format = True

    @property
    def has_sorted_indices(self) -> bool:
        """Whether the indices are sorted

        Returns
            - True: if the indices of the array/matrix are in sorted order
            - False: otherwise
        """
        # first check to see if result was cached
        if not hasattr(self, '_has_sorted_indices'):
            self._has_sorted_indices = bool(
                _sparsetools.csr_has_sorted_indices(
                    len(self.indptr) - 1, self.indptr, self.indices)
                )
        return self._has_sorted_indices

    @has_sorted_indices.setter
    def has_sorted_indices(self, val: bool):
        self._has_sorted_indices = bool(val)


    def sorted_indices(self):
        """Return a copy of this array/matrix with sorted indices
        """
        A = self.copy()
        A.sort_indices()
        return A

        # an alternative that has linear complexity is the following
        # although the previous option is typically faster
        # return self.toother().toother()

    def sort_indices(self):
        """Sort the indices of this array/matrix *in place*
        """

        if not self.has_sorted_indices:
            _sparsetools.csr_sort_indices(len(self.indptr) - 1, self.indptr,
                                          self.indices, self.data)
            self.has_sorted_indices = True

    def prune(self):
        """Remove empty space after all non-zero elements.
        """
        major_dim = self._swap(self._shape_as_2d)[0]

        if len(self.indptr) != major_dim + 1:
            raise ValueError('index pointer has invalid length')
        if len(self.indices) < self.nnz:
            raise ValueError('indices array has fewer than nnz elements')
        if len(self.data) < self.nnz:
            raise ValueError('data array has fewer than nnz elements')

        self.indices = _prune_array(self.indices[:self.nnz])
        self.data = _prune_array(self.data[:self.nnz])

    def resize(self, *shape):
        shape = check_shape(shape, allow_nd=self._allow_nd)

        if hasattr(self, 'blocksize'):
            bm, bn = self.blocksize
            new_M, rm = divmod(shape[0], bm)
            new_N, rn = divmod(shape[1], bn)
            if rm or rn:
                raise ValueError(f"shape must be divisible into {self.blocksize}"
                                 f" blocks. Got {shape}")
            M, N = self.shape[0] // bm, self.shape[1] // bn
        else:
            new_M, new_N = self._swap(shape if len(shape)>1 else (1, shape[0]))
            M, N = self._swap(self._shape_as_2d)

        if new_M < M:
            self.indices = self.indices[:self.indptr[new_M]]
            self.data = self.data[:self.indptr[new_M]]
            self.indptr = self.indptr[:new_M + 1]
        elif new_M > M:
            self.indptr = np.resize(self.indptr, new_M + 1)
            self.indptr[M + 1:].fill(self.indptr[M])

        if new_N < N:
            mask = self.indices < new_N
            if not np.all(mask):
                self.indices = self.indices[mask]
                self.data = self.data[mask]
                major_index, val = self._minor_reduce(np.add, mask)
                self.indptr.fill(0)
                self.indptr[1:][major_index] = val
                np.cumsum(self.indptr, out=self.indptr)

        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__

    ###################
    # utility methods #
    ###################

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            return self.__class__((data, self.indices.copy(),
                                   self.indptr.copy()),
                                  shape=self.shape,
                                  dtype=data.dtype)
        else:
            return self.__class__((data, self.indices, self.indptr),
                                  shape=self.shape, dtype=data.dtype)

    def _binopt(self, other, op):
        """apply the binary operation fn to two sparse matrices."""
        other = self.__class__(other)

        # e.g. csr_plus_csr, csr_minus_csr, etc.
        fn = getattr(_sparsetools, self.format + op + self.format)

        maxnnz = self.nnz + other.nnz
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=maxnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(maxnnz, dtype=idx_dtype)

        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(maxnnz, dtype=np.bool_)
        else:
            data = np.empty(maxnnz, dtype=upcast(self.dtype, other.dtype))

        M, N = self._shape_as_2d
        fn(M, N,
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        A = self.__class__((data, indices, indptr), shape=self.shape)
        A.prune()

        return A

    def _divide_sparse(self, other):
        """
        Divide this matrix by a second sparse matrix.
        """
        if other.shape != self.shape:
            raise ValueError('inconsistent shapes')

        r = self._binopt(other, '_eldiv_')

        if np.issubdtype(r.dtype, np.inexact):
            # Eldiv leaves entries outside the combined sparsity
            # pattern empty, so they must be filled manually.
            # Everything outside of other's sparsity is NaN, and everything
            # inside it is either zero or defined by eldiv.
            out = np.empty(self.shape, dtype=self.dtype)
            out.fill(np.nan)
            coords = other.nonzero()
            if self.ndim == 1:
                coords = (coords[-1],)
            out[coords] = 0
            r = r.tocoo()
            out[r.coords] = r.data
            return self._container(out)
        else:
            # integers types go with nan <-> 0
            out = r
            return out

    def _broadcast_to(self, shape, copy=False):
        if self.shape == shape:
            return self.copy() if copy else self

        shape = check_shape(shape, allow_nd=(self._allow_nd))

        if broadcast_shapes(self.shape, shape) != shape:
            raise ValueError("cannot be broadcast")

        if len(self.shape) == 1 and len(shape) == 1:
            self.sum_duplicates()
            if self.nnz == 0: # array has no non zero elements
                return self.__class__(shape, dtype=self.dtype, copy=False)

            N = shape[0]
            data = np.full(N, self.data[0])
            indices = np.arange(0,N)
            indptr = np.array([0, N])
            return self._csr_container((data, indices, indptr), shape=shape, copy=False)

        # treat 1D as a 2D row
        old_shape = self._shape_as_2d

        if len(shape) != 2:
            ndim = len(shape)
            raise ValueError(f'CSR/CSC broadcast_to cannot have shape >2D. Got {ndim}D')

        if self.nnz == 0: # array has no non zero elements
            return self.__class__(shape, dtype=self.dtype, copy=False)

        self.sum_duplicates()
        M, N = self._swap(shape)
        oM, oN = self._swap(old_shape)
        if all(s == 1 for s in old_shape):
            # Broadcast a single element to the entire shape
            data = np.full(M * N, self.data[0])
            indices = np.tile(np.arange(N), M)
            indptr = np.arange(0, len(data) + 1, N)
        elif oM == 1 and oN == N:
            # Broadcast row-wise (columns for CSC)
            data = np.tile(self.data, M)
            indices = np.tile(self.indices, M)
            indptr = np.arange(0, len(data) + 1, len(self.data))
        elif oN == 1 and oM == M:
            # Broadcast column-wise (rows for CSC)
            data = np.repeat(self.data, N)
            indices = np.tile(np.arange(N), len(self.data))
            indptr = self.indptr * N
        return self.__class__((data, indices, indptr), shape=shape, copy=False)


def _make_diagonal_csr(data, is_array=False):
    """build diagonal csc_array/csr_array => self._csr_container

    Parameter `data` should be a raveled numpy array holding the
    values on the diagonal of the resulting sparse matrix.
    """
    from ._csr import csr_array, csr_matrix
    csr_array = csr_array if is_array else csr_matrix

    N = len(data)
    idx_dtype = get_index_dtype(maxval=N)
    indptr = np.arange(N + 1, dtype=idx_dtype)
    indices = indptr[:-1]

    return csr_array((data, indices, indptr), shape=(N, N))


def _process_slice(sl, num):
    if sl is None:
        i0, i1 = 0, num
    elif isinstance(sl, slice):
        i0, i1, stride = sl.indices(num)
        if stride != 1:
            raise ValueError('slicing with step != 1 not supported')
        i0 = min(i0, i1)  # give an empty slice when i0 > i1
    elif isintlike(sl):
        if sl < 0:
            sl += num
        i0, i1 = sl, sl + 1
        if i0 < 0 or i1 > num:
            raise IndexError(f'index out of bounds: 0 <= {i0} < {i1} <= {num}')
    else:
        raise TypeError('expected slice or scalar')

    return i0, i1
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="sparsetools",
                                   private_modules=["_sparsetools"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
===================================
Sparse arrays (:mod:`scipy.sparse`)
===================================

.. currentmodule:: scipy.sparse

.. toctree::
   :hidden:

   sparse.csgraph
   sparse.linalg
   sparse.migration_to_sparray

SciPy 2-D sparse array package for numeric data.

.. note::

   This package is switching to an array interface, compatible with
   NumPy arrays, from the older matrix interface.  We recommend that
   you use the array objects (`bsr_array`, `coo_array`, etc.) for
   all new work.

   When using the array interface, please note that:

   - ``x * y`` no longer performs matrix multiplication, but
     element-wise multiplication (just like with NumPy arrays).  To
     make code work with both arrays and matrices, use ``x @ y`` for
     matrix multiplication.
   - Operations such as ``sum``, that used to produce dense matrices, now
     produce arrays, whose multiplication behavior differs similarly.
   - Sparse arrays use array style *slicing* operations, returning scalars,
     1D, or 2D sparse arrays. If you need 2D results, use an appropriate index.
     E.g. ``A[:, i, None]`` or ``A[:, [i]]``.

   The construction utilities (`eye`, `kron`, `random`, `diags`, etc.)
   have appropriate replacements (see :ref:`sparse-construction-functions`).

   For more information see
   :ref:`Migration from spmatrix to sparray <migration_to_sparray>`.


Submodules
==========

.. autosummary::

   csgraph - Compressed sparse graph routines
   linalg - Sparse linear algebra routines


Sparse array classes
====================

.. autosummary::
   :toctree: generated/

   bsr_array - Block Sparse Row array
   coo_array - A sparse array in COOrdinate format
   csc_array - Compressed Sparse Column array
   csr_array - Compressed Sparse Row array
   dia_array - Sparse array with DIAgonal storage
   dok_array - Dictionary Of Keys based sparse array
   lil_array - Row-based list of lists sparse array
   sparray - Sparse array base class

.. _sparse-construction-functions:

Building sparse arrays
----------------------

.. autosummary::
   :toctree: generated/

   diags_array - Return a sparse array from diagonals
   eye_array - Sparse MxN array whose k-th diagonal is all ones
   random_array - Random values in a given shape array
   block_array - Build a sparse array from sub-blocks

.. _combining-arrays:

Combining arrays
----------------

.. autosummary::
   :toctree: generated/

   kron - Kronecker product of two sparse arrays
   kronsum - Kronecker sum of sparse arrays
   block_diag - Build a block diagonal sparse array
   tril - Lower triangular portion of a sparse array
   triu - Upper triangular portion of a sparse array
   hstack - Stack sparse arrays horizontally (column wise)
   vstack - Stack sparse arrays vertically (row wise)

Sparse tools
------------

.. autosummary::
   :toctree: generated/

   save_npz - Save a sparse array to a file using ``.npz`` format.
   load_npz - Load a sparse array from a file using ``.npz`` format.
   find - Return the indices and values of the nonzero elements
   get_index_dtype - determine a good dtype for index arrays.
   safely_cast_index_arrays - cast index array dtype or raise if shape too big

Identifying sparse arrays
-------------------------

.. autosummary::
   :toctree: generated/

   issparse - Check if the argument is a sparse object (array or matrix).


Sparse matrix classes
=====================

.. autosummary::
   :toctree: generated/

   bsr_matrix - Block Sparse Row matrix
   coo_matrix - A sparse matrix in COOrdinate format
   csc_matrix - Compressed Sparse Column matrix
   csr_matrix - Compressed Sparse Row matrix
   dia_matrix - Sparse matrix with DIAgonal storage
   dok_matrix - Dictionary Of Keys based sparse matrix
   lil_matrix - Row-based list of lists sparse matrix
   spmatrix - Sparse matrix base class

Building sparse matrices
------------------------

.. autosummary::
   :toctree: generated/

   eye - Sparse MxN matrix whose k-th diagonal is all ones
   identity - Identity matrix in sparse matrix format
   diags - Return a sparse matrix from diagonals
   spdiags - Return a sparse matrix from diagonals
   bmat - Build a sparse matrix from sparse sub-blocks
   random - Random values in a given shape matrix
   rand - Random values in a given shape matrix (old interface)

**Combining matrices use the same functions as for** :ref:`combining-arrays`.

Identifying sparse matrices
---------------------------

.. autosummary::
   :toctree: generated/

   issparse
   isspmatrix
   isspmatrix_csc
   isspmatrix_csr
   isspmatrix_bsr
   isspmatrix_lil
   isspmatrix_dok
   isspmatrix_coo
   isspmatrix_dia


Warnings
========

.. autosummary::
   :toctree: generated/

   SparseEfficiencyWarning
   SparseWarning


Usage information
=================

There are seven available sparse array types:

    1. csc_array: Compressed Sparse Column format
    2. csr_array: Compressed Sparse Row format
    3. bsr_array: Block Sparse Row format
    4. lil_array: List of Lists format
    5. dok_array: Dictionary of Keys format
    6. coo_array: COOrdinate format (aka IJV, triplet format)
    7. dia_array: DIAgonal format

To construct an array efficiently, use any of `coo_array`,
`dok_array` or `lil_array`. `dok_array` and `lil_array`
support basic slicing and fancy indexing with a similar syntax
to NumPy arrays. The COO format does not support indexing (yet)
but can also be used to efficiently construct arrays using coord
and value info.

Despite their similarity to NumPy arrays, it is **strongly discouraged**
to use NumPy functions directly on these arrays because NumPy typically
treats them as generic Python objects rather than arrays, leading to
unexpected (and incorrect) results. If you do want to apply a NumPy
function to these arrays, first check if SciPy has its own implementation
for the given sparse array class, or **convert the sparse array to
a NumPy array** (e.g., using the `toarray` method of the class)
before applying the method.

All conversions among the CSR, CSC, and COO formats are efficient,
linear-time operations.

To perform manipulations such as multiplication or inversion, first
convert the array to either CSC or CSR format. The `lil_array`
format is row-based, so conversion to CSR is efficient, whereas
conversion to CSC is less so.

Matrix vector product
---------------------

To do a vector product between a 2D sparse array and a vector use
the matmul operator (i.e., ``@``) which performs a dot product (like the
``dot`` method):

>>> import numpy as np
>>> from scipy.sparse import csr_array
>>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A @ v
array([ 1, -3, -1], dtype=int64)

The CSR format is especially suitable for fast matrix vector products.

Example 1
---------

Construct a 1000x1000 `lil_array` and add some values to it:

>>> from scipy.sparse import lil_array
>>> from scipy.sparse.linalg import spsolve
>>> from numpy.linalg import solve, norm
>>> from numpy.random import rand

>>> A = lil_array((1000, 1000))
>>> A[0, :100] = rand(100)
>>> A.setdiag(rand(1000))

Now convert it to CSR format and solve A x = b for x:

>>> A = A.tocsr()
>>> b = rand(1000)
>>> x = spsolve(A, b)

Convert it to a dense array and solve, and check that the result
is the same:

>>> x_ = solve(A.toarray(), b)

Now we can compute norm of the error with:

>>> err = norm(x-x_)
>>> err < 1e-10
True

It should be small :)


Example 2
---------

Construct an array in COO format:

>>> from scipy import sparse
>>> from numpy import array
>>> I = array([0,3,1,0])
>>> J = array([0,3,1,2])
>>> V = array([4,5,7,9])
>>> A = sparse.coo_array((V,(I,J)),shape=(4,4))

Notice that the indices do not need to be sorted.

Duplicate (i,j) entries are summed when converting to CSR or CSC.

>>> I = array([0,0,1,3,1,0,0])
>>> J = array([0,2,1,3,1,0,0])
>>> V = array([1,1,1,1,1,1,1])
>>> B = sparse.coo_array((V,(I,J)),shape=(4,4)).tocsr()

This is useful for constructing finite-element stiffness and mass matrices.

Further details
---------------

CSR column indices are not necessarily sorted. Likewise for CSC row
indices. Use the ``.sorted_indices()`` and ``.sort_indices()`` methods when
sorted indices are required (e.g., when passing data to other libraries).

"""

# Original code by Travis Oliphant.
# Modified and extended by Ed Schofield, Robert Cimrman,
# Nathan Bell, and Jake Vanderplas.

import warnings as _warnings

from ._base import *
from ._csr import *
from ._csc import *
from ._lil import *
from ._dok import *
from ._coo import *
from ._dia import *
from ._bsr import *
from ._construct import *
from ._extract import *
from ._matrix import spmatrix
from ._matrix_io import *
from ._sputils import get_index_dtype, safely_cast_index_arrays

# For backward compatibility with v0.19.
from . import csgraph

# Deprecated namespaces, to be removed in v2.0.0
from . import (
    base, bsr, compressed, construct, coo, csc, csr, data, dia, dok, extract,
    lil, sparsetools, sputils
)

__all__ = [s for s in dir() if not s.startswith('_')]

# Filter PendingDeprecationWarning for np.matrix introduced with numpy 1.15
msg = 'the matrix subclass is not the recommended way'
_warnings.filterwarnings('ignore', message=msg)

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""Sparse DIAgonal format"""

__docformat__ = "restructuredtext en"

__all__ = ['dia_array', 'dia_matrix', 'isspmatrix_dia']

import numpy as np

from .._lib._util import copy_if_needed
from ._matrix import spmatrix
from ._base import issparse, _formats, _spbase, sparray
from ._data import _data_matrix
from ._sputils import (
    isshape, upcast_char, getdtype, get_sum_dtype, validateaxis, check_shape
)
from ._sparsetools import dia_matvec


class _dia_base(_data_matrix):
    _format = 'dia'

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _data_matrix.__init__(self, arg1, maxprint=maxprint)

        if issparse(arg1):
            if arg1.format == "dia":
                if copy:
                    arg1 = arg1.copy()
                self.data = arg1.data
                self.offsets = arg1.offsets
                self._shape = check_shape(arg1.shape)
            else:
                if arg1.format == self.format and copy:
                    A = arg1.copy()
                else:
                    A = arg1.todia()
                self.data = A.data
                self.offsets = A.offsets
                self._shape = check_shape(A.shape)
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self._shape = check_shape(arg1)
                self.data = np.zeros((0,0), getdtype(dtype, default=float))
                idx_dtype = self._get_index_dtype(maxval=max(self.shape))
                self.offsets = np.zeros((0), dtype=idx_dtype)
            else:
                try:
                    # Try interpreting it as (data, offsets)
                    data, offsets = arg1
                except Exception as e:
                    message = 'unrecognized form for dia_array constructor'
                    raise ValueError(message) from e
                else:
                    if shape is None:
                        raise ValueError('expected a shape argument')
                    if not copy:
                        copy = copy_if_needed
                    self.data = np.atleast_2d(np.array(arg1[0], dtype=dtype, copy=copy))
                    offsets = np.array(arg1[1],
                                       dtype=self._get_index_dtype(maxval=max(shape)),
                                       copy=copy)
                    self.offsets = np.atleast_1d(offsets)
                    self._shape = check_shape(shape)
        else:
            # must be dense, convert to COO first, then to DIA
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise ValueError("unrecognized form for "
                                 f"{self.format}_matrix constructor") from e
            if isinstance(self, sparray) and arg1.ndim != 2:
                raise ValueError(f"DIA arrays don't support {arg1.ndim}D input. Use 2D")
            A = self._coo_container(arg1, dtype=dtype, shape=shape).todia()
            self.data = A.data
            self.offsets = A.offsets
            self._shape = check_shape(A.shape)

        if dtype is not None:
            newdtype = getdtype(dtype)
            self.data = self.data.astype(newdtype)

        # check format
        if self.offsets.ndim != 1:
            raise ValueError('offsets array must have rank 1')

        if self.data.ndim != 2:
            raise ValueError('data array must have rank 2')

        if self.data.shape[0] != len(self.offsets):
            raise ValueError('number of diagonals (%d) '
                    'does not match the number of offsets (%d)'
                    % (self.data.shape[0], len(self.offsets)))

        if len(np.unique(self.offsets)) != len(self.offsets):
            raise ValueError('offset array contains duplicate values')

    def __repr__(self):
        _, fmt = _formats[self.format]
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        d = self.data.shape[0]
        return (
            f"<{fmt} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements ({d} diagonals) and shape {self.shape}>"
        )

    def _data_mask(self):
        """Returns a mask of the same shape as self.data, where
        mask[i,j] is True when data[i,j] corresponds to a stored element."""
        num_rows, num_cols = self.shape
        offset_inds = np.arange(self.data.shape[1])
        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        return mask

    def count_nonzero(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "count_nonzero over an axis is not implemented for DIA format"
            )
        mask = self._data_mask()
        return np.count_nonzero(self.data[mask])

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("_getnnz over an axis is not implemented "
                                      "for DIA format")
        M,N = self.shape
        nnz = 0
        for k in self.offsets:
            if k > 0:
                nnz += min(M,N-k)
            else:
                nnz += min(M+k,N)
        return int(nnz)

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def sum(self, axis=None, dtype=None, out=None):
        validateaxis(axis)

        if axis is not None and axis < 0:
            axis += 2

        res_dtype = get_sum_dtype(self.dtype)
        num_rows, num_cols = self.shape
        ret = None

        if axis == 0:
            mask = self._data_mask()
            x = (self.data * mask).sum(axis=0)
            if x.shape[0] == num_cols:
                res = x
            else:
                res = np.zeros(num_cols, dtype=x.dtype)
                res[:x.shape[0]] = x
            ret = self._ascontainer(res, dtype=res_dtype)

        else:
            row_sums = np.zeros((num_rows, 1), dtype=res_dtype)
            one = np.ones(num_cols, dtype=res_dtype)
            dia_matvec(num_rows, num_cols, len(self.offsets),
                       self.data.shape[1], self.offsets, self.data, one, row_sums)

            row_sums = self._ascontainer(row_sums)

            if axis is None:
                return row_sums.sum(dtype=dtype, out=out)

            ret = self._ascontainer(row_sums.sum(axis=axis))

        return ret.sum(axis=(), dtype=dtype, out=out)

    sum.__doc__ = _spbase.sum.__doc__

    def _add_sparse(self, other):
        # If other is not DIA format, let them handle us instead.
        if not isinstance(other, _dia_base):
            return other._add_sparse(self)

        # Fast path for exact equality of the sparsity structure.
        if np.array_equal(self.offsets, other.offsets):
            return self._with_data(self.data + other.data)

        # Find the union of the offsets (which will be sorted and unique).
        new_offsets = np.union1d(self.offsets, other.offsets)
        self_idx = np.searchsorted(new_offsets, self.offsets)
        other_idx = np.searchsorted(new_offsets, other.offsets)

        self_d = self.data.shape[1]
        other_d = other.data.shape[1]
        # Fast path for a sparsity structure where the final offsets are a
        # permutation of the existing offsets and the diagonal lengths match.
        if self_d == other_d and len(new_offsets) == len(self.offsets):
            new_data = self.data[_invert_index(self_idx)]
            new_data[other_idx, :] += other.data
        elif self_d == other_d and len(new_offsets) == len(other.offsets):
            new_data = other.data[_invert_index(other_idx)]
            new_data[self_idx, :] += self.data
        else:
            # Maximum diagonal length of the result.
            d = min(self.shape[0] + new_offsets[-1], self.shape[1])

            # Add all diagonals to a freshly-allocated data array.
            new_data = np.zeros(
                (len(new_offsets), d),
                dtype=np.result_type(self.data, other.data),
            )
            new_data[self_idx, :self_d] += self.data[:, :d]
            new_data[other_idx, :other_d] += other.data[:, :d]
        return self._dia_container((new_data, new_offsets), shape=self.shape)

    def _mul_scalar(self, other):
        return self._with_data(self.data * other)

    def _matmul_vector(self, other):
        x = other

        y = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
                                                       x.dtype.char))

        L = self.data.shape[1]

        M,N = self.shape

        dia_matvec(M,N, len(self.offsets), L, self.offsets, self.data,
                   x.ravel(), y.ravel())

        return y

    def _setdiag(self, values, k=0):
        M, N = self.shape

        if values.ndim == 0:
            # broadcast
            values_n = np.inf
        else:
            values_n = len(values)

        if k < 0:
            n = min(M + k, N, values_n)
            min_index = 0
            max_index = n
        else:
            n = min(M, N - k, values_n)
            min_index = k
            max_index = k + n

        if values.ndim != 0:
            # allow also longer sequences
            values = values[:n]

        data_rows, data_cols = self.data.shape
        if k in self.offsets:
            if max_index > data_cols:
                data = np.zeros((data_rows, max_index), dtype=self.data.dtype)
                data[:, :data_cols] = self.data
                self.data = data
            self.data[self.offsets == k, min_index:max_index] = values
        else:
            self.offsets = np.append(self.offsets, self.offsets.dtype.type(k))
            m = max(max_index, data_cols)
            data = np.zeros((data_rows + 1, m), dtype=self.data.dtype)
            data[:-1, :data_cols] = self.data
            data[-1, min_index:max_index] = values
            self.data = data

    def todia(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    todia.__doc__ = _spbase.todia.__doc__

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        num_rows, num_cols = self.shape
        max_dim = max(self.shape)

        # flip diagonal offsets
        offsets = -self.offsets

        # re-align the data matrix
        r = np.arange(len(offsets), dtype=np.intc)[:, None]
        c = np.arange(num_rows, dtype=np.intc) - (offsets % max_dim)[:, None]
        pad_amount = max(0, max_dim-self.data.shape[1])
        data = np.hstack((self.data, np.zeros((self.data.shape[0], pad_amount),
                                              dtype=self.data.dtype)))
        data = data[r, c]
        return self._dia_container((data, offsets), shape=(
            num_cols, num_rows), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        idx, = np.nonzero(self.offsets == k)
        first_col = max(0, k)
        last_col = min(rows + k, cols)
        result_size = last_col - first_col
        if idx.size == 0:
            return np.zeros(result_size, dtype=self.data.dtype)
        result = self.data[idx[0], first_col:last_col]
        padding = result_size - len(result)
        if padding > 0:
            result = np.pad(result, (0, padding), mode='constant')
        return result

    diagonal.__doc__ = _spbase.diagonal.__doc__

    def tocsc(self, copy=False):
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = np.arange(offset_len)

        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)

        idx_dtype = self._get_index_dtype(maxval=max(self.shape))
        indptr = np.zeros(num_cols + 1, dtype=idx_dtype)
        indptr[1:offset_len+1] = np.cumsum(mask.sum(axis=0)[:num_cols])
        if offset_len < num_cols:
            indptr[offset_len+1:] = indptr[offset_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return self._csc_container((data, indices, indptr), shape=self.shape,
                                   dtype=self.dtype)

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tocoo(self, copy=False):
        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = np.arange(offset_len)

        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)
        row = row[mask]
        col = np.tile(offset_inds, num_offsets)[mask.ravel()]
        idx_dtype = self._get_index_dtype(
            arrays=(self.offsets,), maxval=max(self.shape)
        )
        row = row.astype(idx_dtype, copy=False)
        col = col.astype(idx_dtype, copy=False)
        data = self.data[mask]
        # Note: this cannot set has_canonical_format=True, because despite the
        # lack of duplicates, we do not generate sorted indices.
        return self._coo_container(
            (data, (row, col)), shape=self.shape, dtype=self.dtype, copy=False
        )

    tocoo.__doc__ = _spbase.tocoo.__doc__

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays are copied.
        """
        if copy:
            return self._dia_container(
                (data, self.offsets.copy()), shape=self.shape
            )
        else:
            return self._dia_container(
                (data, self.offsets), shape=self.shape
            )

    def resize(self, *shape):
        shape = check_shape(shape)
        M, N = shape
        # we do not need to handle the case of expanding N
        self.data = self.data[:, :N]

        if (M > self.shape[0] and
                np.any(self.offsets + self.shape[0] < self.data.shape[1])):
            # explicitly clear values that were previously hidden
            mask = (self.offsets[:, None] + self.shape[0] <=
                    np.arange(self.data.shape[1]))
            self.data[mask] = 0

        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__


def _invert_index(idx):
    """Helper function to invert an index array."""
    inv = np.zeros_like(idx)
    inv[idx] = np.arange(len(idx))
    return inv


def isspmatrix_dia(x):
    """Is `x` of dia_matrix type?

    Parameters
    ----------
    x
        object to check for being a dia matrix

    Returns
    -------
    bool
        True if `x` is a dia matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dia_array, dia_matrix, coo_matrix, isspmatrix_dia
    >>> isspmatrix_dia(dia_matrix([[5]]))
    True
    >>> isspmatrix_dia(dia_array([[5]]))
    False
    >>> isspmatrix_dia(coo_matrix([[5]]))
    False
    """
    return isinstance(x, dia_matrix)


# This namespace class separates array from matrix with isinstance
class dia_array(_dia_base, sparray):
    """
    Sparse array with DIAgonal storage.

    This can be instantiated in several ways:
        dia_array(D)
            where D is a 2-D ndarray

        dia_array(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_array((M, N), [dtype])
            to construct an empty array with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_array((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

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
        DIA format data array of the array
    offsets
        DIA format offset array of the array
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.
    Sparse arrays with DIAgonal storage do not support slicing.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_array
    >>> dia_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_array((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_array
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_array((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
    """


class dia_matrix(spmatrix, _dia_base):
    """
    Sparse matrix with DIAgonal storage.

    This can be instantiated in several ways:
        dia_matrix(D)
            where D is a 2-D ndarray

        dia_matrix(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_matrix((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

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
        DIA format data array of the matrix
    offsets
        DIA format offset array of the matrix
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.
    Sparse matrices with DIAgonal storage do not support slicing.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_matrix
    >>> dia_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_matrix((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_matrix
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_matrix((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
    """
<|endoftext|>""" Functions that operate on sparse matrices
"""

__all__ = ['count_blocks','estimate_blocksize']

from ._base import issparse
from ._csr import csr_array
from ._sparsetools import csr_count_blocks


def estimate_blocksize(A,efficiency=0.7):
    """Attempt to determine the blocksize of a sparse matrix

    Returns a blocksize=(r,c) such that
        - A.nnz / A.tobsr( (r,c) ).nnz > efficiency
    """
    if not (issparse(A) and A.format in ("csc", "csr")):
        A = csr_array(A)

    if A.nnz == 0:
        return (1,1)

    if not 0 < efficiency < 1.0:
        raise ValueError('efficiency must satisfy 0.0 < efficiency < 1.0')

    high_efficiency = (1.0 + efficiency) / 2.0
    nnz = float(A.nnz)
    M,N = A.shape

    if M % 2 == 0 and N % 2 == 0:
        e22 = nnz / (4 * count_blocks(A,(2,2)))
    else:
        e22 = 0.0

    if M % 3 == 0 and N % 3 == 0:
        e33 = nnz / (9 * count_blocks(A,(3,3)))
    else:
        e33 = 0.0

    if e22 > high_efficiency and e33 > high_efficiency:
        e66 = nnz / (36 * count_blocks(A,(6,6)))
        if e66 > efficiency:
            return (6,6)
        else:
            return (3,3)
    else:
        if M % 4 == 0 and N % 4 == 0:
            e44 = nnz / (16 * count_blocks(A,(4,4)))
        else:
            e44 = 0.0

        if e44 > efficiency:
            return (4,4)
        elif e33 > efficiency:
            return (3,3)
        elif e22 > efficiency:
            return (2,2)
        else:
            return (1,1)


def count_blocks(A,blocksize):
    """For a given blocksize=(r,c) count the number of occupied
    blocks in a sparse matrix A
    """
    r,c = blocksize
    if r < 1 or c < 1:
        raise ValueError('r and c must be positive')

    if issparse(A):
        if A.format == "csr":
            M,N = A.shape
            return csr_count_blocks(M,N,r,c,A.indptr,A.indices)
        elif A.format == "csc":
            return count_blocks(A.T,(c,r))
    return count_blocks(csr_array(A),blocksize)
<|endoftext|>"""Compressed Sparse Column matrix format"""
__docformat__ = "restructuredtext en"

__all__ = ['csc_array', 'csc_matrix', 'isspmatrix_csc']


import numpy as np

from ._matrix import spmatrix
from ._base import _spbase, sparray
from ._sparsetools import csc_tocsr, expandptr
from ._sputils import upcast

from ._compressed import _cs_matrix


class _csc_base(_cs_matrix):
    _format = 'csc'

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        M, N = self.shape

        return self._csr_container((self.data, self.indices,
                                    self.indptr), (N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def __iter__(self):
        yield from self.tocsr()

    def tocsc(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tocsr(self, copy=False):
        M,N = self.shape
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csc_tocsr(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        A = self._csr_container(
            (data, indices, indptr),
            shape=self.shape, copy=False
        )
        A.has_sorted_indices = True
        return A

    tocsr.__doc__ = _spbase.tocsr.__doc__

    def nonzero(self):
        # CSC can't use _cs_matrix's .nonzero method because it
        # returns the indices sorted for self transposed.

        # Get row and col indices, from _cs_matrix.tocoo
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))

        # Remove explicit zeros
        nz_mask = self.data != 0
        row = row[nz_mask]
        col = col[nz_mask]

        # Sort them to be in C-style order
        ind = np.argsort(row, kind='mergesort')
        row = row[ind]
        col = col[ind]

        return row, col

    nonzero.__doc__ = _cs_matrix.nonzero.__doc__

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(minor=i).tocsr()

    def _getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSC matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        return self._get_submatrix(major=i, copy=True)

    def _get_intXarray(self, row, col):
        return self._major_index_fancy(col)._get_submatrix(minor=row)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._major_slice(col)._get_submatrix(minor=row)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._get_submatrix(major=col)._minor_slice(row)

    def _get_sliceXarray(self, row, col):
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        res = self._get_submatrix(major=col)._minor_index_fancy(row)
        if row.ndim > 1:
            return res.reshape(row.shape)
        return res

    def _get_arrayXslice(self, row, col):
        return self._major_slice(col)._minor_index_fancy(row)

    # these functions are used by the parent class (_cs_matrix)
    # to remove redundancy between csc_array and csr_matrix
    @staticmethod
    def _swap(x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x[1], x[0]


def isspmatrix_csc(x):
    """Is `x` of csc_matrix type?

    Parameters
    ----------
    x
        object to check for being a csc matrix

    Returns
    -------
    bool
        True if `x` is a csc matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import csc_array, csc_matrix, coo_matrix, isspmatrix_csc
    >>> isspmatrix_csc(csc_matrix([[5]]))
    True
    >>> isspmatrix_csc(csc_array([[5]]))
    False
    >>> isspmatrix_csc(coo_matrix([[5]]))
    False
    """
    return isinstance(x, csc_matrix)


# This namespace class separates array from matrix with isinstance
class csc_array(_csc_base, sparray):
    """
    Compressed Sparse Column array.

    This can be instantiated in several ways:
        csc_array(D)
            where D is a 2-D ndarray

        csc_array(S)
            with another sparse array or matrix S (equivalent to S.tocsc())

        csc_array((M, N), [dtype])
            to construct an empty array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csc_array((data, indices, indptr), [shape=(M, N)])
            is the standard CSC representation where the row indices for
            column i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding values are stored in
            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
            not supplied, the array dimensions are inferred from
            the index arrays.

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
        CSC format data array of the array
    indices
        CSC format index array of the array
    indptr
        CSC format index pointer array of the array
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical format
      - Within each column, indices are sorted by row.
      - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> csc_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_array((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """


class csc_matrix(spmatrix, _csc_base):
    """
    Compressed Sparse Column matrix.

    This can be instantiated in several ways:
        csc_matrix(D)
            where D is a 2-D ndarray

        csc_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocsc())

        csc_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csc_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSC representation where the row indices for
            column i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding values are stored in
            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
            not supplied, the matrix dimensions are inferred from
            the index arrays.

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
        CSC format data array of the matrix
    indices
        CSC format index array of the matrix
    indptr
        CSC format index pointer array of the matrix
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical format
      - Within each column, indices are sorted by row.
      - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> csc_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """

<|endoftext|>"""Compressed Block Sparse Row format"""

__docformat__ = "restructuredtext en"

__all__ = ['bsr_array', 'bsr_matrix', 'isspmatrix_bsr']

from warnings import warn

import numpy as np

from scipy._lib._util import copy_if_needed
from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
                       check_shape)
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
                           bsr_matmat, bsr_transpose, bsr_sort_indices,
                           bsr_tocsr)


class _bsr_base(_cs_matrix, _minmax_mixin):
    _format = 'bsr'

    def __init__(self, arg1, shape=None, dtype=None, copy=False,
                 blocksize=None, *, maxprint=None):
        _data_matrix.__init__(self, arg1, maxprint=maxprint)

        if issparse(arg1):
            if arg1.format == self.format and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.tobsr(blocksize=blocksize)
            self.indptr, self.indices, self.data, self._shape = (
                arg1.indptr, arg1.indices, arg1.data, arg1._shape
            )

        elif isinstance(arg1,tuple):
            if isshape(arg1):
                # it's a tuple of matrix dimensions (M,N)
                self._shape = check_shape(arg1)
                M,N = self.shape
                # process blocksize
                if blocksize is None:
                    blocksize = (1,1)
                else:
                    if not isshape(blocksize):
                        raise ValueError(f'invalid blocksize={blocksize}')
                    blocksize = tuple(blocksize)
                self.data = np.zeros((0,) + blocksize, getdtype(dtype, default=float))

                R,C = blocksize
                if (M % R) != 0 or (N % C) != 0:
                    raise ValueError('shape must be multiple of blocksize')

                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                idx_dtype = self._get_index_dtype(maxval=max(M//R, N//C, R, C))
                self.indices = np.zeros(0, dtype=idx_dtype)
                self.indptr = np.zeros(M//R + 1, dtype=idx_dtype)

            elif len(arg1) == 2:
                # (data,(row,col)) format
                coo = self._coo_container(arg1, dtype=dtype, shape=shape)
                bsr = coo.tobsr(blocksize=blocksize)
                self.indptr, self.indices, self.data, self._shape = (
                    bsr.indptr, bsr.indices, bsr.data, bsr._shape
                )

            elif len(arg1) == 3:
                # (data,indices,indptr) format
                (data, indices, indptr) = arg1

                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                maxval = 1
                if shape is not None:
                    maxval = max(shape)
                if blocksize is not None:
                    maxval = max(maxval, max(blocksize))
                idx_dtype = self._get_index_dtype((indices, indptr), maxval=maxval,
                                                  check_contents=True)
                if not copy:
                    copy = copy_if_needed
                self.indices = np.array(indices, copy=copy, dtype=idx_dtype)
                self.indptr = np.array(indptr, copy=copy, dtype=idx_dtype)
                self.data = getdata(data, copy=copy, dtype=dtype)
                if self.data.ndim != 3:
                    raise ValueError(
                        f'BSR data must be 3-dimensional, got shape={self.data.shape}'
                    )
                if blocksize is not None:
                    if not isshape(blocksize):
                        raise ValueError(f'invalid blocksize={blocksize}')
                    if tuple(blocksize) != self.data.shape[1:]:
                        raise ValueError(
                            f'mismatching blocksize={blocksize}'
                            f' vs {self.data.shape[1:]}'
                        )
            else:
                raise ValueError('unrecognized bsr_array constructor usage')
        else:
            # must be dense
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise ValueError("unrecognized form for "
                                 f"{self.format}_matrix constructor") from e
            if isinstance(self, sparray) and arg1.ndim != 2:
                raise ValueError(f"BSR arrays don't support {arg1.ndim}D input. Use 2D")
            arg1 = self._coo_container(arg1, dtype=dtype).tobsr(blocksize=blocksize)
            self.indptr, self.indices, self.data, self._shape = (
                arg1.indptr, arg1.indices, arg1.data, arg1._shape
            )

        if shape is not None:
            self._shape = check_shape(shape)
        else:
            if self.shape is None:
                # shape not already set, try to infer dimensions
                try:
                    M = len(self.indptr) - 1
                    N = self.indices.max() + 1
                except Exception as e:
                    raise ValueError('unable to infer matrix dimensions') from e
                else:
                    R,C = self.blocksize
                    self._shape = check_shape((M*R,N*C))

        if self.shape is None:
            if shape is None:
                # TODO infer shape here
                raise ValueError('need to infer shape')
            else:
                self._shape = check_shape(shape)

        if dtype is not None:
            self.data = self.data.astype(getdtype(dtype, self.data), copy=False)

        self.check_format(full_check=False)

    def check_format(self, full_check=True):
        """Check whether the array/matrix respects the BSR format.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, run rigorous check, scanning arrays for valid values.
            Note that activating those check might copy arrays for casting,
            modifying indices and index pointers' inplace.
            If `False`, run basic checks on attributes. O(1) operations.
            Default is `True`.
        """
        M,N = self.shape
        R,C = self.blocksize

        # index arrays should have integer data types
        if self.indptr.dtype.kind != 'i':
            warn(f"indptr array has non-integer dtype ({self.indptr.dtype.name})",
                 stacklevel=2)
        if self.indices.dtype.kind != 'i':
            warn(f"indices array has non-integer dtype ({self.indices.dtype.name})",
                 stacklevel=2)

        # check array shapes
        if self.indices.ndim != 1 or self.indptr.ndim != 1:
            raise ValueError("indices, and indptr should be 1-D")
        if self.data.ndim != 3:
            raise ValueError("data should be 3-D")

        # check index pointer
        if (len(self.indptr) != M//R + 1):
            raise ValueError("index pointer size (%d) should be (%d)" %
                                (len(self.indptr), M//R + 1))
        if (self.indptr[0] != 0):
            raise ValueError("index pointer should start with 0")

        # check index and data arrays
        if (len(self.indices) != len(self.data)):
            raise ValueError("indices and data should have the same size")
        if (self.indptr[-1] > len(self.indices)):
            raise ValueError("Last value of index pointer should be less than "
                                "the size of index and data arrays")

        self.prune()

        if full_check:
            # check format validity (more expensive)
            if self.nnz > 0:
                if self.indices.max() >= N//C:
                    raise ValueError("column index values must be < %d (now max %d)"
                                     % (N//C, self.indices.max()))
                if self.indices.min() < 0:
                    raise ValueError("column index values must be >= 0")
                if np.diff(self.indptr).min() < 0:
                    raise ValueError("index pointer values must form a "
                                        "non-decreasing sequence")

            idx_dtype = self._get_index_dtype((self.indices, self.indptr))
            self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
            self.indices = np.asarray(self.indices, dtype=idx_dtype)
            self.data = to_native(self.data)
        # if not self.has_sorted_indices():
        #    warn('Indices were not in sorted order. Sorting indices.')
        #    self.sort_indices(check_first=False)

    @property
    def blocksize(self) -> tuple:
        """Block size of the matrix."""
        return self.data.shape[1:]

    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("_getnnz over an axis is not implemented "
                                      "for BSR format")
        R, C = self.blocksize
        return int(self.indptr[-1]) * R * C

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def count_nonzero(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "count_nonzero over axis is not implemented for BSR format."
            )
        return np.count_nonzero(self._deduped_data())

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def __repr__(self):
        _, fmt = _formats[self.format]
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        b = 'x'.join(str(x) for x in self.blocksize)
        return (
            f"<{fmt} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements (blocksize={b}) and shape {self.shape}>"
        )

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        R, C = self.blocksize
        y = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                     dtype=upcast(self.dtype))
        _sparsetools.bsr_diagonal(k, rows // R, cols // C, R, C,
                                  self.indptr, self.indices,
                                  np.ravel(self.data), y)
        return y

    diagonal.__doc__ = _spbase.diagonal.__doc__

    ##########################
    # NotImplemented methods #
    ##########################

    def __getitem__(self,key):
        raise NotImplementedError

    def __setitem__(self,key,val):
        raise NotImplementedError

    ######################
    # Arithmetic methods #
    ######################

    def _add_dense(self, other):
        return self.tocoo(copy=False)._add_dense(other)

    def _matmul_vector(self, other):
        M,N = self.shape
        R,C = self.blocksize

        result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))

        bsr_matvec(M//R, N//C, R, C,
            self.indptr, self.indices, self.data.ravel(),
            other, result)

        return result

    def _matmul_multivector(self,other):
        R,C = self.blocksize
        M,N = self.shape
        n_vecs = other.shape[1]  # number of column vectors

        result = np.zeros((M,n_vecs), dtype=upcast(self.dtype,other.dtype))

        bsr_matvecs(M//R, N//C, n_vecs, R, C,
                self.indptr, self.indices, self.data.ravel(),
                other.ravel(), result.ravel())

        return result

    def _matmul_sparse(self, other):
        M, K1 = self.shape
        K2, N = other.shape

        R,n = self.blocksize

        # convert to this format
        if other.format == "bsr":
            C = other.blocksize[1]
        else:
            C = 1

        if other.format == "csr" and n == 1:
            other = other.tobsr(blocksize=(n,C), copy=False)  # lightweight conversion
        else:
            other = other.tobsr(blocksize=(n,C))

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices))

        bnnz = csr_matmat_maxnnz(M//R, N//C,
                                 self.indptr.astype(idx_dtype),
                                 self.indices.astype(idx_dtype),
                                 other.indptr.astype(idx_dtype),
                                 other.indices.astype(idx_dtype))

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=bnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(bnnz, dtype=idx_dtype)
        data = np.empty(R*C*bnnz, dtype=upcast(self.dtype,other.dtype))

        bsr_matmat(bnnz, M//R, N//C, R, C, n,
                   self.indptr.astype(idx_dtype),
                   self.indices.astype(idx_dtype),
                   np.ravel(self.data),
                   other.indptr.astype(idx_dtype),
                   other.indices.astype(idx_dtype),
                   np.ravel(other.data),
                   indptr,
                   indices,
                   data)

        data = data.reshape(-1,R,C)

        # TODO eliminate zeros

        return self._bsr_container(
            (data, indices, indptr), shape=(M, N), blocksize=(R, C)
        )

    ######################
    # Conversion methods #
    ######################

    def tobsr(self, blocksize=None, copy=False):
        """Convert this array/matrix into Block Sparse Row Format.

        With copy=False, the data/indices may be shared between this
        array/matrix and the resultant bsr_array/bsr_matrix.

        If blocksize=(R, C) is provided, it will be used for determining
        block size of the bsr_array/bsr_matrix.
        """
        if blocksize not in [None, self.blocksize]:
            return self.tocsr().tobsr(blocksize=blocksize)
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self, copy=False):
        M, N = self.shape
        R, C = self.blocksize
        nnz = self.nnz
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                          maxval=max(nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype))

        bsr_tocsr(M // R,  # n_brow
                  N // C,  # n_bcol
                  R, C,
                  self.indptr.astype(idx_dtype, copy=False),
                  self.indices.astype(idx_dtype, copy=False),
                  self.data,
                  indptr,
                  indices,
                  data)
        return self._csr_container((data, indices, indptr), shape=self.shape)

    tocsr.__doc__ = _spbase.tocsr.__doc__

    def tocsc(self, copy=False):
        return self.tocsr(copy=False).tocsc(copy=copy)

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tocoo(self, copy=True):
        """Convert this array/matrix to COOrdinate format.

        When copy=False the data array will be shared between
        this array/matrix and the resultant coo_array/coo_matrix.
        """

        M,N = self.shape
        R,C = self.blocksize

        indptr_diff = np.diff(self.indptr)
        if indptr_diff.dtype.itemsize > np.dtype(np.intp).itemsize:
            # Check for potential overflow
            indptr_diff_limited = indptr_diff.astype(np.intp)
            if np.any(indptr_diff_limited != indptr_diff):
                raise ValueError("Matrix too big to convert")
            indptr_diff = indptr_diff_limited

        idx_dtype = self._get_index_dtype(maxval=max(M, N))
        row = (R * np.arange(M//R, dtype=idx_dtype)).repeat(indptr_diff)
        row = row.repeat(R*C).reshape(-1,R,C)
        row += np.tile(np.arange(R, dtype=idx_dtype).reshape(-1,1), (1,C))
        row = row.reshape(-1)

        col = ((C * self.indices).astype(idx_dtype, copy=False)
               .repeat(R*C).reshape(-1,R,C))
        col += np.tile(np.arange(C, dtype=idx_dtype), (R,1))
        col = col.reshape(-1)

        data = self.data.reshape(-1)

        if copy:
            data = data.copy()

        return self._coo_container(
            (data, (row, col)), shape=self.shape
        )

    def toarray(self, order=None, out=None):
        return self.tocoo(copy=False).toarray(order=order, out=out)

    toarray.__doc__ = _spbase.toarray.__doc__

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        R, C = self.blocksize
        M, N = self.shape
        NBLK = self.nnz//(R*C)

        if self.nnz == 0:
            return self._bsr_container((N, M), blocksize=(C, R),
                                       dtype=self.dtype, copy=copy)

        indptr = np.empty(N//C + 1, dtype=self.indptr.dtype)
        indices = np.empty(NBLK, dtype=self.indices.dtype)
        data = np.empty((NBLK, C, R), dtype=self.data.dtype)

        bsr_transpose(M//R, N//C, R, C,
                      self.indptr, self.indices, self.data.ravel(),
                      indptr, indices, data.ravel())

        return self._bsr_container((data, indices, indptr),
                                   shape=(N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    ##############################################################
    # methods that examine or modify the internal data structure #
    ##############################################################

    def eliminate_zeros(self):
        """Remove zero elements in-place."""

        if not self.nnz:
            return  # nothing to do

        R,C = self.blocksize
        M,N = self.shape

        mask = (self.data != 0).reshape(-1,R*C).sum(axis=1)  # nonzero blocks

        nonzero_blocks = mask.nonzero()[0]

        self.data[:len(nonzero_blocks)] = self.data[nonzero_blocks]

        # modifies self.indptr and self.indices *in place*
        _sparsetools.csr_eliminate_zeros(M//R, N//C, self.indptr,
                                         self.indices, mask)
        self.prune()

    def sum_duplicates(self):
        """Eliminate duplicate array/matrix entries by adding them together

        The is an *in place* operation
        """
        if self.has_canonical_format:
            return
        self.sort_indices()
        R, C = self.blocksize
        M, N = self.shape

        # port of _sparsetools.csr_sum_duplicates
        n_row = M // R
        nnz = 0
        row_end = 0
        for i in range(n_row):
            jj = row_end
            row_end = self.indptr[i+1]
            while jj < row_end:
                j = self.indices[jj]
                x = self.data[jj]
                jj += 1
                while jj < row_end and self.indices[jj] == j:
                    x += self.data[jj]
                    jj += 1
                self.indices[nnz] = j
                self.data[nnz] = x
                nnz += 1
            self.indptr[i+1] = nnz

        self.prune()  # nnz may have changed
        self.has_canonical_format = True

    def sort_indices(self):
        """Sort the indices of this array/matrix *in place*
        """
        if self.has_sorted_indices:
            return

        R,C = self.blocksize
        M,N = self.shape

        bsr_sort_indices(M//R, N//C, R, C, self.indptr, self.indices, self.data.ravel())

        self.has_sorted_indices = True

    def prune(self):
        """Remove empty space after all non-zero elements.
        """

        R,C = self.blocksize
        M,N = self.shape

        if len(self.indptr) != M//R + 1:
            raise ValueError("index pointer has invalid length")

        bnnz = self.indptr[-1]

        if len(self.indices) < bnnz:
            raise ValueError("indices array has too few elements")
        if len(self.data) < bnnz:
            raise ValueError("data array has too few elements")

        self.data = self.data[:bnnz]
        self.indices = self.indices[:bnnz]

    # utility functions
    def _binopt(self, other, op, in_shape=None, out_shape=None):
        """Apply the binary operation fn to two sparse matrices."""

        # Ideally we'd take the GCDs of the blocksize dimensions
        # and explode self and other to match.
        other = self.__class__(other, blocksize=self.blocksize)

        # e.g. bsr_plus_bsr, etc.
        fn = getattr(_sparsetools, self.format + op + self.format)

        R,C = self.blocksize

        max_bnnz = len(self.data) + len(other.data)
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=max_bnnz)
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(max_bnnz, dtype=idx_dtype)

        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(R*C*max_bnnz, dtype=np.bool_)
        else:
            data = np.empty(R*C*max_bnnz, dtype=upcast(self.dtype,other.dtype))

        fn(self.shape[0]//R, self.shape[1]//C, R, C,
           self.indptr.astype(idx_dtype),
           self.indices.astype(idx_dtype),
           self.data,
           other.indptr.astype(idx_dtype),
           other.indices.astype(idx_dtype),
           np.ravel(other.data),
           indptr,
           indices,
           data)

        actual_bnnz = indptr[-1]
        indices = indices[:actual_bnnz]
        data = data[:R*C*actual_bnnz]

        if actual_bnnz < max_bnnz/2:
            indices = indices.copy()
            data = data.copy()

        data = data.reshape(-1,R,C)

        return self.__class__((data, indices, indptr), shape=self.shape)

    # needed by _data_matrix
    def _with_data(self,data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            return self.__class__((data,self.indices.copy(),self.indptr.copy()),
                                   shape=self.shape,dtype=data.dtype)
        else:
            return self.__class__((data,self.indices,self.indptr),
                                   shape=self.shape,dtype=data.dtype)

#    # these functions are used by the parent class
#    # to remove redundancy between bsc_matrix and bsr_matrix
#    def _swap(self,x):
#        """swap the members of x if this is a column-oriented matrix
#        """
#        return (x[0],x[1])

    def _broadcast_to(self, shape, copy=False):
        return _spbase._broadcast_to(self, shape, copy)


def isspmatrix_bsr(x):
    """Is `x` of a bsr_matrix type?

    Parameters
    ----------
    x
        object to check for being a bsr matrix

    Returns
    -------
    bool
        True if `x` is a bsr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import bsr_array, bsr_matrix, csr_matrix, isspmatrix_bsr
    >>> isspmatrix_bsr(bsr_matrix([[5]]))
    True
    >>> isspmatrix_bsr(bsr_array([[5]]))
    False
    >>> isspmatrix_bsr(csr_matrix([[5]]))
    False
    """
    return isinstance(x, bsr_matrix)


# This namespace class separates array from matrix with isinstance
class bsr_array(_bsr_base, sparray):
    """
    Block Sparse Row format sparse array.

    This can be instantiated in several ways:
        bsr_array(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_array(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_array((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_array((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_array((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the array dimensions are inferred from the index arrays.

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
        BSR format data array of the array
    indices
        BSR format index array of the array
    indptr
        BSR format index pointer array of the array
    blocksize
        Block size
    has_sorted_indices : bool
        Whether indices are sorted
    has_canonical_format : bool
    T

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    **Summary of BSR format**

    The Block Sparse Row (BSR) format is very similar to the Compressed
    Sparse Row (CSR) format. BSR is appropriate for sparse matrices with dense
    sub matrices like the last example below. Such sparse block matrices often
    arise in vector-valued finite element discretizations. In such cases, BSR is
    considerably more efficient than CSR and CSC for many sparse arithmetic
    operations.

    **Blocksize**

    The blocksize (R,C) must evenly divide the shape of the sparse array (M,N).
    That is, R and C must satisfy the relationship ``M % R = 0`` and
    ``N % C = 0``.

    If no blocksize is specified, a simple heuristic is applied to determine
    an appropriate blocksize.

    **Canonical Format**

    In canonical format, there are no duplicate blocks and indices are sorted
    per row.

    **Limitations**

    Block Sparse Row format sparse arrays do not support slicing.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_array
    >>> bsr_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3 ,4, 5, 6])
    >>> bsr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> bsr_array((data,indices,indptr), shape=(6, 6)).toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])

    """


class bsr_matrix(spmatrix, _bsr_base):
    """
    Block Sparse Row format sparse matrix.

    This can be instantiated in several ways:
        bsr_matrix(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_matrix(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_matrix((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_matrix((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the matrix dimensions are inferred from the index arrays.

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
        BSR format data array of the matrix
    indices
        BSR format index array of the matrix
    indptr
        BSR format index pointer array of the matrix
    blocksize
        Block size
    has_sorted_indices : bool
        Whether indices are sorted
    has_canonical_format : bool
    T

    Notes
    -----
    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    **Summary of BSR format**

    The Block Sparse Row (BSR) format is very similar to the Compressed
    Sparse Row (CSR) format. BSR is appropriate for sparse matrices with dense
    sub matrices like the last example below. Such sparse block matrices often
    arise in vector-valued finite element discretizations. In such cases, BSR is
    considerably more efficient than CSR and CSC for many sparse arithmetic
    operations.

    **Blocksize**

    The blocksize (R,C) must evenly divide the shape of the sparse matrix (M,N).
    That is, R and C must satisfy the relationship ``M % R = 0`` and
    ``N % C = 0``.

    If no blocksize is specified, a simple heuristic is applied to determine
    an appropriate blocksize.

    **Canonical Format**

    In canonical format, there are no duplicate blocks and indices are sorted
    per row.

    **Limitations**

    Block Sparse Row format sparse matrices do not support slicing.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_matrix
    >>> bsr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3 ,4, 5, 6])
    >>> bsr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])

    """

<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'block_diag',
    'bmat',
    'bsr_matrix',
    'check_random_state',
    'coo_matrix',
    'csc_matrix',
    'csr_hstack',
    'csr_matrix',
    'dia_matrix',
    'diags',
    'eye',
    'get_index_dtype',
    'hstack',
    'identity',
    'isscalarlike',
    'issparse',
    'kron',
    'kronsum',
    'numbers',
    'rand',
    'random',
    'rng_integers',
    'spdiags',
    'upcast',
    'vstack',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="construct",
                                   private_modules=["_construct"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'coo_matrix',
    'find',
    'tril',
    'triu',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="extract",
                                   private_modules=["_extract"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Functions to construct sparse matrices and arrays
"""

__docformat__ = "restructuredtext en"

__all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum',
           'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag',
           'diags_array', 'block_array', 'eye_array', 'random_array']

import numbers
import math
import numpy as np

from scipy._lib._util import check_random_state, rng_integers, _transition_to_rng
from ._sputils import upcast, get_index_dtype, isscalarlike

from ._sparsetools import csr_hstack
from ._bsr import bsr_matrix, bsr_array
from ._coo import coo_matrix, coo_array
from ._csc import csc_matrix, csc_array
from ._csr import csr_matrix, csr_array
from ._dia import dia_matrix, dia_array

from ._base import issparse, sparray


def spdiags(data, diags, m=None, n=None, format=None):
    """
    Return a sparse matrix from diagonals.

    Parameters
    ----------
    data : array_like
        Matrix diagonals stored row-wise
    diags : sequence of int or an int
        Diagonals to set:

        * k = 0  the main diagonal
        * k > 0  the kth upper diagonal
        * k < 0  the kth lower diagonal
    m, n : int, tuple, optional
        Shape of the result. If `n` is None and `m` is a given tuple,
        the shape is this tuple. If omitted, the matrix is square and
        its shape is len(data[0]).
    format : str, optional
        Format of the result. By default (format=None) an appropriate sparse
        matrix format is returned. This choice is subject to change.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``dia_array`` to take advantage
        of the sparse array functionality.

    Notes
    -----
    This function can be replaced by an equivalent call to ``dia_matrix``
    as::

        dia_matrix((data, diags), shape=(m, n)).asformat(format)

    See Also
    --------
    diags_array : more convenient form of this function
    diags : matrix version of diags_array
    dia_matrix : the sparse DIAgonal format.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> diags = np.array([0, -1, 2])
    >>> spdiags(data, diags, 4, 4).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    """
    if m is None and n is None:
        m = n = len(data[0])
    elif n is None:
        m, n = m
    return dia_matrix((data, diags), shape=(m, n)).asformat(format)


def diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse array from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the array diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set (repeated offsets are not allowed):
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square array large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse array format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the array.

    Notes
    -----
    Repeated diagonal offsets are disallowed.

    The result from `diags_array` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    ``diags_array`` differs from `dia_array` in the way it handles off-diagonals.
    Specifically, `dia_array` assumes the data input includes padding
    (ignored values) at the start/end of the rows for positive/negative
    offset, while ``diags_array` assumes the input data has no padding.
    Each value in the input ``diagonals`` is used.

    .. versionadded:: 1.11

    Examples
    --------
    >>> from scipy.sparse import diags_array
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags_array(diagonals, offsets=[0, -1, 2]).toarray()
    array([[1., 0., 1., 0.],
           [1., 2., 0., 2.],
           [0., 2., 3., 0.],
           [0., 0., 3., 4.]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags_array([1, -2, 1], offsets=[-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags_array([1, 2, 3], offsets=1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])

    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if isscalarlike(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or isscalarlike(diagonals[0]):
            diagonals = [np.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(np.atleast_1d, diagonals))

    offsets = np.atleast_1d(offsets)

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        dtype = np.common_type(*diagonals)

    # Construct data array
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset)
             for offset in offsets])
    M = max(0, M)
    data_arr = np.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError(f"Offset {offset} (index {j}) out of bounds")
        try:
            data_arr[j, k:k+length] = diagonal[...,:length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    f"Diagonal length (index {j}: {len(diagonal)} at"
                    f" offset {offset}) does not agree with array size ({m}, {n})."
                ) from e
            raise

    return dia_array((data_arr, offsets), shape=(m, n)).asformat(format)


def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``diags_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set (repeated offsets are not allowed):
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.

    See Also
    --------
    spdiags : construct matrix from diagonals
    diags_array : construct sparse array instead of sparse matrix

    Notes
    -----
    Repeated diagonal offsets are disallowed.

    The result from `diags` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    ``diags`` differs from ``dia_matrix`` in the way it handles off-diagonals.
    Specifically, `dia_matrix` assumes the data input includes padding
    (ignored values) at the start/end of the rows for positive/negative
    offset, while ``diags` assumes the input data has no padding.
    Each value in the input ``diagonals`` is used.

    .. versionadded:: 0.11

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1., 0., 1., 0.],
           [1., 2., 0., 2.],
           [0., 2., 3., 0.],
           [0., 0., 3., 4.]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])

    """
    A = diags_array(diagonals, offsets=offsets, shape=shape, dtype=dtype)
    return dia_matrix(A).asformat(format)


def identity(n, dtype='d', format=None):
    """Identity matrix in sparse format

    Returns an identity matrix with shape (n,n) using a given
    sparse format and dtype. This differs from `eye_array` in
    that it has a square shape with ones only on the main diagonal.
    It is thus the multiplicative identity. `eye_array` allows
    rectangular shapes and the diagonal can be offset from the main one.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    n : int
        Shape of the identity matrix.
    dtype : dtype, optional
        Data type of the matrix
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import scipy as sp
    >>> sp.sparse.identity(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.identity(3, dtype='int8', format='dia')
    <DIAgonal sparse matrix of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>
    >>> sp.sparse.eye_array(3, dtype='int8', format='dia')
    <DIAgonal sparse array of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>

    """
    return eye(n, n, dtype=dtype, format=format)


def eye_array(m, n=None, *, k=0, dtype=float, format=None):
    """Identity matrix in sparse array format

    Return a sparse array with ones on diagonal.
    Specifically a sparse array (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int
        Number of rows requested.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the array
    format : str, optional (default: "dia")
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye_array(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye_array(3, dtype=np.int8)
    <DIAgonal sparse array of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>

    """
    # TODO: delete next 15 lines [combine with _eye()] once spmatrix removed
    return _eye(m, n, k, dtype, format)


def _eye(m, n, k, dtype, format, as_sparray=True):
    if as_sparray:
        csr_sparse = csr_array
        csc_sparse = csc_array
        coo_sparse = coo_array
        diags_sparse = diags_array
    else:
        csr_sparse = csr_matrix
        csc_sparse = csc_matrix
        coo_sparse = coo_matrix
        diags_sparse = diags

    if n is None:
        n = m
    m, n = int(m), int(n)

    if m == n and k == 0:
        # fast branch for special formats
        if format in ['csr', 'csc']:
            idx_dtype = get_index_dtype(maxval=n)
            indptr = np.arange(n+1, dtype=idx_dtype)
            indices = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            cls = {'csr': csr_sparse, 'csc': csc_sparse}[format]
            return cls((data, indices, indptr), (n, n))

        elif format == 'coo':
            idx_dtype = get_index_dtype(maxval=n)
            row = np.arange(n, dtype=idx_dtype)
            col = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            return coo_sparse((data, (row, col)), (n, n))

    data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)


def eye(m, n=None, k=0, dtype=float, format=None):
    """Sparse matrix with ones on diagonal

    Returns a sparse matrix (m x n) where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the matrix.
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``eye_array`` to take advantage
        of the sparse array functionality.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> sp.sparse.eye(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sp.sparse.eye(3, dtype=np.int8)
    <DIAgonal sparse matrix of dtype 'int8'
        with 3 stored elements (1 diagonals) and shape (3, 3)>

    """
    return _eye(m, n, k, dtype, format, False)


def kron(A, B, format=None):
    """kronecker product of sparse matrices A and B

    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional (default: 'bsr' or 'coo')
        format of the result (e.g. "csr")
        If None, choose 'bsr' for relatively dense array and 'coo' for others

    Returns
    -------
    kronecker product in a sparse format.
    Returns a sparse matrix unless either A or B is a
    sparse array in which case returns a sparse array.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> A = sp.sparse.csr_array(np.array([[0, 2], [5, 0]]))
    >>> B = sp.sparse.csr_array(np.array([[1, 2], [3, 4]]))
    >>> sp.sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    >>> sp.sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    """
    # TODO: delete next 10 lines and replace _sparse with _array when spmatrix removed
    if isinstance(A, sparray) or isinstance(B, sparray):
        # convert to local variables
        bsr_sparse = bsr_array
        csr_sparse = csr_array
        coo_sparse = coo_array
    else:  # use spmatrix
        bsr_sparse = bsr_matrix
        csr_sparse = csr_matrix
        coo_sparse = coo_matrix

    B = coo_sparse(B)
    if B.ndim != 2:
        raise ValueError(f"kron requires 2D input arrays. `B` is {B.ndim}D.")

    # B is fairly dense, use BSR
    if (format is None or format == "bsr") and 2*B.nnz >= B.shape[0] * B.shape[1]:
        A = csr_sparse(A,copy=True)
        if A.ndim != 2:
            raise ValueError(f"kron requires 2D input arrays. `A` is {A.ndim}D.")
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_sparse(output_shape).asformat(format)

        B = B.toarray()
        data = A.data.repeat(B.size).reshape(-1,B.shape[0],B.shape[1])
        data = data * B

        return bsr_sparse((data,A.indices,A.indptr), shape=output_shape)
    else:
        # use COO
        A = coo_sparse(A)
        if A.ndim != 2:
            raise ValueError(f"kron requires 2D input arrays. `A` is {A.ndim}D.")
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_sparse(output_shape).asformat(format)

        # expand entries of a into blocks
        idx_dtype = get_index_dtype(A.coords, maxval=max(output_shape))
        row = np.asarray(A.row, dtype=idx_dtype).repeat(B.nnz)
        col = np.asarray(A.col, dtype=idx_dtype).repeat(B.nnz)
        data = A.data.repeat(B.nnz)

        row *= B.shape[0]
        col *= B.shape[1]

        # increment block indices
        row,col = row.reshape(-1,B.nnz),col.reshape(-1,B.nnz)
        row += B.row
        col += B.col
        row,col = row.reshape(-1),col.reshape(-1)

        # compute block entries
        data = data.reshape(-1,B.nnz) * B.data
        data = data.reshape(-1)

        return coo_sparse((data,(row,col)), shape=output_shape).asformat(format)


def kronsum(A, B, format=None):
    """kronecker sum of square sparse matrices A and B

    Kronecker sum of two sparse matrices is a sum of two Kronecker
    products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
    and B has shape (n,n) and I_m and I_n are identity matrices
    of shape (m,m) and (n,n), respectively.

    Parameters
    ----------
    A
        square matrix
    B
        square matrix
    format : str
        format of the result (e.g. "csr")

    Returns
    -------
    kronecker sum in a sparse matrix format

    """
    # TODO: delete next 8 lines and replace _sparse with _array when spmatrix removed
    if isinstance(A, sparray) or isinstance(B, sparray):
        # convert to local variables
        coo_sparse = coo_array
        identity_sparse = eye_array
    else:
        coo_sparse = coo_matrix
        identity_sparse = identity

    A = coo_sparse(A)
    B = coo_sparse(B)

    if A.ndim != 2:
        raise ValueError(f"kronsum requires 2D inputs. `A` is {A.ndim}D.")
    if B.ndim != 2:
        raise ValueError(f"kronsum requires 2D inputs. `B` is {B.ndim}D.")
    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square')
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square')

    dtype = upcast(A.dtype, B.dtype)

    I_n = identity_sparse(A.shape[0], dtype=dtype)
    I_m = identity_sparse(B.shape[0], dtype=dtype)
    L = kron(I_m, A, format='coo')
    R = kron(B, I_n, format='coo')

    return (L + R).asformat(format)


def _compressed_sparse_stack(blocks, axis, return_spmatrix):
    """
    Stacking fast path for CSR/CSC matrices or arrays
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = np.concatenate([b.data for b in blocks])
    constant_dim = blocks[0]._shape_as_2d[other_axis]
    idx_dtype = get_index_dtype(arrays=[b.indptr for b in blocks],
                                maxval=max(data.size, constant_dim))
    indices = np.empty(data.size, dtype=idx_dtype)
    indptr = np.empty(sum(b._shape_as_2d[axis] for b in blocks) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b._shape_as_2d[other_axis] != constant_dim:
            raise ValueError(f'incompatible dimensions for axis {other_axis}')
        indices[sum_indices:sum_indices+b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b._shape_as_2d[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b._shape_as_2d[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    # TODO remove this if-structure when sparse matrices removed
    if return_spmatrix:
        if axis == 0:
            return csr_matrix((data, indices, indptr),
                              shape=(sum_dim, constant_dim))
        else:
            return csc_matrix((data, indices, indptr),
                              shape=(constant_dim, sum_dim))

    if axis == 0:
        return csr_array((data, indices, indptr),
                          shape=(sum_dim, constant_dim))
    else:
        return csc_array((data, indices, indptr),
                          shape=(constant_dim, sum_dim))


def _stack_along_minor_axis(blocks, axis):
    """
    Stacking fast path for CSR/CSC matrices along the minor axis
    (i) hstack for CSR, (ii) vstack for CSC.
    """
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise ValueError('Missing block matrices')

    if n_blocks == 1:
        return blocks[0]

    # check for incompatible dimensions
    other_axis = 1 if axis == 0 else 0
    other_axis_dims = {b._shape_as_2d[other_axis] for b in blocks}
    if len(other_axis_dims) > 1:
        raise ValueError(f'Mismatching dimensions along axis {other_axis}: '
                         f'{other_axis_dims}')
    constant_dim, = other_axis_dims

    # Do the stacking
    indptr_list = [b.indptr for b in blocks]
    data_cat = np.concatenate([b.data for b in blocks])

    # Need to check if any indices/indptr, would be too large post-
    # concatenation for np.int32:
    # - The max value of indices is the output array's stacking-axis length - 1
    # - The max value in indptr is the number of non-zero entries. This is
    #   exceedingly unlikely to require int64, but is checked out of an
    #   abundance of caution.
    sum_dim = sum(b._shape_as_2d[axis] for b in blocks)
    nnz = sum(len(b.indices) for b in blocks)
    idx_dtype = get_index_dtype(indptr_list, maxval=max(sum_dim - 1, nnz))
    stack_dim_cat = np.array([b._shape_as_2d[axis] for b in blocks], dtype=idx_dtype)
    if data_cat.size > 0:
        indptr_cat = np.concatenate(indptr_list, dtype=idx_dtype)
        indices_cat = np.concatenate([b.indices for b in blocks], dtype=idx_dtype)
        indptr = np.empty(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty_like(indices_cat)
        data = np.empty_like(data_cat)
        csr_hstack(n_blocks, constant_dim, stack_dim_cat,
                   indptr_cat, indices_cat, data_cat,
                   indptr, indices, data)
    else:
        indptr = np.zeros(constant_dim + 1, dtype=idx_dtype)
        indices = np.empty(0, dtype=idx_dtype)
        data = np.empty(0, dtype=data_cat.dtype)

    if axis == 0:
        return blocks[0]._csc_container((data, indices, indptr),
                          shape=(sum_dim, constant_dim))
    else:
        return blocks[0]._csr_container((data, indices, indptr),
                          shape=(constant_dim, sum_dim))


def hstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices horizontally (column wise)

    Parameters
    ----------
    blocks
        sequence of sparse matrices with compatible shapes
    format : str
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use ``block(hstack(blocks))`` or convert one block
        e.g. ``blocks[0] = csr_array(blocks[0])``.

    See Also
    --------
    vstack : stack sparse matrices vertically (row wise)

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, hstack
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> hstack([A,B]).toarray()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block([blocks], format, dtype)
    else:
        return _block([blocks], format, dtype, return_spmatrix=True)


def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse arrays vertically (row wise)

    Parameters
    ----------
    blocks
        sequence of sparse arrays with compatible shapes
    format : str, optional
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    new_array : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use ``block(vstack(blocks))`` or convert one block
        e.g. `blocks[0] = csr_array(blocks[0])`.

    See Also
    --------
    hstack : stack sparse matrices horizontally (column wise)

    Examples
    --------
    >>> from scipy.sparse import coo_array, vstack
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5, 6]])
    >>> vstack([A, B]).toarray()
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block([[b] for b in blocks], format, dtype)
    else:
        return _block([[b] for b in blocks], format, dtype, return_spmatrix=True)


def bmat(blocks, format=None, dtype=None):
    """
    Build a sparse array or matrix from sparse sub-blocks

    Note: `block_array` is preferred over `bmat`. They are the same function
    except that `bmat` can return a deprecated sparse matrix.
    `bmat` returns a coo_matrix if none of the inputs are a sparse array.

    .. warning::

        This function returns a sparse matrix -- not a sparse array.
        You are encouraged to use ``block_array`` to take advantage
        of the sparse array functionality.

    Parameters
    ----------
    blocks : array_like
        Grid of sparse matrices with compatible shapes.
        An entry of None implies an all-zero matrix.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    bmat : sparse matrix or array
        If any block in blocks is a sparse array, return a sparse array.
        Otherwise return a sparse matrix.

        If you want a sparse array built from blocks that are not sparse
        arrays, use ``block_array()``.

    See Also
    --------
    block_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, bmat
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> bmat([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> bmat([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    blocks = np.asarray(blocks, dtype='object')
    if any(isinstance(b, sparray) for b in blocks.flat):
        return _block(blocks, format, dtype)
    else:
        return _block(blocks, format, dtype, return_spmatrix=True)


def block_array(blocks, *, format=None, dtype=None):
    """
    Build a sparse array from sparse sub-blocks

    Parameters
    ----------
    blocks : array_like
        Grid of sparse arrays with compatible shapes.
        An entry of None implies an all-zero array.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse array format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output array. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    block : sparse array

    See Also
    --------
    block_diag : specify blocks along the main diagonals
    diags : specify (possibly offset) diagonals

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_array
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_array([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> block_array([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """
    return _block(blocks, format, dtype)


def _block(blocks, format, dtype, return_spmatrix=False):
    blocks = np.asarray(blocks, dtype='object')

    if blocks.ndim != 2:
        raise ValueError('blocks must be 2-D')

    M,N = blocks.shape

    # check for fast path cases
    if (format in (None, 'csr') and
        all(issparse(b) and b.format == 'csr' for b in blocks.flat)
    ):
        if N > 1:
            # stack along columns (axis 1): must have shape (M, 1)
            blocks = [[_stack_along_minor_axis(blocks[b, :], 1)] for b in range(M)]
            blocks = np.asarray(blocks, dtype='object')

        # stack along rows (axis 0):
        A = _compressed_sparse_stack(blocks[:, 0], 0, return_spmatrix)
        if dtype is not None:
            A = A.astype(dtype, copy=False)
        return A
    elif (format in (None, 'csc') and
          all(issparse(b) and b.format == 'csc' for b in blocks.flat)
    ):
        if M > 1:
            # stack along rows (axis 0): must have shape (1, N)
            blocks = [[_stack_along_minor_axis(blocks[:, b], 0) for b in range(N)]]
            blocks = np.asarray(blocks, dtype='object')

        # stack along columns (axis 1):
        A = _compressed_sparse_stack(blocks[0, :], 1, return_spmatrix)
        if dtype is not None:
            A = A.astype(dtype, copy=False)
        return A

    block_mask = np.zeros(blocks.shape, dtype=bool)
    brow_lengths = np.zeros(M, dtype=np.int64)
    bcol_lengths = np.zeros(N, dtype=np.int64)

    # convert everything to COO format
    for i in range(M):
        for j in range(N):
            if blocks[i,j] is not None:
                A = coo_array(blocks[i,j])
                blocks[i,j] = A
                block_mask[i,j] = True

                if brow_lengths[i] == 0:
                    brow_lengths[i] = A._shape_as_2d[0]
                elif brow_lengths[i] != A._shape_as_2d[0]:
                    msg = (f'blocks[{i},:] has incompatible row dimensions. '
                           f'Got blocks[{i},{j}].shape[0] == {A._shape_as_2d[0]}, '
                           f'expected {brow_lengths[i]}.')
                    raise ValueError(msg)

                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A._shape_as_2d[1]
                elif bcol_lengths[j] != A._shape_as_2d[1]:
                    msg = (f'blocks[:,{j}] has incompatible column '
                           f'dimensions. '
                           f'Got blocks[{i},{j}].shape[1] == {A._shape_as_2d[1]}, '
                           f'expected {bcol_lengths[j]}.')
                    raise ValueError(msg)

    nnz = sum(block.nnz for block in blocks[block_mask])
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks[block_mask]]
        dtype = upcast(*all_dtypes) if all_dtypes else None

    row_offsets = np.append(0, np.cumsum(brow_lengths))
    col_offsets = np.append(0, np.cumsum(bcol_lengths))

    shape = (row_offsets[-1], col_offsets[-1])

    data = np.empty(nnz, dtype=dtype)
    idx_dtype = get_index_dtype([b.coords[0] for b in blocks[block_mask]],
                                maxval=max(shape))
    row = np.empty(nnz, dtype=idx_dtype)
    col = np.empty(nnz, dtype=idx_dtype)

    nnz = 0
    ii, jj = np.nonzero(block_mask)
    for i, j in zip(ii, jj):
        B = blocks[i, j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        np.add(B.row, row_offsets[i], out=row[idx], dtype=idx_dtype)
        np.add(B.col, col_offsets[j], out=col[idx], dtype=idx_dtype)
        nnz += B.nnz

    if return_spmatrix:
        return coo_matrix((data, (row, col)), shape=shape).asformat(format)
    return coo_array((data, (row, col)), shape=shape).asformat(format)


def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix or array from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices or arrays
        Input matrices or arrays.
    format : str, optional
        The sparse format of the result (e.g., "csr"). If not given, the result
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix or array
        If at least one input is a sparse array, the output is a sparse array.
        Otherwise the output is a sparse matrix.

    Notes
    -----

    .. versionadded:: 0.11.0

    See Also
    --------
    block_array
    diags_array

    Examples
    --------
    >>> from scipy.sparse import coo_array, block_diag
    >>> A = coo_array([[1, 2], [3, 4]])
    >>> B = coo_array([[5], [6]])
    >>> C = coo_array([[7]])
    >>> block_diag((A, B, C)).toarray()
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 7]])

    """
    if any(isinstance(a, sparray) for a in mats):
        container = coo_array
    else:
        container = coo_matrix

    row = []
    col = []
    data = []
    idx_arrays = []  # track idx_dtype of incoming sparse arrays
    r_idx = 0
    c_idx = 0
    for a in mats:
        if isinstance(a, (list | numbers.Number)):
            a = coo_array(np.atleast_2d(a))
        if issparse(a):
            a = a.tocoo()
            if not idx_arrays and a.coords[0].dtype == np.int64:
                idx_arrays.append(a.coords[0])
            nrows, ncols = a._shape_as_2d
            row.append(a.row + r_idx)
            col.append(a.col + c_idx)
            data.append(a.data)
