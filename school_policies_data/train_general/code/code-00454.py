def _dm_QQ_numers_denoms(Mq):
    """Returns the numerators and denominators of a DomainMatrix over QQ."""
    elements = _dm_elements(Mq)
    numers = [e.numerator for e in elements]
    denoms = [e.denominator for e in elements]
    return numers, denoms


def _to_field(M):
    """Convert a DomainMatrix to a field if possible."""
    K = M.domain
    if K.has_assoc_Field:
        return M.to_field()
    else:
        return M
<|endoftext|>from __future__ import annotations

from math import floor as mfloor

from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError


def _ddm_lll(x, delta=QQ(3, 4), return_transform=False):
    if QQ(1, 4) >= delta or delta >= QQ(1, 1):
        raise DMValueError("delta must lie in range (0.25, 1)")
    if x.shape[0] > x.shape[1]:
        raise DMShapeError("input matrix must have shape (m, n) with m <= n")
    if x.domain != ZZ:
        raise DMDomainError("input matrix domain must be ZZ")
    m = x.shape[0]
    n = x.shape[1]
    k = 1
    y = x.copy()
    y_star = x.zeros((m, n), QQ)
    mu = x.zeros((m, m), QQ)
    g_star = [QQ(0, 1) for _ in range(m)]
    half = QQ(1, 2)
    T = x.eye(m, ZZ) if return_transform else None
    linear_dependent_error = "input matrix contains linearly dependent rows"

    def closest_integer(x):
        return ZZ(mfloor(x + half))

    def lovasz_condition(k: int) -> bool:
        return g_star[k] >= ((delta - mu[k][k - 1] ** 2) * g_star[k - 1])

    def mu_small(k: int, j: int) -> bool:
        return abs(mu[k][j]) <= half

    def dot_rows(x, y, rows: tuple[int, int]):
        return sum(x[rows[0]][z] * y[rows[1]][z] for z in range(x.shape[1]))

    def reduce_row(T, mu, y, rows: tuple[int, int]):
        r = closest_integer(mu[rows[0]][rows[1]])
        y[rows[0]] = [y[rows[0]][z] - r * y[rows[1]][z] for z in range(n)]
        mu[rows[0]][:rows[1]] = [mu[rows[0]][z] - r * mu[rows[1]][z] for z in range(rows[1])]
        mu[rows[0]][rows[1]] -= r
        if return_transform:
            T[rows[0]] = [T[rows[0]][z] - r * T[rows[1]][z] for z in range(m)]

    for i in range(m):
        y_star[i] = [QQ.convert_from(z, ZZ) for z in y[i]]
        for j in range(i):
            row_dot = dot_rows(y, y_star, (i, j))
            try:
                mu[i][j] = row_dot / g_star[j]
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)
            y_star[i] = [y_star[i][z] - mu[i][j] * y_star[j][z] for z in range(n)]
        g_star[i] = dot_rows(y_star, y_star, (i, i))
    while k < m:
        if not mu_small(k, k - 1):
            reduce_row(T, mu, y, (k, k - 1))
        if lovasz_condition(k):
            for l in range(k - 2, -1, -1):
                if not mu_small(k, l):
                    reduce_row(T, mu, y, (k, l))
            k += 1
        else:
            nu = mu[k][k - 1]
            alpha = g_star[k] + nu ** 2 * g_star[k - 1]
            try:
                beta = g_star[k - 1] / alpha
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)
            mu[k][k - 1] = nu * beta
            g_star[k] = g_star[k] * beta
            g_star[k - 1] = alpha
            y[k], y[k - 1] = y[k - 1], y[k]
            mu[k][:k - 1], mu[k - 1][:k - 1] = mu[k - 1][:k - 1], mu[k][:k - 1]
            for i in range(k + 1, m):
                xi = mu[i][k]
                mu[i][k] = mu[i][k - 1] - nu * xi
                mu[i][k - 1] = mu[k][k - 1] * mu[i][k] + xi
            if return_transform:
                T[k], T[k - 1] = T[k - 1], T[k]
            k = max(k - 1, 1)
    assert all(lovasz_condition(i) for i in range(1, m))
    assert all(mu_small(i, j) for i in range(m) for j in range(i))
    return y, T


def ddm_lll(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=False)[0]


def ddm_lll_transform(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=True)
<|endoftext|>"""

Routines for computing eigenvectors with DomainMatrix.

"""
from sympy.core.symbol import Dummy

from ..agca.extensions import FiniteExtension
from ..factortools import dup_factor_list
from ..polyroots import roots
from ..polytools import Poly
from ..rootoftools import CRootOf

from .domainmatrix import DomainMatrix


def dom_eigenvects(A, l=Dummy('lambda')):
    charpoly = A.charpoly()
    rows, cols = A.shape
    domain = A.domain
    _, factors = dup_factor_list(charpoly, domain)

    rational_eigenvects = []
    algebraic_eigenvects = []
    for base, exp in factors:
        if len(base) == 2:
            field = domain
            eigenval = -base[1] / base[0]

            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            basis = (A - EE).nullspace(divide_last=True)
            rational_eigenvects.append((field, eigenval, exp, basis))
        else:
            minpoly = Poly.from_list(base, l, domain=domain)
            field = FiniteExtension(minpoly)
            eigenval = field(l)

            AA_items = [
                [Poly.from_list([item], l, domain=domain).rep for item in row]
                for row in A.rep.to_ddm()]
            AA_items = [[field(item) for item in row] for row in AA_items]
            AA = DomainMatrix(AA_items, (rows, cols), field)
            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            basis = (AA - EE).nullspace(divide_last=True)
            algebraic_eigenvects.append((field, minpoly, exp, basis))

    return rational_eigenvects, algebraic_eigenvects


def dom_eigenvects_to_sympy(
    rational_eigenvects, algebraic_eigenvects,
    Matrix, **kwargs
):
    result = []

    for field, eigenvalue, multiplicity, eigenvects in rational_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        eigenvalue = field.to_sympy(eigenvalue)
        new_eigenvects = [
            Matrix([field.to_sympy(x) for x in vect])
            for vect in eigenvects]
        result.append((eigenvalue, multiplicity, new_eigenvects))

    for field, minpoly, multiplicity, eigenvects in algebraic_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        l = minpoly.gens[0]

        eigenvects = [[field.to_sympy(x) for x in vect] for vect in eigenvects]

        degree = minpoly.degree()
        minpoly = minpoly.as_expr()
        eigenvals = roots(minpoly, l, **kwargs)
        if len(eigenvals) != degree:
            eigenvals = [CRootOf(minpoly, l, idx) for idx in range(degree)]

        for eigenvalue in eigenvals:
            new_eigenvects = [
                Matrix([x.subs(l, eigenvalue) for x in vect])
                for vect in eigenvects]
            result.append((eigenvalue, multiplicity, new_eigenvects))

    return result
<|endoftext|>"""

Module for the SDM class.

"""

from operator import add, neg, pos, sub, mul
from collections import defaultdict

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import _strongly_connected_components

from .exceptions import DMBadInputError, DMDomainError, DMShapeError

from sympy.polys.domains import QQ

from .ddm import DDM


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['SDM.to_dfm', 'SDM.to_dfm_or_ddm']


class SDM(dict):
    r"""Sparse matrix based on polys domain elements

    This is a dict subclass and is a wrapper for a dict of dicts that supports
    basic matrix arithmetic +, -, *, **.


    In order to create a new :py:class:`~.SDM`, a dict
    of dicts mapping non-zero elements to their
    corresponding row and column in the matrix is needed.

    We also need to specify the shape and :py:class:`~.Domain`
    of our :py:class:`~.SDM` object.

    We declare a 2x2 :py:class:`~.SDM` matrix belonging
    to QQ domain as shown below.
    The 2x2 Matrix in the example is

    .. math::
           A = \left[\begin{array}{ccc}
                0 & \frac{1}{2} \\
                0 & 0 \end{array} \right]


    >>> from sympy.polys.matrices.sdm import SDM
    >>> from sympy import QQ
    >>> elemsdict = {0:{1:QQ(1, 2)}}
    >>> A = SDM(elemsdict, (2, 2), QQ)
    >>> A
    {0: {1: 1/2}}

    We can manipulate :py:class:`~.SDM` the same way
    as a Matrix class

    >>> from sympy import ZZ
    >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
    >>> A + B
    {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

    Multiplication

    >>> A*B
    {0: {1: 8}, 1: {0: 3}}
    >>> A*ZZ(2)
    {0: {1: 4}, 1: {0: 2}}

    """

    fmt = 'sparse'
    is_DFM = False
    is_DDM = False

    def __init__(self, elemsdict, shape, domain):
        super().__init__(elemsdict)
        self.shape = self.rows, self.cols = m, n = shape
        self.domain = domain

        if not all(0 <= r < m for r in self):
            raise DMBadInputError("Row out of range")
        if not all(0 <= c < n for row in self.values() for c in row):
            raise DMBadInputError("Column out of range")

    def getitem(self, i, j):
        try:
            return self[i][j]
        except KeyError:
            m, n = self.shape
            if -m <= i < m and -n <= j < n:
                try:
                    return self[i % m][j % n]
                except KeyError:
                    return self.domain.zero
            else:
                raise IndexError("index out of range")

    def setitem(self, i, j, value):
        m, n = self.shape
        if not (-m <= i < m and -n <= j < n):
            raise IndexError("index out of range")
        i, j = i % m, j % n
        if value:
            try:
                self[i][j] = value
            except KeyError:
                self[i] = {j: value}
        else:
            rowi = self.get(i, None)
            if rowi is not None:
                try:
                    del rowi[j]
                except KeyError:
                    pass
                else:
                    if not rowi:
                        del self[i]

    def extract_slice(self, slice1, slice2):
        m, n = self.shape
        ri = range(m)[slice1]
        ci = range(n)[slice2]

        sdm = {}
        for i, row in self.items():
            if i in ri:
                row = {ci.index(j): e for j, e in row.items() if j in ci}
                if row:
                    sdm[ri.index(i)] = row

        return self.new(sdm, (len(ri), len(ci)), self.domain)

    def extract(self, rows, cols):
        if not (self and rows and cols):
            return self.zeros((len(rows), len(cols)), self.domain)

        m, n = self.shape
        if not (-m <= min(rows) <= max(rows) < m):
            raise IndexError('Row index out of range')
        if not (-n <= min(cols) <= max(cols) < n):
            raise IndexError('Column index out of range')

        # rows and cols can contain duplicates e.g. M[[1, 2, 2], [0, 1]]
        # Build a map from row/col in self to list of rows/cols in output
        rowmap = defaultdict(list)
        colmap = defaultdict(list)
        for i2, i1 in enumerate(rows):
            rowmap[i1 % m].append(i2)
        for j2, j1 in enumerate(cols):
            colmap[j1 % n].append(j2)

        # Used to efficiently skip zero rows/cols
        rowset = set(rowmap)
        colset = set(colmap)

        sdm1 = self
        sdm2 = {}
        for i1 in rowset & sdm1.keys():
            row1 = sdm1[i1]
            row2 = {}
            for j1 in colset & row1.keys():
                row1_j1 = row1[j1]
                for j2 in colmap[j1]:
                    row2[j2] = row1_j1
            if row2:
                for i2 in rowmap[i1]:
                    sdm2[i2] = row2.copy()

        return self.new(sdm2, (len(rows), len(cols)), self.domain)

    def __str__(self):
        rowsstr = []
        for i, row in self.items():
            elemsstr = ', '.join('%s: %s' % (j, elem) for j, elem in row.items())
            rowsstr.append('%s: {%s}' % (i, elemsstr))
        return '{%s}' % ', '.join(rowsstr)

    def __repr__(self):
        cls = type(self).__name__
        rows = dict.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    @classmethod
    def new(cls, sdm, shape, domain):
        """

        Parameters
        ==========

        sdm: A dict of dicts for non-zero elements in SDM
        shape: tuple representing dimension of SDM
        domain: Represents :py:class:`~.Domain` of SDM

        Returns
        =======

        An :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1: QQ(2)}}
        >>> A = SDM.new(elemsdict, (2, 2), QQ)
        >>> A
        {0: {1: 2}}

        """
        return cls(sdm, shape, domain)

    def copy(A):
        """
        Returns the copy of a :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> B = A.copy()
        >>> B
        {0: {1: 2}, 1: {}}

        """
        Ac = {i: Ai.copy() for i, Ai in A.items()}
        return A.new(Ac, A.shape, A.domain)

    @classmethod
    def from_list(cls, ddm, shape, domain):
        """
        Create :py:class:`~.SDM` object from a list of lists.

        Parameters
        ==========

        ddm:
            list of lists containing domain elements
        shape:
            Dimensions of :py:class:`~.SDM` matrix
        domain:
            Represents :py:class:`~.Domain` of :py:class:`~.SDM` object

        Returns
        =======

        :py:class:`~.SDM` containing elements of ddm

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = [[QQ(1, 2), QQ(0)], [QQ(0), QQ(3, 4)]]
        >>> A = SDM.from_list(ddm, (2, 2), QQ)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}

        See Also
        ========

        to_list
        from_list_flat
        from_dok
        from_ddm
        """

        m, n = shape
        if not (len(ddm) == m and all(len(row) == n for row in ddm)):
            raise DMBadInputError("Inconsistent row-list/shape")
        getrow = lambda i: {j:ddm[i][j] for j in range(n) if ddm[i][j]}
        irows = ((i, getrow(i)) for i in range(m))
        sdm = {i: row for i, row in irows if row}
        return cls(sdm, shape, domain)

    @classmethod
    def from_ddm(cls, ddm):
        """
        Create :py:class:`~.SDM` from a :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)
        >>> A = SDM.from_ddm(ddm)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}
        >>> SDM.from_ddm(ddm).to_ddm() == ddm
        True

        See Also
        ========

        to_ddm
        from_list
        from_list_flat
        from_dok
        """
        return cls.from_list(ddm, ddm.shape, ddm.domain)

    def to_list(M):
        """
        Convert a :py:class:`~.SDM` object to a list of lists.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> A.to_list()
        [[0, 2], [0, 0]]


        """
        m, n = M.shape
        zero = M.domain.zero
        ddm = [[zero] * n for _ in range(m)]
        for i, row in M.items():
            for j, e in row.items():
                ddm[i][j] = e
        return ddm

    def to_list_flat(M):
        """
        Convert :py:class:`~.SDM` to a flat list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_list_flat()
        [0, 2, 3, 0]
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        from_list_flat
        to_list
        to_dok
        to_ddm
        """
        m, n = M.shape
        zero = M.domain.zero
        flat = [zero] * (m * n)
        for i, row in M.items():
            for j, e in row.items():
                flat[i*n + j] = e
        return flat

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :py:class:`~.SDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.from_list_flat([QQ(0), QQ(2), QQ(0), QQ(0)], (2, 2), QQ)
        >>> A
        {0: {1: 2}}
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        from_list
        from_dok
        from_ddm
        """
        m, n = shape
        if len(elements) != m * n:
            raise DMBadInputError("Inconsistent flat-list shape")
        sdm = defaultdict(dict)
        for inj, element in enumerate(elements):
            if element:
                i, j = divmod(inj, n)
                sdm[i][j] = element
        return cls(sdm, shape, domain)

    def to_flat_nz(M):
        """
        Convert :class:`SDM` to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a modified matrix with elements in the same positions using
        :meth:`from_flat_nz`. Zero elements are omitted from the list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [2, 3]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        to_list_flat
        sympy.polys.matrices.ddm.DDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        dok = M.to_dok()
        indices = tuple(dok)
        elements = list(dok.values())
        data = (indices, M.shape)
        return elements, data

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`~.SDM` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        from_list_flat
        sympy.polys.matrices.ddm.DDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        indices, shape = data
        dok = dict(zip(indices, elements))
        return cls.from_dok(dok, shape, domain)

    def to_dod(M):
        """
        Convert to dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dod()
        {0: {1: 2}, 1: {0: 3}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        return {i: row.copy() for i, row in M.items()}

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dod = {0: {1: QQ(2)}, 1: {0: QQ(3)}}
        >>> A = SDM.from_dod(dod, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dod(A.to_dod(), A.shape, A.domain)
        True

        See Also
        ========

        to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        sdm = defaultdict(dict)
        for i, row in dod.items():
            for j, e in row.items():
                if e:
                    sdm[i][j] = e
        return cls(sdm, shape, domain)

    def to_dok(M):
        """
        Convert to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dok()
        {(0, 1): 2, (1, 0): 3}

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_ddm
        """
        return {(i, j): e for i, row in M.items() for j, e in row.items()}

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dok = {(0, 1): QQ(2), (1, 0): QQ(3)}
        >>> A = SDM.from_dok(dok, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        to_dok
        from_list
        from_list_flat
        from_ddm
        """
        sdm = defaultdict(dict)
        for (i, j), e in dok.items():
            if e:
                sdm[i][j] = e
        return cls(sdm, shape, domain)

    def iter_values(M):
        """
        Iterate over the nonzero values of a :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_values())
        [2, 3]

        """
        for row in M.values():
            yield from row.values()

    def iter_items(M):
        """
        Iterate over indices and values of the nonzero elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 1), 2), ((1, 0), 3)]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        for i, row in M.items():
            for j, e in row.items():
                yield (i, j), e

    def to_ddm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_ddm()
        [[0, 2], [0, 0]]

        """
        return DDM(M.to_list(), M.shape, M.domain)

    def to_sdm(M):
        """
        Convert to :py:class:`~.SDM` format (returns self).
        """
        return M

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DFM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm()
        [[0, 2], [0, 0]]

        See Also
        ========

        to_ddm
        to_dfm_or_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm
        """
        return M.to_ddm().to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(M):
        """
        Convert to :py:class:`~.DFM` if possible, else :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[0, 2], [0, 0]]
        >>> type(A.to_dfm_or_ddm())  # depends on the ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dfm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return M.to_ddm().to_dfm_or_ddm()

    @classmethod
    def zeros(cls, shape, domain):
        r"""

        Returns a :py:class:`~.SDM` of size shape,
        belonging to the specified domain

        In the example below we declare a matrix A where,

        .. math::
            A := \left[\begin{array}{ccc}
            0 & 0 & 0 \\
            0 & 0 & 0 \end{array} \right]

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.zeros((2, 3), QQ)
        >>> A
        {}

        """
        return cls({}, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        row = dict(zip(range(n), [one]*n))
        sdm = {i: row.copy() for i in range(m)}
        return cls(sdm, shape, domain)

    @classmethod
    def eye(cls, shape, domain):
        """

        Returns a identity :py:class:`~.SDM` matrix of dimensions
        size x size, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> I = SDM.eye((2, 2), QQ)
        >>> I
        {0: {0: 1}, 1: {1: 1}}

        """
        if isinstance(shape, int):
            rows, cols = shape, shape
        else:
            rows, cols = shape
        one = domain.one
        sdm = {i: {i: one} for i in range(min(rows, cols))}
        return cls(sdm, (rows, cols), domain)

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        if shape is None:
            shape = (len(diagonal), len(diagonal))
        sdm = {i: {i: v} for i, v in enumerate(diagonal) if v}
        return cls(sdm, shape, domain)

    def transpose(M):
        """

        Returns the transpose of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.transpose()
        {1: {0: 2}}

        """
        MT = sdm_transpose(M)
        return M.new(MT, M.shape[::-1], M.domain)

    def __add__(A, B):
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s + %s" % (A.shape, B.shape))
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s - %s" % (A.shape, B.shape))
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, SDM):
            return A.matmul(B)
        elif B in A.domain:
            return A.mul(B)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if b in a.domain:
            return a.rmul(b)
        else:
            return NotImplemented

    def matmul(A, B):
        """
        Performs matrix multiplication of two SDM matrices

        Parameters
        ==========

        A, B: SDM to multiply

        Returns
        =======

        SDM
            SDM after multiplication

        Raises
        ======

        DomainError
            If domain of A does not match
            with that of B

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0:ZZ(2), 1:ZZ(3)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)
        >>> A.matmul(B)
        {0: {0: 8}, 1: {0: 2, 1: 3}}

        """
        if A.domain != B.domain:
            raise DMDomainError
        m, n = A.shape
        n2, o = B.shape
        if n != n2:
            raise DMShapeError
        C = sdm_matmul(A, B, A.domain, m, o)
        return A.new(C, (m, o), A.domain)

    def mul(A, b):
        """
        Multiplies each element of A with a scalar b

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.mul(ZZ(3))
        {0: {1: 6}, 1: {0: 3}}

        """
        Csdm = unop_dict(A, lambda aij: aij*b)
        return A.new(Csdm, A.shape, A.domain)

    def rmul(A, b):
        Csdm = unop_dict(A, lambda aij: b*aij)
        return A.new(Csdm, A.shape, A.domain)

    def mul_elementwise(A, B):
        if A.domain != B.domain:
            raise DMDomainError
        if A.shape != B.shape:
            raise DMShapeError
        zero = A.domain.zero
        fzero = lambda e: zero
        Csdm = binop_dict(A, B, mul, fzero, fzero)
        return A.new(Csdm, A.shape, A.domain)

    def add(A, B):
        """

        Adds two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.add(B)
        {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

        """
        Csdm = binop_dict(A, B, add, pos, pos)
        return A.new(Csdm, A.shape, A.domain)

    def sub(A, B):
        """

        Subtracts two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.sub(B)
        {0: {0: -3, 1: 2}, 1: {0: 1, 1: -4}}

        """
        Csdm = binop_dict(A, B, sub, pos, neg)
        return A.new(Csdm, A.shape, A.domain)

    def neg(A):
        """

        Returns the negative of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.neg()
        {0: {1: -2}, 1: {0: -1}}

        """
        Csdm = unop_dict(A, neg)
        return A.new(Csdm, A.shape, A.domain)

    def convert_to(A, K):
        """
        Converts the :py:class:`~.Domain` of a :py:class:`~.SDM` matrix to K

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.convert_to(QQ)
        {0: {1: 2}, 1: {0: 1}}

        """
        Kold = A.domain
        if K == Kold:
            return A.copy()
        Ak = unop_dict(A, lambda e: K.convert_from(e, Kold))
        return A.new(Ak, A.shape, K)

    def nnz(A):
        """Number of non-zero elements in the :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.nnz()
        2

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(map(len, A.values()))

    def scc(A):
        """Strongly connected components of a square matrix *A*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0: ZZ(2)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc
        """
        rows, cols = A.shape
        assert rows == cols
        V = range(rows)
        Emap = {v: list(A.get(v, [])) for v in V}
        return _strongly_connected_components(V, Emap)

    def rref(A):
        """

        Returns reduced-row echelon form and list of pivots for the :py:class:`~.SDM`

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref()
        ({0: {0: 1, 1: 2}}, [0])

        """
        B, pivots, _ = sdm_irref(A)
        return A.new(B, A.shape, A.domain), pivots

    def rref_den(A):
        """

        Returns reduced-row echelon form (RREF) with denominator and pivots.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref_den()
        ({0: {0: 1, 1: 2}}, 1, [0])

        """
        K = A.domain
        A_rref_sdm, denom, pivots = sdm_rref_den(A, K)
        A_rref = A.new(A_rref_sdm, A.shape, A.domain)
        return A_rref, denom, pivots

    def inv(A):
        """

        Returns inverse of a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.inv()
        {0: {0: -2, 1: 1}, 1: {0: 3/2, 1: -1/2}}

        """
        return A.to_dfm_or_ddm().inv().to_sdm()

    def det(A):
        """
        Returns determinant of A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.det()
        -2

        """
        # It would be better to have a sparse implementation of det for use
        # with very sparse matrices. Extremely sparse matrices probably just
        # have determinant zero and we could probably detect that very quickly.
        # In the meantime, we convert to a dense matrix and use ddm_idet.
        #
        # If GROUND_TYPES=flint though then we will use Flint's implementation
        # if possible (dfm).
        return A.to_dfm_or_ddm().det()

    def lu(A):
        """

        Returns LU decomposition for a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.lu()
        ({0: {0: 1}, 1: {0: 3, 1: 1}}, {0: {0: 1, 1: 2}, 1: {1: -2}}, [])

        """
        L, U, swaps = A.to_ddm().lu()
        return A.from_ddm(L), A.from_ddm(U), swaps

    def qr(self):
        """
        QR decomposition for SDM (Sparse Domain Matrix).

        Returns:
            - Q: Orthogonal matrix as a SDM.
            - R: Upper triangular matrix as a SDM.
        """
        ddm_q, ddm_r = self.to_ddm().qr()
        Q = ddm_q.to_sdm()
        R = ddm_r.to_sdm()
        return Q, R

    def lu_solve(A, b):
        """

        Uses LU decomposition to solve Ax = b,

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
        >>> A.lu_solve(b)
        {1: {0: 1/2}}

        """
        return A.from_ddm(A.to_ddm().lu_solve(b.to_ddm()))

    def fflu(self):
        """
        Fraction free LU decomposition of SDM.

        Uses DDM implementation.

        See Also
        ========

        sympy.polys.matrices.ddm.DDM.fflu
        """
        ddm_p, ddm_l, ddm_d, ddm_u = self.to_dfm_or_ddm().fflu()
        P = ddm_p.to_sdm()
        L = ddm_l.to_sdm()
        D = ddm_d.to_sdm()
        U = ddm_u.to_sdm()
        return P, L, D, U

    def nullspace(A):
        """
        Nullspace of a :py:class:`~.SDM` matrix A.

        The domain of the matrix must be a field.

        It is better to use the :meth:`~.DomainMatrix.nullspace` method rather
        than this method which is otherwise no longer used.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A.nullspace()
        ({0: {0: -2, 1: 1}}, [1])


        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The preferred way to get the nullspace of a matrix.

        """
        ncols = A.shape[1]
        one = A.domain.one
        B, pivots, nzcols = sdm_irref(A)
        K, nonpivots = sdm_nullspace_from_rref(B, one, ncols, pivots, nzcols)
        K = dict(enumerate(K))
        shape = (len(K), ncols)
        return A.new(K, shape, A.domain), nonpivots

    def nullspace_from_rref(A, pivots=None):
        """
        Returns nullspace for a :py:class:`~.SDM` matrix ``A`` in RREF.

        The domain of the matrix can be any domain.

        The matrix must already be in reduced row echelon form (RREF).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A_rref, pivots = A.rref()
        >>> A_null, nonpivots = A_rref.nullspace_from_rref(pivots)
        >>> A_null
        {0: {0: -2, 1: 1}}
        >>> pivots
        [0]
        >>> nonpivots
        [1]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher-level function that would usually be called instead of
            calling this one directly.

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace_from_rref
            The higher-level direct equivalent of this function.

        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
            The equivalent function for dense :py:class:`~.DDM` matrices.

        """
        m, n = A.shape
        K = A.domain

        if pivots is None:
            pivots = sorted(map(min, A.values()))

        if not pivots:
            return A.eye((n, n), K), list(range(n))
        elif len(pivots) == n:
            return A.zeros((0, n), K), []

        # In fraction-free RREF the nonzero entry inserted for the pivots is
        # not necessarily 1.
        pivot_val = A[0][pivots[0]]
        assert not K.is_zero(pivot_val)

        pivots_set = set(pivots)

        # Loop once over all nonzero entries making a map from column indices
        # to the nonzero entries in that column along with the row index of the
        # nonzero entry. This is basically the transpose of the matrix.
        nonzero_cols = defaultdict(list)
        for i, Ai in A.items():
            for j, Aij in Ai.items():
                nonzero_cols[j].append((i, Aij))

        # Usually in SDM we want to avoid looping over the dimensions of the
        # matrix because it is optimised to support extremely sparse matrices.
        # Here in nullspace though every zero column becomes a nonzero column
        # so we need to loop once over the columns at least (range(n)) rather
        # than just the nonzero entries of the matrix. We can still avoid
        # an inner loop over the rows though by using the nonzero_cols map.
        basis = []
        nonpivots = []
        for j in range(n):
            if j in pivots_set:
                continue
            nonpivots.append(j)

            vec = {j: pivot_val}
            for ip, Aij in nonzero_cols[j]:
                vec[pivots[ip]] = -Aij

            basis.append(vec)

        sdm = dict(enumerate(basis))
        A_null = A.new(sdm, (len(basis), n), K)

        return (A_null, nonpivots)

    def particular(A):
        ncols = A.shape[1]
        B, pivots, nzcols = sdm_irref(A)
        P = sdm_particular_from_rref(B, ncols, pivots)
        rep = {0:P} if P else {}
        return A.new(rep, (1, ncols-1), A.domain)

    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.hstack(B)
        {0: {0: 1, 1: 2, 2: 5, 3: 6}, 1: {0: 3, 1: 4, 2: 7, 3: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.hstack(B, C)
        {0: {0: 1, 1: 2, 2: 5, 3: 6, 4: 9, 5: 10}, 1: {0: 3, 1: 4, 2: 7, 3: 8, 4: 11, 5: 12}}
        """
        Anew = dict(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain

            for i, Bki in Bk.items():
                Ai = Anew.get(i, None)
                if Ai is None:
                    Anew[i] = Ai = {}
                for j, Bkij in Bki.items():
                    Ai[j + cols] = Bkij
            cols += Bkcols

        return A.new(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.vstack(B)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.vstack(B, C)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}, 4: {0: 9, 1: 10}, 5: {0: 11, 1: 12}}
        """
        Anew = dict(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain

            for i, Bki in Bk.items():
                Anew[i + rows] = Bki
            rows += Bkrows

        return A.new(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        sdm = {i: {j: func(e) for j, e in row.items()} for i, row in self.items()}
        return self.new(sdm, self.shape, domain)

    def charpoly(A):
        """
        Returns the coefficients of the characteristic polynomial
        of the :py:class:`~.SDM` matrix. These elements will be domain elements.
        The domain of the elements will be same as domain of the :py:class:`~.SDM`.

        Examples
        ========

        >>> from sympy import QQ, Symbol
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy.polys import Poly
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.charpoly()
        [1, -5, -2]

        We can create a polynomial using the
        coefficients using :py:class:`~.Poly`

        >>> x = Symbol('x')
        >>> p = Poly(A.charpoly(), x, domain=A.domain)
        >>> p
        Poly(x**2 - 5*x - 2, x, domain='QQ')

        """
        K = A.domain
        n, _ = A.shape
        pdict = sdm_berk(A, n, K)
        plist = [K.zero] * (n + 1)
        for i, pi in pdict.items():
            plist[i] = pi
        return plist

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        return not self

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i <= j for i, row in self.items() for j in row)

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i >= j for i, row in self.items() for j in row)

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned
        even if the matrix is not square.
        """
        return all(i == j for i, row in self.items() for j in row)

    def diagonal(self):
        """
        Returns the diagonal of the matrix as a list.
        """
        m, n = self.shape
        zero = self.domain.zero
        return [row.get(i, zero) for i, row in self.items() if i < n]

    def lll(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis for the :py:class:`~.SDM` matrix.
        """
        return A.to_dfm_or_ddm().lll(delta=delta).to_sdm()

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis and transformation matrix.
        """
        reduced, transform = A.to_dfm_or_ddm().lll_transform(delta=delta)
        return reduced.to_sdm(), transform.to_sdm()


def binop_dict(A, B, fab, fa, fb):
    Anz, Bnz = set(A), set(B)
    C = {}

    for i in Anz & Bnz:
        Ai, Bi = A[i], B[i]
        Ci = {}
        Anzi, Bnzi = set(Ai), set(Bi)
        for j in Anzi & Bnzi:
            Cij = fab(Ai[j], Bi[j])
            if Cij:
                Ci[j] = Cij
        for j in Anzi - Bnzi:
            Cij = fa(Ai[j])
            if Cij:
                Ci[j] = Cij
        for j in Bnzi - Anzi:
            Cij = fb(Bi[j])
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    for i in Anz - Bnz:
        Ai = A[i]
        Ci = {}
        for j, Aij in Ai.items():
            Cij = fa(Aij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    for i in Bnz - Anz:
        Bi = B[i]
        Ci = {}
        for j, Bij in Bi.items():
            Cij = fb(Bij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    return C


def unop_dict(A, f):
    B = {}
    for i, Ai in A.items():
        Bi = {}
        for j, Aij in Ai.items():
            Bij = f(Aij)
            if Bij:
                Bi[j] = Bij
        if Bi:
            B[i] = Bi
    return B


def sdm_transpose(M):
    MT = {}
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            try:
                MT[j][i] = Mij
            except KeyError:
                MT[j] = {i: Mij}
    return MT


def sdm_dotvec(A, B, K):
    return K.sum(A[j] * B[j] for j in A.keys() & B.keys())


def sdm_matvecmul(A, B, K):
    C = {}
    for i, Ai in A.items():
        Ci = sdm_dotvec(Ai, B, K)
        if Ci:
            C[i] = Ci
    return C


def sdm_matmul(A, B, K, m, o):
    #
    # Should be fast if A and B are very sparse.
    # Consider e.g. A = B = eye(1000).
    #
    # The idea here is that we compute C = A*B in terms of the rows of C and
    # B since the dict of dicts representation naturally stores the matrix as
    # rows. The ith row of C (Ci) is equal to the sum of Aik * Bk where Bk is
    # the kth row of B. The algorithm below loops over each nonzero element
    # Aik of A and if the corresponding row Bj is nonzero then we do
    #    Ci += Aik * Bk.
    # To make this more efficient we don't need to loop over all elements Aik.
    # Instead for each row Ai we compute the intersection of the nonzero
    # columns in Ai with the nonzero rows in B. That gives the k such that
    # Aik and Bk are both nonzero. In Python the intersection of two sets
    # of int can be computed very efficiently.
    #
    if K.is_EXRAW:
        return sdm_matmul_exraw(A, B, K, m, o)

    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci = {}
        Ai_knz = set(Ai)
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            for j, Bkj in B[k].items():
                Cij = Ci.get(j, None)
                if Cij is not None:
                    Cij = Cij + Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
                    else:
                        Ci.pop(j)
                else:
                    Cij = Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
        if Ci:
            C[i] = Ci
    return C


def sdm_matmul_exraw(A, B, K, m, o):
    #
    # Like sdm_matmul above except that:
    #
    # - Handles cases like 0*oo -> nan (sdm_matmul skips multiplication by zero)
    # - Uses K.sum (Add(*items)) for efficient addition of Expr
    #
    zero = K.zero
    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci_list = defaultdict(list)
        Ai_knz = set(Ai)

        # Nonzero row/column pair
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            if zero * Aik == zero:
                # This is the main inner loop:
                for j, Bkj in B[k].items():
                    Ci_list[j].append(Aik * Bkj)
            else:
                for j in range(o):
                    Ci_list[j].append(Aik * B[k].get(j, zero))

        # Zero row in B, check for infinities in A
        for k in Ai_knz - B_knz:
            zAik = zero * Ai[k]
            if zAik != zero:
                for j in range(o):
                    Ci_list[j].append(zAik)

        # Add terms using K.sum (Add(*terms)) for efficiency
        Ci = {}
        for j, Cij_list in Ci_list.items():
            Cij = K.sum(Cij_list)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    # Find all infinities in B
    for k, Bk in B.items():
        for j, Bkj in Bk.items():
            if zero * Bkj != zero:
                for i in range(m):
                    Aik = A.get(i, {}).get(k, zero)
                    # If Aik is not zero then this was handled above
                    if Aik == zero:
                        Ci = C.get(i, {})
                        Cij = Ci.get(j, zero) + Aik * Bkj
                        if Cij != zero:
                            Ci[j] = Cij
                            C[i] = Ci
                        else:
                            Ci.pop(j, None)
                            if Ci:
                                C[i] = Ci
                            else:
                                C.pop(i, None)

    return C


def sdm_irref(A):
    """RREF and pivots of a sparse matrix *A*.

    Compute the reduced row echelon form (RREF) of the matrix *A* and return a
    list of the pivot columns. This routine does not work in place and leaves
    the original matrix *A* unmodified.

    The domain of the matrix must be a field.

    Examples
    ========

    This routine works with a dict of dicts sparse representation of a matrix:

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.sdm import sdm_irref
    >>> A = {0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}
    >>> Arref, pivots, _ = sdm_irref(A)
    >>> Arref
    {0: {0: 1}, 1: {1: 1}}
    >>> pivots
    [0, 1]

    The analogous calculation with :py:class:`~.MutableDenseMatrix` would be

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    Matrix([
    [1, 0],
    [0, 1]])
    >>> pivots
    (0, 1)

    Notes
    =====

    The cost of this algorithm is determined purely by the nonzero elements of
    the matrix. No part of the cost of any step in this algorithm depends on
    the number of rows or columns in the matrix. No step depends even on the
    number of nonzero rows apart from the primary loop over those rows. The
    implementation is much faster than ddm_rref for sparse matrices. In fact
    at the time of writing it is also (slightly) faster than the dense
    implementation even if the input is a fully dense matrix so it seems to be
    faster in all cases.

    The elements of the matrix should support exact division with ``/``. For
    example elements of any domain that is a field (e.g. ``QQ``) should be
    fine. No attempt is made to handle inexact arithmetic.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The higher-level function that would normally be used to call this
        routine.
    sympy.polys.matrices.dense.ddm_irref
        The dense equivalent of this routine.
    sdm_rref_den
        Fraction-free version of this routine.
    """
    #
    # Any zeros in the matrix are not stored at all so an element is zero if
    # its row dict has no index at that key. A row is entirely zero if its
    # row index is not in the outer dict. Since rref reorders the rows and
    # removes zero rows we can completely discard the row indices. The first
    # step then copies the row dicts into a list sorted by the index of the
    # first nonzero column in each row.
    #
    # The algorithm then processes each row Ai one at a time. Previously seen
    # rows are used to cancel their pivot columns from Ai. Then a pivot from
    # Ai is chosen and is cancelled from all previously seen rows. At this
    # point Ai joins the previously seen rows. Once all rows are seen all
    # elimination has occurred and the rows are sorted by pivot column index.
    #
    # The previously seen rows are stored in two separate groups. The reduced
    # group consists of all rows that have been reduced to a single nonzero
    # element (the pivot). There is no need to attempt any further reduction
    # with these. Rows that still have other nonzeros need to be considered
    # when Ai is cancelled from the previously seen rows.
    #
    # A dict nonzerocolumns is used to map from a column index to a set of
    # previously seen rows that still have a nonzero element in that column.
    # This means that we can cancel the pivot from Ai into the previously seen
    # rows without needing to loop over each row that might have a zero in
    # that column.
    #

    # Row dicts sorted by index of first nonzero column
    # (Maybe sorting is not needed/useful.)
    Arows = sorted((Ai.copy() for Ai in A.values()), key=min)

    # Each processed row has an associated pivot column.
    # pivot_row_map maps from the pivot column index to the row dict.
    # This means that we can represent a set of rows purely as a set of their
    # pivot indices.
    pivot_row_map = {}

    # Set of pivot indices for rows that are fully reduced to a single nonzero.
    reduced_pivots = set()

    # Set of pivot indices for rows not fully reduced
    nonreduced_pivots = set()

    # Map from column index to a set of pivot indices representing the rows
    # that have a nonzero at that column.
    nonzero_columns = defaultdict(set)

    while Arows:
        # Select pivot element and row
        Ai = Arows.pop()

        # Nonzero columns from fully reduced pivot rows can be removed
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced_pivots}

        # Others require full row cancellation
        for j in nonreduced_pivots & set(Ai):
            Aj = pivot_row_map[j]
            Aij = Ai[j]
            Ainz = set(Ai)
            Ajnz = set(Aj)
            for k in Ajnz - Ainz:
                Ai[k] = - Aij * Aj[k]
            Ai.pop(j)
            Ainz.remove(j)
            for k in Ajnz & Ainz:
                Aik = Ai[k] - Aij * Aj[k]
                if Aik:
                    Ai[k] = Aik
                else:
                    Ai.pop(k)

        # We have now cancelled previously seen pivots from Ai.
        # If it is zero then discard it.
        if not Ai:
            continue

        # Choose a pivot from Ai:
        j = min(Ai)
        Aij = Ai[j]
        pivot_row_map[j] = Ai
        Ainz = set(Ai)

        # Normalise the pivot row to make the pivot 1.
        #
        # This approach is slow for some domains. Cross cancellation might be
        # better for e.g. QQ(x) with division delayed to the final steps.
        Aijinv = Aij**-1
        for l in Ai:
            Ai[l] *= Aijinv

        # Use Aij to cancel column j from all previously seen rows
        for k in nonzero_columns.pop(j, ()):
            Ak = pivot_row_map[k]
            Akj = Ak[j]
            Aknz = set(Ak)
            for l in Ainz - Aknz:
                Ak[l] = - Akj * Ai[l]
                nonzero_columns[l].add(k)
            Ak.pop(j)
            Aknz.remove(j)
            for l in Ainz & Aknz:
                Akl = Ak[l] - Akj * Ai[l]
                if Akl:
                    Ak[l] = Akl
                else:
                    # Drop nonzero elements
                    Ak.pop(l)
                    if l != j:
                        nonzero_columns[l].remove(k)
            if len(Ak) == 1:
                reduced_pivots.add(k)
                nonreduced_pivots.remove(k)

        if len(Ai) == 1:
            reduced_pivots.add(j)
        else:
            nonreduced_pivots.add(j)
            for l in Ai:
                if l != j:
                    nonzero_columns[l].add(j)

    # All done!
    pivots = sorted(reduced_pivots | nonreduced_pivots)
    pivot2row = {p: n for n, p in enumerate(pivots)}
    nonzero_columns = {c: {pivot2row[p] for p in s} for c, s in nonzero_columns.items()}
    rows = [pivot_row_map[i] for i in pivots]
    rref = dict(enumerate(rows))
    return rref, pivots, nonzero_columns


def sdm_rref_den(A, K):
    """
    Return the reduced row echelon form (RREF) of A with denominator.

    The RREF is computed using fraction-free Gauss-Jordan elimination.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic. This implementation is also
    optimized for sparse matrices.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.sdm import sdm_rref_den
    >>> from sympy.polys.domains import ZZ
    >>> A = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    >>> A_rref, den, pivots = sdm_rref_den(A, ZZ)
    >>> A_rref
    {0: {0: -2}, 1: {1: -2}}
    >>> den
    -2
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher-level interface to ``sdm_rref_den`` that would usually be used
        instead of calling this function directly.
    sympy.polys.matrices.sdm.sdm_rref_den
        The ``SDM`` method that uses this function.
    sdm_irref
        Computes RREF using field division.
    ddm_irref_den
        The dense version of this algorithm.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # We represent each row of the matrix as a dict mapping column indices to
    # nonzero elements. We will build the RREF matrix starting from an empty
    # matrix and appending one row at a time. At each step we will have the
    # RREF of the rows we have processed so far.
    #
    # Our representation of the RREF divides it into three parts:
    #
    # 1. Fully reduced rows having only a single nonzero element (the pivot).
    # 2. Partially reduced rows having nonzeros after the pivot.
    # 3. The current denominator and divisor.
    #
    # For example if the incremental RREF might be:
    #
    #   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #   [0, 0, 2, 0, 0, 0, 7, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
    #
    # Here the second row is partially reduced and the other rows are fully
    # reduced. The denominator would be 2 in this case. We distinguish the
    # fully reduced rows because we can handle them more efficiently when
    # adding a new row.
    #
    # When adding a new row we need to multiply it by the current denominator.
    # Then we reduce the new row by cross cancellation with the previous rows.
    # Then if it is not reduced to zero we take its leading entry as the new
    # pivot, cross cancel the new row from the previous rows and update the
    # denominator. In the fraction-free version this last step requires
    # multiplying and dividing the whole matrix by the new pivot and the
    # current divisor. The advantage of building the RREF one row at a time is
    # that in the sparse case we only need to work with the relatively sparse
    # upper rows of the matrix. The simple version of FFGJ in [1] would
    # multiply and divide all the dense lower rows at each step.

    # Handle the trivial cases.
    if not A:
        return ({}, K.one, [])
    elif len(A) == 1:
        Ai, = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])

    # For inexact domains like RR[x] we use quo and discard the remainder.
    # Maybe it would be better for K.exquo to do this automatically.
    if K.is_Exact:
        exquo = K.exquo
    else:
        exquo = K.quo

    # Make sure we have the rows in order to make this deterministic from the
    # outset.
    _, rows_in_order = zip(*sorted(A.items()))

    col_to_row_reduced = {}
    col_to_row_unreduced = {}
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()

    # Our representation of the RREF so far.
    A_rref_rows = []
    denom = None
    divisor = None

    # The rows that remain to be added to the RREF. These are sorted by the
    # column index of their leading entry. Note that sorted() is stable so the
    # previous sort by unique row index is still needed to make this
    # deterministic (there may be multiple rows with the same leading column).
    A_rows = sorted(rows_in_order, key=min)

    for Ai in A_rows:

        # All fully reduced columns can be immediately discarded.
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced}

        # We need to multiply the new row by the current denominator to bring
        # it into the same scale as the previous rows and then cross-cancel to
        # reduce it wrt the previous unreduced rows. All pivots in the previous
        # rows are equal to denom so the coefficients we need to make a linear
        # combination of the previous rows to cancel into the new row are just
        # the ones that are already in the new row *before* we multiply by
        # denom. We compute that linear combination first and then multiply the
        # new row by denom before subtraction.
        Ai_cancel = {}

        for j in unreduced & Ai.keys():
            # Remove the pivot column from the new row since it would become
            # zero anyway.
            Aij = Ai.pop(j)

            Aj = A_rref_rows[col_to_row_unreduced[j]]

            for k, Ajk in Aj.items():
                Aik_cancel = Ai_cancel.get(k)
                if Aik_cancel is None:
                    Ai_cancel[k] = Aij * Ajk
                else:
                    Aik_cancel = Aik_cancel + Aij * Ajk
                    if Aik_cancel:
                        Ai_cancel[k] = Aik_cancel
                    else:
                        Ai_cancel.pop(k)

        # Multiply the new row by the current denominator and subtract.
        Ai_nz = set(Ai)
        Ai_cancel_nz = set(Ai_cancel)

        d = denom or K.one

        for k in Ai_cancel_nz - Ai_nz:
            Ai[k] = -Ai_cancel[k]

        for k in Ai_nz - Ai_cancel_nz:
            Ai[k] = Ai[k] * d

        for k in Ai_cancel_nz & Ai_nz:
            Aik = Ai[k] * d - Ai_cancel[k]
            if Aik:
                Ai[k] = Aik
            else:
                Ai.pop(k)

        # Now Ai has the same scale as the other rows and is reduced wrt the
        # unreduced rows.

        # If the row is reduced to zero then discard it.
        if not Ai:
            continue

        # Choose a pivot for this row.
        j = min(Ai)
        Aij = Ai.pop(j)

        # Cross cancel the unreduced rows by the new row.
        #     a[k][l] = (a[i][j]*a[k][l] - a[k][j]*a[i][l]) / divisor
        for pk, k in list(col_to_row_unreduced.items()):

            Ak = A_rref_rows[k]

            if j not in Ak:
                # This row is already reduced wrt the new row but we need to
                # bring it to the same scale as the new denominator. This step
                # is not needed in sdm_irref.
                for l, Akl in Ak.items():
                    Akl = Akl * Aij
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                continue

            Akj = Ak.pop(j)
            Ai_nz = set(Ai)
            Ak_nz = set(Ak)

            for l in Ai_nz - Ak_nz:
                Ak[l] = - Akj * Ai[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            # This loop also not needed in sdm_irref.
            for l in Ak_nz - Ai_nz:
                Ak[l] = Aij * Ak[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            for l in Ai_nz & Ak_nz:
                Akl = Aij * Ak[l] - Akj * Ai[l]
                if Akl:
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                else:
                    Ak.pop(l)

            if not Ak:
                col_to_row_unreduced.pop(pk)
                col_to_row_reduced[pk] = k

        i = len(A_rref_rows)
        A_rref_rows.append(Ai)
        if Ai:
            col_to_row_unreduced[j] = i
        else:
            col_to_row_reduced[j] = i

        # Update the denominator.
        if not K.is_one(Aij):
            if denom is None:
                denom = Aij
            else:
                denom *= Aij

        if divisor is not None:
            denom = exquo(denom, divisor)

        # Update the divisor.
        divisor = denom

    if denom is None:
        denom = K.one

    # Sort the rows by their leading column index.
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}
    row_to_col = {i: j for j, i in col_to_row.items()}
    A_rref_rows_col = [(row_to_col[i], Ai) for i, Ai in enumerate(A_rref_rows)]
    pivots, A_rref = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)

    # Insert the pivot values
    for i, Ai in enumerate(A_rref):
        Ai[pivots[i]] = denom

    A_rref_sdm = dict(enumerate(A_rref))

    return A_rref_sdm, denom, pivots


def sdm_nullspace_from_rref(A, one, ncols, pivots, nonzero_cols):
    """Get nullspace from A which is in RREF"""
    nonpivots = sorted(set(range(ncols)) - set(pivots))

    K = []
    for j in nonpivots:
        Kj = {j:one}
        for i in nonzero_cols.get(j, ()):
            Kj[pivots[i]] = -A[i][j]
        K.append(Kj)

    return K, nonpivots


def sdm_particular_from_rref(A, ncols, pivots):
    """Get a particular solution from A which is in RREF"""
    P = {}
    for i, j in enumerate(pivots):
        Ain = A[i].get(ncols-1, None)
        if Ain is not None:
            P[j] = Ain / A[i][j]
    return P


def sdm_berk(M, n, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring. This implementation is for sparse
    matrices represented in a dict-of-dicts format (like :class:`SDM`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.sdm import sdm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    >>> sdm_berk(M, 2, ZZ)
    {0: 1, 1: -5, 2: -2}
    >>> Matrix([[1, 2], [3, 4]]).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.
    sympy.polys.matrices.dense.ddm_berk
        The dense version of this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    zero = K.zero
    one = K.one

    if n == 0:
        return {0: one}
    elif n == 1:
        pdict = {0: one}
        if M00 := M.get(0, {}).get(0, zero):
            pdict[1] = -M00

    # M = [[a, R],
    #      [C, A]]
    a, R, C, A = K.zero, {}, {}, defaultdict(dict)
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            if i and j:
                A[i-1][j-1] = Mij
            elif i:
                C[i-1] = Mij
            elif j:
                R[j-1] = Mij
            else:
                a = Mij

    # T = [       1,      0,    0,  0, 0, ... ]
    #     [      -a,      1,    0,  0, 0, ... ]
    #     [    -R*C,     -a,    1,  0, 0, ... ]
    #     [  -R*A*C,   -R*C,   -a,  1, 0, ... ]
    #     [-R*A^2*C, -R*A*C, -R*C, -a, 1, ... ]
    #     [ ...                               ]
    # T is (n+1) x n
    #
    # In the sparse case we might have A^m*C = 0 for some m making T banded
    # rather than triangular so we just compute the nonzero entries of the
    # first column rather than constructing the matrix explicitly.

    AnC = C
    RC = sdm_dotvec(R, C, K)

    Tvals = [one, -a, -RC]
    for i in range(3, n+1):
        AnC = sdm_matvecmul(A, AnC, K)
        if not AnC:
            break
        RAnC = sdm_dotvec(R, AnC, K)
        Tvals.append(-RAnC)

    # Strip trailing zeros
    while Tvals and not Tvals[-1]:
        Tvals.pop()

    q = sdm_berk(A, n-1, K)

    # This would be the explicit multiplication T*q but we can do better:
    #
    #  T = {}
    #  for i in range(n+1):
    #      Ti = {}
    #      for j in range(max(0, i-len(Tvals)+1), min(i+1, n)):
    #          Ti[j] = Tvals[i-j]
    #      T[i] = Ti
    #  Tq = sdm_matvecmul(T, q, K)
    #
    # In the sparse case q might be mostly zero. We know that T[i,j] is nonzero
    # for i <= j < i + len(Tvals) so if q does not have a nonzero entry in that
    # range then Tq[j] must be zero. We exploit this potential banded
    # structure and the potential sparsity of q to compute Tq more efficiently.

    Tvals = Tvals[::-1]

    Tq = {}

    for i in range(min(q), min(max(q)+len(Tvals), n+1)):
        Ti = dict(enumerate(Tvals, i-len(Tvals)+1))
        if Tqi := sdm_dotvec(Ti, q, K):
            Tq[i] = Tqi

    return Tq
<|endoftext|>#
# sympy.polys.matrices.linsolve module
#
# This module defines the _linsolve function which is the internal workhorse
# used by linsolve. This computes the solution of a system of linear equations
# using the SDM sparse matrix implementation in sympy.polys.matrices.sdm. This
# is a replacement for solve_lin_sys in sympy.polys.solvers which is
# inefficient for large sparse systems due to the use of a PolyRing with many
# generators:
#
#     https://github.com/sympy/sympy/issues/20857
#
# The implementation of _linsolve here handles:
#
# - Extracting the coefficients from the Expr/Eq input equations.
# - Constructing a domain and converting the coefficients to
#   that domain.
# - Using the SDM.rref, SDM.nullspace etc methods to generate the full
#   solution working with arithmetic only in the domain of the coefficients.
#
# The routines here are particularly designed to be efficient for large sparse
# systems of linear equations although as well as dense systems. It is
# possible that for some small dense systems solve_lin_sys which uses the
# dense matrix implementation DDM will be more efficient. With smaller systems
# though the bulk of the time is spent just preprocessing the inputs and the
# relative time spent in rref is too small to be noticeable.
#

from collections import defaultdict

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S

from sympy.polys.constructor import construct_domain
from sympy.polys.solvers import PolyNonlinearError

from .sdm import (
    SDM,
    sdm_irref,
    sdm_particular_from_rref,
    sdm_nullspace_from_rref
)

from sympy.utilities.misc import filldedent


def _linsolve(eqs, syms):

    """Solve a linear system of equations.

    Examples
    ========

    Solve a linear system with a unique solution:

    >>> from sympy import symbols, Eq
    >>> from sympy.polys.matrices.linsolve import _linsolve
    >>> x, y = symbols('x, y')
    >>> eqs = [Eq(x + y, 1), Eq(x - y, 2)]
    >>> _linsolve(eqs, [x, y])
    {x: 3/2, y: -1/2}

    In the case of underdetermined systems the solution will be expressed in
    terms of the unknown symbols that are unconstrained:

    >>> _linsolve([Eq(x + y, 0)], [x, y])
    {x: -y, y: y}

    """
    # Number of unknowns (columns in the non-augmented matrix)
    nsyms = len(syms)

    # Convert to sparse augmented matrix (len(eqs) x (nsyms+1))
    eqsdict, const = _linear_eq_to_dict(eqs, syms)
    Aaug = sympy_dict_to_dm(eqsdict, const, syms)
    K = Aaug.domain

    # sdm_irref has issues with float matrices. This uses the ddm_rref()
    # function. When sdm_rref() can handle float matrices reasonably this
    # should be removed...
    if K.is_RealField or K.is_ComplexField:
        Aaug = Aaug.to_ddm().rref()[0].to_sdm()

    # Compute reduced-row echelon form (RREF)
    Arref, pivots, nzcols = sdm_irref(Aaug)

    # No solution:
    if pivots and pivots[-1] == nsyms:
        return None

    # Particular solution for non-homogeneous system:
    P = sdm_particular_from_rref(Arref, nsyms+1, pivots)

    # Nullspace - general solution to homogeneous system
    # Note: using nsyms not nsyms+1 to ignore last column
    V, nonpivots = sdm_nullspace_from_rref(Arref, K.one, nsyms, pivots, nzcols)

    # Collect together terms from particular and nullspace:
    sol = defaultdict(list)
    for i, v in P.items():
        sol[syms[i]].append(K.to_sympy(v))
    for npi, Vi in zip(nonpivots, V):
        sym = syms[npi]
        for i, v in Vi.items():
            sol[syms[i]].append(sym * K.to_sympy(v))

    # Use a single call to Add for each term:
    sol = {s: Add(*terms) for s, terms in sol.items()}

    # Fill in the zeros:
    zero = S.Zero
    for s in set(syms) - set(sol):
        sol[s] = zero

    # All done!
    return sol


def sympy_dict_to_dm(eqs_coeffs, eqs_rhs, syms):
    """Convert a system of dict equations to a sparse augmented matrix"""
    elems = set(eqs_rhs).union(*(e.values() for e in eqs_coeffs))
    K, elems_K = construct_domain(elems, field=True, extension=True)
    elem_map = dict(zip(elems, elems_K))
    neqs = len(eqs_coeffs)
    nsyms = len(syms)
    sym2index = dict(zip(syms, range(nsyms)))
    eqsdict = []
    for eq, rhs in zip(eqs_coeffs, eqs_rhs):
        eqdict = {sym2index[s]: elem_map[c] for s, c in eq.items()}
        if rhs:
            eqdict[nsyms] = -elem_map[rhs]
        if eqdict:
            eqsdict.append(eqdict)
    sdm_aug = SDM(enumerate(eqsdict), (neqs, nsyms + 1), K)
    return sdm_aug


def _linear_eq_to_dict(eqs, syms):
    """Convert a system Expr/Eq equations into dict form, returning
    the coefficient dictionaries and a list of syms-independent terms
    from each expression in ``eqs```.

    Examples
    ========

    >>> from sympy.polys.matrices.linsolve import _linear_eq_to_dict
    >>> from sympy.abc import x
    >>> _linear_eq_to_dict([2*x + 3], {x})
    ([{x: 2}], [3])
    """
    coeffs = []
    ind = []
    symset = set(syms)
    for e in eqs:
        if e.is_Equality:
            coeff, terms = _lin_eq2dict(e.lhs, symset)
            cR, tR = _lin_eq2dict(e.rhs, symset)
            # there were no nonlinear errors so now
            # cancellation is allowed
            coeff -= cR
            for k, v in tR.items():
                if k in terms:
                    terms[k] -= v
                else:
                    terms[k] = -v
            # don't store coefficients of 0, however
            terms = {k: v for k, v in terms.items() if v}
            c, d = coeff, terms
        else:
            c, d = _lin_eq2dict(e, symset)
        coeffs.append(d)
        ind.append(c)
    return coeffs, ind


def _lin_eq2dict(a, symset):
    """return (c, d) where c is the sym-independent part of ``a`` and
    ``d`` is an efficiently calculated dictionary mapping symbols to
    their coefficients. A PolyNonlinearError is raised if non-linearity
    is detected.

    The values in the dictionary will be non-zero.

    Examples
    ========

    >>> from sympy.polys.matrices.linsolve import _lin_eq2dict
    >>> from sympy.abc import x, y
    >>> _lin_eq2dict(x + 2*y + 3, {x, y})
    (3, {x: 1, y: 2})
    """
    if a in symset:
        return S.Zero, {a: S.One}
    elif a.is_Add:
        terms_list = defaultdict(list)
        coeff_list = []
        for ai in a.args:
            ci, ti = _lin_eq2dict(ai, symset)
            coeff_list.append(ci)
            for mij, cij in ti.items():
                terms_list[mij].append(cij)
        coeff = Add(*coeff_list)
        terms = {sym: Add(*coeffs) for sym, coeffs in terms_list.items()}
        return coeff, terms
    elif a.is_Mul:
        terms = terms_coeff = None
        coeff_list = []
        for ai in a.args:
            ci, ti = _lin_eq2dict(ai, symset)
            if not ti:
                coeff_list.append(ci)
            elif terms is None:
                terms = ti
                terms_coeff = ci
            else:
                # since ti is not null and we already have
                # a term, this is a cross term
                raise PolyNonlinearError(filldedent('''
                    nonlinear cross-term: %s''' % a))
        coeff = Mul._from_args(coeff_list)
        if terms is None:
            return coeff, {}
        else:
            terms = {sym: coeff * c for sym, c in terms.items()}
            return  coeff * terms_coeff, terms
    elif not a.has_xfree(symset):
        return a, {}
    else:
        raise PolyNonlinearError('nonlinear term: %s' % a)
<|endoftext|>"""
sympy.polys.matrices.dfm

Provides the :class:`DFM` class if ``GROUND_TYPES=flint'``. Otherwise, ``DFM``
is a placeholder class that raises NotImplementedError when instantiated.
"""

from sympy.external.gmpy import GROUND_TYPES

if GROUND_TYPES == "flint":  # pragma: no cover
    # When python-flint is installed we will try to use it for dense matrices
    # if the domain is supported by python-flint.
    from ._dfm import DFM

else: # pragma: no cover
    # Other code should be able to import this and it should just present as a
    # version of DFM that does not support any domains.
    class DFM_dummy:
        """
        Placeholder class for DFM when python-flint is not installed.
        """
        def __init__(*args, **kwargs):
            raise NotImplementedError("DFM requires GROUND_TYPES=flint.")

        @classmethod
        def _supports_domain(cls, domain):
            return False

        @classmethod
        def _get_flint_func(cls, domain):
            raise NotImplementedError("DFM requires GROUND_TYPES=flint.")

    # mypy really struggles with this kind of conditional type assignment.
    # Maybe there is a better way to annotate this rather than type: ignore.
    DFM = DFM_dummy # type: ignore
<|endoftext|>"""

Module to define exceptions to be used in sympy.polys.matrices modules and
classes.

Ideally all exceptions raised in these modules would be defined and documented
here and not e.g. imported from matrices. Also ideally generic exceptions like
ValueError/TypeError would not be raised anywhere.

"""


class DMError(Exception):
    """Base class for errors raised by DomainMatrix"""
    pass


class DMBadInputError(DMError):
    """list of lists is inconsistent with shape"""
    pass


class DMDomainError(DMError):
    """domains do not match"""
    pass


class DMNotAField(DMDomainError):
    """domain is not a field"""
    pass


class DMFormatError(DMError):
    """mixed dense/sparse not supported"""
    pass


class DMNonInvertibleMatrixError(DMError):
    """The matrix in not invertible"""
    pass


class DMRankError(DMError):
    """matrix does not have expected rank"""
    pass


class DMShapeError(DMError):
    """shapes are inconsistent"""
    pass


class DMNonSquareMatrixError(DMShapeError):
    """The matrix is not square"""
    pass


class DMValueError(DMError):
    """The value passed is invalid"""
    pass


__all__ = [
    'DMError', 'DMBadInputError', 'DMDomainError', 'DMFormatError',
    'DMRankError', 'DMShapeError', 'DMNotAField',
    'DMNonInvertibleMatrixError', 'DMNonSquareMatrixError', 'DMValueError'
]
<|endoftext|>"""Implementation of :class:`PythonIntegerRing` class. """


from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import (
    PythonInteger, SymPyInteger, sqrt as python_sqrt,
    factorial as python_factorial, python_gcdex, python_gcd, python_lcm,
)
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class PythonIntegerRing(IntegerRing):
    """Integer ring based on Python's ``int`` type.

    This will be used as :ref:`ZZ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of the standard Python ``int`` type.
    """

    dtype = PythonInteger
    zero = dtype(0)
    one = dtype(1)
    alias = 'ZZ_python'

    def __init__(self):
        """Allow instantiation of this domain. """

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(a)

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return PythonInteger(a.p)
        elif int_valued(a):
            return PythonInteger(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to Python's ``int``. """
        return K0.to_int(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to Python's ``int``. """
        return a

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to Python's ``int``. """
        if a.denominator == 1:
            return a.numerator

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to Python's ``int``. """
        if a.denominator == 1:
            return a.numerator

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to Python's ``int``. """
        return PythonInteger(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to Python's ``int``. """
        return PythonInteger(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpq`` to Python's ``int``. """
        if a.denom() == 1:
            return PythonInteger(a.numer())

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to Python's ``int``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return PythonInteger(p)

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        return python_gcdex(a, b)

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return python_gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return python_lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return python_sqrt(a)

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return python_factorial(a)
<|endoftext|>"""Implementation of :class:`PythonRationalField` class. """


from sympy.polys.domains.groundtypes import PythonInteger, PythonRational, SymPyRational
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class PythonRationalField(RationalField):
    """Rational field based on :ref:`MPQ`.

    This will be used as :ref:`QQ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of :ref:`MPQ`.
    """

    dtype = PythonRational
    zero = dtype(0)
    one = dtype(1)
    alias = 'QQ_python'

    def __init__(self):
        pass

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import PythonIntegerRing
        return PythonIntegerRing()

    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
        return SymPyRational(a.numerator, a.denominator)

    def from_sympy(self, a):
        """Convert SymPy's Rational to `dtype`. """
        if a.is_Rational:
            return PythonRational(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            p, q = RR.to_rational(a)
            return PythonRational(int(p), int(q))
        else:
            raise CoercionFailed("expected `Rational` object, got %s" % a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return PythonRational(a)

    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return a

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        return PythonRational(PythonInteger(a))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        return PythonRational(PythonInteger(a.numer()),
                              PythonInteger(a.denom()))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        p, q = K0.to_rational(a)
        return PythonRational(int(p), int(q))

    def numer(self, a):
        """Returns numerator of `a`. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of `a`. """
        return a.denominator
<|endoftext|>#
# This module is deprecated and should not be used any more. The actual
# implementation of RR and CC now uses mpmath's mpf and mpc types directly.
#
"""Real and complex elements. """


from sympy.external.gmpy import MPQ
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public

from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
    round_nearest, mpf_mul, repr_dps, int_types,
    from_int, from_float, from_str, to_rational)


@public
class RealElement(_mpf, DomainElement):
    """An element of a real domain. """

    __slots__ = ('__mpf__',)

    def _set_mpf(self, val):
        self.__mpf__ = val

    _mpf_ = property(lambda self: self.__mpf__, _set_mpf)

    def parent(self):
        return self.context._parent

@public
class ComplexElement(_mpc, DomainElement):
    """An element of a complex domain. """

    __slots__ = ('__mpc__',)

    def _set_mpc(self, val):
        self.__mpc__ = val

    _mpc_ = property(lambda self: self.__mpc__, _set_mpc)

    def parent(self):
        return self.context._parent

new = object.__new__

@public
class MPContext(PythonMPContext):

    def __init__(ctx, prec=53, dps=None, tol=None, real=False):
        ctx._prec_rounding = [prec, round_nearest]

        if dps is None:
            ctx._set_prec(prec)
        else:
            ctx._set_dps(dps)

        ctx.mpf = RealElement
        ctx.mpc = ComplexElement
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]

        if real:
            ctx.mpf.context = ctx
        else:
            ctx.mpc.context = ctx

        ctx.constant = _constant
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.constant.context = ctx

        ctx.types = [ctx.mpf, ctx.mpc, ctx.constant]
        ctx.trap_complex = True
        ctx.pretty = True

        if tol is None:
            ctx.tol = ctx._make_tol()
        elif tol is False:
            ctx.tol = fzero
        else:
            ctx.tol = ctx._convert_tol(tol)

        ctx.tolerance = ctx.make_mpf(ctx.tol)

        if not ctx.tolerance:
            ctx.max_denom = 1000000
        else:
            ctx.max_denom = int(1/ctx.tolerance)

        ctx.zero = ctx.make_mpf(fzero)
        ctx.one = ctx.make_mpf(fone)
        ctx.j = ctx.make_mpc((fzero, fone))
        ctx.inf = ctx.make_mpf(finf)
        ctx.ninf = ctx.make_mpf(fninf)
        ctx.nan = ctx.make_mpf(fnan)

    def _make_tol(ctx):
        hundred = (0, 25, 2, 5)
        eps = (0, MPZ_ONE, 1-ctx.prec, 1)
        return mpf_mul(hundred, eps)

    def make_tol(ctx):
        return ctx.make_mpf(ctx._make_tol())

    def _convert_tol(ctx, tol):
        if isinstance(tol, int_types):
            return from_int(tol)
        if isinstance(tol, float):
            return from_float(tol)
        if hasattr(tol, "_mpf_"):
            return tol._mpf_
        prec, rounding = ctx._prec_rounding
        if isinstance(tol, str):
            return from_str(tol, prec, rounding)
        raise ValueError("expected a real number, got %s" % tol)

    def _convert_fallback(ctx, x, strings):
        raise TypeError("cannot create mpf from " + repr(x))

    @property
    def _repr_digits(ctx):
        return repr_dps(ctx._prec)

    @property
    def _str_digits(ctx):
        return ctx._dps

    def to_rational(ctx, s, limit=True):
        p, q = to_rational(s._mpf_)

        # Needed for GROUND_TYPES=flint if gmpy2 is installed because mpmath's
        # to_rational() function returns a gmpy2.mpz instance and if MPQ is
        # flint.fmpq then MPQ(p, q) will fail.
        p = int(p)

        if not limit or q <= ctx.max_denom:
            return p, q

        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = p, q

        while True:
            a = n//d
            q2 = q0 + a*q1
            if q2 > ctx.max_denom:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
            n, d = d, n - a*d

        k = (ctx.max_denom - q0)//q1

        number = MPQ(p, q)
        bound1 = MPQ(p0 + k*p1, q0 + k*q1)
        bound2 = MPQ(p1, q1)

        if not bound2 or not bound1:
            return p, q
        elif abs(bound2 - number) <= abs(bound1 - number):
            return bound2.numerator, bound2.denominator
        else:
            return bound1.numerator, bound1.denominator

    def almosteq(ctx, s, t, rel_eps=None, abs_eps=None):
        t = ctx.convert(t)
        if abs_eps is None and rel_eps is None:
            rel_eps = abs_eps = ctx.tolerance or ctx.make_tol()
        if abs_eps is None:
            abs_eps = ctx.convert(rel_eps)
        elif rel_eps is None:
            rel_eps = ctx.convert(abs_eps)
        diff = abs(s-t)
        if diff <= abs_eps:
            return True
        abss = abs(s)
        abst = abs(t)
        if abss < abst:
            err = diff/abst
        else:
            err = diff/abss
        return err <= rel_eps
<|endoftext|>"""Implementation of :class:`GMPYIntegerRing` class. """


from sympy.polys.domains.groundtypes import (
    GMPYInteger, SymPyInteger,
    factorial as gmpy_factorial,
    gmpy_gcdex, gmpy_gcd, gmpy_lcm, sqrt as gmpy_sqrt,
)
from sympy.core.numbers import int_valued
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class GMPYIntegerRing(IntegerRing):
    """Integer ring based on GMPY's ``mpz`` type.

    This will be the implementation of :ref:`ZZ` if ``gmpy`` or ``gmpy2`` is
    installed. Elements will be of type ``gmpy.mpz``.
    """

    dtype = GMPYInteger
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    alias = 'ZZ_gmpy'

    def __init__(self):
        """Allow instantiation of this domain. """

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return GMPYInteger(a.p)
        elif int_valued(a):
            return GMPYInteger(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return K0.to_int(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return GMPYInteger(a)

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return GMPYInteger(a.numerator)

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return GMPYInteger(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        return K0.to_int(a)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to GMPY's ``mpz``. """
        return a

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY ``mpq`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return a.numerator

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to GMPY's ``mpz``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return GMPYInteger(p)

    def from_GaussianIntegerRing(K1, a, K0):
        if a.y == 0:
            return a.x

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        h, s, t = gmpy_gcdex(a, b)
        return s, t, h

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return gmpy_gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return gmpy_lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return gmpy_sqrt(a)

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return gmpy_factorial(a)
<|endoftext|>"""Implementation of :class:`CompositeDomain` class. """


from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import GeneratorsError

from sympy.utilities import public

@public
class CompositeDomain(Domain):
    """Base class for composite domains, e.g. ZZ[x], ZZ(X). """

    is_Composite = True

    gens, ngens, symbols, domain = [None]*4

    def inject(self, *symbols):
        """Inject generators into this domain.  """
        if not (set(self.symbols) & set(symbols)):
            return self.__class__(self.domain, self.symbols + symbols, self.order)
        else:
            raise GeneratorsError("common generators in %s and %s" % (self.symbols, symbols))

    def drop(self, *symbols):
        """Drop generators from this domain. """
        symset = set(symbols)
        newsyms = tuple(s for s in self.symbols if s not in symset)
        domain = self.domain.drop(*symbols)
        if not newsyms:
            return domain
        else:
            return self.__class__(domain, newsyms, self.order)

    def set_domain(self, domain):
        """Set the ground domain of this domain. """
        return self.__class__(domain, self.symbols, self.order)

    @property
    def is_Exact(self):
        """Returns ``True`` if this domain is exact. """
        return self.domain.is_Exact

    def get_exact(self):
        """Returns an exact version of this domain. """
        return self.set_domain(self.domain.get_exact())

    @property
    def has_CharacteristicZero(self):
        return self.domain.has_CharacteristicZero

    def characteristic(self):
        return self.domain.characteristic()
<|endoftext|>"""Implementation of :class:`AlgebraicField` class. """


from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, symbols
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, IsomorphismFailed
from sympy.utilities import public

@public
class AlgebraicField(Field, CharacteristicZero, SimpleDomain):
    r"""Algebraic number field :ref:`QQ(a)`

    A :ref:`QQ(a)` domain represents an `algebraic number field`_
    `\mathbb{Q}(a)` as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression involving `algebraic
    numbers`_ will treat the algebraic numbers as generators if the generators
    argument is not specified.

    >>> from sympy import Poly, Symbol, sqrt
    >>> x = Symbol('x')
    >>> Poly(x**2 + sqrt(2))
    Poly(x**2 + (sqrt(2)), x, sqrt(2), domain='ZZ')

    That is a multivariate polynomial with ``sqrt(2)`` treated as one of the
    generators (variables). If the generators are explicitly specified then
    ``sqrt(2)`` will be considered to be a coefficient but by default the
    :ref:`EX` domain is used. To make a :py:class:`~.Poly` with a :ref:`QQ(a)`
    domain the argument ``extension=True`` can be given.

    >>> Poly(x**2 + sqrt(2), x)
    Poly(x**2 + sqrt(2), x, domain='EX')
    >>> Poly(x**2 + sqrt(2), x, extension=True)
    Poly(x**2 + sqrt(2), x, domain='QQ<sqrt(2)>')

    A generator of the algebraic field extension can also be specified
    explicitly which is particularly useful if the coefficients are all
    rational but an extension field is needed (e.g. to factor the
    polynomial).

    >>> Poly(x**2 + 1)
    Poly(x**2 + 1, x, domain='ZZ')
    >>> Poly(x**2 + 1, extension=sqrt(2))
    Poly(x**2 + 1, x, domain='QQ<sqrt(2)>')

    It is possible to factorise a polynomial over a :ref:`QQ(a)` domain using
    the ``extension`` argument to :py:func:`~.factor` or by specifying the domain
    explicitly.

    >>> from sympy import factor, QQ
    >>> factor(x**2 - 2)
    x**2 - 2
    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))
    >>> factor(x**2 - 2, domain='QQ<sqrt(2)>')
    (x - sqrt(2))*(x + sqrt(2))
    >>> factor(x**2 - 2, domain=QQ.algebraic_field(sqrt(2)))
    (x - sqrt(2))*(x + sqrt(2))

    The ``extension=True`` argument can be used but will only create an
    extension that contains the coefficients which is usually not enough to
    factorise the polynomial.

    >>> p = x**3 + sqrt(2)*x**2 - 2*x - 2*sqrt(2)
    >>> factor(p)                         # treats sqrt(2) as a symbol
    (x + sqrt(2))*(x**2 - 2)
    >>> factor(p, extension=True)
    (x - sqrt(2))*(x + sqrt(2))**2
    >>> factor(x**2 - 2, extension=True)  # all rational coefficients
    x**2 - 2

    It is also possible to use :ref:`QQ(a)` with the :py:func:`~.cancel`
    and :py:func:`~.gcd` functions.

    >>> from sympy import cancel, gcd
    >>> cancel((x**2 - 2)/(x - sqrt(2)))
    (x**2 - 2)/(x - sqrt(2))
    >>> cancel((x**2 - 2)/(x - sqrt(2)), extension=sqrt(2))
    x + sqrt(2)
    >>> gcd(x**2 - 2, x - sqrt(2))
    1
    >>> gcd(x**2 - 2, x - sqrt(2), extension=sqrt(2))
    x - sqrt(2)

    When using the domain directly :ref:`QQ(a)` can be used as a constructor
    to create instances which then support the operations ``+,-,*,**,/``. The
    :py:meth:`~.Domain.algebraic_field` method is used to construct a
    particular :ref:`QQ(a)` domain. The :py:meth:`~.Domain.from_sympy` method
    can be used to create domain elements from normal SymPy expressions.

    >>> K = QQ.algebraic_field(sqrt(2))
    >>> K
    QQ<sqrt(2)>
    >>> xk = K.from_sympy(3 + 4*sqrt(2))
    >>> xk  # doctest: +SKIP
    ANP([4, 3], [1, 0, -2], QQ)

    Elements of :ref:`QQ(a)` are instances of :py:class:`~.ANP` which have
    limited printing support. The raw display shows the internal
    representation of the element as the list ``[4, 3]`` representing the
    coefficients of ``1`` and ``sqrt(2)`` for this element in the form
    ``a * sqrt(2) + b * 1`` where ``a`` and ``b`` are elements of :ref:`QQ`.
    The minimal polynomial for the generator ``(x**2 - 2)`` is also shown in
    the :ref:`dup-representation` as the list ``[1, 0, -2]``. We can use
    :py:meth:`~.Domain.to_sympy` to get a better printed form for the
    elements and to see the results of operations.

    >>> xk = K.from_sympy(3 + 4*sqrt(2))
    >>> yk = K.from_sympy(2 + 3*sqrt(2))
    >>> xk * yk  # doctest: +SKIP
    ANP([17, 30], [1, 0, -2], QQ)
    >>> K.to_sympy(xk * yk)
    17*sqrt(2) + 30
    >>> K.to_sympy(xk + yk)
    5 + 7*sqrt(2)
    >>> K.to_sympy(xk ** 2)
    24*sqrt(2) + 41
    >>> K.to_sympy(xk / yk)
    sqrt(2)/14 + 9/7

    Any expression representing an algebraic number can be used to generate
    a :ref:`QQ(a)` domain provided its `minimal polynomial`_ can be computed.
    The function :py:func:`~.minpoly` function is used for this.

    >>> from sympy import exp, I, pi, minpoly
    >>> g = exp(2*I*pi/3)
    >>> g
    exp(2*I*pi/3)
    >>> g.is_algebraic
    True
    >>> minpoly(g, x)
    x**2 + x + 1
    >>> factor(x**3 - 1, extension=g)
    (x - 1)*(x - exp(2*I*pi/3))*(x + 1 + exp(2*I*pi/3))

    It is also possible to make an algebraic field from multiple extension
    elements.

    >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
    >>> K
    QQ<sqrt(2) + sqrt(3)>
    >>> p = x**4 - 5*x**2 + 6
    >>> factor(p)
    (x**2 - 3)*(x**2 - 2)
    >>> factor(p, domain=K)
    (x - sqrt(2))*(x + sqrt(2))*(x - sqrt(3))*(x + sqrt(3))
    >>> factor(p, extension=[sqrt(2), sqrt(3)])
    (x - sqrt(2))*(x + sqrt(2))*(x - sqrt(3))*(x + sqrt(3))

    Multiple extension elements are always combined together to make a single
    `primitive element`_. In the case of ``[sqrt(2), sqrt(3)]`` the primitive
    element chosen is ``sqrt(2) + sqrt(3)`` which is why the domain displays
    as ``QQ<sqrt(2) + sqrt(3)>``. The minimal polynomial for the primitive
    element is computed using the :py:func:`~.primitive_element` function.

    >>> from sympy import primitive_element
    >>> primitive_element([sqrt(2), sqrt(3)], x)
    (x**4 - 10*x**2 + 1, [1, 1])
    >>> minpoly(sqrt(2) + sqrt(3), x)
    x**4 - 10*x**2 + 1

    The extension elements that generate the domain can be accessed from the
    domain using the :py:attr:`~.ext` and :py:attr:`~.orig_ext` attributes as
    instances of :py:class:`~.AlgebraicNumber`. The minimal polynomial for
    the primitive element as a :py:class:`~.DMP` instance is available as
    :py:attr:`~.mod`.

    >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
    >>> K
    QQ<sqrt(2) + sqrt(3)>
    >>> K.ext
    sqrt(2) + sqrt(3)
    >>> K.orig_ext
    (sqrt(2), sqrt(3))
    >>> K.mod  # doctest: +SKIP
    DMP_Python([1, 0, -10, 0, 1], QQ)

    The `discriminant`_ of the field can be obtained from the
    :py:meth:`~.discriminant` method, and an `integral basis`_ from the
    :py:meth:`~.integral_basis` method. The latter returns a list of
    :py:class:`~.ANP` instances by default, but can be made to return instances
    of :py:class:`~.Expr` or :py:class:`~.AlgebraicNumber` by passing a ``fmt``
    argument. The maximal order, or ring of integers, of the field can also be
    obtained from the :py:meth:`~.maximal_order` method, as a
    :py:class:`~sympy.polys.numberfields.modules.Submodule`.

    >>> zeta5 = exp(2*I*pi/5)
    >>> K = QQ.algebraic_field(zeta5)
    >>> K
    QQ<exp(2*I*pi/5)>
    >>> K.discriminant()
    125
    >>> K = QQ.algebraic_field(sqrt(5))
    >>> K
    QQ<sqrt(5)>
    >>> K.integral_basis(fmt='sympy')
    [1, 1/2 + sqrt(5)/2]
    >>> K.maximal_order()
    Submodule[[2, 0], [1, 1]]/2

    The factorization of a rational prime into prime ideals of the field is
    computed by the :py:meth:`~.primes_above` method, which returns a list
    of :py:class:`~sympy.polys.numberfields.primes.PrimeIdeal` instances.

    >>> zeta7 = exp(2*I*pi/7)
    >>> K = QQ.algebraic_field(zeta7)
    >>> K
    QQ<exp(2*I*pi/7)>
    >>> K.primes_above(11)
    [(11, _x**3 + 5*_x**2 + 4*_x - 1), (11, _x**3 - 4*_x**2 - 5*_x - 1)]

    The Galois group of the Galois closure of the field can be computed (when
    the minimal polynomial of the field is of sufficiently small degree).

    >>> K.galois_group(by_name=True)[0]
    S6TransitiveSubgroups.C6

    Notes
    =====

    It is not currently possible to generate an algebraic extension over any
    domain other than :ref:`QQ`. Ideally it would be possible to generate
    extensions like ``QQ(x)(sqrt(x**2 - 2))``. This is equivalent to the
    quotient ring ``QQ(x)[y]/(y**2 - x**2 + 2)`` and there are two
    implementations of this kind of quotient ring/extension in the
    :py:class:`~.QuotientRing` and :py:class:`~.MonogenicFiniteExtension`
    classes.  Each of those implementations needs some work to make them fully
    usable though.

    .. _algebraic number field: https://en.wikipedia.org/wiki/Algebraic_number_field
    .. _algebraic numbers: https://en.wikipedia.org/wiki/Algebraic_number
    .. _discriminant: https://en.wikipedia.org/wiki/Discriminant_of_an_algebraic_number_field
    .. _integral basis: https://en.wikipedia.org/wiki/Algebraic_number_field#Integral_basis
    .. _minimal polynomial: https://en.wikipedia.org/wiki/Minimal_polynomial_(field_theory)
    .. _primitive element: https://en.wikipedia.org/wiki/Primitive_element_theorem
    """

    dtype = ANP

    is_AlgebraicField = is_Algebraic = True
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self, dom, *ext, alias=None):
        r"""
        Parameters
        ==========

        dom : :py:class:`~.Domain`
            The base field over which this is an extension field.
            Currently only :ref:`QQ` is accepted.

        *ext : One or more :py:class:`~.Expr`
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the :py:class:`~.AlgebraicField`.
            If ``None``, while ``ext`` consists of exactly one
            :py:class:`~.AlgebraicNumber`, its alias (if any) will be used.
        """
        if not dom.is_QQ:
            raise DomainError("ground domain must be a rational field")

        from sympy.polys.numberfields import to_number_field
        if len(ext) == 1 and isinstance(ext[0], tuple):
            orig_ext = ext[0][1:]
        else:
            orig_ext = ext

        if alias is None and len(ext) == 1:
            alias = getattr(ext[0], 'alias', None)

        self.orig_ext = orig_ext
        """
        Original elements given to generate the extension.

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
        >>> K.orig_ext
        (sqrt(2), sqrt(3))
        """

        self.ext = to_number_field(ext, alias=alias)
        """
        Primitive element used for the extension.

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2), sqrt(3))
        >>> K.ext
        sqrt(2) + sqrt(3)
        """

        self.mod = self.ext.minpoly.rep
        """
        Minimal polynomial for the primitive element of the extension.

        >>> from sympy import QQ, sqrt
        >>> K = QQ.algebraic_field(sqrt(2))
        >>> K.mod
        DMP([1, 0, -2], QQ)
        """

        self.domain = self.dom = dom

        self.ngens = 1
        self.symbols = self.gens = (self.ext,)
        self.unit = self([dom(1), dom(0)])

        self.zero = self.dtype.zero(self.mod.to_list(), dom)
        self.one = self.dtype.one(self.mod.to_list(), dom)

        self._maximal_order = None
        self._discriminant = None
        self._nilradicals_mod_p = {}

    def new(self, element):
        return self.dtype(element, self.mod.to_list(), self.dom)

    def __str__(self):
        return str(self.dom) + '<' + str(self.ext) + '>'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom, self.ext))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, AlgebraicField):
            return self.dtype == other.dtype and self.ext == other.ext
        else:
            return NotImplemented

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`. """
        return AlgebraicField(self.dom, *((self.ext,) + extension), alias=alias)

    def to_alg_num(self, a):
        """Convert ``a`` of ``dtype`` to an :py:class:`~.AlgebraicNumber`. """
        return self.ext.field_element(a)

    def to_sympy(self, a):
        """Convert ``a`` of ``dtype`` to a SymPy object. """
        # Precompute a converter to be reused:
        if not hasattr(self, '_converter'):
            self._converter = _make_converter(self)

        return self._converter(a)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        try:
            return self([self.dom.from_sympy(a)])
        except CoercionFailed:
            pass

        from sympy.polys.numberfields import to_number_field

        try:
            return self(to_number_field(a, self.ext).native_coeffs())
        except (NotAlgebraic, IsomorphismFailed):
            raise CoercionFailed(
                "%s is not a valid algebraic number in %s" % (a, self))

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError('there is no ring associated with %s' % self)

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return self.dom.is_nonnegative(a.LC())

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a

    def denom(self, a):
        """Returns denominator of ``a``. """
        return self.one

    def from_AlgebraicField(K1, a, K0):
        """Convert AlgebraicField element 'a' to another AlgebraicField """
        return K1.from_sympy(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a GaussianInteger element 'a' to ``dtype``. """
        return K1.from_sympy(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a GaussianRational element 'a' to ``dtype``. """
        return K1.from_sympy(K0.to_sympy(a))

    def _do_round_two(self):
        from sympy.polys.numberfields.basis import round_two
        ZK, dK = round_two(self, radicals=self._nilradicals_mod_p)
        self._maximal_order = ZK
        self._discriminant = dK

    def maximal_order(self):
        """
        Compute the maximal order, or ring of integers, of the field.

        Returns
        =======

        :py:class:`~sympy.polys.numberfields.modules.Submodule`.

        See Also
        ========

        integral_basis

        """
        if self._maximal_order is None:
            self._do_round_two()
        return self._maximal_order

    def integral_basis(self, fmt=None):
        r"""
        Get an integral basis for the field.

        Parameters
        ==========

        fmt : str, None, optional (default=None)
            If ``None``, return a list of :py:class:`~.ANP` instances.
            If ``"sympy"``, convert each element of the list to an
            :py:class:`~.Expr`, using ``self.to_sympy()``.
            If ``"alg"``, convert each element of the list to an
            :py:class:`~.AlgebraicNumber`, using ``self.to_alg_num()``.

        Examples
        ========

        >>> from sympy import QQ, AlgebraicNumber, sqrt
        >>> alpha = AlgebraicNumber(sqrt(5), alias='alpha')
        >>> k = QQ.algebraic_field(alpha)
        >>> B0 = k.integral_basis()
        >>> B1 = k.integral_basis(fmt='sympy')
        >>> B2 = k.integral_basis(fmt='alg')
        >>> print(B0[1])  # doctest: +SKIP
        ANP([mpq(1,2), mpq(1,2)], [mpq(1,1), mpq(0,1), mpq(-5,1)], QQ)
        >>> print(B1[1])
        1/2 + alpha/2
        >>> print(B2[1])
        alpha/2 + 1/2

        In the last two cases we get legible expressions, which print somewhat
        differently because of the different types involved:

        >>> print(type(B1[1]))
        <class 'sympy.core.add.Add'>
        >>> print(type(B2[1]))
        <class 'sympy.core.numbers.AlgebraicNumber'>

        See Also
        ========

        to_sympy
        to_alg_num
        maximal_order
        """
        ZK = self.maximal_order()
        M = ZK.QQ_matrix
        n = M.shape[1]
        B = [self.new(list(reversed(M[:, j].flat()))) for j in range(n)]
        if fmt == 'sympy':
            return [self.to_sympy(b) for b in B]
        elif fmt == 'alg':
            return [self.to_alg_num(b) for b in B]
        return B

    def discriminant(self):
        """Get the discriminant of the field."""
        if self._discriminant is None:
            self._do_round_two()
        return self._discriminant

    def primes_above(self, p):
        """Compute the prime ideals lying above a given rational prime *p*."""
        from sympy.polys.numberfields.primes import prime_decomp
        ZK = self.maximal_order()
        dK = self.discriminant()
        rad = self._nilradicals_mod_p.get(p)
        return prime_decomp(p, ZK=ZK, dK=dK, radical=rad)

    def galois_group(self, by_name=False, max_tries=30, randomize=False):
        """
        Compute the Galois group of the Galois closure of this field.

        Examples
        ========

        If the field is Galois, the order of the group will equal the degree
        of the field:

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> k = QQ.alg_field_from_poly(x**4 + 1)
        >>> G, _ = k.galois_group()
        >>> G.order()
        4

        If the field is not Galois, then its Galois closure is a proper
        extension, and the order of the Galois group will be greater than the
        degree of the field:

        >>> k = QQ.alg_field_from_poly(x**4 - 2)
        >>> G, _ = k.galois_group()
        >>> G.order()
        8

        See Also
        ========

        sympy.polys.numberfields.galoisgroups.galois_group

        """
        return self.ext.minpoly_of_element().galois_group(
            by_name=by_name, max_tries=max_tries, randomize=randomize)


def _make_converter(K):
    """Construct the converter to convert back to Expr"""
    # Precompute the effect of converting to SymPy and expanding expressions
    # like (sqrt(2) + sqrt(3))**2. Asking Expr to do the expansion on every
    # conversion from K to Expr is slow. Here we compute the expansions for
    # each power of the generator and collect together the resulting algebraic
    # terms and the rational coefficients into a matrix.

    ext = K.ext.as_expr()
    todom = K.dom.from_sympy
    toexpr = K.dom.to_sympy

    if not ext.is_Add:
        powers = [ext**n for n in range(K.mod.degree())]
    else:
        # primitive_element generates a QQ-linear combination of lower degree
        # algebraic numbers to generate the higher degree extension e.g.
        # QQ<sqrt(2)+sqrt(3)> That means that we end up having high powers of low
        # degree algebraic numbers that can be reduced. Here we will use the
        # minimal polynomials of the algebraic numbers to reduce those powers
        # before converting to Expr.
        from sympy.polys.numberfields.minpoly import minpoly

        # Decompose ext as a linear combination of gens and make a symbol for
        # each gen.
        gens, coeffs = zip(*ext.as_coefficients_dict().items())
        syms = symbols(f'a:{len(gens)}', cls=Dummy)
        sym2gen = dict(zip(syms, gens))

        # Make a polynomial ring that can express ext and minpolys of all gens
        # in terms of syms.
        R = K.dom[syms]
        monoms = [R.ring.monomial_basis(i) for i in range(R.ngens)]
        ext_dict = {m: todom(c) for m, c in zip(monoms, coeffs)}
        ext_poly = R.ring.from_dict(ext_dict)
        minpolys = [R.from_sympy(minpoly(g, s)) for s, g in sym2gen.items()]

        # Compute all powers of ext_poly reduced modulo minpolys
        powers = [R.one, ext_poly]
        for n in range(2, K.mod.degree()):
            ext_poly_n = (powers[-1] * ext_poly).rem(minpolys)
            powers.append(ext_poly_n)

        # Convert the powers back to Expr. This will recombine some things like
        # sqrt(2)*sqrt(3) -> sqrt(6).
        powers = [p.as_expr().xreplace(sym2gen) for p in powers]

    # This also expands some rational powers
    powers = [p.expand() for p in powers]

    # Collect the rational coefficients and algebraic Expr that can
    # map the ANP coefficients into an expanded SymPy expression
    terms = [dict(t.as_coeff_Mul()[::-1] for t in Add.make_args(p)) for p in powers]
    algebraics = set().union(*terms)
    matrix = [[todom(t.get(a, S.Zero)) for t in terms] for a in algebraics]

    # Create a function to do the conversion efficiently:

    def converter(a):
        """Convert a to Expr using converter"""
        ai = a.to_list()[::-1]
        coeffs_dom = [sum(mij*aj for mij, aj in zip(mi, ai)) for mi in matrix]
        coeffs_sympy = [toexpr(c) for c in coeffs_dom]
        res = Add(*(Mul(c, a) for c, a in zip(coeffs_sympy, algebraics)))
        return res

    return converter
<|endoftext|>"""Trait for implementing domain elements. """


from sympy.utilities import public

@public
class DomainElement:
    """
    Represents an element of a domain.

    Mix in this trait into a class whose instances should be recognized as
    elements of a domain. Method ``parent()`` gives that domain.
    """

    __slots__ = ()

    def parent(self):
        """Get the domain associated with ``self``

        Examples
        ========

        >>> from sympy import ZZ, symbols
        >>> x, y = symbols('x, y')
        >>> K = ZZ[x,y]
        >>> p = K(x)**2 + K(y)**2
        >>> p
        x**2 + y**2
        >>> p.parent()
        ZZ[x,y]

        Notes
        =====

        This is used by :py:meth:`~.Domain.convert` to identify the domain
        associated with a domain element.
        """
        raise NotImplementedError("abstract method")
<|endoftext|>"""Implementation of :class:`SimpleDomain` class. """


from sympy.polys.domains.domain import Domain
from sympy.utilities import public

@public
class SimpleDomain(Domain):
    """Base class for simple domains, e.g. ZZ, QQ. """

    is_Simple = True

    def inject(self, *gens):
        """Inject generators into this domain. """
        return self.poly_ring(*gens)
<|endoftext|>"""Implementation of :class:`FractionField` class. """


from sympy.polys.domains.field import Field
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.polyclasses import DMF
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public

@public
class FractionField(Field, CompositeDomain):
    """A class for representing rational function fields. """

    dtype = DMF
    is_FractionField = is_Frac = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self, dom, *gens):
        if not gens:
            raise GeneratorsNeeded("generators not specified")

        lev = len(gens) - 1
        self.ngens = len(gens)

        self.zero = self.dtype.zero(lev, dom)
        self.one = self.dtype.one(lev, dom)

        self.domain = self.dom = dom
        self.symbols = self.gens = gens

    def set_domain(self, dom):
        """Make a new fraction field with given domain. """
        return self.__class__(dom, *self.gens)

    def new(self, element):
        return self.dtype(element, self.dom, len(self.gens) - 1)

    def __str__(self):
        return str(self.dom) + '(' + ','.join(map(str, self.gens)) + ')'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom, self.gens))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, FractionField) and \
            self.dtype == other.dtype and self.dom == other.dom and self.gens == other.gens

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return (basic_from_dict(a.numer().to_sympy_dict(), *self.gens) /
                basic_from_dict(a.denom().to_sympy_dict(), *self.gens))

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        p, q = a.as_numer_denom()

        num, _ = dict_from_basic(p, gens=self.gens)
        den, _ = dict_from_basic(q, gens=self.gens)

        for k, v in num.items():
            num[k] = self.dom.from_sympy(v)

        for k, v in den.items():
            den[k] = self.dom.from_sympy(v)

        return self((num, den)).cancel()

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K1.dom.convert(a, K0))

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a ``DMF`` object to ``dtype``. """
        if K1.gens == K0.gens:
            if K1.dom == K0.dom:
                return K1(a.to_list())
            else:
                return K1(a.convert(K1.dom).to_list())
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1(dict(zip(monoms, coeffs)))

    def from_FractionField(K1, a, K0):
        """
        Convert a fraction field element to another fraction field.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMF
        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.abc import x

        >>> f = DMF(([ZZ(1), ZZ(2)], [ZZ(1), ZZ(1)]), ZZ)

        >>> QQx = QQ.old_frac_field(x)
        >>> ZZx = ZZ.old_frac_field(x)

        >>> QQx.from_FractionField(f, ZZx)
        DMF([1, 2], [1, 1], QQ)

        """
        if K1.gens == K0.gens:
            if K1.dom == K0.dom:
                return a
            else:
                return K1((a.numer().convert(K1.dom).to_list(),
                           a.denom().convert(K1.dom).to_list()))
        elif set(K0.gens).issubset(K1.gens):
            nmonoms, ncoeffs = _dict_reorder(
                a.numer().to_dict(), K0.gens, K1.gens)
            dmonoms, dcoeffs = _dict_reorder(
                a.denom().to_dict(), K0.gens, K1.gens)

            if K1.dom != K0.dom:
                ncoeffs = [ K1.dom.convert(c, K0.dom) for c in ncoeffs ]
                dcoeffs = [ K1.dom.convert(c, K0.dom) for c in dcoeffs ]

            return K1((dict(zip(nmonoms, ncoeffs)), dict(zip(dmonoms, dcoeffs))))

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        from sympy.polys.domains import PolynomialRing
        return PolynomialRing(self.dom, *self.gens)

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. `K[X]`. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. `K(X)`. """
        raise NotImplementedError('nested domains not allowed')

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return self.dom.is_positive(a.numer().LC())

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return self.dom.is_negative(a.numer().LC())

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return self.dom.is_nonpositive(a.numer().LC())

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return self.dom.is_nonnegative(a.numer().LC())

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numer()

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denom()

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return self.dtype(self.dom.factorial(a))
<|endoftext|>"""Implementation of :class:`Domain` class. """

from __future__ import annotations
from typing import Any

from sympy.core.numbers import AlgebraicNumber
from sympy.core import Basic, sympify
from sympy.core.sorting import ordered
from sympy.external.gmpy import GROUND_TYPES
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence


@public
class Domain:
    """Superclass for all domains in the polys domains system.

    See :ref:`polys-domainsintro` for an introductory explanation of the
    domains system.

    The :py:class:`~.Domain` class is an abstract base class for all of the
    concrete domain types. There are many different :py:class:`~.Domain`
    subclasses each of which has an associated ``dtype`` which is a class
    representing the elements of the domain. The coefficients of a
    :py:class:`~.Poly` are elements of a domain which must be a subclass of
    :py:class:`~.Domain`.

    Examples
    ========

    The most common example domains are the integers :ref:`ZZ` and the
    rationals :ref:`QQ`.

    >>> from sympy import Poly, symbols, Domain
    >>> x, y = symbols('x, y')
    >>> p = Poly(x**2 + y)
    >>> p
    Poly(x**2 + y, x, y, domain='ZZ')
    >>> p.domain
    ZZ
    >>> isinstance(p.domain, Domain)
    True
    >>> Poly(x**2 + y/2)
    Poly(x**2 + 1/2*y, x, y, domain='QQ')

    The domains can be used directly in which case the domain object e.g.
    (:ref:`ZZ` or :ref:`QQ`) can be used as a constructor for elements of
    ``dtype``.

    >>> from sympy import ZZ, QQ
    >>> ZZ(2)
    2
    >>> ZZ.dtype  # doctest: +SKIP
    <class 'int'>
    >>> type(ZZ(2))  # doctest: +SKIP
    <class 'int'>
    >>> QQ(1, 2)
    1/2
    >>> type(QQ(1, 2))  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    The corresponding domain elements can be used with the arithmetic
    operations ``+,-,*,**`` and depending on the domain some combination of
    ``/,//,%`` might be usable. For example in :ref:`ZZ` both ``//`` (floor
    division) and ``%`` (modulo division) can be used but ``/`` (true
    division) cannot. Since :ref:`QQ` is a :py:class:`~.Field` its elements
    can be used with ``/`` but ``//`` and ``%`` should not be used. Some
    domains have a :py:meth:`~.Domain.gcd` method.

    >>> ZZ(2) + ZZ(3)
    5
    >>> ZZ(5) // ZZ(2)
    2
    >>> ZZ(5) % ZZ(2)
    1
    >>> QQ(1, 2) / QQ(2, 3)
    3/4
    >>> ZZ.gcd(ZZ(4), ZZ(2))
    2
    >>> QQ.gcd(QQ(2,7), QQ(5,3))
    1/21
    >>> ZZ.is_Field
    False
    >>> QQ.is_Field
    True

    There are also many other domains including:

        1. :ref:`GF(p)` for finite fields of prime order.
        2. :ref:`RR` for real (floating point) numbers.
        3. :ref:`CC` for complex (floating point) numbers.
        4. :ref:`QQ(a)` for algebraic number fields.
        5. :ref:`K[x]` for polynomial rings.
        6. :ref:`K(x)` for rational function fields.
        7. :ref:`EX` for arbitrary expressions.

    Each domain is represented by a domain object and also an implementation
    class (``dtype``) for the elements of the domain. For example the
    :ref:`K[x]` domains are represented by a domain object which is an
    instance of :py:class:`~.PolynomialRing` and the elements are always
    instances of :py:class:`~.PolyElement`. The implementation class
    represents particular types of mathematical expressions in a way that is
    more efficient than a normal SymPy expression which is of type
    :py:class:`~.Expr`. The domain methods :py:meth:`~.Domain.from_sympy` and
    :py:meth:`~.Domain.to_sympy` are used to convert from :py:class:`~.Expr`
    to a domain element and vice versa.

    >>> from sympy import Symbol, ZZ, Expr
    >>> x = Symbol('x')
    >>> K = ZZ[x]           # polynomial ring domain
    >>> K
    ZZ[x]
    >>> type(K)             # class of the domain
    <class 'sympy.polys.domains.polynomialring.PolynomialRing'>
    >>> K.dtype             # doctest: +SKIP
    <class 'sympy.polys.rings.PolyElement'>
    >>> p_expr = x**2 + 1   # Expr
    >>> p_expr
    x**2 + 1
    >>> type(p_expr)
    <class 'sympy.core.add.Add'>
    >>> isinstance(p_expr, Expr)
    True
    >>> p_domain = K.from_sympy(p_expr)
    >>> p_domain            # domain element
    x**2 + 1
    >>> type(p_domain)
    <class 'sympy.polys.rings.PolyElement'>
    >>> K.to_sympy(p_domain) == p_expr
    True

    The :py:meth:`~.Domain.convert_from` method is used to convert domain
    elements from one domain to another.

    >>> from sympy import ZZ, QQ
    >>> ez = ZZ(2)
    >>> eq = QQ.convert_from(ez, ZZ)
    >>> type(ez)  # doctest: +SKIP
    <class 'int'>
    >>> type(eq)  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    Elements from different domains should not be mixed in arithmetic or other
    operations: they should be converted to a common domain first.  The domain
    method :py:meth:`~.Domain.unify` is used to find a domain that can
    represent all the elements of two given domains.

    >>> from sympy import ZZ, QQ, symbols
    >>> x, y = symbols('x, y')
    >>> ZZ.unify(QQ)
    QQ
    >>> ZZ[x].unify(QQ)
    QQ[x]
    >>> ZZ[x].unify(QQ[y])
    QQ[x,y]

    If a domain is a :py:class:`~.Ring` then is might have an associated
    :py:class:`~.Field` and vice versa. The :py:meth:`~.Domain.get_field` and
    :py:meth:`~.Domain.get_ring` methods will find or create the associated
    domain.

    >>> from sympy import ZZ, QQ, Symbol
    >>> x = Symbol('x')
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ
    >>> QQ.has_assoc_Ring
    True
    >>> QQ.get_ring()
    ZZ
    >>> K = QQ[x]
    >>> K
    QQ[x]
    >>> K.get_field()
    QQ(x)

    See also
    ========

    DomainElement: abstract base class for domain elements
    construct_domain: construct a minimal domain for some expressions

    """

    dtype: type | None = None
    """The type (class) of the elements of this :py:class:`~.Domain`:

    >>> from sympy import ZZ, QQ, Symbol
    >>> ZZ.dtype
    <class 'int'>
    >>> z = ZZ(2)
    >>> z
    2
    >>> type(z)
    <class 'int'>
    >>> type(z) == ZZ.dtype
    True

    Every domain has an associated **dtype** ("datatype") which is the
    class of the associated domain elements.

    See also
    ========

    of_type
    """

    zero: Any = None
    """The zero element of the :py:class:`~.Domain`:

    >>> from sympy import QQ
    >>> QQ.zero
    0
    >>> QQ.of_type(QQ.zero)
    True

    See also
    ========

    of_type
    one
    """

    one: Any = None
    """The one element of the :py:class:`~.Domain`:

    >>> from sympy import QQ
    >>> QQ.one
    1
    >>> QQ.of_type(QQ.one)
    True

    See also
    ========

    of_type
    zero
    """

    is_Ring = False
    """Boolean flag indicating if the domain is a :py:class:`~.Ring`.

    >>> from sympy import ZZ
    >>> ZZ.is_Ring
    True

    Basically every :py:class:`~.Domain` represents a ring so this flag is
    not that useful.

    See also
    ========

    is_PID
    is_Field
    get_ring
    has_assoc_Ring
    """

    is_Field = False
    """Boolean flag indicating if the domain is a :py:class:`~.Field`.

    >>> from sympy import ZZ, QQ
    >>> ZZ.is_Field
    False
    >>> QQ.is_Field
    True

    See also
    ========

    is_PID
    is_Ring
    get_field
    has_assoc_Field
    """

    has_assoc_Ring = False
    """Boolean flag indicating if the domain has an associated
    :py:class:`~.Ring`.

    >>> from sympy import QQ
    >>> QQ.has_assoc_Ring
    True
    >>> QQ.get_ring()
    ZZ

    See also
    ========

    is_Field
    get_ring
    """

    has_assoc_Field = False
    """Boolean flag indicating if the domain has an associated
    :py:class:`~.Field`.

    >>> from sympy import ZZ
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ

    See also
    ========

    is_Field
    get_field
    """

    is_FiniteField = is_FF = False
    is_IntegerRing = is_ZZ = False
    is_RationalField = is_QQ = False
    is_GaussianRing = is_ZZ_I = False
    is_GaussianField = is_QQ_I = False
    is_RealField = is_RR = False
    is_ComplexField = is_CC = False
    is_AlgebraicField = is_Algebraic = False
    is_PolynomialRing = is_Poly = False
    is_FractionField = is_Frac = False
    is_SymbolicDomain = is_EX = False
    is_SymbolicRawDomain = is_EXRAW = False
    is_FiniteExtension = False

    is_Exact = True
    is_Numerical = False

    is_Simple = False
    is_Composite = False

    is_PID = False
    """Boolean flag indicating if the domain is a `principal ideal domain`_.

    >>> from sympy import ZZ
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ

    .. _principal ideal domain: https://en.wikipedia.org/wiki/Principal_ideal_domain

    See also
    ========

    is_Field
    get_field
    """

    has_CharacteristicZero = False

    rep: str | None = None
    alias: str | None = None

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return self.rep

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype))

    def new(self, *args):
        return self.dtype(*args)

    @property
    def tp(self):
        """Alias for :py:attr:`~.Domain.dtype`"""
        return self.dtype

    def __call__(self, *args):
        """Construct an element of ``self`` domain from ``args``. """
        return self.new(*args)

    def normal(self, *args):
        return self.dtype(*args)

    def convert_from(self, element, base):
        """Convert ``element`` to ``self.dtype`` given the base domain. """
        if base.alias is not None:
            method = "from_" + base.alias
        else:
            method = "from_" + base.__class__.__name__

        _convert = getattr(self, method)

        if _convert is not None:
            result = _convert(element, base)

            if result is not None:
                return result

        raise CoercionFailed("Cannot convert %s of type %s from %s to %s" % (element, type(element), base, self))

    def convert(self, element, base=None):
        """Convert ``element`` to ``self.dtype``. """

        if base is not None:
            if _not_a_coeff(element):
                raise CoercionFailed('%s is not in any domain' % element)
            return self.convert_from(element, base)

        if self.of_type(element):
            return element

        if _not_a_coeff(element):
            raise CoercionFailed('%s is not in any domain' % element)

        from sympy.polys.domains import ZZ, QQ, RealField, ComplexField

        if ZZ.of_type(element):
            return self.convert_from(element, ZZ)

        if isinstance(element, int):
            return self.convert_from(ZZ(element), ZZ)

        if GROUND_TYPES != 'python':
            if isinstance(element, ZZ.tp):
                return self.convert_from(element, ZZ)
            if isinstance(element, QQ.tp):
                return self.convert_from(element, QQ)

        if isinstance(element, float):
            parent = RealField()
            return self.convert_from(parent(element), parent)

        if isinstance(element, complex):
            parent = ComplexField()
            return self.convert_from(parent(element), parent)

        if type(element).__name__ == 'mpf':
            parent = RealField()
            return self.convert_from(parent(element), parent)

        if type(element).__name__ == 'mpc':
            parent = ComplexField()
            return self.convert_from(parent(element), parent)

        if isinstance(element, DomainElement):
            return self.convert_from(element, element.parent())

        # TODO: implement this in from_ methods
        if self.is_Numerical and getattr(element, 'is_ground', False):
            return self.convert(element.LC())

        if isinstance(element, Basic):
            try:
                return self.from_sympy(element)
            except (TypeError, ValueError):
                pass
        else: # TODO: remove this branch
            if not is_sequence(element):
                try:
                    element = sympify(element, strict=True)
                    if isinstance(element, Basic):
                        return self.from_sympy(element)
                except (TypeError, ValueError):
                    pass

        raise CoercionFailed("Cannot convert %s of type %s to %s" % (element, type(element), self))

    def of_type(self, element):
        """Check if ``a`` is of type ``dtype``. """
        return isinstance(element, self.tp)

    def __contains__(self, a):
        """Check if ``a`` belongs to this domain. """
        try:
            if _not_a_coeff(a):
                raise CoercionFailed
            self.convert(a)  # this might raise, too
        except CoercionFailed:
            return False

        return True

    def to_sympy(self, a):
        """Convert domain element *a* to a SymPy expression (Expr).

        Explanation
        ===========

        Convert a :py:class:`~.Domain` element *a* to :py:class:`~.Expr`. Most
        public SymPy functions work with objects of type :py:class:`~.Expr`.
        The elements of a :py:class:`~.Domain` have a different internal
        representation. It is not possible to mix domain elements with
        :py:class:`~.Expr` so each domain has :py:meth:`~.Domain.to_sympy` and
        :py:meth:`~.Domain.from_sympy` methods to convert its domain elements
        to and from :py:class:`~.Expr`.

        Parameters
        ==========

        a: domain element
            An element of this :py:class:`~.Domain`.

        Returns
        =======

        expr: Expr
            A normal SymPy expression of type :py:class:`~.Expr`.

        Examples
        ========

        Construct an element of the :ref:`QQ` domain and then convert it to
        :py:class:`~.Expr`.

        >>> from sympy import QQ, Expr
        >>> q_domain = QQ(2)
        >>> q_domain
        2
        >>> q_expr = QQ.to_sympy(q_domain)
        >>> q_expr
        2

        Although the printed forms look similar these objects are not of the
        same type.

        >>> isinstance(q_domain, Expr)
        False
        >>> isinstance(q_expr, Expr)
        True

        Construct an element of :ref:`K[x]` and convert to
        :py:class:`~.Expr`.

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> K = QQ[x]
        >>> x_domain = K.gens[0]  # generator x as a domain element
        >>> p_domain = x_domain**2/3 + 1
        >>> p_domain
        1/3*x**2 + 1
        >>> p_expr = K.to_sympy(p_domain)
        >>> p_expr
        x**2/3 + 1

        The :py:meth:`~.Domain.from_sympy` method is used for the opposite
        conversion from a normal SymPy expression to a domain element.

        >>> p_domain == p_expr
        False
        >>> K.from_sympy(p_expr) == p_domain
        True
        >>> K.to_sympy(p_domain) == p_expr
        True
        >>> K.from_sympy(K.to_sympy(p_domain)) == p_domain
        True
        >>> K.to_sympy(K.from_sympy(p_expr)) == p_expr
        True

        The :py:meth:`~.Domain.from_sympy` method makes it easier to construct
        domain elements interactively.

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> K = QQ[x]
        >>> K.from_sympy(x**2/3 + 1)
        1/3*x**2 + 1

        See also
        ========

        from_sympy
        convert_from
        """
        raise NotImplementedError

    def from_sympy(self, a):
        """Convert a SymPy expression to an element of this domain.

        Explanation
        ===========

        See :py:meth:`~.Domain.to_sympy` for explanation and examples.

        Parameters
        ==========

        expr: Expr
            A normal SymPy expression of type :py:class:`~.Expr`.

        Returns
        =======

        a: domain element
            An element of this :py:class:`~.Domain`.

        See also
        ========

        to_sympy
        convert_from
        """
        raise NotImplementedError

    def sum(self, args):
        return sum(args, start=self.zero)

    def from_FF(K1, a, K0):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return None

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return None

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return None

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return None

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to ``dtype``. """
        return None

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return None

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return None

    def from_RealField(K1, a, K0):
        """Convert a real element object to ``dtype``. """
        return None

    def from_ComplexField(K1, a, K0):
        """Convert a complex element to ``dtype``. """
        return None

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        return None

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        if a.is_ground:
            return K1.convert(a.LC, K0.dom)

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        return None

    def from_MonogenicFiniteExtension(K1, a, K0):
        """Convert an ``ExtensionElement`` to ``dtype``. """
        return K1.convert_from(a.rep, K0.ring)

    def from_ExpressionDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return K1.from_sympy(a.ex)

    def from_ExpressionRawDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return K1.from_sympy(a)

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        if a.degree() <= 0:
            return K1.convert(a.LC(), K0.dom)

    def from_GeneralizedPolynomialRing(K1, a, K0):
        return K1.from_FractionField(a, K0)

    def unify_with_symbols(K0, K1, symbols):
        if (K0.is_Composite and (set(K0.symbols) & set(symbols))) or (K1.is_Composite and (set(K1.symbols) & set(symbols))):
            raise UnificationFailed("Cannot unify %s with %s, given %s generators" % (K0, K1, tuple(symbols)))

        return K0.unify(K1)

    def unify_composite(K0, K1):
        """Unify two domains where at least one is composite."""
        K0_ground = K0.dom if K0.is_Composite else K0
        K1_ground = K1.dom if K1.is_Composite else K1

        K0_symbols = K0.symbols if K0.is_Composite else ()
        K1_symbols = K1.symbols if K1.is_Composite else ()

        domain = K0_ground.unify(K1_ground)
        symbols = _unify_gens(K0_symbols, K1_symbols)
        order = K0.order if K0.is_Composite else K1.order

        # E.g. ZZ[x].unify(QQ.frac_field(x)) -> ZZ.frac_field(x)
        if ((K0.is_FractionField and K1.is_PolynomialRing or
             K1.is_FractionField and K0.is_PolynomialRing) and
             (not K0_ground.is_Field or not K1_ground.is_Field) and domain.is_Field
             and domain.has_assoc_Ring):
            domain = domain.get_ring()

        if K0.is_Composite and (not K1.is_Composite or K0.is_FractionField or K1.is_PolynomialRing):
            cls = K0.__class__
        else:
            cls = K1.__class__

        # Here cls might be PolynomialRing, FractionField, GlobalPolynomialRing
        # (dense/old Polynomialring) or dense/old FractionField.

        from sympy.polys.domains.old_polynomialring import GlobalPolynomialRing
        if cls == GlobalPolynomialRing:
            return cls(domain, symbols)

        return cls(domain, symbols, order)

    def unify(K0, K1, symbols=None):
        """
        Construct a minimal domain that contains elements of ``K0`` and ``K1``.

        Known domains (from smallest to largest):

        - ``GF(p)``
        - ``ZZ``
        - ``QQ``
        - ``RR(prec, tol)``
        - ``CC(prec, tol)``
        - ``ALG(a, b, c)``
        - ``K[x, y, z]``
        - ``K(x, y, z)``
        - ``EX``

        """
        if symbols is not None:
            return K0.unify_with_symbols(K1, symbols)

        if K0 == K1:
            return K0

        if not (K0.has_CharacteristicZero and K1.has_CharacteristicZero):
            # Reject unification of domains with different characteristics.
            if K0.characteristic() != K1.characteristic():
                raise UnificationFailed("Cannot unify %s with %s" % (K0, K1))

            # We do not get here if K0 == K1. The two domains have the same
            # characteristic but are unequal so at least one is composite and
            # we are unifying something like GF(3).unify(GF(3)[x]).
            return K0.unify_composite(K1)

        # From here we know both domains have characteristic zero and it can be
        # acceptable to fall back on EX.

        if K0.is_EXRAW:
            return K0
        if K1.is_EXRAW:
            return K1

        if K0.is_EX:
            return K0
        if K1.is_EX:
            return K1

        if K0.is_FiniteExtension or K1.is_FiniteExtension:
            if K1.is_FiniteExtension:
                K0, K1 = K1, K0
            if K1.is_FiniteExtension:
                # Unifying two extensions.
                # Try to ensure that K0.unify(K1) == K1.unify(K0)
                if list(ordered([K0.modulus, K1.modulus]))[1] == K0.modulus:
                    K0, K1 = K1, K0
                return K1.set_domain(K0)
            else:
                # Drop the generator from other and unify with the base domain
                K1 = K1.drop(K0.symbol)
                K1 = K0.domain.unify(K1)
                return K0.set_domain(K1)

        if K0.is_Composite or K1.is_Composite:
            return K0.unify_composite(K1)

        if K1.is_ComplexField:
            K0, K1 = K1, K0
        if K0.is_ComplexField:
            if K1.is_ComplexField or K1.is_RealField:
                if K0.precision >= K1.precision:
                    return K0
                else:
                    from sympy.polys.domains.complexfield import ComplexField
                    return ComplexField(prec=K1.precision)
            else:
                return K0

        if K1.is_RealField:
            K0, K1 = K1, K0
        if K0.is_RealField:
            if K1.is_RealField:
                if K0.precision >= K1.precision:
                    return K0
                else:
                    return K1
            elif K1.is_GaussianRing or K1.is_GaussianField:
                from sympy.polys.domains.complexfield import ComplexField
                return ComplexField(prec=K0.precision)
            else:
                return K0

        if K1.is_AlgebraicField:
            K0, K1 = K1, K0
        if K0.is_AlgebraicField:
            if K1.is_GaussianRing:
                K1 = K1.get_field()
            if K1.is_GaussianField:
                K1 = K1.as_AlgebraicField()
            if K1.is_AlgebraicField:
                return K0.__class__(K0.dom.unify(K1.dom), *_unify_gens(K0.orig_ext, K1.orig_ext))
            else:
                return K0

        if K0.is_GaussianField:
            return K0
        if K1.is_GaussianField:
            return K1

        if K0.is_GaussianRing:
            if K1.is_RationalField:
                K0 = K0.get_field()
            return K0
        if K1.is_GaussianRing:
            if K0.is_RationalField:
                K1 = K1.get_field()
            return K1

        if K0.is_RationalField:
            return K0
        if K1.is_RationalField:
            return K1

        if K0.is_IntegerRing:
            return K0
        if K1.is_IntegerRing:
            return K1

        from sympy.polys.domains import EX
        return EX

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        # XXX: Remove this.
        return isinstance(other, Domain) and self.dtype == other.dtype

    def __ne__(self, other):
        """Returns ``False`` if two domains are equivalent. """
        return not self == other

    def map(self, seq):
        """Rersively apply ``self`` to all elements of ``seq``. """
        result = []

        for elt in seq:
            if isinstance(elt, list):
                result.append(self.map(elt))
            else:
                result.append(self(elt))

        return result

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError('there is no ring associated with %s' % self)

    def get_field(self):
        """Returns a field associated with ``self``. """
        raise DomainError('there is no field associated with %s' % self)

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        return self

    def __getitem__(self, symbols):
        """The mathematical way to make a polynomial ring. """
        if hasattr(symbols, '__iter__'):
            return self.poly_ring(*symbols)
        else:
            return self.poly_ring(symbols)

    def poly_ring(self, *symbols, order=lex):
        """Returns a polynomial ring, i.e. `K[X]`. """
        from sympy.polys.domains.polynomialring import PolynomialRing
        return PolynomialRing(self, symbols, order)

    def frac_field(self, *symbols, order=lex):
        """Returns a fraction field, i.e. `K(X)`. """
        from sympy.polys.domains.fractionfield import FractionField
        return FractionField(self, symbols, order)

    def old_poly_ring(self, *symbols, **kwargs):
        """Returns a polynomial ring, i.e. `K[X]`. """
        from sympy.polys.domains.old_polynomialring import PolynomialRing
        return PolynomialRing(self, *symbols, **kwargs)

    def old_frac_field(self, *symbols, **kwargs):
        """Returns a fraction field, i.e. `K(X)`. """
        from sympy.polys.domains.old_fractionfield import FractionField
        return FractionField(self, *symbols, **kwargs)

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `K(\alpha, \ldots)`. """
        raise DomainError("Cannot create algebraic field over %s" % self)

    def alg_field_from_poly(self, poly, alias=None, root_index=-1):
        r"""
        Convenience method to construct an algebraic extension on a root of a
        polynomial, chosen by root index.

        Parameters
        ==========

        poly : :py:class:`~.Poly`
            The polynomial whose root generates the extension.
        alias : str, optional (default=None)
            Symbol name for the generator of the extension.
            E.g. "alpha" or "theta".
        root_index : int, optional (default=-1)
            Specifies which root of the polynomial is desired. The ordering is
            as defined by the :py:class:`~.ComplexRootOf` class. The default of
            ``-1`` selects the most natural choice in the common cases of
            quadratic and cyclotomic fields (the square root on the positive
            real or imaginary axis, resp. $\mathrm{e}^{2\pi i/n}$).

        Examples
        ========

        >>> from sympy import QQ, Poly
        >>> from sympy.abc import x
        >>> f = Poly(x**2 - 2)
        >>> K = QQ.alg_field_from_poly(f)
        >>> K.ext.minpoly == f
        True
        >>> g = Poly(8*x**3 - 6*x - 1)
        >>> L = QQ.alg_field_from_poly(g, "alpha")
        >>> L.ext.minpoly == g
        True
        >>> L.to_sympy(L([1, 1, 1]))
        alpha**2 + alpha + 1

        """
        from sympy.polys.rootoftools import CRootOf
        root = CRootOf(poly, root_index)
        alpha = AlgebraicNumber(root, alias=alias)
        return self.algebraic_field(alpha, alias=alias)

    def cyclotomic_field(self, n, ss=False, alias="zeta", gen=None, root_index=-1):
        r"""
        Convenience method to construct a cyclotomic field.

        Parameters
        ==========

        n : int
            Construct the nth cyclotomic field.
        ss : boolean, optional (default=False)
            If True, append *n* as a subscript on the alias string.
        alias : str, optional (default="zeta")
            Symbol name for the generator.
        gen : :py:class:`~.Symbol`, optional (default=None)
            Desired variable for the cyclotomic polynomial that defines the
            field. If ``None``, a dummy variable will be used.
        root_index : int, optional (default=-1)
            Specifies which root of the polynomial is desired. The ordering is
            as defined by the :py:class:`~.ComplexRootOf` class. The default of
            ``-1`` selects the root $\mathrm{e}^{2\pi i/n}$.

        Examples
        ========

        >>> from sympy import QQ, latex
        >>> K = QQ.cyclotomic_field(5)
        >>> K.to_sympy(K([-1, 1]))
        1 - zeta
        >>> L = QQ.cyclotomic_field(7, True)
        >>> a = L.to_sympy(L([-1, 1]))
        >>> print(a)
        1 - zeta7
        >>> print(latex(a))
        1 - \zeta_{7}

        """
        from sympy.polys.specialpolys import cyclotomic_poly
        if ss:
            alias += str(n)
        return self.alg_field_from_poly(cyclotomic_poly(n, gen), alias=alias,
                                        root_index=root_index)

    def inject(self, *symbols):
        """Inject generators into this domain. """
        raise NotImplementedError

    def drop(self, *symbols):
        """Drop generators from this domain. """
        if self.is_Simple:
            return self
        raise NotImplementedError  # pragma: no cover

    def is_zero(self, a):
        """Returns True if ``a`` is zero. """
        return not a

    def is_one(self, a):
        """Returns True if ``a`` is one. """
        return a == self.one

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return a > 0

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return a < 0

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return a <= 0

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return a >= 0

    def canonical_unit(self, a):
        if self.is_negative(a):
            return -self.one
        else:
            return self.one

    def abs(self, a):
        """Absolute value of ``a``, implies ``__abs__``. """
        return abs(a)

    def neg(self, a):
        """Returns ``a`` negated, implies ``__neg__``. """
        return -a

    def pos(self, a):
        """Returns ``a`` positive, implies ``__pos__``. """
        return +a

    def add(self, a, b):
        """Sum of ``a`` and ``b``, implies ``__add__``.  """
        return a + b

    def sub(self, a, b):
        """Difference of ``a`` and ``b``, implies ``__sub__``.  """
        return a - b

    def mul(self, a, b):
        """Product of ``a`` and ``b``, implies ``__mul__``.  """
        return a * b

    def pow(self, a, b):
        """Raise ``a`` to power ``b``, implies ``__pow__``.  """
        return a ** b

    def exquo(self, a, b):
        """Exact quotient of *a* and *b*. Analogue of ``a / b``.

        Explanation
        ===========

        This is essentially the same as ``a / b`` except that an error will be
        raised if the division is inexact (if there is any remainder) and the
        result will always be a domain element. When working in a
        :py:class:`~.Domain` that is not a :py:class:`~.Field` (e.g. :ref:`ZZ`
        or :ref:`K[x]`) ``exquo`` should be used instead of ``/``.

        The key invariant is that if ``q = K.exquo(a, b)`` (and ``exquo`` does
        not raise an exception) then ``a == b*q``.

        Examples
        ========

        We can use ``K.exquo`` instead of ``/`` for exact division.

        >>> from sympy import ZZ
        >>> ZZ.exquo(ZZ(4), ZZ(2))
        2
        >>> ZZ.exquo(ZZ(5), ZZ(2))
        Traceback (most recent call last):
            ...
        ExactQuotientFailed: 2 does not divide 5 in ZZ

        Over a :py:class:`~.Field` such as :ref:`QQ`, division (with nonzero
        divisor) is always exact so in that case ``/`` can be used instead of
        :py:meth:`~.Domain.exquo`.

        >>> from sympy import QQ
        >>> QQ.exquo(QQ(5), QQ(2))
        5/2
        >>> QQ(5) / QQ(2)
        5/2

        Parameters
        ==========

        a: domain element
            The dividend
        b: domain element
            The divisor

        Returns
        =======

        q: domain element
            The exact quotient

        Raises
        ======

        ExactQuotientFailed: if exact division is not possible.
        ZeroDivisionError: when the divisor is zero.

        See also
        ========

        quo: Analogue of ``a // b``
        rem: Analogue of ``a % b``
        div: Analogue of ``divmod(a, b)``

        Notes
        =====

        Since the default :py:attr:`~.Domain.dtype` for :ref:`ZZ` is ``int``
        (or ``mpz``) division as ``a / b`` should not be used as it would give
        a ``float`` which is not a domain element.

        >>> ZZ(4) / ZZ(2) # doctest: +SKIP
        2.0
        >>> ZZ(5) / ZZ(2) # doctest: +SKIP
        2.5

        On the other hand with `SYMPY_GROUND_TYPES=flint` elements of :ref:`ZZ`
        are ``flint.fmpz`` and division would raise an exception:

        >>> ZZ(4) / ZZ(2) # doctest: +SKIP
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for /: 'fmpz' and 'fmpz'

        Using ``/`` with :ref:`ZZ` will lead to incorrect results so
        :py:meth:`~.Domain.exquo` should be used instead.

        """
        raise NotImplementedError

    def quo(self, a, b):
        """Quotient of *a* and *b*. Analogue of ``a // b``.

        ``K.quo(a, b)`` is equivalent to ``K.div(a, b)[0]``. See
        :py:meth:`~.Domain.div` for more explanation.

        See also
        ========

        rem: Analogue of ``a % b``
        div: Analogue of ``divmod(a, b)``
        exquo: Analogue of ``a / b``
        """
        raise NotImplementedError

    def rem(self, a, b):
        """Modulo division of *a* and *b*. Analogue of ``a % b``.

        ``K.rem(a, b)`` is equivalent to ``K.div(a, b)[1]``. See
        :py:meth:`~.Domain.div` for more explanation.

        See also
        ========

        quo: Analogue of ``a // b``
        div: Analogue of ``divmod(a, b)``
        exquo: Analogue of ``a / b``
        """
        raise NotImplementedError

    def div(self, a, b):
        """Quotient and remainder for *a* and *b*. Analogue of ``divmod(a, b)``

        Explanation
        ===========

        This is essentially the same as ``divmod(a, b)`` except that is more
        consistent when working over some :py:class:`~.Field` domains such as
        :ref:`QQ`. When working over an arbitrary :py:class:`~.Domain` the
        :py:meth:`~.Domain.div` method should be used instead of ``divmod``.

        The key invariant is that if ``q, r = K.div(a, b)`` then
        ``a == b*q + r``.

        The result of ``K.div(a, b)`` is the same as the tuple
        ``(K.quo(a, b), K.rem(a, b))`` except that if both quotient and
        remainder are needed then it is more efficient to use
        :py:meth:`~.Domain.div`.

        Examples
        ========

        We can use ``K.div`` instead of ``divmod`` for floor division and
        remainder.

        >>> from sympy import ZZ, QQ
        >>> ZZ.div(ZZ(5), ZZ(2))
        (2, 1)

        If ``K`` is a :py:class:`~.Field` then the division is always exact
        with a remainder of :py:attr:`~.Domain.zero`.

        >>> QQ.div(QQ(5), QQ(2))
        (5/2, 0)

        Parameters
        ==========

        a: domain element
            The dividend
        b: domain element
            The divisor

        Returns
        =======

        (q, r): tuple of domain elements
            The quotient and remainder

        Raises
        ======

        ZeroDivisionError: when the divisor is zero.

        See also
        ========

        quo: Analogue of ``a // b``
        rem: Analogue of ``a % b``
        exquo: Analogue of ``a / b``

        Notes
        =====

        If ``gmpy`` is installed then the ``gmpy.mpq`` type will be used as
        the :py:attr:`~.Domain.dtype` for :ref:`QQ`. The ``gmpy.mpq`` type
        defines ``divmod`` in a way that is undesirable so
        :py:meth:`~.Domain.div` should be used instead of ``divmod``.

        >>> a = QQ(1)
        >>> b = QQ(3, 2)
        >>> a               # doctest: +SKIP
        mpq(1,1)
        >>> b               # doctest: +SKIP
        mpq(3,2)
        >>> divmod(a, b)    # doctest: +SKIP
        (mpz(0), mpq(1,1))
        >>> QQ.div(a, b)    # doctest: +SKIP
        (mpq(2,3), mpq(0,1))

        Using ``//`` or ``%`` with :ref:`QQ` will lead to incorrect results so
        :py:meth:`~.Domain.div` should be used instead.

        """
        raise NotImplementedError

    def invert(self, a, b):
        """Returns inversion of ``a mod b``, implies something. """
        raise NotImplementedError

    def revert(self, a):
        """Returns ``a**(-1)`` if possible. """
        raise NotImplementedError

    def numer(self, a):
        """Returns numerator of ``a``. """
        raise NotImplementedError

    def denom(self, a):
        """Returns denominator of ``a``. """
        raise NotImplementedError

    def half_gcdex(self, a, b):
        """Half extended GCD of ``a`` and ``b``. """
        s, t, h = self.gcdex(a, b)
        return s, h

    def gcdex(self, a, b):
        """Extended GCD of ``a`` and ``b``. """
        raise NotImplementedError

    def cofactors(self, a, b):
        """Returns GCD and cofactors of ``a`` and ``b``. """
        gcd = self.gcd(a, b)
        cfa = self.quo(a, gcd)
        cfb = self.quo(b, gcd)
        return gcd, cfa, cfb

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        raise NotImplementedError

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        raise NotImplementedError

    def log(self, a, b):
        """Returns b-base logarithm of ``a``. """
        raise NotImplementedError

    def sqrt(self, a):
        """Returns a (possibly inexact) square root of ``a``.

        Explanation
        ===========
        There is no universal definition of "inexact square root" for all
        domains. It is not recommended to implement this method for domains
        other then :ref:`ZZ`.

        See also
        ========
        exsqrt
        """
        raise NotImplementedError

    def is_square(self, a):
        """Returns whether ``a`` is a square in the domain.

        Explanation
        ===========
        Returns ``True`` if there is an element ``b`` in the domain such that
        ``b * b == a``, otherwise returns ``False``. For inexact domains like
        :ref:`RR` and :ref:`CC`, a tiny difference in this equality can be
        tolerated.

        See also
        ========
        exsqrt
        """
        raise NotImplementedError

    def exsqrt(self, a):
        """Principal square root of a within the domain if ``a`` is square.

        Explanation
        ===========
        The implementation of this method should return an element ``b`` in the
        domain such that ``b * b == a``, or ``None`` if there is no such ``b``.
        For inexact domains like :ref:`RR` and :ref:`CC`, a tiny difference in
        this equality can be tolerated. The choice of a "principal" square root
        should follow a consistent rule whenever possible.

        See also
        ========
        sqrt, is_square
        """
        raise NotImplementedError

    def evalf(self, a, prec=None, **options):
        """Returns numerical approximation of ``a``. """
        return self.to_sympy(a).evalf(prec, **options)

    n = evalf

    def real(self, a):
        return a

    def imag(self, a):
        return self.zero

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return a == b

    def characteristic(self):
        """Return the characteristic of this domain. """
        raise NotImplementedError('characteristic()')


__all__ = ['Domain']
<|endoftext|>"""Implementation of :class:`ExpressionRawDomain` class. """


from sympy.core import Expr, S, sympify, Add
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public


@public
class ExpressionRawDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions but without automatic simplification. """

    is_SymbolicRawDomain = is_EXRAW = True

    dtype = Expr

    zero = S.Zero
    one = S.One

    rep = 'EXRAW'

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        pass

    @classmethod
    def new(self, a):
        return sympify(a)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        if not isinstance(a, Expr):
            raise CoercionFailed(f"Expecting an Expr instance but found: {type(a).__name__}")
        return a

    def convert_from(self, a, K):
        """Convert a domain element from another domain to EXRAW"""
        return K.to_sympy(a)

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def sum(self, items):
        return Add(*items)


EXRAW = ExpressionRawDomain()
<|endoftext|>"""Implementation of :class:`ModularInteger` class. """

from __future__ import annotations
from typing import Any

import operator

from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domainelement import DomainElement

from sympy.utilities import public
from sympy.utilities.exceptions import sympy_deprecation_warning

@public
class ModularInteger(PicklableWithSlots, DomainElement):
    """A class representing a modular integer. """

    mod, dom, sym, _parent = None, None, None, None

    __slots__ = ('val',)

    def parent(self):
        return self._parent

    def __init__(self, val):
        if isinstance(val, self.__class__):
            self.val = val.val % self.mod
        else:
            self.val = self.dom.convert(val) % self.mod

    def modulus(self):
        return self.mod

    def __hash__(self):
        return hash((self.val, self.mod))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.val)

    def __str__(self):
        return "%s mod %s" % (self.val, self.mod)

    def __int__(self):
        return int(self.val)

    def to_int(self):

        sympy_deprecation_warning(
            """ModularInteger.to_int() is deprecated.

            Use int(a) or K = GF(p) and K.to_int(a) instead of a.to_int().
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="modularinteger-to-int",
        )

        if self.sym:
            if self.val <= self.mod // 2:
                return self.val
            else:
                return self.val - self.mod
        else:
            return self.val

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(-self.val)

    @classmethod
    def _get_val(cls, other):
        if isinstance(other, cls):
            return other.val
        else:
            try:
                return cls.dom.convert(other)
            except CoercionFailed:
                return None

    def __add__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val + val)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val - val)
        else:
            return NotImplemented

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val * val)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val * self._invert(val))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        return self.invert().__mul__(other)

    def __mod__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val % val)
        else:
            return NotImplemented

    def __rmod__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(val % self.val)
        else:
            return NotImplemented

    def __pow__(self, exp):
        if not exp:
            return self.__class__(self.dom.one)

        if exp < 0:
            val, exp = self.invert().val, -exp
        else:
            val = self.val

        return self.__class__(pow(val, int(exp), self.mod))

    def _compare(self, other, op):
        val = self._get_val(other)

        if val is None:
            return NotImplemented

        return op(self.val, val % self.mod)

    def _compare_deprecated(self, other, op):
        val = self._get_val(other)

        if val is None:
            return NotImplemented

        sympy_deprecation_warning(
            """Ordered comparisons with modular integers are deprecated.

            Use e.g. int(a) < int(b) instead of a < b.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="modularinteger-compare",
            stacklevel=4,
        )

        return op(self.val, val % self.mod)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def __lt__(self, other):
        return self._compare_deprecated(other, operator.lt)

    def __le__(self, other):
        return self._compare_deprecated(other, operator.le)

    def __gt__(self, other):
        return self._compare_deprecated(other, operator.gt)

    def __ge__(self, other):
        return self._compare_deprecated(other, operator.ge)

    def __bool__(self):
        return bool(self.val)

    @classmethod
    def _invert(cls, value):
        return cls.dom.invert(value, cls.mod)

    def invert(self):
        return self.__class__(self._invert(self.val))

_modular_integer_cache: dict[tuple[Any, Any, Any], type[ModularInteger]] = {}

def ModularIntegerFactory(_mod, _dom, _sym, parent):
    """Create custom class for specific integer modulus."""
    try:
        _mod = _dom.convert(_mod)
    except CoercionFailed:
        ok = False
    else:
        ok = True

    if not ok or _mod < 1:
        raise ValueError("modulus must be a positive integer, got %s" % _mod)

    key = _mod, _dom, _sym

    try:
        cls = _modular_integer_cache[key]
    except KeyError:
        class cls(ModularInteger):
            mod, dom, sym = _mod, _dom, _sym
            _parent = parent

        if _sym:
            cls.__name__ = "SymmetricModularIntegerMod%s" % _mod
        else:
            cls.__name__ = "ModularIntegerMod%s" % _mod

        _modular_integer_cache[key] = cls

    return cls
<|endoftext|>"""Implementation of :class:`PythonFiniteField` class. """


from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.pythonintegerring import PythonIntegerRing

from sympy.utilities import public

@public
class PythonFiniteField(FiniteField):
    """Finite field based on Python's integers. """

    alias = 'FF_python'

    def __init__(self, mod, symmetric=True):
        super().__init__(mod, PythonIntegerRing(), symmetric)
<|endoftext|>"""Implementation of :class:`PolynomialRing` class. """


from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
        CoercionFailed, ExactQuotientFailed, NotReversible)
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable


@public
class PolynomialRingBase(Ring, CompositeDomain):
    """
    Base class for generalized polynomial rings.

    This base class should be used for uniform access to generalized polynomial
    rings. Subclasses only supply information about the element storage etc.

    Do not instantiate.
    """

    has_assoc_Ring = True
    has_assoc_Field = True

    default_order = "grevlex"

    def __init__(self, dom, *gens, **opts):
        if not gens:
            raise GeneratorsNeeded("generators not specified")

        lev = len(gens) - 1
        self.ngens = len(gens)

        self.zero = self.dtype.zero(lev, dom)
        self.one = self.dtype.one(lev, dom)

        self.domain = self.dom = dom
        self.symbols = self.gens = gens
        # NOTE 'order' may not be set if inject was called through CompositeDomain
        self.order = opts.get('order', monomial_key(self.default_order))

    def set_domain(self, dom):
        """Return a new polynomial ring with given domain. """
        return self.__class__(dom, *self.gens, order=self.order)

    def new(self, element):
        return self.dtype(element, self.dom, len(self.gens) - 1)

    def _ground_new(self, element):
        return self.one.ground_new(element)

    def _from_dict(self, element):
        return DMP.from_dict(element, len(self.gens) - 1, self.dom)

    def __str__(self):
        s_order = str(self.order)
        orderstr = (
            " order=" + s_order) if s_order != self.default_order else ""
        return str(self.dom) + '[' + ','.join(map(str, self.gens)) + orderstr + ']'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom,
                     self.gens, self.order))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, PolynomialRingBase) and \
            self.dtype == other.dtype and self.dom == other.dom and \
            self.gens == other.gens and self.order == other.order

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert a ``ANP`` object to ``dtype``. """
        if K1.dom == K0:
            return K1._ground_new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``PolyElement`` object to ``dtype``. """
        if K1.gens == K0.symbols:
            if K1.dom == K0.dom:
                return K1(dict(a))  # set the correct ring
            else:
                convert_dom = lambda c: K1.dom.convert_from(c, K0.dom)
                return K1._from_dict({m: convert_dom(c) for m, c in a.items()})
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.symbols, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1._from_dict(dict(zip(monoms, coeffs)))

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        if K1.gens == K0.gens:
            if K1.dom != K0.dom:
                a = a.convert(K1.dom)
            return K1(a.to_list())
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1(dict(zip(monoms, coeffs)))

    def get_field(self):
        """Returns a field associated with ``self``. """
        return FractionField(self.dom, *self.gens)

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        try:
            return self.exquo(self.one, a)
        except (ExactQuotientFailed, ZeroDivisionError):
            raise NotReversible('%s is not a unit' % a)

    def gcdex(self, a, b):
        """Extended GCD of ``a`` and ``b``. """
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a.lcm(b)

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return self.dtype(self.dom.factorial(a))

    def _vector_to_sdm(self, v, order):
        """
        For internal use by the modules class.

        Convert an iterable of elements of this ring into a sparse distributed
        module element.
        """
        raise NotImplementedError

    def _sdm_to_dics(self, s, n):
        """Helper for _sdm_to_vector."""
        from sympy.polys.distributedmodules import sdm_to_dict
        dic = sdm_to_dict(s)
        res = [{} for _ in range(n)]
        for k, v in dic.items():
            res[k[0]][k[1:]] = v
        return res

    def _sdm_to_vector(self, s, n):
        """
        For internal use by the modules class.

        Convert a sparse distributed module into a list of length ``n``.

        Examples
        ========

        >>> from sympy import QQ, ilex
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> L = [((1, 1, 1), QQ(1)), ((0, 1, 0), QQ(1)), ((0, 0, 1), QQ(2))]
        >>> R._sdm_to_vector(L, 2)
        [DMF([[1], [2, 0]], [[1]], QQ), DMF([[1, 0], []], [[1]], QQ)]
        """
        dics = self._sdm_to_dics(s, n)
        # NOTE this works for global and local rings!
        return [self(x) for x in dics]

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2)
        QQ[x]**2
        """
        return FreeModulePolyRing(self, rank)


def _vector_to_sdm_helper(v, order):
    """Helper method for common code in Global and Local poly rings."""
    from sympy.polys.distributedmodules import sdm_from_dict
    d = {}
    for i, e in enumerate(v):
        for key, value in e.to_dict().items():
            d[(i,) + key] = value
    return sdm_from_dict(d, order)


@public
class GlobalPolynomialRing(PolynomialRingBase):
    """A true polynomial ring, with objects DMP. """

    is_PolynomialRing = is_Poly = True
    dtype = DMP

    def new(self, element):
        if isinstance(element, dict):
            return DMP.from_dict(element, len(self.gens) - 1, self.dom)
        elif element in self.dom:
            return self._ground_new(self.dom.convert(element))
        else:
            return self.dtype(element, self.dom, len(self.gens) - 1)

    def from_FractionField(K1, a, K0):
        """
        Convert a ``DMF`` object to ``DMP``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP, DMF
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.abc import x

        >>> f = DMF(([ZZ(1), ZZ(1)], [ZZ(1)]), ZZ)
        >>> K = ZZ.old_frac_field(x)

        >>> F = ZZ.old_poly_ring(x).from_FractionField(f, K)

        >>> F == DMP([ZZ(1), ZZ(1)], ZZ)
        True
        >>> type(F)  # doctest: +SKIP
        <class 'sympy.polys.polyclasses.DMP_Python'>

        """
        if a.denom().is_one:
            return K1.from_GlobalPolynomialRing(a.numer(), K0)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return basic_from_dict(a.to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        try:
            rep, _ = dict_from_basic(a, gens=self.gens)
        except PolynomialError:
            raise CoercionFailed("Cannot convert %s to type %s" % (a, self))

        for k, v in rep.items():
            rep[k] = self.dom.from_sympy(v)

        return DMP.from_dict(rep, self.ngens - 1, self.dom)

    def is_positive(self, a):
        """Returns True if ``LC(a)`` is positive. """
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if ``LC(a)`` is negative. """
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if ``LC(a)`` is non-positive. """
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if ``LC(a)`` is non-negative. """
        return self.dom.is_nonnegative(a.LC())

    def _vector_to_sdm(self, v, order):
        """
        Examples
        ========

        >>> from sympy import lex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y)
        >>> f = R.convert(x + 2*y)
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], lex)
        [((1, 1, 1), 1), ((0, 1, 0), 1), ((0, 0, 1), 2)]
        """
        return _vector_to_sdm_helper(v, order)


class GeneralizedPolynomialRing(PolynomialRingBase):
    """A generalized polynomial ring, with objects DMF. """

    dtype = DMF

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        res = self.dtype(a, self.dom, len(self.gens) - 1)

        # make sure res is actually in our ring
        if res.denom().terms(order=self.order)[0][0] != (0,)*len(self.gens):
            from sympy.printing.str import sstr
            raise CoercionFailed("denominator %s not allowed in %s"
                                 % (sstr(res), self))
        return res

    def __contains__(self, a):
        try:
            a = self.convert(a)
        except CoercionFailed:
            return False
        return a.denom().terms(order=self.order)[0][0] == (0,)*len(self.gens)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return (basic_from_dict(a.numer().to_sympy_dict(), *self.gens) /
                basic_from_dict(a.denom().to_sympy_dict(), *self.gens))

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        p, q = a.as_numer_denom()

        num, _ = dict_from_basic(p, gens=self.gens)
        den, _ = dict_from_basic(q, gens=self.gens)

        for k, v in num.items():
            num[k] = self.dom.from_sympy(v)

        for k, v in den.items():
            den[k] = self.dom.from_sympy(v)

        return self((num, den)).cancel()

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``. """
        # Elements are DMF that will always divide (except 0). The result is
        # not guaranteed to be in this ring, so we have to check that.
        r = a / b

        try:
            r = self.new((r.num, r.den))
        except CoercionFailed:
            raise ExactQuotientFailed(a, b, self)

        return r

    def from_FractionField(K1, a, K0):
        dmf = K1.get_field().from_FractionField(a, K0)
        return K1((dmf.num, dmf.den))

    def _vector_to_sdm(self, v, order):
        """
        Turn an iterable into a sparse distributed module.

        Note that the vector is multiplied by a unit first to make all entries
        polynomials.

        Examples
        ========

        >>> from sympy import ilex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> f = R.convert((x + 2*y) / (1 + x))
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], ilex)
        [((0, 0, 1), 2), ((0, 1, 0), 1), ((1, 1, 1), 1), ((1,
          2, 1), 1)]
        """
        # NOTE this is quite inefficient...
        u = self.one.numer()
        for x in v:
            u *= x.denom()
        return _vector_to_sdm_helper([x.numer()*u/x.denom() for x in v], order)


@public
def PolynomialRing(dom, *gens, **opts):
    r"""
    Create a generalized multivariate polynomial ring.

    A generalized polynomial ring is defined by a ground field `K`, a set
    of generators (typically `x_1, \ldots, x_n`) and a monomial order `<`.
    The monomial order can be global, local or mixed. In any case it induces
    a total ordering on the monomials, and there exists for every (non-zero)
    polynomial `f \in K[x_1, \ldots, x_n]` a well-defined "leading monomial"
    `LM(f) = LM(f, >)`. One can then define a multiplicative subset
    `S = S_> = \{f \in K[x_1, \ldots, x_n] | LM(f) = 1\}`. The generalized
    polynomial ring corresponding to the monomial order is
    `R = S^{-1}K[x_1, \ldots, x_n]`.

    If `>` is a so-called global order, that is `1` is the smallest monomial,
    then we just have `S = K` and `R = K[x_1, \ldots, x_n]`.

    Examples
    ========

    A few examples may make this clearer.

    >>> from sympy.abc import x, y
    >>> from sympy import QQ

    Our first ring uses global lexicographic order.

    >>> R1 = QQ.old_poly_ring(x, y, order=(("lex", x, y),))

    The second ring uses local lexicographic order. Note that when using a
    single (non-product) order, you can just specify the name and omit the
    variables:

    >>> R2 = QQ.old_poly_ring(x, y, order="ilex")

    The third and fourth rings use a mixed orders:

    >>> o1 = (("ilex", x), ("lex", y))
    >>> o2 = (("lex", x), ("ilex", y))
    >>> R3 = QQ.old_poly_ring(x, y, order=o1)
    >>> R4 = QQ.old_poly_ring(x, y, order=o2)

    We will investigate what elements of `K(x, y)` are contained in the various
    rings.

    >>> L = [x, 1/x, y/(1 + x), 1/(1 + y), 1/(1 + x*y)]
    >>> test = lambda R: [f in R for f in L]

    The first ring is just `K[x, y]`:

    >>> test(R1)
    [True, False, False, False, False]

    The second ring is R1 localised at the maximal ideal (x, y):

    >>> test(R2)
    [True, False, True, True, True]

    The third ring is R1 localised at the prime ideal (x):

    >>> test(R3)
    [True, False, True, False, True]

    Finally the fourth ring is R1 localised at `S = K[x, y] \setminus yK[y]`:

    >>> test(R4)
    [True, False, False, True, False]
    """

    order = opts.get("order", GeneralizedPolynomialRing.default_order)
    if iterable(order):
        order = build_product_order(order, gens)
    order = monomial_key(order)
    opts['order'] = order

    if order.is_global:
        return GlobalPolynomialRing(dom, *gens, **opts)
    else:
        return GeneralizedPolynomialRing(dom, *gens, **opts)
<|endoftext|>"""Domains of Gaussian type."""

from __future__ import annotations
from sympy.core.numbers import I
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring


class GaussianElement(DomainElement):
    """Base class for elements of Gaussian type domains."""
    base: Domain
    _parent: Domain

    __slots__ = ('x', 'y')

    def __new__(cls, x, y=0):
        conv = cls.base.convert
        return cls.new(conv(x), conv(y))

    @classmethod
    def new(cls, x, y):
        """Create a new GaussianElement of the same domain."""
        obj = super().__new__(cls)
        obj.x = x
        obj.y = y
        return obj

    def parent(self):
        """The domain that this is an element of (ZZ_I or QQ_I)"""
        return self._parent

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, GaussianElement):
            return NotImplemented
        return [self.y, self.x] < [other.y, other.x]

    def __pos__(self):
        return self

    def __neg__(self):
        return self.new(-self.x, -self.y)

    def __repr__(self):
        return "%s(%s, %s)" % (self._parent.rep, self.x, self.y)

    def __str__(self):
        return str(self._parent.to_sympy(self))

    @classmethod
    def _get_xy(cls, other):
        if not isinstance(other, cls):
            try:
                other = cls._parent.convert(other)
            except CoercionFailed:
                return None, None
        return other.x, other.y

    def __add__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x + x, self.y + y)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x - x, self.y - y)
        else:
            return NotImplemented

    def __rsub__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(x - self.x, y - self.y)
        else:
            return NotImplemented

    def __mul__(self, other):
        x, y = self._get_xy(other)
        if x is not None:
            return self.new(self.x*x - self.y*y, self.x*y + self.y*x)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __pow__(self, exp):
        if exp == 0:
            return self.new(1, 0)
        if exp < 0:
            self, exp = 1/self, -exp
        if exp == 1:
            return self
        pow2 = self
        prod = self if exp % 2 else self._parent.one
        exp //= 2
        while exp:
            pow2 *= pow2
            if exp % 2:
                prod *= pow2
            exp //= 2
        return prod

    def __bool__(self):
        return bool(self.x) or bool(self.y)

    def quadrant(self):
        """Return quadrant index 0-3.

        0 is included in quadrant 0.
        """
        if self.y > 0:
            return 0 if self.x > 0 else 1
        elif self.y < 0:
            return 2 if self.x < 0 else 3
        else:
            return 0 if self.x >= 0 else 2

    def __rdivmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__divmod__(self)

    def __rtruediv__(self, other):
        try:
            other = QQ_I.convert(other)
        except CoercionFailed:
            return NotImplemented
        else:
            return other.__truediv__(self)

    def __floordiv__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __rfloordiv__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[0]

    def __mod__(self, other):
        qr = self.__divmod__(other)
        return qr if qr is NotImplemented else qr[1]

    def __rmod__(self, other):
        qr = self.__rdivmod__(other)
        return qr if qr is NotImplemented else qr[1]


class GaussianInteger(GaussianElement):
    """Gaussian integer: domain element for :ref:`ZZ_I`

        >>> from sympy import ZZ_I
        >>> z = ZZ_I(2, 3)
        >>> z
        (2 + 3*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianInteger'>
    """
    base = ZZ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        return QQ_I.convert(self)/other

    def __divmod__(self, other):
        if not other:
            raise ZeroDivisionError('divmod({}, 0)'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented

        # multiply self and other by x - I*y
        # self/other == (a + I*b)/c
        a, b = self.x*x + self.y*y, -self.x*y + self.y*x
        c = x*x + y*y

        # find integers qx and qy such that
        # |a - qx*c| <= c/2 and |b - qy*c| <= c/2
        qx = (2*a + c) // (2*c)  # -c <= 2*a - qx*2*c < c
        qy = (2*b + c) // (2*c)

        q = GaussianInteger(qx, qy)
        # |self/other - q| < 1 since
        # |a/c - qx|**2 + |b/c - qy|**2 <= 1/4 + 1/4 < 1

        return q, self - q*other  # |r| < |other|


class GaussianRational(GaussianElement):
    """Gaussian rational: domain element for :ref:`QQ_I`

        >>> from sympy import QQ_I, QQ
        >>> z = QQ_I(QQ(2, 3), QQ(4, 5))
        >>> z
        (2/3 + 4/5*I)
        >>> type(z)
        <class 'sympy.polys.domains.gaussiandomains.GaussianRational'>
    """
    base = QQ

    def __truediv__(self, other):
        """Return a Gaussian rational."""
        if not other:
            raise ZeroDivisionError('{} / 0'.format(self))
        x, y = self._get_xy(other)
        if x is None:
            return NotImplemented
        c = x*x + y*y

        return GaussianRational((self.x*x + self.y*y)/c,
                                (-self.x*y + self.y*x)/c)

    def __divmod__(self, other):
        try:
            other = self._parent.convert(other)
        except CoercionFailed:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('{} % 0'.format(self))
        else:
            return self/other, QQ_I.zero


class GaussianDomain():
    """Base class for Gaussian domains."""
    dom: Domain

    is_Numerical = True
    is_Exact = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        conv = self.dom.to_sympy
        return conv(a.x) + I*conv(a.y)

    def from_sympy(self, a):
        """Convert a SymPy object to ``self.dtype``."""
        r, b = a.as_coeff_Add()
        x = self.dom.from_sympy(r)  # may raise CoercionFailed
        if not b:
            return self.new(x, 0)
        r, b = b.as_coeff_Mul()
        y = self.dom.from_sympy(r)
        if b is I:
            return self.new(x, y)
        else:
            raise CoercionFailed("{} is not Gaussian".format(a))

    def inject(self, *gens):
        """Inject generators into this domain. """
        return self.poly_ring(*gens)

    def canonical_unit(self, d):
        unit = self.units[-d.quadrant()]  # - for inverse power
        return unit

    def is_negative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_positive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

