        =======

        DomainMatrix
            DomainMatrix after term by term multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> b = ZZ(2)

        >>> A.mul(b)
        DomainMatrix([[2, 4], [6, 8]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        return A.from_rep(A.rep.mul(b))

    def rmul(A, b):
        return A.from_rep(A.rep.rmul(b))

    def matmul(A, B):
        r"""
        Performs matrix multiplication of two DomainMatrix matrices

        Parameters
        ==========

        A, B: DomainMatrix
            to multiply

        Returns
        =======

        DomainMatrix
            DomainMatrix after multiplication

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.matmul(B)
        DomainMatrix([[1, 3], [3, 7]], (2, 2), ZZ)

        See Also
        ========

        mul, pow, add, sub

        """

        A._check('*', B, A.shape[1], B.shape[0])
        return A.from_rep(A.rep.matmul(B.rep))

    def _scalarmul(A, lamda, reverse):
        if lamda == A.domain.zero:
            return DomainMatrix.zeros(A.shape, A.domain)
        elif lamda == A.domain.one:
            return A.copy()
        elif reverse:
            return A.rmul(lamda)
        else:
            return A.mul(lamda)

    def scalarmul(A, lamda):
        return A._scalarmul(lamda, reverse=False)

    def rscalarmul(A, lamda):
        return A._scalarmul(lamda, reverse=True)

    def mul_elementwise(A, B):
        assert A.domain == B.domain
        return A.from_rep(A.rep.mul_elementwise(B.rep))

    def __truediv__(A, lamda):
        """ Method for Scalar Division"""
        if isinstance(lamda, int) or ZZ.of_type(lamda):
            lamda = DomainScalar(ZZ(lamda), ZZ)
        elif A.domain.is_Field and lamda in A.domain:
            K = A.domain
            lamda = DomainScalar(K.convert(lamda), K)

        if not isinstance(lamda, DomainScalar):
            return NotImplemented

        A, lamda = A.to_field().unify(lamda)
        if lamda.element == lamda.domain.zero:
            raise ZeroDivisionError
        if lamda.element == lamda.domain.one:
            return A

        return A.mul(1 / lamda.element)

    def pow(A, n):
        r"""
        Computes A**n

        Parameters
        ==========

        A : DomainMatrix

        n : exponent for A

        Returns
        =======

        DomainMatrix
            DomainMatrix on computing A**n

        Raises
        ======

        NotImplementedError
            if n is negative.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)

        >>> A.pow(2)
        DomainMatrix([[1, 2], [0, 1]], (2, 2), ZZ)

        See Also
        ========

        matmul

        """
        nrows, ncols = A.shape
        if nrows != ncols:
            raise DMNonSquareMatrixError('Power of a nonsquare matrix')
        if n < 0:
            raise NotImplementedError('Negative powers')
        elif n == 0:
            return A.eye(nrows, A.domain)
        elif n == 1:
            return A
        elif n % 2 == 1:
            return A * A**(n - 1)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn

    def scc(self):
        """Compute the strongly connected components of a DomainMatrix

        Explanation
        ===========

        A square matrix can be considered as the adjacency matrix for a
        directed graph where the row and column indices are the vertices. In
        this graph if there is an edge from vertex ``i`` to vertex ``j`` if
        ``M[i, j]`` is nonzero. This routine computes the strongly connected
        components of that graph which are subsets of the rows and columns that
        are connected by some nonzero element of the matrix. The strongly
        connected components are useful because many operations such as the
        determinant can be computed by working with the submatrices
        corresponding to each component.

        Examples
        ========

        Find the strongly connected components of a matrix:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = DomainMatrix([[ZZ(1), ZZ(0), ZZ(2)],
        ...                   [ZZ(0), ZZ(3), ZZ(0)],
        ...                   [ZZ(4), ZZ(6), ZZ(5)]], (3, 3), ZZ)
        >>> M.scc()
        [[1], [0, 2]]

        Compute the determinant from the components:

        >>> MM = M.to_Matrix()
        >>> MM
        Matrix([
        [1, 0, 2],
        [0, 3, 0],
        [4, 6, 5]])
        >>> MM[[1], [1]]
        Matrix([[3]])
        >>> MM[[0, 2], [0, 2]]
        Matrix([
        [1, 2],
        [4, 5]])
        >>> MM.det()
        -9
        >>> MM[[1], [1]].det() * MM[[0, 2], [0, 2]].det()
        -9

        The components are given in reverse topological order and represent a
        permutation of the rows and columns that will bring the matrix into
        block lower-triangular form:

        >>> MM[[1, 0, 2], [1, 0, 2]]
        Matrix([
        [3, 0, 0],
        [0, 1, 2],
        [6, 4, 5]])

        Returns
        =======

        List of lists of integers
            Each list represents a strongly connected component.

        See also
        ========

        sympy.matrices.matrixbase.MatrixBase.strongly_connected_components
        sympy.utilities.iterables.strongly_connected_components

        """
        if not self.is_square:
            raise DMNonSquareMatrixError('Matrix must be square for scc')

        return self.rep.scc()

    def clear_denoms(self, convert=False):
        """
        Clear denominators, but keep the domain unchanged.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3)], [(1,4), (1,5)]], QQ)
        >>> den, Anum = A.clear_denoms()
        >>> den.to_sympy()
        60
        >>> Anum.to_Matrix()
        Matrix([
        [30, 20],
        [15, 12]])
        >>> den * A == Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms()[1].domain
        QQ
        >>> A.clear_denoms(convert=True)[1].domain
        ZZ

        The denominator is always in the associated ring:

        >>> A.clear_denoms()[0].domain
        ZZ
        >>> A.domain.get_ring()
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms_rowwise
        """
        elems0, data = self.to_flat_nz()

        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        den, elems1 = dup_clear_denoms(elems0, K0, K1, convert=convert)

        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        den = DomainScalar(den, Kden)
        num = self.from_flat_nz(elems1, data, Knum)

        return den, num

    def clear_denoms_rowwise(self, convert=False):
        """
        Clear denominators from each row of the matrix.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[(1,2), (1,3), (1,4)], [(1,5), (1,6), (1,7)]], QQ)
        >>> den, Anum = A.clear_denoms_rowwise()
        >>> den.to_Matrix()
        Matrix([
        [12,   0],
        [ 0, 210]])
        >>> Anum.to_Matrix()
        Matrix([
        [ 6,  4,  3],
        [42, 35, 30]])

        The denominator matrix is a diagonal matrix with the denominators of
        each row on the diagonal. The invariants are:

        >>> den * A == Anum
        True
        >>> A == den.to_field().inv() * Anum
        True

        The numerator matrix will be in the same domain as the original matrix
        unless ``convert`` is set to ``True``:

        >>> A.clear_denoms_rowwise()[1].domain
        QQ
        >>> A.clear_denoms_rowwise(convert=True)[1].domain
        ZZ

        The domain of the denominator matrix is the associated ring:

        >>> A.clear_denoms_rowwise()[0].domain
        ZZ

        See Also
        ========

        sympy.polys.polytools.Poly.clear_denoms
        clear_denoms
        """
        dod = self.to_dod()

        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0

        diagonals = [K0.one] * self.shape[0]
        dod_num = {}
        for i, rowi in dod.items():
            indices, elems = zip(*rowi.items())
            den, elems_num = dup_clear_denoms(elems, K0, K1, convert=convert)
            rowi_num = dict(zip(indices, elems_num))
            diagonals[i] = den
            dod_num[i] = rowi_num

        if convert:
            Kden, Knum = K1, K1
        else:
            Kden, Knum = K1, K0

        den = self.diag(diagonals, Kden)
        num = self.from_dod_like(dod_num, Knum)

        return den, num

    def cancel_denom(self, denom):
        """
        Cancel factors between a matrix and a denominator.

        Returns a matrix and denominator on lowest terms.

        Requires ``gcd`` in the ground domain.

        Methods like :meth:`solve_den`, :meth:`inv_den` and :meth:`rref_den`
        return a matrix and denominator but not necessarily on lowest terms.
        Reduction to lowest terms without fractions can be performed with
        :meth:`cancel_denom`.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 2, 0],
        ...         [0, 2, 2],
        ...         [0, 0, 2]], ZZ)
        >>> Minv, den = M.inv_den()
        >>> Minv.to_Matrix()
        Matrix([
        [1, -1,  1],
        [0,  1, -1],
        [0,  0,  1]])
        >>> den
        2
        >>> Minv_reduced, den_reduced = Minv.cancel_denom(den)
        >>> Minv_reduced.to_Matrix()
        Matrix([
        [1, -1,  1],
        [0,  1, -1],
        [0,  0,  1]])
        >>> den_reduced
        2
        >>> Minv_reduced.to_field() / den_reduced == Minv.to_field() / den
        True

        The denominator is made canonical with respect to units (e.g. a
        negative denominator is made positive):

        >>> M = DM([[2, 2, 0]], ZZ)
        >>> den = ZZ(-4)
        >>> M.cancel_denom(den)
        (DomainMatrix([[-1, -1, 0]], (1, 3), ZZ), 2)

        Any factor common to _all_ elements will be cancelled but there can
        still be factors in common between _some_ elements of the matrix and
        the denominator. To cancel factors between each element and the
        denominator, use :meth:`cancel_denom_elementwise` or otherwise convert
        to a field and use division:

        >>> M = DM([[4, 6]], ZZ)
        >>> den = ZZ(12)
        >>> M.cancel_denom(den)
        (DomainMatrix([[2, 3]], (1, 2), ZZ), 6)
        >>> numers, denoms = M.cancel_denom_elementwise(den)
        >>> numers
        DomainMatrix([[1, 1]], (1, 2), ZZ)
        >>> denoms
        DomainMatrix([[3, 2]], (1, 2), ZZ)
        >>> M.to_field() / den
        DomainMatrix([[1/3, 1/2]], (1, 2), QQ)

        See Also
        ========

        solve_den
        inv_den
        rref_den
        cancel_denom_elementwise
        """
        M = self
        K = self.domain

        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            return (M.copy(), denom)

        elements, data = M.to_flat_nz()

        # First canonicalize the denominator (e.g. multiply by -1).
        if K.is_negative(denom):
            u = -K.one
        else:
            u = K.canonical_unit(denom)

        # Often after e.g. solve_den the denominator will be much more
        # complicated than the elements of the numerator. Hopefully it will be
        # quicker to find the gcd of the numerator and if there is no content
        # then we do not need to look at the denominator at all.
        content = dup_content(elements, K)
        common = K.gcd(content, denom)

        if not K.is_one(content):

            common = K.gcd(content, denom)

            if not K.is_one(common):
                elements = dup_quo_ground(elements, common, K)
                denom = K.quo(denom, common)

        if not K.is_one(u):
            elements = dup_mul_ground(elements, u, K)
            denom = u * denom
        elif K.is_one(common):
            return (M.copy(), denom)

        M_cancelled = M.from_flat_nz(elements, data, K)

        return M_cancelled, denom

    def cancel_denom_elementwise(self, denom):
        """
        Cancel factors between the elements of a matrix and a denominator.

        Returns a matrix of numerators and matrix of denominators.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 3], [4, 12]], ZZ)
        >>> denom = ZZ(6)
        >>> numers, denoms = M.cancel_denom_elementwise(denom)
        >>> numers.to_Matrix()
        Matrix([
        [1, 1],
        [2, 2]])
        >>> denoms.to_Matrix()
        Matrix([
        [3, 2],
        [3, 1]])
        >>> M_frac = (M.to_field() / denom).to_Matrix()
        >>> M_frac
        Matrix([
        [1/3, 1/2],
        [2/3,   2]])
        >>> denoms_inverted = denoms.to_Matrix().applyfunc(lambda e: 1/e)
        >>> numers.to_Matrix().multiply_elementwise(denoms_inverted) == M_frac
        True

        Use :meth:`cancel_denom` to cancel factors between the matrix and the
        denominator while preserving the form of a matrix with a scalar
        denominator.

        See Also
        ========

        cancel_denom
        """
        K = self.domain
        M = self

        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            M_numers = M.copy()
            M_denoms = M.ones(M.shape, M.domain)
            return (M_numers, M_denoms)

        elements, data = M.to_flat_nz()

        cofactors = [K.cofactors(numer, denom) for numer in elements]
        gcds, numers, denoms = zip(*cofactors)

        M_numers = M.from_flat_nz(list(numers), data, K)
        M_denoms = M.from_flat_nz(list(denoms), data, K)

        return (M_numers, M_denoms)

    def content(self):
        """
        Return the gcd of the elements of the matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> M.content()
        2

        See Also
        ========

        primitive
        cancel_denom
        """
        K = self.domain
        elements, _ = self.to_flat_nz()
        return dup_content(elements, K)

    def primitive(self):
        """
        Factor out gcd of the elements of a matrix.

        Requires ``gcd`` in the ground domain.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[2, 4], [4, 12]], ZZ)
        >>> content, M_primitive = M.primitive()
        >>> content
        2
        >>> M_primitive
        DomainMatrix([[1, 2], [2, 6]], (2, 2), ZZ)
        >>> content * M_primitive == M
        True
        >>> M_primitive.content() == ZZ(1)
        True

        See Also
        ========

        content
        cancel_denom
        """
        K = self.domain
        elements, data = self.to_flat_nz()
        content, prims = dup_primitive(elements, K)
        M_primitive = self.from_flat_nz(prims, data, K)
        return content, M_primitive

    def rref(self, *, method='auto'):
        r"""
        Returns reduced-row echelon form (RREF) and list of pivots.

        If the domain is not a field then it will be converted to a field. See
        :meth:`rref_den` for the fraction-free version of this routine that
        returns RREF with denominator instead.

        The domain must either be a field or have an associated fraction field
        (see :meth:`to_field`).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)

        >>> rref_matrix, rref_pivots = A.rref()
        >>> rref_matrix
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> rref_pivots
        (0, 1, 2)

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination but if
              the domain is not a field then it will be converted to a field
              at the end and divided by the denominator. This is most efficient
              for dense matrices or for matrices with simple denominators.

            - ``A.rref(method='CD')`` clears the denominators before using
              fraction-free Gauss-Jordan elimination in the associated ring.
              This is most efficient for dense matrices with very simple
              denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and its
            domain will always be the field of fractions of the input domain.

        Returns
        =======

        (DomainMatrix, list)
            reduced-row echelon form and list of pivots for the DomainMatrix

        See Also
        ========

        rref_den
            RREF with denominator
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'`` and
            by ``method='GJ'`` when the original domain is not a field.

        """
        return _dm_rref(self, method=method)

    def rref_den(self, *, method='auto', keep_domain=True):
        r"""
        Returns reduced-row echelon form with denominator and list of pivots.

        Requires exact division in the ground domain (``exquo``).

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)

        >>> A_rref, denom, pivots = A.rref_den()
        >>> A_rref
        DomainMatrix([[6, 0, 0], [0, 6, 0], [0, 0, 6]], (3, 3), ZZ)
        >>> denom
        6
        >>> pivots
        (0, 1, 2)
        >>> A_rref.to_field() / denom
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> A_rref.to_field() / denom == A.convert_to(QQ).rref()[0]
        True

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination and the
              result is always returned as a matrix over the current domain.
              This is most efficient for dense matrices or for matrices with
              simple denominators.

            - ``A.rref(method='CD')`` clears denominators before using
              fraction-free Gauss-Jordan elimination in the associated ring.
              The result will be converted back to the original domain unless
              ``keep_domain=False`` is passed in which case the result will be
              over the ring used for elimination. This is most efficient for
              dense matrices with very simple denominators.

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. The result is
              converted back to the original domain by clearing denominators
              unless ``keep_domain=False`` is passed in which case the result
              will be over the field used for elimination. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and if
            ``keep_domain=True`` its domain will always be the same as the
            input.

        keep_domain : bool, optional
            If True (the default), the domain of the returned matrix and
            denominator are the same as the domain of the input matrix. If
            False, the domain of the returned matrix might be changed to an
            associated ring or field if the algorithm used a different domain.
            This is useful for efficiency if the caller does not need the
            result to be in the original domain e.g. it avoids clearing
            denominators in the case of ``A.rref(method='GJ')``.

        Returns
        =======

        (DomainMatrix, scalar, list)
            Reduced-row echelon form, denominator and list of pivot indices.

        See Also
        ========

        rref
            RREF without denominator for field domains.
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'``.

        """
        return _dm_rref_den(self, method=method, keep_domain=keep_domain)

    def columnspace(self):
        r"""
        Returns the columnspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The columns of this matrix form a basis for the columnspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.columnspace()
        DomainMatrix([[1], [2]], (2, 1), QQ)

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        rref, pivots = self.rref()
        rows, cols = self.shape
        return self.extract(range(rows), pivots)

    def rowspace(self):
        r"""
        Returns the rowspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the rowspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.rowspace()
        DomainMatrix([[1, -1]], (1, 2), QQ)

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        rref, pivots = self.rref()
        rows, cols = self.shape
        return self.extract(range(len(pivots)), range(cols))

    def nullspace(self, divide_last=False):
        r"""
        Returns the nullspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The rows of this matrix form a basis for the nullspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...    [QQ(2), QQ(-2)],
        ...    [QQ(4), QQ(-4)]], QQ)
        >>> A.nullspace()
        DomainMatrix([[1, 1]], (1, 2), QQ)

        The returned matrix is a basis for the nullspace:

        >>> A_null = A.nullspace().transpose()
        >>> A * A_null
        DomainMatrix([[0], [0]], (2, 1), QQ)
        >>> rows, cols = A.shape
        >>> nullity = rows - A.rank()
        >>> A_null.shape == (cols, nullity)
        True

        Nullspace can also be computed for non-field rings. If the ring is not
        a field then division is not used. Setting ``divide_last`` to True will
        raise an error in this case:

        >>> from sympy import ZZ
        >>> B = DM([[6, -3],
        ...         [4, -2]], ZZ)
        >>> B.nullspace()
        DomainMatrix([[3, 6]], (1, 2), ZZ)
        >>> B.nullspace(divide_last=True)
        Traceback (most recent call last):
        ...
        DMNotAField: Cannot normalize vectors over a non-field

        Over a ring with ``gcd`` defined the nullspace can potentially be
        reduced with :meth:`primitive`:

        >>> B.nullspace().primitive()
        (3, DomainMatrix([[1, 2]], (1, 2), ZZ))

        A matrix over a ring can often be normalized by converting it to a
        field but it is often a bad idea to do so:

        >>> from sympy.abc import a, b, c
        >>> from sympy import Matrix
        >>> M = Matrix([[        a*b,       b + c,        c],
        ...             [      a - b,         b*c,     c**2],
        ...             [a*b + a - b, b*c + b + c, c**2 + c]])
        >>> M.to_DM().domain
        ZZ[a,b,c]
        >>> M.to_DM().nullspace().to_Matrix().transpose()
        Matrix([
        [                             c**3],
        [            -a*b*c**2 + a*c - b*c],
        [a*b**2*c - a*b - a*c + b**2 + b*c]])

        The unnormalized form here is nicer than the normalized form that
        spreads a large denominator throughout the matrix:

        >>> M.to_DM().to_field().nullspace(divide_last=True).to_Matrix().transpose()
        Matrix([
        [                   c**3/(a*b**2*c - a*b - a*c + b**2 + b*c)],
        [(-a*b*c**2 + a*c - b*c)/(a*b**2*c - a*b - a*c + b**2 + b*c)],
        [                                                          1]])

        Parameters
        ==========

        divide_last : bool, optional
            If False (the default), the vectors are not normalized and the RREF
            is computed using :meth:`rref_den` and the denominator is
            discarded. If True, then each row is divided by its final element;
            the domain must be a field in this case.

        See Also
        ========

        nullspace_from_rref
        rref
        rref_den
        rowspace
        """
        A = self
        K = A.domain

        if divide_last and not K.is_Field:
            raise DMNotAField("Cannot normalize vectors over a non-field")

        if divide_last:
            A_rref, pivots = A.rref()
        else:
            A_rref, den, pivots = A.rref_den()

            # Ensure that the sign is canonical before discarding the
            # denominator. Then M.nullspace().primitive() is canonical.
            u = K.canonical_unit(den)
            if u != K.one:
                A_rref *= u

        A_null = A_rref.nullspace_from_rref(pivots)

        return A_null

    def nullspace_from_rref(self, pivots=None):
        """
        Compute nullspace from rref and pivots.

        The domain of the matrix can be any domain.

        The matrix must be in reduced row echelon form already. Otherwise the
        result will be incorrect. Use :meth:`rref` or :meth:`rref_den` first
        to get the reduced row echelon form or use :meth:`nullspace` instead.

        See Also
        ========

        nullspace
        rref
        rref_den
        sympy.polys.matrices.sdm.SDM.nullspace_from_rref
        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
        """
        null_rep, nonpivots = self.rep.nullspace_from_rref(pivots)
        return self.from_rep(null_rep)

    def inv(self):
        r"""
        Finds the inverse of the DomainMatrix if exists

        Returns
        =======

        DomainMatrix
            DomainMatrix after inverse

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        DMNonSquareMatrixError
            If the DomainMatrix is not a not Square DomainMatrix

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [QQ(2), QQ(-1), QQ(0)],
        ...     [QQ(-1), QQ(2), QQ(-1)],
        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)
        >>> A.inv()
        DomainMatrix([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [0, 0, 1/2]], (3, 3), QQ)

        See Also
        ========

        neg

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        m, n = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        inv = self.rep.inv()
        return self.from_rep(inv)

    def det(self):
        r"""
        Returns the determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        determinant: DomainElement
            Determinant of the matrix.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix is not a Field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.det()
        -2

        """
        m, n = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        return self.rep.det()

    def adj_det(self):
        """
        Adjugate and determinant of a square :class:`DomainMatrix`.

        Returns
        =======

        (adjugate, determinant) : (DomainMatrix, DomainScalar)
            The adjugate matrix and determinant of this matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([
        ...     [ZZ(1), ZZ(2)],
        ...     [ZZ(3), ZZ(4)]], ZZ)
        >>> adjA, detA = A.adj_det()
        >>> adjA
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)
        >>> detA
        -2

        See Also
        ========

        adjugate
            Returns only the adjugate matrix.
        det
            Returns only the determinant.
        inv_den
            Returns a matrix/denominator pair representing the inverse matrix
            but perhaps differing from the adjugate and determinant by a common
            factor.
        """
        m, n = self.shape
        I_m = self.eye((m, m), self.domain)
        adjA, detA = self.solve_den_charpoly(I_m, check=False)
        if self.rep.fmt == "dense":
            adjA = adjA.to_dense()
        return adjA, detA

    def adjugate(self):
        """
        Adjugate of a square :class:`DomainMatrix`.

        The adjugate matrix is the transpose of the cofactor matrix and is
        related to the inverse by::

            adj(A) = det(A) * A.inv()

        Unlike the inverse matrix the adjugate matrix can be computed and
        expressed without division or fractions in the ground domain.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> A.adjugate()
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)

        Returns
        =======

        DomainMatrix
            The adjugate matrix of this matrix with the same domain.

        See Also
        ========

        adj_det
        """
        adjA, detA = self.adj_det()
        return adjA

    def inv_den(self, method=None):
        """
        Return the inverse as a :class:`DomainMatrix` with denominator.

        Returns
        =======

        (inv, den) : (:class:`DomainMatrix`, :class:`~.DomainElement`)
            The inverse matrix and its denominator.

        This is more or less equivalent to :meth:`adj_det` except that ``inv``
        and ``den`` are not guaranteed to be the adjugate and inverse. The
        ratio ``inv/den`` is equivalent to ``adj/det`` but some factors
        might be cancelled between ``inv`` and ``den``. In simple cases this
        might just be a minus sign so that ``(inv, den) == (-adj, -det)`` but
        factors more complicated than ``-1`` can also be cancelled.
        Cancellation is not guaranteed to be complete so ``inv`` and ``den``
        may not be on lowest terms. The denominator ``den`` will be zero if and
        only if the determinant is zero.

        If the actual adjugate and determinant are needed, use :meth:`adj_det`
        instead. If the intention is to compute the inverse matrix or solve a
        system of equations then :meth:`inv_den` is more efficient.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)
        >>> Ainv, den = A.inv_den()
        >>> den
        6
        >>> Ainv
        DomainMatrix([[4, 2, 1], [2, 4, 2], [0, 0, 3]], (3, 3), ZZ)
        >>> A * Ainv == den * A.eye(A.shape, A.domain).to_dense()
        True

        Parameters
        ==========

        method : str, optional
            The method to use to compute the inverse. Can be one of ``None``,
            ``'rref'`` or ``'charpoly'``. If ``None`` then the method is
            chosen automatically (see :meth:`solve_den` for details).

        See Also
        ========

        inv
        det
        adj_det
        solve_den
        """
        I = self.eye(self.shape, self.domain)
        return self.solve_den(I, method=method)

    def solve_den(self, b, method=None):
        """
        Solve matrix equation $Ax = b$ without fractions in the ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        Solve a matrix equation over a polynomial ring:

        >>> from sympy import ZZ
        >>> from sympy.abc import x, y, z, a, b
        >>> R = ZZ[x, y, z, a, b]
        >>> M = DM([[x*y, x*z], [y*z, x*z]], R)
        >>> b = DM([[a], [b]], R)
        >>> M.to_Matrix()
        Matrix([
        [x*y, x*z],
        [y*z, x*z]])
        >>> b.to_Matrix()
        Matrix([
        [a],
        [b]])
        >>> xnum, xden = M.solve_den(b)
        >>> xden
        x**2*y*z - x*y*z**2
        >>> xnum.to_Matrix()
        Matrix([
        [ a*x*z - b*x*z],
        [-a*y*z + b*x*y]])
        >>> M * xnum == xden * b
        True

        The solution can be expressed over a fraction field which will cancel
        gcds between the denominator and the elements of the numerator:

        >>> xsol = xnum.to_field() / xden
        >>> xsol.to_Matrix()
        Matrix([
        [           (a - b)/(x*y - y*z)],
        [(-a*z + b*x)/(x**2*z - x*z**2)]])
        >>> (M * xsol).to_Matrix() == b.to_Matrix()
        True

        When solving a large system of equations this cancellation step might
        be a lot slower than :func:`solve_den` itself. The solution can also be
        expressed as a ``Matrix`` without attempting any polynomial
        cancellation between the numerator and denominator giving a less
        simplified result more quickly:

        >>> xsol_uncancelled = xnum.to_Matrix() / xnum.domain.to_sympy(xden)
        >>> xsol_uncancelled
        Matrix([
        [ (a*x*z - b*x*z)/(x**2*y*z - x*y*z**2)],
        [(-a*y*z + b*x*y)/(x**2*y*z - x*y*z**2)]])
        >>> from sympy import cancel
        >>> cancel(xsol_uncancelled) == xsol.to_Matrix()
        True

        Parameters
        ==========

        self : :class:`DomainMatrix`
            The ``m x n`` matrix $A$ in the equation $Ax = b$. Underdetermined
            systems are not supported so ``m >= n``: $A$ should be square or
            have more rows than columns.
        b : :class:`DomainMatrix`
            The ``n x m`` matrix $b$ for the rhs.
        cp : list of :class:`~.DomainElement`, optional
            The characteristic polynomial of the matrix $A$. If not given, it
            will be computed using :meth:`charpoly`.
        method: str, optional
            The method to use for solving the system. Can be one of ``None``,
            ``'charpoly'`` or ``'rref'``. If ``None`` (the default) then the
            method will be chosen automatically.

            The ``charpoly`` method uses :meth:`solve_den_charpoly` and can
            only be used if the matrix is square. This method is division free
            and can be used with any domain.

            The ``rref`` method is fraction free but requires exact division
            in the ground domain (``exquo``). This is also suitable for most
            domains. This method can be used with overdetermined systems (more
            equations than unknowns) but not underdetermined systems as a
            unique solution is sought.

        Returns
        =======

        (xnum, xden) : (DomainMatrix, DomainElement)
            The solution of the equation $Ax = b$ as a pair consisting of an
            ``n x m`` matrix numerator ``xnum`` and a scalar denominator
            ``xden``.

        The solution $x$ is given by ``x = xnum / xden``. The division free
        invariant is ``A * xnum == xden * b``. If $A$ is square then the
        denominator ``xden`` will be a divisor of the determinant $det(A)$.

        Raises
        ======

        DMNonInvertibleMatrixError
            If the system $Ax = b$ does not have a unique solution.

        See Also
        ========

        solve_den_charpoly
        solve_den_rref
        inv_den
        """
        m, n = self.shape
        bm, bn = b.shape

        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        if method is None:
            method = 'rref'
        elif method == 'charpoly' and m != n:
            raise DMNonSquareMatrixError("method='charpoly' requires a square matrix.")

        if method == 'charpoly':
            xnum, xden = self.solve_den_charpoly(b)
        elif method == 'rref':
            xnum, xden = self.solve_den_rref(b)
        else:
            raise DMBadInputError("method should be 'rref' or 'charpoly'")

        return xnum, xden

    def solve_den_rref(self, b):
        """
        Solve matrix equation $Ax = b$ using fraction-free RREF

        Solves the matrix equation $Ax = b$ for $x$ and returns the solution
        as a numerator/denominator pair.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, xden = A.solve_den_rref(b)
        >>> xden
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == xden * b
        True

        See Also
        ========

        solve_den
        solve_den_charpoly
        """
        A = self
        m, n = A.shape
        bm, bn = b.shape

        if m != bm:
            raise DMShapeError("Matrix equation shape mismatch.")

        if m < n:
            raise DMShapeError("Underdetermined matrix equation.")

        Aaug = A.hstack(b)
        Aaug_rref, denom, pivots = Aaug.rref_den()

        # XXX: We check here if there are pivots after the last column. If
        # there were than it possibly means that rref_den performed some
        # unnecessary elimination. It would be better if rref methods had a
        # parameter indicating how many columns should be used for elimination.
        if len(pivots) != n or pivots and pivots[-1] >= n:
            raise DMNonInvertibleMatrixError("Non-unique solution.")

        xnum = Aaug_rref[:n, n:]
        xden = denom

        return xnum, xden

    def solve_den_charpoly(self, b, cp=None, check=True):
        """
        Solve matrix equation $Ax = b$ using the characteristic polynomial.

        This method solves the square matrix equation $Ax = b$ for $x$ using
        the characteristic polynomial without any division or fractions in the
        ground domain.

        Examples
        ========

        Solve a matrix equation over the integers:

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)
        >>> xnum, detA = A.solve_den_charpoly(b)
        >>> detA
        -2
        >>> xnum
        DomainMatrix([[8], [-9]], (2, 1), ZZ)
        >>> A * xnum == detA * b
        True

        Parameters
        ==========

        self : DomainMatrix
            The ``n x n`` matrix `A` in the equation `Ax = b`. Must be square
            and invertible.
        b : DomainMatrix
            The ``n x m`` matrix `b` for the rhs.
        cp : list, optional
            The characteristic polynomial of the matrix `A` if known. If not
            given, it will be computed using :meth:`charpoly`.
        check : bool, optional
            If ``True`` (the default) check that the determinant is not zero
            and raise an error if it is. If ``False`` then if the determinant
            is zero the return value will be equal to ``(A.adjugate()*b, 0)``.

        Returns
        =======

        (xnum, detA) : (DomainMatrix, DomainElement)
            The solution of the equation `Ax = b` as a matrix numerator and
            scalar denominator pair. The denominator is equal to the
            determinant of `A` and the numerator is ``adj(A)*b``.

        The solution $x$ is given by ``x = xnum / detA``. The division free
        invariant is ``A * xnum == detA * b``.

        If ``b`` is the identity matrix, then ``xnum`` is the adjugate matrix
        and we have ``A * adj(A) == detA * I``.

        See Also
        ========

        solve_den
            Main frontend for solving matrix equations with denominator.
        solve_den_rref
            Solve matrix equations using fraction-free RREF.
        inv_den
            Invert a matrix using the characteristic polynomial.
        """
        A, b = self.unify(b)
        m, n = self.shape
        mb, nb = b.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if mb != m:
            raise DMShapeError("Matrix and vector must have the same number of rows")

        f, detA = self.adj_poly_det(cp=cp)

        if check and not detA:
            raise DMNonInvertibleMatrixError("Matrix is not invertible")

        # Compute adj(A)*b = det(A)*inv(A)*b using Horner's method without
        # constructing inv(A) explicitly.
        adjA_b = self.eval_poly_mul(f, b)

        return (adjA_b, detA)

    def adj_poly_det(self, cp=None):
        """
        Return the polynomial $p$ such that $p(A) = adj(A)$ and also the
        determinant of $A$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p, detA = A.adj_poly_det()
        >>> p
        [-1, 5]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[4, -2], [-3, 1]], (2, 2), QQ)
        >>> p[0]*A**1 + p[1]*A**0 == p_A
        True
        >>> p_A == A.adjugate()
        True
        >>> A * A.adjugate() == detA * A.eye(A.shape, A.domain).to_dense()
        True

        See Also
        ========

        adjugate
        eval_poly
        adj_det
        """

        # Cayley-Hamilton says that a matrix satisfies its own minimal
        # polynomial
        #
        #   p[0]*A^n + p[1]*A^(n-1) + ... + p[n]*I = 0
        #
        # with p[0]=1 and p[n]=(-1)^n*det(A) or
        #
        #   det(A)*I = -(-1)^n*(p[0]*A^(n-1) + p[1]*A^(n-2) + ... + p[n-1]*A).
        #
        # Define a new polynomial f with f[i] = -(-1)^n*p[i] for i=0..n-1. Then
        #
        #   det(A)*I = f[0]*A^n + f[1]*A^(n-1) + ... + f[n-1]*A.
        #
        # Multiplying on the right by inv(A) gives
        #
        #   det(A)*inv(A) = f[0]*A^(n-1) + f[1]*A^(n-2) + ... + f[n-1].
        #
        # So adj(A) = det(A)*inv(A) = f(A)

        A = self
        m, n = self.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if cp is None:
            cp = A.charpoly()

        if len(cp) % 2:
            # n is even
            detA = cp[-1]
            f = [-cpi for cpi in cp[:-1]]
        else:
            # n is odd
            detA = -cp[-1]
            f = cp[:-1]

        return f, detA

    def eval_poly(self, p):
        """
        Evaluate polynomial function of a matrix $p(A)$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A = A.eval_poly(p)
        >>> p_A
        DomainMatrix([[12, 14], [21, 33]], (2, 2), QQ)
        >>> p_A == p[0]*A**2 + p[1]*A + p[2]*A**0
        True

        See Also
        ========

        eval_poly_mul
        """
        A = self
        m, n = A.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if not p:
            return self.zeros(self.shape, self.domain)
        elif len(p) == 1:
            return p[0] * self.eye(self.shape, self.domain)

        # Evaluate p(A) using Horner's method:
        # XXX: Use Paterson-Stockmeyer method?
        I = A.eye(A.shape, A.domain)
        p_A = p[0] * I
        for pi in p[1:]:
            p_A = A*p_A + pi*I

        return p_A

    def eval_poly_mul(self, p, B):
        r"""
        Evaluate polynomial matrix product $p(A) \times B$.

        Evaluate the polynomial matrix product $p(A) \times B$ using Horner's
        method without creating the matrix $p(A)$ explicitly. If $B$ is a
        column matrix then this method will only use matrix-vector multiplies
        and no matrix-matrix multiplies are needed.

        If $B$ is square or wide or if $A$ can be represented in a simpler
        domain than $B$ then it might be faster to evaluate $p(A)$ explicitly
        (see :func:`eval_poly`) and then multiply with $B$.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)
        >>> b = DM([[QQ(5)], [QQ(6)]], QQ)
        >>> p = [QQ(1), QQ(2), QQ(3)]
        >>> p_A_b = A.eval_poly_mul(p, b)
        >>> p_A_b
        DomainMatrix([[144], [303]], (2, 1), QQ)
        >>> p_A_b == p[0]*A**2*b + p[1]*A*b + p[2]*b
        True
        >>> A.eval_poly_mul(p, b) == A.eval_poly(p)*b
        True

        See Also
        ========

        eval_poly
        solve_den_charpoly
        """
        A = self
        m, n = A.shape
        mb, nb = B.shape

        if m != n:
            raise DMNonSquareMatrixError("Matrix must be square")

        if mb != n:
            raise DMShapeError("Matrices are not aligned")

        if A.domain != B.domain:
            raise DMDomainError("Matrices must have the same domain")

        # Given a polynomial p(x) = p[0]*x^n + p[1]*x^(n-1) + ... + p[n-1]
        # and matrices A and B we want to find
        #
        #   p(A)*B = p[0]*A^n*B + p[1]*A^(n-1)*B + ... + p[n-1]*B
        #
        # Factoring out A term by term we get
        #
        #   p(A)*B = A*(...A*(A*(A*(p[0]*B) + p[1]*B) + p[2]*B) + ...) + p[n-1]*B
        #
        # where each pair of brackets represents one iteration of the loop
        # below starting from the innermost p[0]*B. If B is a column matrix
        # then products like A*(...) are matrix-vector multiplies and products
        # like p[i]*B are scalar-vector multiplies so there are no
        # matrix-matrix multiplies.

        if not p:
            return B.zeros(B.shape, B.domain, fmt=B.rep.fmt)

        p_A_B = p[0]*B

        for p_i in p[1:]:
            p_A_B = A*p_A_B + p_i*B

        return p_A_B

    def lu(self):
        r"""
        Returns Lower and Upper decomposition of the DomainMatrix

        Returns
        =======

        (L, U, exchange)
            L, U are Lower and Upper decomposition of the DomainMatrix,
            exchange is the list of indices of rows exchanged in the
            decomposition.

        Raises
        ======

        ValueError
            If the domain of DomainMatrix not a Field

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> L, U, exchange = A.lu()
        >>> L
        DomainMatrix([[1, 0], [2, 1]], (2, 2), QQ)
        >>> U
        DomainMatrix([[1, -1], [0, 0]], (2, 2), QQ)
        >>> exchange
        []

        See Also
        ========

        lu_solve

        """
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        L, U, swaps = self.rep.lu()
        return self.from_rep(L), self.from_rep(U), swaps

    def qr(self):
        r"""
        QR decomposition of the DomainMatrix.

        Explanation
        ===========

        The QR decomposition expresses a matrix as the product of an orthogonal
        matrix (Q) and an upper triangular matrix (R). In this implementation,
        Q is not orthonormal: its columns are orthogonal but not normalized to
        unit vectors. This avoids unnecessary divisions and is particularly
        suited for exact arithmetic domains.

        Note
        ====

        This implementation is valid only for matrices over real domains. For
        matrices over complex domains, a proper QR decomposition would require
        handling conjugation to ensure orthogonality.

        Returns
        =======

        (Q, R)
            Q is the orthogonal matrix, and R is the upper triangular matrix
            resulting from the QR decomposition of the DomainMatrix.

        Raises
        ======

        DMDomainError
            If the domain of the DomainMatrix is not a field (e.g., QQ).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[1, 2], [3, 4], [5, 6]], (3, 2), QQ)
        >>> Q, R = A.qr()
        >>> Q
        DomainMatrix([[1, 26/35], [3, 8/35], [5, -2/7]], (3, 2), QQ)
        >>> R
        DomainMatrix([[1, 44/35], [0, 1]], (2, 2), QQ)
        >>> Q * R == A
        True
        >>> (Q.transpose() * Q).is_diagonal
        True
        >>> R.is_upper
        True

        See Also
        ========

        lu

        """
        ddm_q, ddm_r = self.rep.qr()
        Q = self.from_rep(ddm_q)
        R = self.from_rep(ddm_r)
        return Q, R

    def lu_solve(self, rhs):
        r"""
        Solver for DomainMatrix x in the A*x = B

        Parameters
        ==========

        rhs : DomainMatrix B

        Returns
        =======

        DomainMatrix
            x in A*x = B

        Raises
        ======

        DMShapeError
            If the DomainMatrix A and rhs have different number of rows

        ValueError
            If the domain of DomainMatrix A not a Field

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(2)],
        ...    [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> B = DomainMatrix([
        ...    [QQ(1), QQ(1)],
        ...    [QQ(0), QQ(1)]], (2, 2), QQ)

        >>> A.lu_solve(B)
        DomainMatrix([[-2, -1], [3/2, 1]], (2, 2), QQ)

        See Also
        ========

        lu

        """
        if self.shape[0] != rhs.shape[0]:
            raise DMShapeError("Shape")
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_rep(sol)

    def fflu(self):
        """
        Fraction-free LU decomposition of DomainMatrix.

        Explanation
        ===========

        This method computes the PLDU decomposition
        using Gauss-Bareiss elimination in a fraction-free manner,
        it ensures that all intermediate results remain in
        the domain of the input matrix. Unlike standard
        LU decomposition, which introduces division, this approach
        avoids fractions, making it particularly suitable
        for exact arithmetic over integers or polynomials.

        This method satisfies the invariant:

        P * A = L * inv(D) * U

        Returns
        =======

        (P, L, D, U)
            - P (Permutation matrix)
            - L (Lower triangular matrix)
            - D (Diagonal matrix)
            - U (Upper triangular matrix)

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> P, L, D, U = A.fflu()
        >>> P
        DomainMatrix([[1, 0], [0, 1]], (2, 2), ZZ)
        >>> L
        DomainMatrix([[1, 0], [3, -2]], (2, 2), ZZ)
        >>> D
        DomainMatrix([[1, 0], [0, -2]], (2, 2), ZZ)
        >>> U
        DomainMatrix([[1, 2], [0, -2]], (2, 2), ZZ)
        >>> L.is_lower and U.is_upper and D.is_diagonal
        True
        >>> L * D.to_field().inv() * U == P * A.to_field()
        True
        >>> I, d = D.inv_den()
        >>> L * I * U == d * P * A
        True

        See Also
        ========

        sympy.polys.matrices.ddm.DDM.fflu

        References
        ==========

        .. [1]  Nakos, G. C., Turner, P. R., & Williams, R. M. (1997). Fraction-free
                algorithms for linear and polynomial equations. ACM SIGSAM Bulletin,
                31(3), 11-19. https://doi.org/10.1145/271130.271133
        .. [2]  Middeke, J.; Jeffrey, D.J.; Koutschan, C. (2020), "Common Factors
                in Fraction-Free Matrix Decompositions", Mathematics in Computer Science,
                15 (4): 589–608, arXiv:2005.12380, doi:10.1007/s11786-020-00495-9
        .. [3]  https://en.wikipedia.org/wiki/Bareiss_algorithm
        """
        from_rep = self.from_rep
        P, L, D, U = self.rep.fflu()
        return from_rep(P), from_rep(L), from_rep(D), from_rep(U)

    def _solve(A, b):
        # XXX: Not sure about this method or its signature. It is just created
        # because it is needed by the holonomic module.
        if A.shape[0] != b.shape[0]:
            raise DMShapeError("Shape")
        if A.domain != b.domain or not A.domain.is_Field:
            raise DMNotAField('Not a field')
        Aaug = A.hstack(b)
        Arref, pivots = Aaug.rref()
        particular = Arref.from_rep(Arref.rep.particular())
        nullspace_rep, nonpivots = Arref[:,:-1].rep.nullspace()
        nullspace = Arref.from_rep(nullspace_rep)
        return particular, nullspace

    def charpoly(self):
        r"""
        Characteristic polynomial of a square matrix.

        Computes the characteristic polynomial in a fully expanded form using
        division free arithmetic. If a factorization of the characteristic
        polynomial is needed then it is more efficient to call
        :meth:`charpoly_factor_list` than calling :meth:`charpoly` and then
        factorizing the result.

        Returns
        =======

        list: list of DomainElement
            coefficients of the characteristic polynomial

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.charpoly()
        [1, -5, -2]

        See Also
        ========

        charpoly_factor_list
            Compute the factorisation of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently than either the full factorisation or
            the fully expanded polynomial.
        """
        M = self
        K = M.domain

        factors = M.charpoly_factor_blocks()

        cp = [K.one]

        for f, mult in factors:
            for _ in range(mult):
                cp = dup_mul(cp, f, K)

        return cp

    def charpoly_factor_list(self):
        """
        Full factorization of the characteristic polynomial.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        Compute the factorization of the characteristic polynomial:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the unfactorized characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        The same calculations with ``Matrix``:

        >>> M.to_Matrix().charpoly().as_expr()
        lambda**4 - 25*lambda**3 + 203*lambda**2 - 495*lambda - 324
        >>> M.to_Matrix().charpoly().as_expr().factor()
        (lambda - 9)**2*(lambda**2 - 7*lambda - 4)

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A full factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Expanded form of the characteristic polynomial.
        charpoly_factor_blocks
            A partial factorisation of the characteristic polynomial that can
            be computed more efficiently.
        """
        M = self
        K = M.domain

        # It is more efficient to start from the partial factorization provided
        # for free by M.charpoly_factor_blocks than the expanded M.charpoly.
        factors = M.charpoly_factor_blocks()

        factors_irreducible = []

        for factor_i, mult_i in factors:

            _, factors_list = dup_factor_list(factor_i, K)

            for factor_j, mult_j in factors_list:
                factors_irreducible.append((factor_j, mult_i * mult_j))

        return _collect_factors(factors_irreducible)

    def charpoly_factor_blocks(self):
        """
        Partial factorisation of the characteristic polynomial.

        This factorisation arises from a block structure of the matrix (if any)
        and so the factors are not guaranteed to be irreducible. The
        :meth:`charpoly_factor_blocks` method is the most efficient way to get
        a representation of the characteristic polynomial but the result is
        neither fully expanded nor fully factored.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import ZZ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], ZZ)

        This computes a partial factorization using only the block structure of
        the matrix to reveal factors:

        >>> M.charpoly_factor_blocks()
        [([1, -18, 81], 1), ([1, -7, -4], 1)]

        These factors correspond to the two diagonal blocks in the matrix:

        >>> DM([[6, -1], [9, 12]], ZZ).charpoly()
        [1, -18, 81]
        >>> DM([[1, 2], [5, 6]], ZZ).charpoly()
        [1, -7, -4]

        Use :meth:`charpoly_factor_list` to get a complete factorization into
        irreducibles:

        >>> M.charpoly_factor_list()
        [([1, -9], 2), ([1, -7, -4], 1)]

        Use :meth:`charpoly` to get the expanded characteristic polynomial:

        >>> M.charpoly()
        [1, -25, 203, -495, -324]

        Returns
        =======

        list: list of pairs (factor, multiplicity)
            A partial factorization of the characteristic polynomial.

        See Also
        ========

        charpoly
            Compute the fully expanded characteristic polynomial.
        charpoly_factor_list
            Compute a full factorization of the characteristic polynomial.
        """
        M = self

        if not M.is_square:
            raise DMNonSquareMatrixError("not square")

        # scc returns indices that permute the matrix into block triangular
        # form and can extract the diagonal blocks. M.charpoly() is equal to
        # the product of the diagonal block charpolys.
        components = M.scc()

        block_factors = []

        for indices in components:
            block = M.extract(indices, indices)
            block_factors.append((block.charpoly_base(), 1))

        return _collect_factors(block_factors)

    def charpoly_base(self):
        """
        Base case for :meth:`charpoly_factor_blocks` after block decomposition.

        This method is used internally by :meth:`charpoly_factor_blocks` as the
        base case for computing the characteristic polynomial of a block. It is
        more efficient to call :meth:`charpoly_factor_blocks`, :meth:`charpoly`
        or :meth:`charpoly_factor_list` rather than call this method directly.

        This will use either the dense or the sparse implementation depending
        on the sparsity of the matrix and will clear denominators if possible
        before calling :meth:`charpoly_berk` to compute the characteristic
        polynomial using the Berkowitz algorithm.

        See Also
        ========

        charpoly
        charpoly_factor_list
        charpoly_factor_blocks
        charpoly_berk
        """
        M = self
        K = M.domain

        # It seems that the sparse implementation is always faster for random
        # matrices with fewer than 50% non-zero entries. This does not seem to
        # depend on domain, size, bit count etc.
        density = self.nnz() / self.shape[0]**2
        if density < 0.5:
            M = M.to_sparse()
        else:
            M = M.to_dense()

        # Clearing denominators is always more efficient if it can be done.
        # Doing it here after block decomposition is good because each block
        # might have a smaller denominator. However it might be better for
        # charpoly and charpoly_factor_list to restore the denominators only at
        # the very end so that they can call e.g. dup_factor_list before
        # restoring the denominators. The methods would need to be changed to
        # return (poly, denom) pairs to make that work though.
        clear_denoms = K.is_Field and K.has_assoc_Ring

        if clear_denoms:
            clear_denoms = True
            d, M = M.clear_denoms(convert=True)
            d = d.element
            K_f = K
            K_r = M.domain

        # Berkowitz algorithm over K_r.
        cp = M.charpoly_berk()

        if clear_denoms:
            # Restore the denominator in the charpoly over K_f.
            #
            # If M = N/d then p_M(x) = p_N(x*d)/d^n.
            cp = dup_convert(cp, K_r, K_f)
            p = [K_f.one, K_f.zero]
            q = [K_f.one/d]
            cp = dup_transform(cp, p, q, K_f)

        return cp

    def charpoly_berk(self):
        """Compute the characteristic polynomial using the Berkowitz algorithm.

        This method directly calls the underlying implementation of the
        Berkowitz algorithm (:meth:`sympy.polys.matrices.dense.ddm_berk` or
        :meth:`sympy.polys.matrices.sdm.sdm_berk`).

        This is used by :meth:`charpoly` and other methods as the base case for
        for computing the characteristic polynomial. However those methods will
        apply other optimizations such as block decomposition, clearing
        denominators and converting between dense and sparse representations
        before calling this method. It is more efficient to call those methods
        instead of this one but this method is provided for direct access to
        the Berkowitz algorithm.

        Examples
        ========

        >>> from sympy.polys.matrices import DM
        >>> from sympy import QQ
        >>> M = DM([[6, -1, 0, 0],
        ...         [9, 12, 0, 0],
        ...         [0,  0, 1, 2],
        ...         [0,  0, 5, 6]], QQ)
        >>> M.charpoly_berk()
        [1, -25, 203, -495, -324]

        See Also
        ========

        charpoly
        charpoly_base
        charpoly_factor_list
        charpoly_factor_blocks
        sympy.polys.matrices.dense.ddm_berk
        sympy.polys.matrices.sdm.sdm_berk
        """
        return self.rep.charpoly()

    @classmethod
    def eye(cls, shape, domain):
        r"""
        Return identity matrix of size n or shape (m, n).

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.eye(3, QQ)
        DomainMatrix({0: {0: 1}, 1: {1: 1}, 2: {2: 1}}, (3, 3), QQ)

        """
        if isinstance(shape, int):
            shape = (shape, shape)
        return cls.from_rep(SDM.eye(shape, domain))

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        r"""
        Return diagonal matrix with entries from ``diagonal``.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> DomainMatrix.diag([ZZ(5), ZZ(6)], ZZ)
        DomainMatrix({0: {0: 5}, 1: {1: 6}}, (2, 2), ZZ)

        """
        if shape is None:
            N = len(diagonal)
            shape = (N, N)
        return cls.from_rep(SDM.diag(diagonal, domain, shape))

    @classmethod
    def zeros(cls, shape, domain, *, fmt='sparse'):
        """Returns a zero DomainMatrix of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.zeros((2, 3), QQ)
        DomainMatrix({}, (2, 3), QQ)

        """
        return cls.from_rep(SDM.zeros(shape, domain))

    @classmethod
    def ones(cls, shape, domain):
        """Returns a DomainMatrix of 1s, of size shape, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.ones((2,3), QQ)
        DomainMatrix([[1, 1, 1], [1, 1, 1]], (2, 3), QQ)

        """
        return cls.from_rep(DDM.ones(shape, domain).to_dfm_or_ddm())

    def __eq__(A, B):
        r"""
        Checks for two DomainMatrix matrices to be equal or not

        Parameters
        ==========

        A, B: DomainMatrix
            to check equality

        Returns
        =======

        Boolean
            True for equal, else False

        Raises
        ======

        NotImplementedError
            If B is not a DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(1), ZZ(1)],
        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.__eq__(A)
        True
        >>> A.__eq__(B)
        False

        """
        if not isinstance(A, type(B)):
            return NotImplemented
        return A.domain == B.domain and A.rep == B.rep

    def unify_eq(A, B):
        if A.shape != B.shape:
            return False
        if A.domain != B.domain:
            A, B = A.unify(B)
        return A == B

    def lll(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.
        See [1]_ and [2]_.

        Parameters
        ==========

        delta : QQ, optional
            The Lovász parameter. Must be in the interval (0.25, 1), with larger
            values producing a more reduced basis. The default is 0.75 for
            historical reasons.

        Returns
        =======

        The reduced basis as a DomainMatrix over ZZ.

        Throws
        ======

        DMValueError: if delta is not in the range (0.25, 1)
        DMShapeError: if the matrix is not of shape (m, n) with m <= n
        DMDomainError: if the matrix domain is not ZZ
        DMRankError: if the matrix contains linearly dependent rows

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> x = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> y = DM([[10, -3, -2, 8, -4],
        ...         [3, -9, 8, 1, -11],
        ...         [-3, 13, -9, -3, -9],
        ...         [-12, -7, -11, 9, -1]], ZZ)
        >>> assert x.lll(delta=QQ(5, 6)) == y

        Notes
        =====

        The implementation is derived from the Maple code given in Figures 4.3
        and 4.4 of [3]_ (pp.68-69). It uses the efficient method of only calculating
        state updates as they are required.

        See also
        ========

        lll_transform

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm
        .. [2] https://web.archive.org/web/20221029115428/https://web.cs.elte.hu/~lovasz/scans/lll.pdf
        .. [3] Murray R. Bremner, "Lattice Basis Reduction: An Introduction to the LLL Algorithm and Its Applications"

        """
        return DomainMatrix.from_rep(A.rep.lll(delta=delta))

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm
        and returns the reduced basis and transformation matrix.

        Explanation
        ===========

        Parameters, algorithm and basis are the same as for :meth:`lll` except that
        the return value is a tuple `(B, T)` with `B` the reduced basis and
        `T` a transformation matrix. The original basis `A` is transformed to
        `B` with `T*A == B`. If only `B` is needed then :meth:`lll` should be
        used as it is a little faster.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ, QQ
        >>> from sympy.polys.matrices import DM
        >>> X = DM([[1, 0, 0, 0, -20160],
        ...         [0, 1, 0, 0, 33768],
        ...         [0, 0, 1, 0, 39578],
        ...         [0, 0, 0, 1, 47757]], ZZ)
        >>> B, T = X.lll_transform(delta=QQ(5, 6))
        >>> T * X == B
        True

        See also
        ========

        lll

        """
        reduced, transform = A.rep.lll_transform(delta=delta)
        return DomainMatrix.from_rep(reduced), DomainMatrix.from_rep(transform)


def _collect_factors(factors_list):
    """
    Collect repeating factors and sort.

    >>> from sympy.polys.matrices.domainmatrix import _collect_factors
    >>> _collect_factors([([1, 2], 2), ([1, 4], 3), ([1, 2], 5)])
    [([1, 4], 3), ([1, 2], 7)]
    """
    factors = Counter()
    for factor, exponent in factors_list:
        factors[tuple(factor)] += exponent

    factors_list = [(list(f), e) for f, e in factors.items()]

    return _sort_factors(factors_list)
<|endoftext|>from typing import TypeVar, Protocol


T = TypeVar('T')


class RingElement(Protocol):
    """A ring element.

    Must support ``+``, ``-``, ``*``, ``**`` and ``-``.
    """
    def __add__(self: T, other: T, /) -> T: ...
    def __sub__(self: T, other: T, /) -> T: ...
    def __mul__(self: T, other: T, /) -> T: ...
    def __pow__(self: T, other: int, /) -> T: ...
    def __neg__(self: T, /) -> T: ...
<|endoftext|>#
# sympy.polys.matrices.dfm
#
# This modules defines the DFM class which is a wrapper for dense flint
# matrices as found in python-flint.
#
# As of python-flint 0.4.1 matrices over the following domains can be supported
# by python-flint:
#
#   ZZ: flint.fmpz_mat
#   QQ: flint.fmpq_mat
#   GF(p): flint.nmod_mat (p prime and p < ~2**62)
#
# The underlying flint library has many more domains, but these are not yet
# supported by python-flint.
#
# The DFM class is a wrapper for the flint matrices and provides a common
# interface for all supported domains that is interchangeable with the DDM
# and SDM classes so that DomainMatrix can be used with any as its internal
# matrix representation.
#

# TODO:
#
# Implement the following methods that are provided by python-flint:
#
# - hnf (Hermite normal form)
# - snf (Smith normal form)
# - minpoly
# - is_hnf
# - is_snf
# - rank
#
# The other types DDM and SDM do not have these methods and the algorithms
# for hnf, snf and rank are already implemented. Algorithms for minpoly,
# is_hnf and is_snf would need to be added.
#
# Add more methods to python-flint to expose more of Flint's functionality
# and also to make some of the above methods simpler or more efficient e.g.
# slicing, fancy indexing etc.

from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import import_module
from sympy.utilities.decorator import doctest_depends_on

from sympy.polys.domains import ZZ, QQ

from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMRankError,
    DMShapeError,
    DMValueError,
)


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['*']


flint = import_module('flint')


__all__ = ['DFM']


@doctest_depends_on(ground_types=['flint'])
class DFM:
    """
    Dense FLINT matrix. This class is a wrapper for matrices from python-flint.

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.matrices.dfm import DFM
    >>> dfm = DFM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> dfm
    [[1, 2], [3, 4]]
    >>> dfm.rep
    [1, 2]
    [3, 4]
    >>> type(dfm.rep)  # doctest: +SKIP
    <class 'flint._flint.fmpz_mat'>

    Usually, the DFM class is not instantiated directly, but is created as the
    internal representation of :class:`~.DomainMatrix`. When
    `SYMPY_GROUND_TYPES` is set to `flint` and `python-flint` is installed, the
    :class:`DFM` class is used automatically as the internal representation of
    :class:`~.DomainMatrix` in dense format if the domain is supported by
    python-flint.

    >>> from sympy.polys.matrices.domainmatrix import DM
    >>> dM = DM([[1, 2], [3, 4]], ZZ)
    >>> dM.rep
    [[1, 2], [3, 4]]

    A :class:`~.DomainMatrix` can be converted to :class:`DFM` by calling the
    :meth:`to_dfm` method:

    >>> dM.to_dfm()
    [[1, 2], [3, 4]]

    """

    fmt = 'dense'
    is_DFM = True
    is_DDM = False

    def __new__(cls, rowslist, shape, domain):
        """Construct from a nested list."""
        flint_mat = cls._get_flint_func(domain)

        if 0 not in shape:
            try:
                rep = flint_mat(rowslist)
            except (ValueError, TypeError):
                raise DMBadInputError(f"Input should be a list of list of {domain}")
        else:
            rep = flint_mat(*shape)

        return cls._new(rep, shape, domain)

    @classmethod
    def _new(cls, rep, shape, domain):
        """Internal constructor from a flint matrix."""
        cls._check(rep, shape, domain)
        obj = object.__new__(cls)
        obj.rep = rep
        obj.shape = obj.rows, obj.cols = shape
        obj.domain = domain
        return obj

    def _new_rep(self, rep):
        """Create a new DFM with the same shape and domain but a new rep."""
        return self._new(rep, self.shape, self.domain)

    @classmethod
    def _check(cls, rep, shape, domain):
        repshape = (rep.nrows(), rep.ncols())
        if repshape != shape:
            raise DMBadInputError("Shape of rep does not match shape of DFM")
        if domain == ZZ and not isinstance(rep, flint.fmpz_mat):
            raise RuntimeError("Rep is not a flint.fmpz_mat")
        elif domain == QQ and not isinstance(rep, flint.fmpq_mat):
            raise RuntimeError("Rep is not a flint.fmpq_mat")
        elif domain.is_FF and not isinstance(rep, (flint.fmpz_mod_mat, flint.nmod_mat)):
            raise RuntimeError("Rep is not a flint.fmpz_mod_mat or flint.nmod_mat")
        elif domain not in (ZZ, QQ) and not domain.is_FF:
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @classmethod
    def _supports_domain(cls, domain):
        """Return True if the given domain is supported by DFM."""
        return domain in (ZZ, QQ) or domain.is_FF and domain._is_flint

    @classmethod
    def _get_flint_func(cls, domain):
        """Return the flint matrix class for the given domain."""
        if domain == ZZ:
            return flint.fmpz_mat
        elif domain == QQ:
            return flint.fmpq_mat
        elif domain.is_FF:
            c = domain.characteristic()
            if isinstance(domain.one, flint.nmod):
                _cls = flint.nmod_mat
                def _func(*e):
                    if len(e) == 1 and isinstance(e[0], flint.nmod_mat):
                        return _cls(e[0])
                    else:
                        return _cls(*e, c)
            else:
                m = flint.fmpz_mod_ctx(c)
                _func = lambda *e: flint.fmpz_mod_mat(*e, m)
            return _func
        else:
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    @property
    def _func(self):
        """Callable to create a flint matrix of the same domain."""
        return self._get_flint_func(self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return str(self.to_ddm())

    def __repr__(self):
        """Return ``repr(self)``."""
        return f'DFM{repr(self.to_ddm())[3:]}'

    def __eq__(self, other):
        """Return ``self == other``."""
        if not isinstance(other, DFM):
            return NotImplemented
        # Compare domains first because we do *not* want matrices with
        # different domains to be equal but e.g. a flint fmpz_mat and fmpq_mat
        # with the same entries will compare equal.
        return self.domain == other.domain and self.rep == other.rep

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """Construct from a nested list."""
        return cls(rowslist, shape, domain)

    def to_list(self):
        """Convert to a nested list."""
        return self.rep.tolist()

    def copy(self):
        """Return a copy of self."""
        return self._new_rep(self._func(self.rep))

    def to_ddm(self):
        """Convert to a DDM."""
        return DDM.from_list(self.to_list(), self.shape, self.domain)

    def to_sdm(self):
        """Convert to a SDM."""
        return SDM.from_list(self.to_list(), self.shape, self.domain)

    def to_dfm(self):
        """Return self."""
        return self

    def to_dfm_or_ddm(self):
        """
        Convert to a :class:`DFM`.

        This :class:`DFM` method exists to parallel the :class:`~.DDM` and
        :class:`~.SDM` methods. For :class:`DFM` it will always return self.

        See Also
        ========

        to_ddm
        to_sdm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return self

    @classmethod
    def from_ddm(cls, ddm):
        """Convert from a DDM."""
        return cls.from_list(ddm.to_list(), ddm.shape, ddm.domain)

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """Inverse of :meth:`to_list_flat`."""
        func = cls._get_flint_func(domain)
        try:
            rep = func(*shape, elements)
        except ValueError:
            raise DMBadInputError(f"Incorrect number of elements for shape {shape}")
        except TypeError:
            raise DMBadInputError(f"Input should be a list of {domain}")
        return cls(rep, shape, domain)

    def to_list_flat(self):
        """Convert to a flat list."""
        return self.rep.entries()

    def to_flat_nz(self):
        """Convert to a flat list of non-zeros."""
        return self.to_ddm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """Inverse of :meth:`to_flat_nz`."""
        return DDM.from_flat_nz(elements, data, domain).to_dfm()

    def to_dod(self):
        """Convert to a DOD."""
        return self.to_ddm().to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """Inverse of :meth:`to_dod`."""
        return DDM.from_dod(dod, shape, domain).to_dfm()

    def to_dok(self):
        """Convert to a DOK."""
        return self.to_ddm().to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """Inverse of :math:`to_dod`."""
        return DDM.from_dok(dok, shape, domain).to_dfm()

    def iter_values(self):
        """Iterate over the non-zero values of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield rep[i, j]

    def iter_items(self):
        """Iterate over indices and values of nonzero elements of the matrix."""
        m, n = self.shape
        rep = self.rep
        for i in range(m):
            for j in range(n):
                repij = rep[i, j]
                if repij:
                    yield ((i, j), repij)

    def convert_to(self, domain):
        """Convert to a new domain."""
        if domain == self.domain:
            return self.copy()
        elif domain == QQ and self.domain == ZZ:
            return self._new(flint.fmpq_mat(self.rep), self.shape, domain)
        elif self._supports_domain(domain):
            # XXX: Use more efficient conversions when possible.
            return self.to_ddm().convert_to(domain).to_dfm()
        else:
            # It is the callers responsibility to convert to DDM before calling
            # this method if the domain is not supported by DFM.
            raise NotImplementedError("Only ZZ and QQ are supported by DFM")

    def getitem(self, i, j):
        """Get the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            return self.rep[i, j]
        except ValueError:
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    def setitem(self, i, j, value):
        """Set the ``(i, j)``-th entry."""
        # XXX: flint matrices do not support negative indices
        # XXX: They also raise ValueError instead of IndexError
        m, n = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            self.rep[i, j] = value
        except ValueError:
            raise IndexError(f"Invalid indices ({i}, {j}) for Matrix of shape {self.shape}")

    def _extract(self, i_indices, j_indices):
        """Extract a submatrix with no checking."""
        # Indices must be positive and in range.
        M = self.rep
        lol = [[M[i, j] for j in j_indices] for i in i_indices]
        shape = (len(i_indices), len(j_indices))
        return self.from_list(lol, shape, self.domain)

    def extract(self, rowslist, colslist):
        """Extract a submatrix."""
        # XXX: flint matrices do not support fancy indexing or negative indices
        #
        # Check and convert negative indices before calling _extract.
        m, n = self.shape

        new_rows = []
        new_cols = []

        for i in rowslist:
            if i < 0:
                i_pos = i + m
            else:
                i_pos = i
            if not 0 <= i_pos < m:
                raise IndexError(f"Invalid row index {i} for Matrix of shape {self.shape}")
            new_rows.append(i_pos)

        for j in colslist:
            if j < 0:
                j_pos = j + n
            else:
                j_pos = j
            if not 0 <= j_pos < n:
                raise IndexError(f"Invalid column index {j} for Matrix of shape {self.shape}")
            new_cols.append(j_pos)

        return self._extract(new_rows, new_cols)

    def extract_slice(self, rowslice, colslice):
        """Slice a DFM."""
        # XXX: flint matrices do not support slicing
        m, n = self.shape
        i_indices = range(m)[rowslice]
        j_indices = range(n)[colslice]
        return self._extract(i_indices, j_indices)

    def neg(self):
        """Negate a DFM matrix."""
        return self._new_rep(-self.rep)

    def add(self, other):
        """Add two DFM matrices."""
        return self._new_rep(self.rep + other.rep)

    def sub(self, other):
        """Subtract two DFM matrices."""
        return self._new_rep(self.rep - other.rep)

    def mul(self, other):
        """Multiply a DFM matrix from the right by a scalar."""
        return self._new_rep(self.rep * other)

    def rmul(self, other):
        """Multiply a DFM matrix from the left by a scalar."""
        return self._new_rep(other * self.rep)

    def mul_elementwise(self, other):
        """Elementwise multiplication of two DFM matrices."""
        # XXX: flint matrices do not support elementwise multiplication
        return self.to_ddm().mul_elementwise(other.to_ddm()).to_dfm()

    def matmul(self, other):
        """Multiply two DFM matrices."""
        shape = (self.rows, other.cols)
        return self._new(self.rep * other.rep, shape, self.domain)

    # XXX: For the most part DomainMatrix does not expect DDM, SDM, or DFM to
    # have arithmetic operators defined. The only exception is negation.
    # Perhaps that should be removed.

    def __neg__(self):
        """Negate a DFM matrix."""
        return self.neg()

    @classmethod
    def zeros(cls, shape, domain):
        """Return a zero DFM matrix."""
        func = cls._get_flint_func(domain)
        return cls._new(func(*shape), shape, domain)

    # XXX: flint matrices do not have anything like ones or eye
    # In the methods below we convert to DDM and then back to DFM which is
    # probably about as efficient as implementing these methods directly.

    @classmethod
    def ones(cls, shape, domain):
        """Return a one DFM matrix."""
        # XXX: flint matrices do not have anything like ones
        return DDM.ones(shape, domain).to_dfm()

    @classmethod
    def eye(cls, n, domain):
        """Return the identity matrix of size n."""
        # XXX: flint matrices do not have anything like eye
        return DDM.eye(n, domain).to_dfm()

    @classmethod
    def diag(cls, elements, domain):
        """Return a diagonal matrix."""
        return DDM.diag(elements, domain).to_dfm()

    def applyfunc(self, func, domain):
        """Apply a function to each entry of a DFM matrix."""
        return self.to_ddm().applyfunc(func, domain).to_dfm()

    def transpose(self):
        """Transpose a DFM matrix."""
        return self._new(self.rep.transpose(), (self.cols, self.rows), self.domain)

    def hstack(self, *others):
        """Horizontally stack matrices."""
        return self.to_ddm().hstack(*[o.to_ddm() for o in others]).to_dfm()

    def vstack(self, *others):
        """Vertically stack matrices."""
        return self.to_ddm().vstack(*[o.to_ddm() for o in others]).to_dfm()

    def diagonal(self):
        """Return the diagonal of a DFM matrix."""
        M = self.rep
        m, n = self.shape
        return [M[i, i] for i in range(min(m, n))]

    def is_upper(self):
        """Return ``True`` if the matrix is upper triangular."""
        M = self.rep
        for i in range(self.rows):
            for j in range(min(i, self.cols)):
                if M[i, j]:
                    return False
        return True

    def is_lower(self):
        """Return ``True`` if the matrix is lower triangular."""
        M = self.rep
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if M[i, j]:
                    return False
        return True

    def is_diagonal(self):
        """Return ``True`` if the matrix is diagonal."""
        return self.is_upper() and self.is_lower()

    def is_zero_matrix(self):
        """Return ``True`` if the matrix is the zero matrix."""
        M = self.rep
        for i in range(self.rows):
            for j in range(self.cols):
                if M[i, j]:
                    return False
        return True

    def nnz(self):
        """Return the number of non-zero elements in the matrix."""
        return self.to_ddm().nnz()

    def scc(self):
        """Return the strongly connected components of the matrix."""
        return self.to_ddm().scc()

    @doctest_depends_on(ground_types='flint')
    def det(self):
        """
        Compute the determinant of the matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.det()
        -2

        Notes
        =====

        Calls the ``.det()`` method of the underlying FLINT matrix.

        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_det`` or
        ``fmpq_mat_det`` respectively.

        At the time of writing the implementation of ``fmpz_mat_det`` uses one
        of several algorithms depending on the size of the matrix and bit size
        of the entries. The algorithms used are:

        - Cofactor for very small (up to 4x4) matrices.
        - Bareiss for small (up to 25x25) matrices.
        - Modular algorithms for larger matrices (up to 60x60) or for larger
          matrices with large bit sizes.
        - Modular "accelerated" for larger matrices (60x60 upwards) if the bit
          size is smaller than the dimensions of the matrix.

        The implementation of ``fmpq_mat_det`` clears denominators from each
        row (not the whole matrix) and then calls ``fmpz_mat_det`` and divides
        by the product of the denominators.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.det
            Higher level interface to compute the determinant of a matrix.
        """
        # XXX: At least the first three algorithms described above should also
        # be implemented in the pure Python DDM and SDM classes which at the
        # time of writng just use Bareiss for all matrices and domains.
        # Probably in Python the thresholds would be different though.
        return self.rep.det()

    @doctest_depends_on(ground_types='flint')
    def charpoly(self):
        """
        Compute the characteristic polynomial of the matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm()  # need ground types = 'flint'
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.charpoly()
        [1, -5, -2]

        Notes
        =====

        Calls the ``.charpoly()`` method of the underlying FLINT matrix.

        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_charpoly`` or
        ``fmpq_mat_charpoly`` respectively.

        At the time of writing the implementation of ``fmpq_mat_charpoly``
        clears a denominator from the whole matrix and then calls
        ``fmpz_mat_charpoly``. The coefficients of the characteristic
        polynomial are then multiplied by powers of the denominator.

        The ``fmpz_mat_charpoly`` method uses a modular algorithm with CRT
        reconstruction. The modular algorithm uses ``nmod_mat_charpoly`` which
        uses Berkowitz for small matrices and non-prime moduli or otherwise
        the Danilevsky method.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
            Higher level interface to compute the characteristic polynomial of
            a matrix.
        """
        # FLINT polynomial coefficients are in reverse order compared to SymPy.
        return self.rep.charpoly().coeffs()[::-1]

    @doctest_depends_on(ground_types='flint')
    def inv(self):
        """
        Compute the inverse of a matrix using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> dfm.inv()
        [[-2, 1], [3/2, -1/2]]
        >>> dfm.matmul(dfm.inv())
        [[1, 0], [0, 1]]

        Notes
        =====

        Calls the ``.inv()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_inv`` and
        ``fmpq_mat_inv`` respectively. The ``fmpz_mat_inv`` method computes an
        inverse with denominator. This is implemented by calling
        ``fmpz_mat_solve`` (see notes in :meth:`lu_solve` about the algorithm).

        The ``fmpq_mat_inv`` method clears denominators from each row and then
        multiplies those into the rhs identity matrix before calling
        ``fmpz_mat_solve``.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.inv
            Higher level method for computing the inverse of a matrix.
        """
        # TODO: Implement similar algorithms for DDM and SDM.
        #
        # XXX: The flint fmpz_mat and fmpq_mat inv methods both return fmpq_mat
        # by default. The fmpz_mat method has an optional argument to return
        # fmpz_mat instead for unimodular matrices.
        #
        # The convention in DomainMatrix is to raise an error if the matrix is
        # not over a field regardless of whether the matrix is invertible over
        # its domain or over any associated field. Maybe DomainMatrix.inv
        # should be changed to always return a matrix over an associated field
        # except with a unimodular argument for returning an inverse over a
        # ring if possible.
        #
        # For now we follow the existing DomainMatrix convention...
        K = self.domain
        m, n = self.shape

        if m != n:
            raise DMNonSquareMatrixError("cannot invert a non-square matrix")

        if K == ZZ:
            raise DMDomainError("field expected, got %s" % K)
        elif K == QQ or K.is_FF:
            try:
                return self._new_rep(self.rep.inv())
            except ZeroDivisionError:
                raise DMNonInvertibleMatrixError("matrix is not invertible")
        else:
            # If more domains are added for DFM then we will need to consider
            # what happens here.
            raise NotImplementedError("DFM.inv() is not implemented for %s" % K)

    def lu(self):
        """Return the LU decomposition of the matrix."""
        L, U, swaps = self.to_ddm().lu()
        return L.to_dfm(), U.to_dfm(), swaps

    def qr(self):
        """Return the QR decomposition of the matrix."""
        Q, R = self.to_ddm().qr()
        return Q.to_dfm(), R.to_dfm()

    # XXX: The lu_solve function should be renamed to solve. Whether or not it
    # uses an LU decomposition is an implementation detail. A method called
    # lu_solve would make sense for a situation in which an LU decomposition is
    # reused several times to solve with different rhs but that would imply a
    # different call signature.
    #
    # The underlying python-flint method has an algorithm= argument so we could
    # use that and have e.g. solve_lu and solve_modular or perhaps also a
    # method= argument to choose between the two. Flint itself has more
    # possible algorithms to choose from than are exposed by python-flint.

    @doctest_depends_on(ground_types='flint')
    def lu_solve(self, rhs):
        """
        Solve a matrix equation using FLINT.

        Examples
        ========

        >>> from sympy import Matrix, QQ
        >>> M = Matrix([[1, 2], [3, 4]])
        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)
        >>> dfm
        [[1, 2], [3, 4]]
        >>> rhs = Matrix([1, 2]).to_DM().to_dfm().convert_to(QQ)
        >>> dfm.lu_solve(rhs)
        [[0], [1/2]]

        Notes
        =====

        Calls the ``.solve()`` method of the underlying FLINT matrix.

        For now this will raise an error if the domain is :ref:`ZZ` but will
        use the FLINT method for :ref:`QQ`.

        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_solve``
        and ``fmpq_mat_solve`` respectively. The ``fmpq_mat_solve`` method
        uses one of two algorithms:

        - For small matrices (<25 rows) it clears denominators between the
          matrix and rhs and uses ``fmpz_mat_solve``.
        - For larger matrices it uses ``fmpq_mat_solve_dixon`` which is a
          modular approach with CRT reconstruction over :ref:`QQ`.

        The ``fmpz_mat_solve`` method uses one of four algorithms:

        - For very small (<= 3x3) matrices it uses a Cramer's rule.
        - For small (<= 15x15) matrices it uses a fraction-free LU solve.
        - Otherwise it uses either Dixon or another multimodular approach.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
            Higher level interface to solve a matrix equation.
        """
        if not self.domain == rhs.domain:
            raise DMDomainError("Domains must match: %s != %s" % (self.domain, rhs.domain))

        # XXX: As for inv we should consider whether to return a matrix over
        # over an associated field or attempt to find a solution in the ring.
        # For now we follow the existing DomainMatrix convention...
        if not self.domain.is_Field:
            raise DMDomainError("Field expected, got %s" % self.domain)

        m, n = self.shape
        j, k = rhs.shape
        if m != j:
            raise DMShapeError("Matrix size mismatch: %s * %s vs %s * %s" % (m, n, j, k))
        sol_shape = (n, k)

        # XXX: The Flint solve method only handles square matrices. Probably
        # Flint has functions that could be used to solve non-square systems
        # but they are not exposed in python-flint yet. Alternatively we could
        # put something here using the features that are available like rref.
        if m != n:
            return self.to_ddm().lu_solve(rhs.to_ddm()).to_dfm()

        try:
            sol = self.rep.solve(rhs.rep)
        except ZeroDivisionError:
            raise DMNonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return self._new(sol, sol_shape, self.domain)

    def fflu(self):
        """
        Fraction-free LU decomposition of DFM.

        Explanation
        ===========

        Uses `python-flint` if possible for a matrix of
        integers otherwise uses the DDM method.

        See Also
        ========

        sympy.polys.matrices.ddm.DDM.fflu
        """
        if self.domain == ZZ:
            fflu = getattr(self.rep, 'fflu', None)
            if fflu is not None:
                P, L, D, U = self.rep.fflu()
                m, n = self.shape
                return (
                    self._new(P, (m, m), self.domain),
                    self._new(L, (m, m), self.domain),
                    self._new(D, (m, m), self.domain),
                    self._new(U, self.shape, self.domain)
                )
        ddm_p, ddm_l, ddm_d, ddm_u = self.to_ddm().fflu()
        P = ddm_p.to_dfm()
        L = ddm_l.to_dfm()
        D = ddm_d.to_dfm()
        U = ddm_u.to_dfm()
        return P, L, D, U

    def nullspace(self):
        """Return a basis for the nullspace of the matrix."""
        # Code to compute nullspace using flint:
        #
        # V, nullity = self.rep.nullspace()
        # V_dfm = self._new_rep(V)._extract(range(self.rows), range(nullity))
        #
        # XXX: That gives the nullspace but does not give us nonpivots. So we
        # use the slower DDM method anyway. It would be better to change the
        # signature of the nullspace method to not return nonpivots.
        #
        # XXX: Also python-flint exposes a nullspace method for fmpz_mat but
        # not for fmpq_mat. This is the reverse of the situation for DDM etc
        # which only allow nullspace over a field. The nullspace method for
        # DDM, SDM etc should be changed to allow nullspace over ZZ as well.
        # The DomainMatrix nullspace method does allow the domain to be a ring
        # but does not directly call the lower-level nullspace methods and uses
        # rref_den instead. Nullspace methods should also be added to all
        # matrix types in python-flint.
        ddm, nonpivots = self.to_ddm().nullspace()
        return ddm.to_dfm(), nonpivots

    def nullspace_from_rref(self, pivots=None):
        """Return a basis for the nullspace of the matrix."""
        # XXX: Use the flint nullspace method!!!
        sdm, nonpivots = self.to_sdm().nullspace_from_rref(pivots=pivots)
        return sdm.to_dfm(), nonpivots

    def particular(self):
        """Return a particular solution to the system."""
        return self.to_ddm().particular().to_dfm()

    def _lll(self, transform=False, delta=0.99, eta=0.51, rep='zbasis', gram='approx'):
        """Call the fmpz_mat.lll() method but check rank to avoid segfaults."""

        # XXX: There are tests that pass e.g. QQ(5,6) for delta. That fails
        # with a TypeError in flint because if QQ is fmpq then conversion with
        # float fails. We handle that here but there are two better fixes:
        #
        # - Make python-flint's fmpq convert with float(x)
        # - Change the tests because delta should just be a float.

        def to_float(x):
            if QQ.of_type(x):
                return float(x.numerator) / float(x.denominator)
            else:
                return float(x)

        delta = to_float(delta)
        eta = to_float(eta)

        if not 0.25 < delta < 1:
            raise DMValueError("delta must be between 0.25 and 1")

        # XXX: The flint lll method segfaults if the matrix is not full rank.
        m, n = self.shape
        if self.rep.rank() != m:
            raise DMRankError("Matrix must have full row rank for Flint LLL.")

        # Actually call the flint method.
        return self.rep.lll(transform=transform, delta=delta, eta=eta, rep=rep, gram=gram)

    @doctest_depends_on(ground_types='flint')
    def lll(self, delta=0.75):
        """Compute LLL-reduced basis using FLINT.

        See :meth:`lll_transform` for more information.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> M.to_DM().to_dfm().lll()
        [[2, 1, 0], [-1, 1, 3]]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
            Higher level interface to compute LLL-reduced basis.
        lll_transform
            Compute LLL-reduced basis and transform matrix.
        """
        if self.domain != ZZ:
            raise DMDomainError("ZZ expected, got %s" % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError("Matrix must not have more rows than columns.")

        rep = self._lll(delta=delta)
        return self._new_rep(rep)

    @doctest_depends_on(ground_types='flint')
    def lll_transform(self, delta=0.75):
        """Compute LLL-reduced basis and transform using FLINT.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6]]).to_DM().to_dfm()
        >>> M_lll, T = M.lll_transform()
        >>> M_lll
        [[2, 1, 0], [-1, 1, 3]]
        >>> T
        [[-2, 1], [3, -1]]
        >>> T.matmul(M) == M_lll
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.lll
            Higher level interface to compute LLL-reduced basis.
        lll
            Compute LLL-reduced basis without transform matrix.
        """
        if self.domain != ZZ:
            raise DMDomainError("ZZ expected, got %s" % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError("Matrix must not have more rows than columns.")

        rep, T = self._lll(transform=True, delta=delta)
        basis = self._new_rep(rep)
        T_dfm = self._new(T, (self.rows, self.rows), self.domain)
        return basis, T_dfm


# Avoid circular imports
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.ddm import SDM
<|endoftext|>"""

Module for the ddm_* routines for operating on a matrix in list of lists
matrix representation.

These routines are used internally by the DDM class which also provides a
friendlier interface for them. The idea here is to implement core matrix
routines in a way that can be applied to any simple list representation
without the need to use any particular matrix class. For example we can
compute the RREF of a matrix like:

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> M = [[1, 2, 3], [4, 5, 6]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1.0, 0.0, -1.0], [0, 1.0, 2.0]]

These are lower-level routines that work mostly in place.The routines at this
level should not need to know what the domain of the elements is but should
ideally document what operations they will use and what functions they need to
be provided with.

The next-level up is the DDM class which uses these routines but wraps them up
with an interface that handles copying etc and keeps track of the Domain of
the elements of the matrix:

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> M = DDM([[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]], (2, 3), QQ)
    >>> M
    [[1, 2, 3], [4, 5, 6]]
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    [[1, 0, -1], [0, 1, 2]]

"""
from __future__ import annotations
from operator import mul
from .exceptions import (
    DMShapeError,
    DMDomainError,
    DMNonInvertibleMatrixError,
    DMNonSquareMatrixError,
)
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement


#: Type variable for the elements of the matrix
T = TypeVar('T')

#: Type variable for the elements of the matrix that are in a ring
R = TypeVar('R', bound=RingElement)


def ddm_transpose(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """matrix transpose"""
    return list(map(list, zip(*matrix)))


def ddm_iadd(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a += b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] += bij


def ddm_isub(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a -= b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] -= bij


def ddm_ineg(a: list[list[R]]) -> None:
    """a <-- -a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = -aij


def ddm_imul(a: list[list[R]], b: R) -> None:
    """a <-- a*b"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = aij * b


def ddm_irmul(a: list[list[R]], b: R) -> None:
    """a <-- b*a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = b * aij


def ddm_imatmul(
    a: list[list[R]], b: Sequence[Sequence[R]], c: Sequence[Sequence[R]]
) -> None:
    """a += b @ c"""
    cT = list(zip(*c))

    for bi, ai in zip(b, a):
        for j, cTj in enumerate(cT):
            ai[j] = sum(map(mul, bi, cTj), ai[j])


def ddm_irref(a, _partial_pivot=False):
    """In-place reduced row echelon form of a matrix.

    Compute the reduced row echelon form of $a$. Modifies $a$ in place and
    returns a list of the pivot columns.

    Uses naive Gauss-Jordan elimination in the ground domain which must be a
    field.

    This routine is only really suitable for use with simple field domains like
    :ref:`GF(p)`, :ref:`QQ` and :ref:`QQ(a)` although even for :ref:`QQ` with
    larger matrices it is possibly more efficient to use fraction free
    approaches.

    This method is not suitable for use with rational function fields
    (:ref:`K(x)`) because the elements will blowup leading to costly gcd
    operations. In this case clearing denominators and using fraction free
    approaches is likely to be more efficient.

    For inexact numeric domains like :ref:`RR` and :ref:`CC` pass
    ``_partial_pivot=True`` to use partial pivoting to control rounding errors.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> from sympy import QQ
    >>> M = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1, 0, -1], [0, 1, 2]]
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        Higher level interface to this routine.
    ddm_irref_den
        The fraction free version of this routine.
    sdm_irref
        A sparse version of this routine.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form
    """
    # We compute aij**-1 below and then use multiplication instead of division
    # in the innermost loop. The domain here is a field so either operation is
    # defined. There are significant performance differences for some domains
    # though. In the case of e.g. QQ or QQ(x) inversion is free but
    # multiplication and division have the same cost so it makes no difference.
    # In cases like GF(p), QQ<sqrt(2)>, RR or CC though multiplication is
    # faster than division so reusing a precomputed inverse for many
    # multiplications can be a lot faster. The biggest win is QQ<a> when
    # deg(minpoly(a)) is large.
    #
    # With domains like QQ(x) this can perform badly for other reasons.
    # Typically the initial matrix has simple denominators and the
    # fraction-free approach with exquo (ddm_irref_den) will preserve that
    # property throughout. The method here causes denominator blowup leading to
    # expensive gcd reductions in the intermediate expressions. With many
    # generators like QQ(x,y,z,...) this is extremely bad.
    #
    # TODO: Use a nontrivial pivoting strategy to control intermediate
    # expression growth. Rearranging rows and/or columns could defer the most
    # complicated elements until the end. If the first pivot is a
    # complicated/large element then the first round of reduction will
    # immediately introduce expression blowup across the whole matrix.

    # a is (m x n)
    m = len(a)
    if not m:
        return []
    n = len(a[0])

    i = 0
    pivots = []

    for j in range(n):
        # Proper pivoting should be used for all domains for performance
        # reasons but it is only strictly needed for RR and CC (and possibly
        # other domains like RR(x)). This path is used by DDM.rref() if the
        # domain is RR or CC. It uses partial (row) pivoting based on the
        # absolute value of the pivot candidates.
        if _partial_pivot:
            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
            a[i], a[ip] = a[ip], a[i]

        # pivot
        aij = a[i][j]

        # zero-pivot
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # next column
                continue

        # normalise row
        ai = a[i]
        aijinv = aij**-1
        for l in range(j, n):
            ai[l] *= aijinv # ai[j] = one

        # eliminate above and below to the right
        for k, ak in enumerate(a):
            if k == i or not ak[j]:
                continue
            akj = ak[j]
            ak[j] -= akj # ak[j] = zero
            for l in range(j+1, n):
                ak[l] -= akj * ai[l]

        # next row
        pivots.append(j)
        i += 1

        # no more rows?
        if i >= m:
            break

    return pivots


def ddm_irref_den(a, K):
    """a  <--  rref(a); return (den, pivots)

    Compute the fraction-free reduced row echelon form (RREF) of $a$. Modifies
    $a$ in place and returns a tuple containing the denominator of the RREF and
    a list of the pivot columns.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref_den
    >>> from sympy import ZZ, Matrix
    >>> M = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)]]
    >>> den, pivots = ddm_irref_den(M, ZZ)
    >>> M
    [[-3, 0, 3], [0, -3, -6]]
    >>> den
    -3
    >>> pivots
    [0, 1]
    >>> Matrix(M).rref()[0]
    Matrix([
    [1, 0, -1],
    [0, 1,  2]])

    See Also
    ========

    ddm_irref
        A version of this routine that uses field division.
    sdm_irref
        A sparse version of :func:`ddm_irref`.
    sdm_rref_den
        A sparse version of :func:`ddm_irref_den`.
    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher level interface.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # A simpler presentation of this algorithm is given in [1]:
    #
    # Given an n x n matrix A and n x 1 matrix b:
    #
    #   for i in range(n):
    #       if i != 0:
    #           d = a[i-1][i-1]
    #       for j in range(n):
    #           if j == i:
    #               continue
    #           b[j] = a[i][i]*b[j] - a[j][i]*b[i]
    #           for k in range(n):
    #               a[j][k] = a[i][i]*a[j][k] - a[j][i]*a[i][k]
    #               if i != 0:
    #                   a[j][k] /= d
    #
    # Our version here is a bit more complicated because:
    #
    #  1. We use row-swaps to avoid zero pivots.
    #  2. We allow for some columns to be missing pivots.
    #  3. We avoid a lot of redundant arithmetic.
    #
    # TODO: Use a non-trivial pivoting strategy. Even just row swapping makes a
    # big difference to performance if e.g. the upper-left entry of the matrix
    # is a huge polynomial.

    # a is (m x n)
    m = len(a)
    if not m:
        return K.one, []
    n = len(a[0])

    d = None
    pivots = []
    no_pivots = []

    # i, j will be the row and column indices of the current pivot
    i = 0
    for j in range(n):
        # next pivot?
        aij = a[i][j]

        # swap rows if zero
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # go to next column
                no_pivots.append(j)
                continue

        # Now aij is the pivot and i,j are the row and column. We need to clear
        # the column above and below but we also need to keep track of the
        # denominator of the RREF which means also multiplying everything above
        # and to the left by the current pivot aij and dividing by d (which we
        # multiplied everything by in the previous iteration so this is an
        # exact division).
        #
        # First handle the upper left corner which is usually already diagonal
        # with all diagonal entries equal to the current denominator but there
        # can be other non-zero entries in any column that has no pivot.

        # Update previous pivots in the matrix
        if pivots:
            pivot_val = aij * a[0][pivots[0]]
            # Divide out the common factor
            if d is not None:
                pivot_val = K.exquo(pivot_val, d)

            # Could defer this until the end but it is pretty cheap and
            # helps when debugging.
            for ip, jp in enumerate(pivots):
                a[ip][jp] = pivot_val

        # Update columns without pivots
        for jnp in no_pivots:
            for ip in range(i):
                aijp = a[ip][jnp]
                if aijp:
                    aijp *= aij
                    if d is not None:
                        aijp = K.exquo(aijp, d)
                    a[ip][jnp] = aijp

        # Eliminate above, below and to the right as in ordinary division free
        # Gauss-Jordan elmination except also dividing out d from every entry.

        for jp, aj in enumerate(a):

            # Skip the current row
            if jp == i:
                continue

            # Eliminate to the right in all rows
            for kp in range(j+1, n):
                ajk = aij * aj[kp] - aj[j] * a[i][kp]
                if d is not None:
                    ajk = K.exquo(ajk, d)
                aj[kp] = ajk

            # Set to zero above and below the pivot
            aj[j] = K.zero

        # next row
        pivots.append(j)
        i += 1

        # no more rows left?
        if i >= m:
            break

        if not K.is_one(aij):
            d = aij
        else:
            d = None

    if not pivots:
        denom = K.one
    else:
        denom = a[0][pivots[0]]

    return denom, pivots


def ddm_idet(a, K):
    """a  <--  echelon(a); return det

    Explanation
    ===========

    Compute the determinant of $a$ using the Bareiss fraction-free algorithm.
    The matrix $a$ is modified in place. Its diagonal elements are the
    determinants of the leading principal minors. The determinant of $a$ is
    returned.

    The domain $K$ must support exact division (``K.exquo``). This method is
    suitable for most exact rings and fields like :ref:`ZZ`, :ref:`QQ` and
    :ref:`QQ(a)` but not for inexact domains like :ref:`RR` and :ref:`CC`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices.ddm import ddm_idet
    >>> a = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]]
    >>> a
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> ddm_idet(a, ZZ)
    0
    >>> a
    [[1, 2, 3], [4, -3, -6], [7, -6, 0]]
    >>> [a[i][i] for i in range(len(a))]
    [1, -3, 0]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.det

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bareiss_algorithm
    .. [2] https://www.math.usm.edu/perry/Research/Thesis_DRL.pdf
    """
    # Bareiss algorithm
    # https://www.math.usm.edu/perry/Research/Thesis_DRL.pdf

    # a is (m x n)
    m = len(a)
    if not m:
        return K.one
    n = len(a[0])

    exquo = K.exquo
    # uf keeps track of the sign change from row swaps
    uf = K.one

    for k in range(n-1):
        if not a[k][k]:
            for i in range(k+1, n):
                if a[i][k]:
                    a[k], a[i] = a[i], a[k]
                    uf = -uf
                    break
            else:
                return K.zero

        akkm1 = a[k-1][k-1] if k else K.one

        for i in range(k+1, n):
            for j in range(k+1, n):
                a[i][j] = exquo(a[i][j]*a[k][k] - a[i][k]*a[k][j], akkm1)

    return uf * a[-1][-1]


def ddm_iinv(ainv, a, K):
    """ainv  <--  inv(a)

    Compute the inverse of a matrix $a$ over a field $K$ using Gauss-Jordan
    elimination. The result is stored in $ainv$.

    Uses division in the ground domain which should be an exact field.

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_iinv, ddm_imatmul
    >>> from sympy import QQ
    >>> a = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> ainv = [[None, None], [None, None]]
    >>> ddm_iinv(ainv, a, QQ)
    >>> ainv
    [[-2, 1], [3/2, -1/2]]
    >>> result = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> ddm_imatmul(result, a, ainv)
    >>> result
    [[1, 0], [0, 1]]

    See Also
    ========

    ddm_irref: the underlying routine.
    """
    if not K.is_Field:
        raise DMDomainError('Not a field')

    # a is (m x n)
    m = len(a)
    if not m:
        return
    n = len(a[0])
    if m != n:
        raise DMNonSquareMatrixError

    eye = [[K.one if i==j else K.zero for j in range(n)] for i in range(n)]
    Aaug = [row + eyerow for row, eyerow in zip(a, eye)]
    pivots = ddm_irref(Aaug)
    if pivots != list(range(n)):
        raise DMNonInvertibleMatrixError('Matrix det == 0; not invertible.')
    ainv[:] = [row[n:] for row in Aaug]


def ddm_ilu_split(L, U, K):
    """L, U  <--  LU(U)

    Compute the LU decomposition of a matrix $L$ in place and store the lower
    and upper triangular matrices in $L$ and $U$, respectively. Returns a list
    of row swaps that were performed.

    Uses division in the ground domain which should be an exact field.

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_ilu_split
    >>> from sympy import QQ
    >>> L = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> swaps = ddm_ilu_split(L, U, QQ)
    >>> swaps
    []
    >>> L
    [[0, 0], [3, 0]]
    >>> U
    [[1, 2], [0, -2]]

    See Also
    ========

    ddm_ilu
    ddm_ilu_solve
    """
    m = len(U)
    if not m:
        return []
    n = len(U[0])

    swaps = ddm_ilu(U)

    zeros = [K.zero] * min(m, n)
    for i in range(1, m):
        j = min(i, n)
        L[i][:j] = U[i][:j]
        U[i][:j] = zeros[:j]

    return swaps


def ddm_ilu(a):
    """a  <--  LU(a)

    Computes the LU decomposition of a matrix in place. Returns a list of
    row swaps that were performed.

    Uses division in the ground domain which should be an exact field.

    This is only suitable for domains like :ref:`GF(p)`, :ref:`QQ`, :ref:`QQ_I`
    and :ref:`QQ(a)`. With a rational function field like :ref:`K(x)` it is
    better to clear denominators and use division-free algorithms. Pivoting is
    used to avoid exact zeros but not for floating point accuracy so :ref:`RR`
    and :ref:`CC` are not suitable (use :func:`ddm_irref` instead).

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_ilu
    >>> from sympy import QQ
    >>> a = [[QQ(1, 2), QQ(1, 3)], [QQ(1, 4), QQ(1, 5)]]
    >>> swaps = ddm_ilu(a)
    >>> swaps
    []
    >>> a
    [[1/2, 1/3], [1/2, 1/30]]

    The same example using ``Matrix``:

    >>> from sympy import Matrix, S
    >>> M = Matrix([[S(1)/2, S(1)/3], [S(1)/4, S(1)/5]])
    >>> L, U, swaps = M.LUdecomposition()
    >>> L
    Matrix([
    [  1, 0],
    [1/2, 1]])
    >>> U
    Matrix([
    [1/2,  1/3],
    [  0, 1/30]])
    >>> swaps
    []

    See Also
    ========

    ddm_irref
    ddm_ilu_solve
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    """
    m = len(a)
    if not m:
        return []
    n = len(a[0])

    swaps = []

    for i in range(min(m, n)):
        if not a[i][i]:
            for ip in range(i+1, m):
                if a[ip][i]:
                    swaps.append((i, ip))
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # M = Matrix([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]])
                continue
        for j in range(i+1, m):
            l_ji = a[j][i] / a[i][i]
            a[j][i] = l_ji
            for k in range(i+1, n):
                a[j][k] -= l_ji * a[i][k]

    return swaps


def ddm_ilu_solve(x, L, U, swaps, b):
    """x  <--  solve(L*U*x = swaps(b))

    Solve a linear system, $A*x = b$, given an LU factorization of $A$.

    Uses division in the ground domain which must be a field.

    Modifies $x$ in place.

    Examples
    ========

    Compute the LU decomposition of $A$ (in place):

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.dense import ddm_ilu, ddm_ilu_solve
    >>> A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> swaps = ddm_ilu(A)
    >>> A
    [[1, 2], [3, -2]]
    >>> L = U = A

    Solve the linear system:

    >>> b = [[QQ(5)], [QQ(6)]]
    >>> x = [[None], [None]]
    >>> ddm_ilu_solve(x, L, U, swaps, b)
    >>> x
    [[-4], [9/2]]

    See Also
    ========

    ddm_ilu
        Compute the LU decomposition of a matrix in place.
    ddm_ilu_split
        Compute the LU decomposition of a matrix and separate $L$ and $U$.
    sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
        Higher level interface to this function.
    """
    m = len(U)
    if not m:
        return
    n = len(U[0])

    m2 = len(b)
    if not m2:
        raise DMShapeError("Shape mismtch")
    o = len(b[0])

    if m != m2:
        raise DMShapeError("Shape mismtch")
    if m < n:
        raise NotImplementedError("Underdetermined")

    if swaps:
        b = [row[:] for row in b]
        for i1, i2 in swaps:
            b[i1], b[i2] = b[i2], b[i1]

    # solve Ly = b
    y = [[None] * o for _ in range(m)]
    for k in range(o):
        for i in range(m):
            rhs = b[i][k]
            for j in range(i):
                rhs -= L[i][j] * y[j][k]
            y[i][k] = rhs

    if m > n:
        for i in range(n, m):
            for j in range(o):
                if y[i][j]:
                    raise DMNonInvertibleMatrixError

    # Solve Ux = y
    for k in range(o):
        for i in reversed(range(n)):
            if not U[i][i]:
                raise DMNonInvertibleMatrixError
            rhs = y[i][k]
            for j in range(i+1, n):
                rhs -= U[i][j] * x[j][k]
            x[i][k] = rhs / U[i][i]


def ddm_berk(M, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.dense import ddm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    >>> ddm_berk(M, ZZ)
    [[1], [-5], [-2]]
    >>> Matrix(M).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    m = len(M)
    if not m:
        return [[K.one]]
    n = len(M[0])

    if m != n:
        raise DMShapeError("Not square")

    if n == 1:
        return [[K.one], [-M[0][0]]]

    a = M[0][0]
    R = [M[0][1:]]
    C = [[row[0]] for row in M[1:]]
    A = [row[1:] for row in M[1:]]

    q = ddm_berk(A, K)

    T = [[K.zero] * n for _ in range(n+1)]
    for i in range(n):
        T[i][i] = K.one
        T[i+1][i] = -a
    for i in range(2, n+1):
        if i == 2:
            AnC = C
        else:
            C = AnC
            AnC = [[K.zero] for row in C]
            ddm_imatmul(AnC, A, C)
        RAnC = [[K.zero]]
        ddm_imatmul(RAnC, R, AnC)
        for j in range(0, n+1-i):
            T[i+j][j] = -RAnC[0][0]

    qout = [[K.zero] for _ in range(n+1)]
    ddm_imatmul(qout, T, q)
    return qout
<|endoftext|>"""

Module for the DomainScalar class.

A DomainScalar represents an element which is in a particular
Domain. The idea is that the DomainScalar class provides the
convenience routines for unifying elements with different domains.

It assists in Scalar Multiplication and getitem for DomainMatrix.

"""
from ..constructor import construct_domain

from sympy.polys.domains import Domain, ZZ


class DomainScalar:
    r"""
    docstring
    """

    def __new__(cls, element, domain):
        if not isinstance(domain, Domain):
            raise TypeError("domain should be of type Domain")
        if not domain.of_type(element):
            raise TypeError("element %s should be in domain %s" % (element, domain))
        return cls.new(element, domain)

    @classmethod
    def new(cls, element, domain):
        obj = super().__new__(cls)
        obj.element = element
        obj.domain = domain
        return obj

    def __repr__(self):
        return repr(self.element)

    @classmethod
    def from_sympy(cls, expr):
        [domain, [element]] = construct_domain([expr])
        return cls.new(element, domain)

    def to_sympy(self):
        return self.domain.to_sympy(self.element)

    def to_domain(self, domain):
        element = domain.convert_from(self.element, self.domain)
        return self.new(element, domain)

    def convert_to(self, domain):
        return self.to_domain(domain)

    def unify(self, other):
        domain = self.domain.unify(other.domain)
        return self.to_domain(domain), other.to_domain(domain)

    def __bool__(self):
        return bool(self.element)

    def __add__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.element + other.element, self.domain)

    def __sub__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.element - other.element, self.domain)

    def __mul__(self, other):
        if not isinstance(other, DomainScalar):
            if isinstance(other, int):
                other = DomainScalar(ZZ(other), ZZ)
            else:
                return NotImplemented

        self, other = self.unify(other)
        return self.new(self.element * other.element, self.domain)

    def __floordiv__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.domain.quo(self.element, other.element), self.domain)

    def __mod__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        return self.new(self.domain.rem(self.element, other.element), self.domain)

    def __divmod__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        self, other = self.unify(other)
        q, r = self.domain.div(self.element, other.element)
        return (self.new(q, self.domain), self.new(r, self.domain))

    def __pow__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        return self.new(self.element**n, self.domain)

    def __pos__(self):
        return self.new(+self.element, self.domain)

    def __neg__(self):
        return self.new(-self.element, self.domain)

    def __eq__(self, other):
        if not isinstance(other, DomainScalar):
            return NotImplemented
        return self.element == other.element and self.domain == other.domain

    def is_zero(self):
        return self.element == self.domain.zero

    def is_one(self):
        return self.element == self.domain.one
<|endoftext|>"""

Module for the DDM class.

The DDM class is an internal representation used by DomainMatrix. The letters
DDM stand for Dense Domain Matrix. A DDM instance represents a matrix using
elements from a polynomial Domain (e.g. ZZ, QQ, ...) in a dense-matrix
representation.

Basic usage:

    >>> from sympy import ZZ, QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> A = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> A.shape
    (2, 2)
    >>> A
    [[0, 1], [-1, 0]]
    >>> type(A)
    <class 'sympy.polys.matrices.ddm.DDM'>
    >>> A @ A
    [[-1, 0], [0, -1]]

The ddm_* functions are designed to operate on DDM as well as on an ordinary
list of lists:

    >>> from sympy.polys.matrices.dense import ddm_idet
    >>> ddm_idet(A, QQ)
    1
    >>> ddm_idet([[0, 1], [-1, 0]], QQ)
    1
    >>> A
    [[-1, 0], [0, -1]]

Note that ddm_idet modifies the input matrix in-place. It is recommended to
use the DDM.det method as a friendlier interface to this instead which takes
care of copying the matrix:

    >>> B = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> B.det()
    1

Normally DDM would not be used directly and is just part of the internal
representation of DomainMatrix which adds further functionality including e.g.
unifying domains.

The dense format used by DDM is a list of lists of elements e.g. the 2x2
identity matrix is like [[1, 0], [0, 1]]. The DDM class itself is a subclass
of list and its list items are plain lists. Elements are accessed as e.g.
ddm[i][j] where ddm[i] gives the ith row and ddm[i][j] gets the element in the
jth column of that row. Subclassing list makes e.g. iteration and indexing
very efficient. We do not override __getitem__ because it would lose that
benefit.

The core routines are implemented by the ddm_* functions defined in dense.py.
Those functions are intended to be able to operate on a raw list-of-lists
representation of matrices with most functions operating in-place. The DDM
class takes care of copying etc and also stores a Domain object associated
with its elements. This makes it possible to implement things like A + B with
domain checking and also shape checking so that the list of lists
representation is friendlier.

"""
from itertools import chain

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from .exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMShapeError,
)

from sympy.polys.domains import QQ

from .dense import (
        ddm_transpose,
        ddm_iadd,
        ddm_isub,
        ddm_ineg,
        ddm_imul,
        ddm_irmul,
        ddm_imatmul,
        ddm_irref,
        ddm_irref_den,
        ddm_idet,
        ddm_iinv,
        ddm_ilu_split,
        ddm_ilu_solve,
        ddm_berk,
        )

from .lll import ddm_lll, ddm_lll_transform


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DDM.to_dfm', 'DDM.to_dfm_or_ddm']


class DDM(list):
    """Dense matrix based on polys domain elements

    This is a list subclass and is a wrapper for a list of lists that supports
    basic matrix arithmetic +, -, *, **.
    """

    fmt = 'dense'
    is_DFM = False
    is_DDM = True

    def __init__(self, rowslist, shape, domain):
        if not (isinstance(rowslist, list) and all(type(row) is list for row in rowslist)):
            raise DMBadInputError("rowslist must be a list of lists")
        m, n = shape
        if len(rowslist) != m or any(len(row) != n for row in rowslist):
            raise DMBadInputError("Inconsistent row-list/shape")

        super().__init__([i.copy() for i in rowslist])
        self.shape = (m, n)
        self.rows = m
        self.cols = n
        self.domain = domain

    def getitem(self, i, j):
        return self[i][j]

    def setitem(self, i, j, value):
        self[i][j] = value

    def extract_slice(self, slice1, slice2):
        ddm = [row[slice2] for row in self[slice1]]
        rows = len(ddm)
        cols = len(ddm[0]) if ddm else len(range(self.shape[1])[slice2])
        return DDM(ddm, (rows, cols), self.domain)

    def extract(self, rows, cols):
        ddm = []
        for i in rows:
            rowi = self[i]
            ddm.append([rowi[j] for j in cols])
        return DDM(ddm, (len(rows), len(cols)), self.domain)

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """
        Create a :class:`DDM` from a list of lists.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        >>> A
        [[0, 1], [-1, 0]]
        >>> A == DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        True

        See Also
        ========

        from_list_flat
        """
        return cls(rowslist, shape, domain)

    @classmethod
    def from_ddm(cls, other):
        return other.copy()

    def to_list(self):
        """
        Convert to a list of lists.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list()
        [[1, 2], [3, 4]]

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list
        """
        return [row[:] for row in self]

    def to_list_flat(self):
        """
        Convert to a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list_flat
        """
        flat = []
        for row in self:
            flat.extend(row)
        return flat

    @classmethod
    def from_list_flat(cls, flat, shape, domain):
        """
        Create a :class:`DDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list_flat([1, 2, 3, 4], (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_list_flat
        """
        assert type(flat) is list
        rows, cols = shape
        if not (len(flat) == rows*cols):
            raise DMBadInputError("Inconsistent flat-list shape")
        lol = [flat[i*cols:(i+1)*cols] for i in range(rows)]
        return cls(lol, shape, domain)

    def flatiter(self):
        return chain.from_iterable(self)

    def flat(self):
        items = []
        for row in self:
            items.extend(row)
        return items

    def to_flat_nz(self):
        """
        Convert to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a matrix using :meth:`from_flat_nz`. Zero elements are
        included in the list but that may change in the future.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        sympy.polys.matrices.sdm.SDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        return self.to_sdm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`DDM` after calling :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        to_flat_nz
        sympy.polys.matrices.sdm.SDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        return SDM.from_flat_nz(elements, data, domain).to_ddm()

    def to_dod(self):
        """
        Convert to a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.sdm.SDM.to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        dod = {}
        for i, row in enumerate(self):
            row = {j:e for j, e in enumerate(row) if e}
            if row:
                dod[i] = row
        return dod

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dod = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> A = DDM.from_dod(dod, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dod
        sympy.polys.matrices.sdm.SDM.from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dod
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for i, row in dod.items():
            for j, element in row.items():
                lol[i][j] = element
        return DDM(lol, shape, domain)

    def to_dok(self):
        """
        Convert :class:`DDM` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dok()
        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

        See Also
        ========

        from_dok
        sympy.polys.matrices.sdm.SDM.to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dok
        """
        dok = {}
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    dok[i, j] = element
        return dok

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dok = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        >>> A = DDM.from_dok(dok, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dok
        sympy.polys.matrices.sdm.SDM.from_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dok
        """
        rows, cols = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for (i, j), element in dok.items():
            lol[i][j] = element
        return DDM(lol, shape, domain)

    def iter_values(self):
        """
        Iterate over the non-zero values of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_values
        """
        for row in self:
            yield from filter(None, row)

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        for i, row in enumerate(self):
            for j, element in enumerate(row):
                if element:
                    yield (i, j), element

    def to_ddm(self):
        """
        Convert to a :class:`DDM`.

        This just returns ``self`` but exists to parallel the corresponding
        method in other matrix types like :class:`~.SDM`.

        See Also
        ========

        to_sdm
        to_dfm
        to_dfm_or_ddm
        sympy.polys.matrices.sdm.SDM.to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_ddm
        """
        return self

    def to_sdm(self):
        """
        Convert to a :class:`~.SDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_sdm()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> type(A.to_sdm())
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        SDM
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        return SDM.from_list(self, self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Convert to :class:`~.DDM` to :class:`~.DFM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        DFM
        sympy.polys.matrices._dfm.DFM.to_ddm
        """
        return DFM(list(self), self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Convert to :class:`~.DFM` if possible or otherwise return self.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm_or_ddm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_dfm
        to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        if DFM._supports_domain(self.domain):
            return self.to_dfm()
        return self

    def convert_to(self, K):
        Kold = self.domain
        if K == Kold:
            return self.copy()
        rows = [[K.convert_from(e, Kold) for e in row] for row in self]
        return DDM(rows, self.shape, K)

    def __str__(self):
        rowsstr = ['[%s]' % ', '.join(map(str, row)) for row in self]
        return '[%s]' % ', '.join(rowsstr)

    def __repr__(self):
        cls = type(self).__name__
        rows = list.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    def __eq__(self, other):
        if not isinstance(other, DDM):
            return False
        return (super().__eq__(other) and self.domain == other.domain)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def zeros(cls, shape, domain):
        z = domain.zero
        m, n = shape
        rowslist = [[z] * n for _ in range(m)]
        return DDM(rowslist, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        rowlist = [[one] * n for _ in range(m)]
        return DDM(rowlist, shape, domain)

    @classmethod
    def eye(cls, size, domain):
        if isinstance(size, tuple):
            m, n = size
        elif isinstance(size, int):
            m = n = size
        one = domain.one
        ddm = cls.zeros((m, n), domain)
        for i in range(min(m, n)):
            ddm[i][i] = one
        return ddm

    def copy(self):
        copyrows = [row[:] for row in self]
        return DDM(copyrows, self.shape, self.domain)

    def transpose(self):
        rows, cols = self.shape
        if rows:
            ddmT = ddm_transpose(self)
        else:
            ddmT = [[]] * cols
        return DDM(ddmT, (cols, rows), self.domain)

    def __add__(a, b):
        if not isinstance(b, DDM):
            return NotImplemented
        return a.add(b)

    def __sub__(a, b):
        if not isinstance(b, DDM):
            return NotImplemented
        return a.sub(b)

    def __neg__(a):
        return a.neg()

    def __mul__(a, b):
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __matmul__(a, b):
        if isinstance(b, DDM):
            return a.matmul(b)
        else:
            return NotImplemented

    @classmethod
    def _check(cls, a, op, b, ashape, bshape):
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)

    def add(a, b):
        """a + b"""
        a._check(a, '+', b, a.shape, b.shape)
        c = a.copy()
        ddm_iadd(c, b)
        return c

    def sub(a, b):
        """a - b"""
        a._check(a, '-', b, a.shape, b.shape)
        c = a.copy()
        ddm_isub(c, b)
        return c

    def neg(a):
        """-a"""
        b = a.copy()
        ddm_ineg(b)
        return b

    def mul(a, b):
        c = a.copy()
        ddm_imul(c, b)
        return c

    def rmul(a, b):
        c = a.copy()
        ddm_irmul(c, b)
        return c

    def matmul(a, b):
        """a @ b (matrix product)"""
        m, o = a.shape
        o2, n = b.shape
        a._check(a, '*', b, o, o2)
        c = a.zeros((m, n), a.domain)
        ddm_imatmul(c, a, b)
        return c

    def mul_elementwise(a, b):
        assert a.shape == b.shape
        assert a.domain == b.domain
        c = [[aij * bij for aij, bij in zip(ai, bi)] for ai, bi in zip(a, b)]
        return DDM(c, a.shape, a.domain)

    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        [[1, 2, 5, 6], [3, 4, 7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
        """
        Anew = list(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain

            cols += Bkcols

            for i, Bki in enumerate(Bk):
                Anew[i].extend(Bki)

        return DDM(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        """
        Anew = list(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain

            rows += Bkrows

            Anew.extend(Bk.copy())

        return DDM(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        elements = [list(map(func, row)) for row in self]
        return DDM(elements, self.shape, domain)

    def nnz(a):
        """Number of non-zero entries in :py:class:`~.DDM` matrix.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(sum(map(bool, row)) for row in a)

    def scc(a):
        """Strongly connected components of a square matrix *a*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> A = DDM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc

        """
        return a.to_sdm().scc()

    @classmethod
    def diag(cls, values, domain):
        """Returns a square diagonal matrix with *values* on the diagonal.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> DDM.diag([ZZ(1), ZZ(2), ZZ(3)], ZZ)
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.diag
        """
        return SDM.diag(values, domain).to_ddm()

    def rref(a):
        """Reduced-row echelon form of a and list of pivots.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref
            The underlying algorithm.
        """
        b = a.copy()
        K = a.domain
        partial_pivot = K.is_RealField or K.is_ComplexField
        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
        return b, pivots

    def rref_den(a):
        """Reduced-row echelon form of a with denominator and list of pivots

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref_den
            The underlying algorithm.
        """
        b = a.copy()
        K = a.domain
        denom, pivots = ddm_irref_den(b, K)
        return b, denom, pivots

    def nullspace(a):
        """Returns a basis for the nullspace of a.

        The domain of the matrix must be a field.

        See Also
        ========

        rref
        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
        """
        rref, pivots = a.rref()
        return rref.nullspace_from_rref(pivots)

    def nullspace_from_rref(a, pivots=None):
        """Compute the nullspace of a matrix from its rref.

        The domain of the matrix can be any domain.

        Returns a tuple (basis, nonpivots).

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher level interface to this function.
        """
        m, n = a.shape
        K = a.domain

        if pivots is None:
            pivots = []
            last_pivot = -1
            for i in range(m):
                ai = a[i]
                for j in range(last_pivot+1, n):
                    if ai[j]:
                        last_pivot = j
                        pivots.append(j)
                        break

        if not pivots:
            return (a.eye(n, K), list(range(n)))

        # After rref the pivots are all one but after rref_den they may not be.
        pivot_val = a[0][pivots[0]]

        basis = []
        nonpivots = []
        for i in range(n):
            if i in pivots:
                continue
            nonpivots.append(i)
            vec = [pivot_val if i == j else K.zero for j in range(n)]
            for ii, jj in enumerate(pivots):
                vec[jj] -= a[ii][i]
            basis.append(vec)

        basis_ddm = DDM(basis, (len(basis), n), K)

        return (basis_ddm, nonpivots)

    def particular(a):
        return a.to_sdm().particular().to_ddm()

    def det(a):
        """Determinant of a"""
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        b = a.copy()
        K = b.domain
        deta = ddm_idet(b, K)
        return deta

    def inv(a):
        """Inverse of a"""
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Determinant of non-square matrix")
        ainv = a.copy()
        K = a.domain
        ddm_iinv(ainv, a, K)
        return ainv

    def lu(a):
        """L, U decomposition of a"""
        m, n = a.shape
        K = a.domain

        U = a.copy()
        L = a.eye(m, K)
        swaps = ddm_ilu_split(L, U, K)

        return L, U, swaps

    def _fflu(self):
        """
        Private method for Phase 1 of fraction-free LU decomposition.
        Performs row operations and elimination to compute U and permutation indices.

        Returns:
            LU : decomposition as a single matrix.
            perm (list): Permutation indices for row swaps.
        """
        rows, cols = self.shape
        K = self.domain

        LU = self.copy()
        perm = list(range(rows))
        rank = 0

        for j in range(min(rows, cols)):
            # Skip columns where all entries are zero
            if all(LU[i][j] == K.zero for i in range(rows)):
                continue

            # Find the first non-zero pivot in the current column
            pivot_row = -1
            for i in range(rank, rows):
                if LU[i][j] != K.zero:
                    pivot_row = i
                    break

            # If no pivot is found, skip column
            if pivot_row == -1:
                continue

            # Swap rows to bring the pivot to the current rank
            if pivot_row != rank:
                LU[rank], LU[pivot_row] = LU[pivot_row], LU[rank]
                perm[rank], perm[pivot_row] = perm[pivot_row], perm[rank]

            # Found pivot - (Gauss-Bareiss elimination)
            pivot = LU[rank][j]
            for i in range(rank + 1, rows):
                multiplier = LU[i][j]
                # Denominator is previous pivot or 1
                denominator = LU[rank - 1][rank - 1] if rank > 0 else K.one
                for k in range(j + 1, cols):
                    LU[i][k] = K.exquo(pivot * LU[i][k] - LU[rank][k] * multiplier, denominator)
                # Keep the multiplier for L matrix
                LU[i][j] = multiplier
            rank += 1

        return LU, perm

    def fflu(self):
        """
        Fraction-free LU decomposition of DDM.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.fflu
            The higher-level interface to this function.
        """
        rows, cols = self.shape
        K = self.domain

        # Phase 1: Perform row operations and get permutation
        U, perm = self._fflu()

        # Phase 2: Construct P, L, D matrices
        # Create P from permutation
        P = self.zeros((rows, rows), K)
        for i, pi in enumerate(perm):
            P[i][pi] = K.one

        # Create L matrix
        L = self.zeros((rows, rows), K)
        i = j = 0
        while i < rows and j < cols:
            if U[i][j] != K.zero:
                # Found non-zero pivot
                # Diagonal entry is the pivot
                L[i][i] = U[i][j]
                for l in range(i + 1, rows):
                    # Off-diagonal entries are the multipliers
                    L[l][i] = U[l][j]
                    # zero out the entries in U
                    U[l][j] = K.zero
                i += 1
            j += 1

        # Fill remaining diagonal of L with ones
        for i in range(i, rows):
            L[i][i] = K.one

        # Create D matrix - using FLINT's approach with accumulator
        D = self.zeros((rows, rows), K)
        if rows >= 1:
            D[0][0] = L[0][0]
        di = K.one
        for i in range(1, rows):
            # Accumulate product of pivots
            di = L[i - 1][i - 1] * L[i][i]
            D[i][i] = di

        return P, L, D, U

    def qr(self):
        """
        QR decomposition for DDM.

        Returns:
            - Q: Orthogonal matrix as a DDM.
            - R: Upper triangular matrix as a DDM.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.qr
            The higher-level interface to this function.
        """
        rows, cols = self.shape
        K = self.domain
        Q = self.copy()
        R = self.zeros((min(rows, cols), cols), K)

        # Check that the domain is a field
        if not K.is_Field:
            raise DMDomainError("QR decomposition requires a field (e.g. QQ).")

        dot_cols = lambda i, j: K.sum(Q[k][i] * Q[k][j] for k in range(rows))

        for j in range(cols):
            for i in range(min(j, rows)):
                dot_ii = dot_cols(i, i)
                if dot_ii != K.zero:
                    R[i][j] = dot_cols(i, j) / dot_ii
                    for k in range(rows):
                        Q[k][j] -= R[i][j] * Q[k][i]

            if j < rows:
                dot_jj = dot_cols(j, j)
                if dot_jj != K.zero:
                    R[j][j] = K.one

        Q = Q.extract(range(rows), range(min(rows, cols)))

        return Q, R

    def lu_solve(a, b):
        """x where a*x = b"""
        m, n = a.shape
        m2, o = b.shape
        a._check(a, 'lu_solve', b, m, m2)
        if not a.domain.is_Field:
            raise DMDomainError("lu_solve requires a field")

        L, U, swaps = a.lu()
        x = a.zeros((n, o), a.domain)
        ddm_ilu_solve(x, L, U, swaps, b)
        return x

    def charpoly(a):
        """Coefficients of characteristic polynomial of a"""
        K = a.domain
        m, n = a.shape
        if m != n:
            raise DMNonSquareMatrixError("Charpoly of non-square matrix")
        vec = ddm_berk(a, K)
        coeffs = [vec[i][0] for i in range(n+1)]
        return coeffs

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        zero = self.domain.zero
        return all(Mij == zero for Mij in self.flatiter())

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        zero = self.domain.zero
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[:i])

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        zero = self.domain.zero
        return all(Mij == zero for i, Mi in enumerate(self) for Mij in Mi[i+1:])

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned even if
        the matrix is not square.
        """
        return self.is_upper() and self.is_lower()

    def diagonal(self):
        """
        Returns a list of the elements from the diagonal of the matrix.
        """
        m, n = self.shape
        return [self[i][i] for i in range(min(m, n))]

    def lll(A, delta=QQ(3, 4)):
        return ddm_lll(A, delta=delta)

    def lll_transform(A, delta=QQ(3, 4)):
        return ddm_lll_transform(A, delta=delta)


from .sdm import SDM
from .dfm import DFM
<|endoftext|>'''Functions returning normal forms of matrices'''

from collections import defaultdict

from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ


# TODO (future work):
#  There are faster algorithms for Smith and Hermite normal forms, which
#  we should implement. See e.g. the Kannan-Bachem algorithm:
#  <https://www.researchgate.net/publication/220617516_Polynomial_Algorithms_for_Computing_the_Smith_and_Hermite_Normal_Forms_of_an_Integer_Matrix>


def smith_normal_form(m):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(smith_normal_form(m).to_Matrix())
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])

    '''
    invs = invariant_factors(m)
    smf = DomainMatrix.diag(invs, m.domain, m.shape)
    return smf


def is_smith_normal_form(m):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    domain = m.domain
    shape = m.shape
    zero = domain.zero
    m = m.to_list()

    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                continue
            if not m[i][j] == zero:
                return False

    upper = min(shape[0], shape[1])
    for i in range(1, upper):
        if m[i-1][i-1] == zero:
            if m[i][i] != zero:
                return False
        else:
            r = domain.div(m[i][i], m[i-1][i-1])[1]
            if r != zero:
                return False

    return True


def add_columns(m, i, j, a, b, c, d):
    # replace m[:, i] by a*m[:, i] + b*m[:, j]
    # and m[:, j] by c*m[:, i] + d*m[:, j]
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = a*e + b*m[k][j]
        m[k][j] = c*e + d*m[k][j]


def invariant_factors(m):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    domain = m.domain
    shape = m.shape
    m = m.to_list()
    return _smith_normal_decomp(m, domain, shape=shape, full=False)


def smith_normal_decomp(m):
    '''
    Return the Smith-Normal form decomposition of matrix `m`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_decomp
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> a, s, t = smith_normal_decomp(m)
    >>> assert a == s * m * t
    '''
    domain = m.domain
    rows, cols = shape = m.shape
    m = m.to_list()

    invs, s, t = _smith_normal_decomp(m, domain, shape=shape, full=True)
    smf = DomainMatrix.diag(invs, domain, shape).to_dense()

    s = DomainMatrix(s, domain=domain, shape=(rows, rows))
    t = DomainMatrix(t, domain=domain, shape=(cols, cols))
    return smf, s, t


def _smith_normal_decomp(m, domain, shape, full):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form). If `full=True` then invertible matrices
    ``s, t`` such that the product ``s, m, t`` is the Smith Normal Form
    are also returned.
    '''
    if not domain.is_PID:
        msg = f"The matrix entries must be over a principal ideal domain, but got {domain}"
        raise ValueError(msg)

    rows, cols = shape
    zero = domain.zero
    one = domain.one

    def eye(n):
        return [[one if i == j else zero for i in range(n)] for j in range(n)]

    if 0 in shape:
        if full:
            return (), eye(rows), eye(cols)
        else:
            return ()

    if full:
        s = eye(rows)
        t = eye(cols)

    def add_rows(m, i, j, a, b, c, d):
        # replace m[i, :] by a*m[i, :] + b*m[j, :]
        # and m[j, :] by c*m[i, :] + d*m[j, :]
        for k in range(len(m[0])):
            e = m[i][k]
            m[i][k] = a*e + b*m[j][k]
            m[j][k] = c*e + d*m[j][k]

    def clear_column():
        # make m[1:, 0] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, rows):
            if m[j][0] == zero:
                continue
            d, r = domain.div(m[j][0], pivot)
            if r == zero:
                add_rows(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_rows(s, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[j][0])
                d_0 = domain.exquo(m[j][0], g)
                d_j = domain.exquo(pivot, g)
                add_rows(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_rows(s, 0, j, a, b, d_0, -d_j)
                pivot = g

    def clear_row():
        # make m[0, 1:] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, cols):
            if m[0][j] == zero:
                continue
            d, r = domain.div(m[0][j], pivot)
            if r == zero:
                add_columns(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_columns(t, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[0][j])
                d_0 = domain.exquo(m[0][j], g)
                d_j = domain.exquo(pivot, g)
                add_columns(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_columns(t, 0, j, a, b, d_0, -d_j)
                pivot = g

    # permute the rows and columns until m[0,0] is non-zero if possible
    ind = [i for i in range(rows) if m[i][0] != zero]
    if ind and ind[0] != zero:
        m[0], m[ind[0]] = m[ind[0]], m[0]
        if full:
            s[0], s[ind[0]] = s[ind[0]], s[0]
    else:
        ind = [j for j in range(cols) if m[0][j] != zero]
        if ind and ind[0] != zero:
            for row in m:
                row[0], row[ind[0]] = row[ind[0]], row[0]
            if full:
                for row in t:
                    row[0], row[ind[0]] = row[ind[0]], row[0]

    # make the first row and column except m[0,0] zero
    while (any(m[0][i] != zero for i in range(1,cols)) or
           any(m[i][0] != zero for i in range(1,rows))):
        clear_column()
        clear_row()

    def to_domain_matrix(m):
        return DomainMatrix(m, shape=(len(m), len(m[0])), domain=domain)

    if m[0][0] != 0:
        c = domain.canonical_unit(m[0][0])
        if domain.is_Field:
            c = 1 / m[0][0]
        if c != domain.one:
            m[0][0] *= c
            if full:
                s[0] = [elem * c for elem in s[0]]

    if 1 in shape:
        invs = ()
    else:
        lower_right = [r[1:] for r in m[1:]]
        ret = _smith_normal_decomp(lower_right, domain,
                shape=(rows - 1, cols - 1), full=full)
        if full:
            invs, s_small, t_small = ret
            s2 = [[1] + [0]*(rows-1)] + [[0] + row for row in s_small]
            t2 = [[1] + [0]*(cols-1)] + [[0] + row for row in t_small]
            s, s2, t, t2 = list(map(to_domain_matrix, [s, s2, t, t2]))
            s = s2 * s
            t = t * t2
            s = s.to_list()
            t = t.to_list()
        else:
            invs = ret

    if m[0][0]:
        result = [m[0][0]]
        result.extend(invs)
        # in case m[0] doesn't divide the invariants of the rest of the matrix
        for i in range(len(result)-1):
            a, b = result[i], result[i+1]
            if b and domain.div(b, a)[1] != zero:
                if full:
                    x, y, d = domain.gcdex(a, b)
                else:
                    d = domain.gcd(a, b)

                alpha = domain.div(a, d)[0]
                if full:
                    beta = domain.div(b, d)[0]
                    add_rows(s, i, i + 1, 1, 0, x, 1)
                    add_columns(t, i, i + 1, 1, y, 0, 1)
                    add_rows(s, i, i + 1, 1, -alpha, 0, 1)
                    add_columns(t, i, i + 1, 1, 0, -beta, 1)
                    add_rows(s, i, i + 1, 0, 1, -1, 0)

                result[i+1] = b * alpha
                result[i] = d
            else:
                break
    else:
        if full:
            if rows > 1:
                s = s[1:] + [s[0]]
            if cols > 1:
                t = [row[1:] + [row[0]] for row in t]
        result = invs + (m[0][0],)

    if full:
        return tuple(result), s, t
    else:
        return tuple(result)


def _gcdex(a, b):
    r"""
    This supports the functions that compute Hermite Normal Form.

    Explanation
    ===========

    Let x, y be the coefficients returned by the extended Euclidean
    Algorithm, so that x*a + y*b = g. In the algorithms for computing HNF,
    it is critical that x, y not only satisfy the condition of being small
    in magnitude -- namely that |x| <= |b|/g, |y| <- |a|/g -- but also that
    y == 0 when a | b.

    """
    x, y, g = ZZ.gcdex(a, b)
    if a != 0 and b % a == 0:
        y = 0
        x = -1 if a < 0 else 1
    return x, y, g


def _hermite_normal_form(A):
    r"""
    Compute the Hermite Normal Form of DomainMatrix *A* over :ref:`ZZ`.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over domain :ref:`ZZ`.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.5.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    # We work one row at a time, starting from the bottom row, and working our
    # way up.
    m, n = A.shape
    A = A.to_ddm().copy()
    # Our goal is to put pivot entries in the rightmost columns.
    # Invariant: Before processing each row, k should be the index of the
    # leftmost column in which we have so far put a pivot.
    k = n
    for i in range(m - 1, -1, -1):
        if k == 0:
            # This case can arise when n < m and we've already found n pivots.
            # We don't need to consider any more rows, because this is already
            # the maximum possible number of pivots.
            break
        k -= 1
        # k now points to the column in which we want to put a pivot.
        # We want zeros in all entries to the left of the pivot column.
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                # Replace cols j, k by lin combs of these cols such that, in row i,
                # col j has 0, while col k has the gcd of their row i entries. Note
                # that this ensures a nonzero entry in col k.
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns(A, k, j, u, v, -s, r)
        b = A[i][k]
        # Do not want the pivot entry to be negative.
        if b < 0:
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        # The pivot entry will be 0 iff the row was 0 from the pivot col all the
        # way to the left. In this case, we are still working on the same pivot
        # col for the next row. Therefore:
        if b == 0:
            k += 1
        # If the pivot entry is nonzero, then we want to reduce all entries to its
        # right in the sense of the division algorithm, i.e. make them all remainders
        # w.r.t. the pivot as divisor.
        else:
            for j in range(k + 1, n):
                q = A[i][j] // b
                add_columns(A, j, k, 1, -q, 0, 1)
    # Finally, the HNF consists of those columns of A in which we succeeded in making
    # a nonzero pivot.
    return DomainMatrix.from_rep(A.to_dfm_or_ddm())[:, k:]


def _hermite_normal_form_modulo_D(A, D):
    r"""
    Perform the mod *D* Hermite Normal Form reduction algorithm on
    :py:class:`~.DomainMatrix` *A*.

    Explanation
    ===========

    If *A* is an $m \times n$ matrix of rank $m$, having Hermite Normal Form
    $W$, and if *D* is any positive integer known in advance to be a multiple
    of $\det(W)$, then the HNF of *A* can be computed by an algorithm that
    works mod *D* in order to prevent coefficient explosion.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over :ref:`ZZ`
        $m \times n$ matrix, having rank $m$.
    D : :ref:`ZZ`
        Positive integer, known to be a multiple of the determinant of the
        HNF of *A*.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.

    DMShapeError
        If the matrix has more rows than columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.8.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    if not ZZ.of_type(D) or D < 1:
        raise DMDomainError('Modulus D must be positive element of domain ZZ.')

    def add_columns_mod_R(m, R, i, j, a, b, c, d):
        # replace m[:, i] by (a*m[:, i] + b*m[:, j]) % R
        # and m[:, j] by (c*m[:, i] + d*m[:, j]) % R
        for k in range(len(m)):
            e = m[k][i]
            m[k][i] = symmetric_residue((a * e + b * m[k][j]) % R, R)
            m[k][j] = symmetric_residue((c * e + d * m[k][j]) % R, R)

    W = defaultdict(dict)

    m, n = A.shape
    if n < m:
        raise DMShapeError('Matrix must have at least as many columns as rows.')
    A = A.to_list()
    k = n
    R = D
    for i in range(m - 1, -1, -1):
        k -= 1
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns_mod_R(A, R, k, j, u, v, -s, r)
        b = A[i][k]
        if b == 0:
            A[i][k] = b = R
        u, v, d = _gcdex(b, R)
        for ii in range(m):
            W[ii][i] = u*A[ii][k] % R
        if W[i][i] == 0:
            W[i][i] = R
        for j in range(i + 1, m):
            q = W[i][j] // W[i][i]
            add_columns(W, j, i, 1, -q, 0, 1)
        R //= d
    return DomainMatrix(W, (m, m), ZZ).to_dense()


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of :py:class:`~.DomainMatrix` *A* over
    :ref:`ZZ`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import hermite_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(hermite_normal_form(m).to_Matrix())
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``DomainMatrix`` over :ref:`ZZ`.

    D : :ref:`ZZ`, optional
        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*
        being any multiple of $\det(W)$ may be provided. In this case, if *A*
        also has rank $m$, then we may use an alternative algorithm that works
        mod *D* in order to prevent coefficient explosion.

    check_rank : boolean, optional (default=False)
        The basic assumption is that, if you pass a value for *D*, then
        you already believe that *A* has rank $m$, so we do not waste time
        checking it for you. If you do want this to be checked (and the
        ordinary, non-modulo *D* algorithm to be used if the check fails), then
        set *check_rank* to ``True``.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.

    DMShapeError
        If the mod *D* algorithm is used but the matrix has more rows than
        columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithms 2.4.5 and 2.4.8.)

    """
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    if D is not None and (not check_rank or A.convert_to(QQ).rank() == A.shape[0]):
        return _hermite_normal_form_modulo_D(A, D)
    else:
        return _hermite_normal_form(A)
<|endoftext|># Algorithms for computing the reduced row echelon form of a matrix.
#
# We need to choose carefully which algorithms to use depending on the domain,
# shape, and sparsity of the matrix as well as things like the bit count in the
# case of ZZ or QQ. This is important because the algorithms have different
# performance characteristics in the extremes of dense vs sparse.
#
# In all cases we use the sparse implementations but we need to choose between
# Gauss-Jordan elimination with division and fraction-free Gauss-Jordan
# elimination. For very sparse matrices over ZZ with low bit counts it is
# asymptotically faster to use Gauss-Jordan elimination with division. For
# dense matrices with high bit counts it is asymptotically faster to use
# fraction-free Gauss-Jordan.
#
# The most important thing is to get the extreme cases right because it can
# make a big difference. In between the extremes though we have to make a
# choice and here we use empirically determined thresholds based on timings
# with random sparse matrices.
#
# In the case of QQ we have to consider the denominators as well. If the
# denominators are small then it is faster to clear them and use fraction-free
# Gauss-Jordan over ZZ. If the denominators are large then it is faster to use
# Gauss-Jordan elimination with division over QQ.
#
# Timings for the various algorithms can be found at
#
#   https://github.com/sympy/sympy/issues/25410
#   https://github.com/sympy/sympy/pull/25443

from sympy.polys.domains import ZZ

from sympy.polys.matrices.sdm import SDM, sdm_irref, sdm_rref_den
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import ddm_irref, ddm_irref_den


def _dm_rref(M, *, method='auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix``.

    This function is the implementation of :meth:`DomainMatrix.rref`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the field associated with the domain
    of the Matrix.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref_den
        Alternative function for computing RREF with denominator.
    """
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=False)

    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'GJ':
        # Use Gauss-Jordan with division over the associated field.
        Mf = _to_field(M)
        M_rref, pivots = _dm_rref_GJ(Mf)

    elif method == 'FF':
        # Use fraction-free GJ over the current domain.
        M_rref_f, den, pivots = _dm_rref_den_FF(M)
        M_rref = _to_field(M_rref_f) / den

    elif method == 'CD':
        # Clear denominators and use fraction-free GJ in the associated ring.
        _, Mr = M.clear_denoms_rowwise(convert=True)
        M_rref_f, den, pivots = _dm_rref_den_FF(Mr)
        M_rref = _to_field(M_rref_f) / den

    else:
        raise ValueError(f"Unknown method for rref: {method}")

    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # Invariants:
    #   - M_rref is in the same format (sparse or dense) as the input matrix.
    #   - M_rref is in the associated field domain and any denominator was
    #     divided in (so is implicitly 1 now).

    return M_rref, pivots


def _dm_rref_den(M, *, keep_domain=True, method='auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix`` with denominator.

    This function is the implementation of :meth:`DomainMatrix.rref_den`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the same domain as the input matrix
    unless ``keep_domain=False`` in which case the result might be over an
    associated ring or field domain.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref
        Alternative function for computing RREF without denominator.
    """
    method, use_fmt = _dm_rref_choose_method(M, method, denominator=True)

    M, old_fmt = _dm_to_fmt(M, use_fmt)

    if method == 'FF':
        # Use fraction-free GJ over the current domain.
        M_rref, den, pivots = _dm_rref_den_FF(M)

    elif method == 'GJ':
        # Use Gauss-Jordan with division over the associated field.
        M_rref_f, pivots = _dm_rref_GJ(_to_field(M))

        # Convert back to the ring?
        if keep_domain and M_rref_f.domain != M.domain:
            _, M_rref = M_rref_f.clear_denoms(convert=True)

            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
        else:
            # Possibly an associated field
            M_rref = M_rref_f
            den = M_rref.domain.one

    elif method == 'CD':
        # Clear denominators and use fraction-free GJ in the associated ring.
        _, Mr = M.clear_denoms_rowwise(convert=True)

        M_rref_r, den, pivots = _dm_rref_den_FF(Mr)

        if keep_domain and M_rref_r.domain != M.domain:
            # Convert back to the field
            M_rref = _to_field(M_rref_r) / den
            den = M.domain.one
        else:
            # Possibly an associated ring
            M_rref = M_rref_r

            if pivots:
                den = M_rref[0, pivots[0]].element
            else:
                den = M_rref.domain.one
    else:
        raise ValueError(f"Unknown method for rref: {method}")

    M_rref, _ = _dm_to_fmt(M_rref, old_fmt)

    # Invariants:
    #   - M_rref is in the same format (sparse or dense) as the input matrix.
    #   - If keep_domain=True then M_rref and den are in the same domain as the
    #     input matrix
    #   - If keep_domain=False then M_rref might be in an associated ring or
    #     field domain but den is always in the same domain as M_rref.

    return M_rref, den, pivots


def _dm_to_fmt(M, fmt):
    """Convert a matrix to the given format and return the old format."""
    old_fmt = M.rep.fmt
    if old_fmt == fmt:
        pass
    elif fmt == 'dense':
        M = M.to_dense()
    elif fmt == 'sparse':
        M = M.to_sparse()
    else:
        raise ValueError(f'Unknown format: {fmt}') # pragma: no cover
    return M, old_fmt


# These are the four basic implementations that we want to choose between:


def _dm_rref_GJ(M):
    """Compute RREF using Gauss-Jordan elimination with division."""
    if M.rep.fmt == 'sparse':
        return _dm_rref_GJ_sparse(M)
    else:
        return _dm_rref_GJ_dense(M)


def _dm_rref_den_FF(M):
    """Compute RREF using fraction-free Gauss-Jordan elimination."""
    if M.rep.fmt == 'sparse':
        return _dm_rref_den_FF_sparse(M)
    else:
        return _dm_rref_den_FF_dense(M)


def _dm_rref_GJ_sparse(M):
    """Compute RREF using sparse Gauss-Jordan elimination with division."""
    M_rref_d, pivots, _ = sdm_irref(M.rep)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_sdm), pivots


def _dm_rref_GJ_dense(M):
    """Compute RREF using dense Gauss-Jordan elimination with division."""
    partial_pivot = M.domain.is_RR or M.domain.is_CC
    ddm = M.rep.to_ddm().copy()
    pivots = ddm_irref(ddm, _partial_pivot=partial_pivot)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_ddm.to_dfm_or_ddm()), pivots


def _dm_rref_den_FF_sparse(M):
    """Compute RREF using sparse fraction-free Gauss-Jordan elimination."""
    M_rref_d, den, pivots = sdm_rref_den(M.rep, M.domain)
    M_rref_sdm = SDM(M_rref_d, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_sdm), den, pivots


def _dm_rref_den_FF_dense(M):
    """Compute RREF using sparse fraction-free Gauss-Jordan elimination."""
    ddm = M.rep.to_ddm().copy()
    den, pivots = ddm_irref_den(ddm, M.domain)
    M_rref_ddm = DDM(ddm, M.shape, M.domain)
    pivots = tuple(pivots)
    return M.from_rep(M_rref_ddm.to_dfm_or_ddm()), den, pivots


def _dm_rref_choose_method(M, method, *, denominator=False):
    """Choose the fastest method for computing RREF for M."""

    if method != 'auto':
        if method.endswith('_dense'):
            method = method[:-len('_dense')]
            use_fmt = 'dense'
        else:
            use_fmt = 'sparse'

    else:
        # The sparse implementations are always faster
        use_fmt = 'sparse'

        K = M.domain

        if K.is_ZZ:
            method = _dm_rref_choose_method_ZZ(M, denominator=denominator)
        elif K.is_QQ:
            method = _dm_rref_choose_method_QQ(M, denominator=denominator)
        elif K.is_RR or K.is_CC:
            # TODO: Add partial pivot support to the sparse implementations.
            method = 'GJ'
            use_fmt = 'dense'
        elif K.is_EX and M.rep.fmt == 'dense' and not denominator:
            # Do not switch to the sparse implementation for EX because the
            # domain does not have proper canonicalization and the sparse
            # implementation gives equivalent but non-identical results over EX
            # from performing arithmetic in a different order. Specifically
            # test_issue_23718 ends up getting a more complicated expression
            # when using the sparse implementation. Probably the best fix for
            # this is something else but for now we stick with the dense
            # implementation for EX if the matrix is already dense.
            method = 'GJ'
            use_fmt = 'dense'
        else:
            # This is definitely suboptimal. More work is needed to determine
            # the best method for computing RREF over different domains.
            if denominator:
                method = 'FF'
            else:
                method = 'GJ'

    return method, use_fmt


def _dm_rref_choose_method_QQ(M, *, denominator=False):
    """Choose the fastest method for computing RREF over QQ."""
    # The same sorts of considerations apply here as in the case of ZZ. Here
    # though a new more significant consideration is what sort of denominators
    # we have and what to do with them so we focus on that.

    # First compute the density. This is the average number of non-zero entries
    # per row but only counting rows that have at least one non-zero entry
    # since RREF can ignore fully zero rows.
    density, _, ncols = _dm_row_density(M)

    # For sparse matrices use Gauss-Jordan elimination over QQ regardless.
    if density < min(5, ncols/2):
        return 'GJ'

    # Compare the bit-length of the lcm of the denominators to the bit length
    # of the numerators.
    #
    # The threshold here is empirical: we prefer rref over QQ if clearing
    # denominators would result in a numerator matrix having 5x the bit size of
    # the current numerators.
    numers, denoms = _dm_QQ_numers_denoms(M)
    numer_bits = max([n.bit_length() for n in numers], default=1)

    denom_lcm = ZZ.one
    for d in denoms:
        denom_lcm = ZZ.lcm(denom_lcm, d)
        if denom_lcm.bit_length() > 5*numer_bits:
            return 'GJ'

    # If we get here then the matrix is dense and the lcm of the denominators
    # is not too large compared to the numerators. For particularly small
    # denominators it is fastest just to clear them and use fraction-free
    # Gauss-Jordan over ZZ. With very small denominators this is a little
    # faster than using rref_den over QQ but there is an intermediate regime
    # where rref_den over QQ is significantly faster. The small denominator
    # case is probably very common because small fractions like 1/2 or 1/3 are
    # often seen in user inputs.

    if denom_lcm.bit_length() < 50:
        return 'CD'
    else:
        return 'FF'


def _dm_rref_choose_method_ZZ(M, *, denominator=False):
    """Choose the fastest method for computing RREF over ZZ."""
    # In the extreme of very sparse matrices and low bit counts it is faster to
    # use Gauss-Jordan elimination over QQ rather than fraction-free
    # Gauss-Jordan over ZZ. In the opposite extreme of dense matrices and high
    # bit counts it is faster to use fraction-free Gauss-Jordan over ZZ. These
    # two extreme cases need to be handled differently because they lead to
    # different asymptotic complexities. In between these two extremes we need
    # a threshold for deciding which method to use. This threshold is
    # determined empirically by timing the two methods with random matrices.

    # The disadvantage of using empirical timings is that future optimisations
    # might change the relative speeds so this can easily become out of date.
    # The main thing is to get the asymptotic complexity right for the extreme
    # cases though so the precise value of the threshold is hopefully not too
    # important.

    # Empirically determined parameter.
    PARAM = 10000

    # First compute the density. This is the average number of non-zero entries
    # per row but only counting rows that have at least one non-zero entry
    # since RREF can ignore fully zero rows.
    density, nrows_nz, ncols = _dm_row_density(M)

    # For small matrices use QQ if more than half the entries are zero.
    if nrows_nz < 10:
        if density < ncols/2:
            return 'GJ'
        else:
            return 'FF'

    # These are just shortcuts for the formula below.
    if density < 5:
        return 'GJ'
    elif density > 5 + PARAM/nrows_nz:
        return 'FF'  # pragma: no cover

    # Maximum bitsize of any entry.
    elements = _dm_elements(M)
    bits = max([e.bit_length() for e in elements], default=1)

    # Wideness parameter. This is 1 for square or tall matrices but >1 for wide
    # matrices.
    wideness = max(1, 2/3*ncols/nrows_nz)

    max_density = (5 + PARAM/(nrows_nz*bits**2)) * wideness

    if density < max_density:
        return 'GJ'
    else:
        return 'FF'


def _dm_row_density(M):
    """Density measure for sparse matrices.

    Defines the "density", ``d`` as the average number of non-zero entries per
    row except ignoring rows that are fully zero. RREF can ignore fully zero
    rows so they are excluded. By definition ``d >= 1`` except that we define
    ``d = 0`` for the zero matrix.

    Returns ``(density, nrows_nz, ncols)`` where ``nrows_nz`` counts the number
    of nonzero rows and ``ncols`` is the number of columns.
    """
    # Uses the SDM dict-of-dicts representation.
    ncols = M.shape[1]
    rows_nz = M.rep.to_sdm().values()
    if not rows_nz:
        return 0, 0, ncols
    else:
        nrows_nz = len(rows_nz)
        density = sum(map(len, rows_nz)) / nrows_nz
        return density, nrows_nz, ncols


def _dm_elements(M):
    """Return nonzero elements of a DomainMatrix."""
    elements, _ = M.to_flat_nz()
    return elements


