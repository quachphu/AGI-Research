        the bottom right hand corner of the previous item: this
        will create a block-diagonal matrix whether the matrices
        are square or not.

        >>> col = [1, 2, 3]
        >>> row = [[4, 5]]
        >>> Matrix.diag(col, row)
        Matrix([
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [0, 4, 5]])

        When `unpack` is False, elements within a list need not all be
        of the same length. Setting `strict` to True would raise a
        ValueError for the following:

        >>> Matrix.diag([[1, 2, 3], [4, 5], [6]], unpack=False)
        Matrix([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0]])

        The type of the returned matrix can be set with the ``cls``
        keyword.

        >>> from sympy import ImmutableMatrix
        >>> from sympy.utilities.misc import func_name
        >>> func_name(Matrix.diag(1, cls=ImmutableMatrix))
        'ImmutableDenseMatrix'

        A zero dimension matrix can be used to position the start of
        the filling at the start of an arbitrary row or column:

        >>> from sympy import ones
        >>> r2 = ones(0, 2)
        >>> Matrix.diag(r2, 1, 2)
        Matrix([
        [0, 0, 1, 0],
        [0, 0, 0, 2]])

        See Also
        ========
        eye
        diagonal
        .dense.diag
        .expressions.blockmatrix.BlockMatrix
        .sparsetools.banded
       """
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.matrices.dense import Matrix
        from sympy.matrices import SparseMatrix
        klass = kwargs.get('cls', kls)
        if unpack and len(args) == 1 and is_sequence(args[0]) and \
                not isinstance(args[0], MatrixBase):
            args = args[0]

        # fill a default dict with the diagonal entries
        diag_entries = defaultdict(int)
        rmax = cmax = 0  # keep track of the biggest index seen
        for m in args:
            if isinstance(m, list):
                if strict:
                    # if malformed, Matrix will raise an error
                    _ = Matrix(m)
                    r, c = _.shape
                    m = _.tolist()
                else:
                    r, c, smat = SparseMatrix._handle_creation_inputs(m)
                    for (i, j), _ in smat.items():
                        diag_entries[(i + rmax, j + cmax)] = _
                    m = []  # to skip process below
            elif hasattr(m, 'shape'):  # a Matrix
                # convert to list of lists
                r, c = m.shape
                m = m.tolist()
            else:  # in this case, we're a single value
                diag_entries[(rmax, cmax)] = m
                rmax += 1
                cmax += 1
                continue
            # process list of lists
            for i, mi in enumerate(m):
                for j, _ in enumerate(mi):
                    diag_entries[(i + rmax, j + cmax)] = _
            rmax += r
            cmax += c
        if rows is None:
            rows, cols = cols, rows
        if rows is None:
            rows, cols = rmax, cmax
        else:
            cols = rows if cols is None else cols
        if rows < rmax or cols < cmax:
            raise ValueError(filldedent('''
                The constructed matrix is {} x {} but a size of {} x {}
                was specified.'''.format(rmax, cmax, rows, cols)))
        return klass._eval_diag(rows, cols, diag_entries)

    @classmethod
    def eye(kls, rows, cols=None, **kwargs):
        """Returns an identity matrix.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_eye(rows, cols)

    @classmethod
    def jordan_block(kls, size=None, eigenvalue=None, *, band='upper', **kwargs):
        """Returns a Jordan block

        Parameters
        ==========

        size : Integer, optional
            Specifies the shape of the Jordan block matrix.

        eigenvalue : Number or Symbol
            Specifies the value for the main diagonal of the matrix.

            .. note::
                The keyword ``eigenval`` is also specified as an alias
                of this keyword, but it is not recommended to use.

                We may deprecate the alias in later release.

        band : 'upper' or 'lower', optional
            Specifies the position of the off-diagonal to put `1` s on.

        cls : Matrix, optional
            Specifies the matrix class of the output form.

            If it is not specified, the class type where the method is
            being executed on will be returned.

        Returns
        =======

        Matrix
            A Jordan block matrix.

        Raises
        ======

        ValueError
            If insufficient arguments are given for matrix size
            specification, or no eigenvalue is given.

        Examples
        ========

        Creating a default Jordan block:

        >>> from sympy import Matrix
        >>> from sympy.abc import x
        >>> Matrix.jordan_block(4, x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        Creating an alternative Jordan block matrix where `1` is on
        lower off-diagonal:

        >>> Matrix.jordan_block(4, x, band='lower')
        Matrix([
        [x, 0, 0, 0],
        [1, x, 0, 0],
        [0, 1, x, 0],
        [0, 0, 1, x]])

        Creating a Jordan block with keyword arguments

        >>> Matrix.jordan_block(size=4, eigenvalue=x)
        Matrix([
        [x, 1, 0, 0],
        [0, x, 1, 0],
        [0, 0, x, 1],
        [0, 0, 0, x]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Jordan_matrix
        """
        klass = kwargs.pop('cls', kls)

        eigenval = kwargs.get('eigenval', None)
        if eigenvalue is None and eigenval is None:
            raise ValueError("Must supply an eigenvalue")
        elif eigenvalue != eigenval and None not in (eigenval, eigenvalue):
            raise ValueError(
                "Inconsistent values are given: 'eigenval'={}, "
                "'eigenvalue'={}".format(eigenval, eigenvalue))
        else:
            if eigenval is not None:
                eigenvalue = eigenval

        if size is None:
            raise ValueError("Must supply a matrix size")

        size = as_int(size)
        return klass._eval_jordan_block(size, eigenvalue, band)

    @classmethod
    def ones(kls, rows, cols=None, **kwargs):
        """Returns a matrix of ones.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_ones(rows, cols)

    @classmethod
    def zeros(kls, rows, cols=None, **kwargs):
        """Returns a matrix of zeros.

        Parameters
        ==========

        rows : rows of the matrix
        cols : cols of the matrix (if None, cols=rows)

        kwargs
        ======
        cls : class of the returned matrix
        """
        if cols is None:
            cols = rows
        if rows < 0 or cols < 0:
            raise ValueError("Cannot create a {} x {} matrix. "
                             "Both dimensions must be positive".format(rows, cols))
        klass = kwargs.get('cls', kls)
        rows, cols = as_int(rows), as_int(cols)

        return klass._eval_zeros(rows, cols)

    @classmethod
    def companion(kls, poly):
        """Returns a companion matrix of a polynomial.

        Examples
        ========

        >>> from sympy import Matrix, Poly, Symbol, symbols
        >>> x = Symbol('x')
        >>> c0, c1, c2, c3, c4 = symbols('c0:5')
        >>> p = Poly(c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + x**5, x)
        >>> Matrix.companion(p)
        Matrix([
        [0, 0, 0, 0, -c0],
        [1, 0, 0, 0, -c1],
        [0, 1, 0, 0, -c2],
        [0, 0, 1, 0, -c3],
        [0, 0, 0, 1, -c4]])
        """
        poly = kls._sympify(poly)
        if not isinstance(poly, Poly):
            raise ValueError("{} must be a Poly instance.".format(poly))
        if not poly.is_monic:
            raise ValueError("{} must be a monic polynomial.".format(poly))
        if not poly.is_univariate:
            raise ValueError(
                "{} must be a univariate polynomial.".format(poly))

        size = poly.degree()
        if not size >= 1:
            raise ValueError(
                "{} must have degree not less than 1.".format(poly))

        coeffs = poly.all_coeffs()
        def entry(i, j):
            if j == size - 1:
                return -coeffs[-1 - i]
            elif i == j + 1:
                return kls.one
            return kls.zero
        return kls._new(size, size, entry)


    @classmethod
    def wilkinson(kls, n, **kwargs):
        """Returns two square Wilkinson Matrix of size 2*n + 1
        $W_{2n + 1}^-, W_{2n + 1}^+ =$ Wilkinson(n)

        Examples
        ========

        >>> from sympy import Matrix
        >>> wminus, wplus = Matrix.wilkinson(3)
        >>> wminus
        Matrix([
        [-3,  1,  0, 0, 0, 0, 0],
        [ 1, -2,  1, 0, 0, 0, 0],
        [ 0,  1, -1, 1, 0, 0, 0],
        [ 0,  0,  1, 0, 1, 0, 0],
        [ 0,  0,  0, 1, 1, 1, 0],
        [ 0,  0,  0, 0, 1, 2, 1],
        [ 0,  0,  0, 0, 0, 1, 3]])
        >>> wplus
        Matrix([
        [3, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 3]])

        References
        ==========

        .. [1] https://blogs.mathworks.com/cleve/2013/04/15/wilkinsons-matrices-2/
        .. [2] J. H. Wilkinson, The Algebraic Eigenvalue Problem, Claredon Press, Oxford, 1965, 662 pp.

        """
        klass = kwargs.get('cls', kls)
        n = as_int(n)
        return klass._eval_wilkinson(n)

    # The RepMatrix subclass uses more efficient sparse implementations of
    # _eval_iter_values and other things.

    def _eval_iter_values(self):
        return (i for i in self if i is not S.Zero)

    def _eval_values(self):
        return list(self.iter_values())

    def _eval_iter_items(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j]:
                    yield (i, j), self[i, j]

    def _eval_atoms(self, *types):
        values = self.values()
        if len(values) < self.rows * self.cols and isinstance(S.Zero, types):
            s = {S.Zero}
        else:
            s = set()
        return s.union(*[v.atoms(*types) for v in values])

    def _eval_free_symbols(self):
        return set().union(*(i.free_symbols for i in set(self.values())))

    def _eval_has(self, *patterns):
        return any(a.has(*patterns) for a in self.iter_values())

    def _eval_is_symbolic(self):
        return self.has(Symbol)

    # _eval_is_hermitian is called by some general SymPy
    # routines and has a different *args signature.  Make
    # sure the names don't clash by adding `_matrix_` in name.
    def _eval_is_matrix_hermitian(self, simpfunc):
        herm = lambda i, j: simpfunc(self[i, j] - self[j, i].adjoint()).is_zero
        return fuzzy_and(herm(i, j) for (i, j), v in self.iter_items())

    def _eval_is_zero_matrix(self):
        return fuzzy_and(v.is_zero for v in self.iter_values())

    def _eval_is_Identity(self) -> FuzzyBool:
        one = self.one
        zero = self.zero
        ident = lambda i, j, v: v is one if i == j else v is zero
        return all(ident(i, j, v) for (i, j), v in self.iter_items())

    def _eval_is_diagonal(self):
        return fuzzy_and(v.is_zero for (i, j), v in self.iter_items() if i != j)

    def _eval_is_lower(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i < j)

    def _eval_is_upper(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i > j)

    def _eval_is_lower_hessenberg(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i + 1 < j)

    def _eval_is_upper_hessenberg(self):
        return all(v.is_zero for (i, j), v in self.iter_items() if i > j + 1)

    def _eval_is_symmetric(self, simpfunc):
        sym = lambda i, j: simpfunc(self[i, j] - self[j, i]).is_zero
        return fuzzy_and(sym(i, j) for (i, j), v in self.iter_items())

    def _eval_is_anti_symmetric(self, simpfunc):
        anti = lambda i, j: simpfunc(self[i, j] + self[j, i]).is_zero
        return fuzzy_and(anti(i, j) for (i, j), v in self.iter_items())

    def _has_positive_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_positive for x in diagonal_entries)

    def _has_nonnegative_diagonals(self):
        diagonal_entries = (self[i, i] for i in range(self.rows))
        return fuzzy_and(x.is_nonnegative for x in diagonal_entries)

    def atoms(self, *types):
        """Returns the atoms that form the current object.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import Matrix
        >>> Matrix([[x]])
        Matrix([[x]])
        >>> _.atoms()
        {x}
        >>> Matrix([[x, y], [y, x]])
        Matrix([
        [x, y],
        [y, x]])
        >>> _.atoms()
        {x, y}
        """

        types = tuple(t if isinstance(t, type) else type(t) for t in types)
        if not types:
            types = (Atom,)
        return self._eval_atoms(*types)

    @property
    def free_symbols(self):
        """Returns the free symbols within the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix([[x], [1]]).free_symbols
        {x}
        """
        return self._eval_free_symbols()

    def has(self, *patterns):
        """Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import Matrix, SparseMatrix, Float
        >>> from sympy.abc import x, y
        >>> A = Matrix(((1, x), (0.2, 3)))
        >>> B = SparseMatrix(((1, x), (0.2, 3)))
        >>> A.has(x)
        True
        >>> A.has(y)
        False
        >>> A.has(Float)
        True
        >>> B.has(x)
        True
        >>> B.has(y)
        False
        >>> B.has(Float)
        True
        """
        return self._eval_has(*patterns)

    def is_anti_symmetric(self, simplify=True):
        """Check if matrix M is an antisymmetric matrix,
        that is, M is a square matrix with all M[i, j] == -M[j, i].

        When ``simplify=True`` (default), the sum M[i, j] + M[j, i] is
        simplified before testing to see if it is zero. By default,
        the SymPy simplify function is used. To use a custom function
        set simplify to a function that accepts a single argument which
        returns a simplified expression. To skip simplification, set
        simplify to False but note that although this will be faster,
        it may induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> m = Matrix(2, 2, [0, 1, -1, 0])
        >>> m
        Matrix([
        [ 0, 1],
        [-1, 0]])
        >>> m.is_anti_symmetric()
        True
        >>> x, y = symbols('x y')
        >>> m = Matrix(2, 3, [0, 0, x, -y, 0, 0])
        >>> m
        Matrix([
        [ 0, 0, x],
        [-y, 0, 0]])
        >>> m.is_anti_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [0, x**2 + 2*x + 1, y,
        ...                   -(x + 1)**2, 0, x*y,
        ...                   -y, -x*y, 0])

        Simplification of matrix elements is done by default so even
        though two elements which should be equal and opposite would not
        pass an equality test, the matrix is still reported as
        anti-symmetric:

        >>> m[0, 1] == -m[1, 0]
        False
        >>> m.is_anti_symmetric()
        True

        If ``simplify=False`` is used for the case when a Matrix is already
        simplified, this will speed things up. Here, we see that without
        simplification the matrix does not appear anti-symmetric:

        >>> print(m.is_anti_symmetric(simplify=False))
        None

        But if the matrix were already expanded, then it would appear
        anti-symmetric and simplification in the is_anti_symmetric routine
        is not needed:

        >>> m = m.expand()
        >>> m.is_anti_symmetric(simplify=False)
        True
        """
        # accept custom simplification
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _utilities_simplify if simplify else lambda x: x

        if not self.is_square:
            return False
        return self._eval_is_anti_symmetric(simpfunc)

    def is_diagonal(self):
        """Check if matrix is diagonal,
        that is matrix in which the entries outside the main diagonal are all zero.

        Examples
        ========

        >>> from sympy import Matrix, diag
        >>> m = Matrix(2, 2, [1, 0, 0, 2])
        >>> m
        Matrix([
        [1, 0],
        [0, 2]])
        >>> m.is_diagonal()
        True

        >>> m = Matrix(2, 2, [1, 1, 0, 2])
        >>> m
        Matrix([
        [1, 1],
        [0, 2]])
        >>> m.is_diagonal()
        False

        >>> m = diag(1, 2, 3)
        >>> m
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])
        >>> m.is_diagonal()
        True

        See Also
        ========

        is_lower
        is_upper
        sympy.matrices.matrixbase.MatrixBase.is_diagonalizable
        diagonalize
        """
        return self._eval_is_diagonal()

    @property
    def is_weakly_diagonally_dominant(self):
        r"""Tests if the matrix is row weakly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row weakly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| \ge \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_weakly_diagonally_dominant
        True

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_weakly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_weakly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_nonnegative

        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_strongly_diagonally_dominant(self):
        r"""Tests if the matrix is row strongly diagonally dominant.

        Explanation
        ===========

        A $n, n$ matrix $A$ is row strongly diagonally dominant if

        .. math::
            \left|A_{i, i}\right| > \sum_{j = 0, j \neq i}^{n-1}
            \left|A_{i, j}\right| \quad {\text{for all }}
            i \in \{ 0, ..., n-1 \}

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[3, -2, 1], [1, -3, 2], [-1, 2, 4]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-2, 2, 1], [1, 3, 2], [1, -2, 0]])
        >>> A.is_strongly_diagonally_dominant
        False

        >>> A = Matrix([[-4, 2, 1], [1, 6, 2], [1, -2, 5]])
        >>> A.is_strongly_diagonally_dominant
        True

        Notes
        =====

        If you want to test whether a matrix is column diagonally
        dominant, you can apply the test after transposing the matrix.
        """
        if not self.is_square:
            return False

        rows, cols = self.shape

        def test_row(i):
            summation = self.zero
            for j in range(cols):
                if i != j:
                    summation += Abs(self[i, j])
            return (Abs(self[i, i]) - summation).is_positive

        return fuzzy_and(test_row(i) for i in range(rows))

    @property
    def is_hermitian(self):
        """Checks if the matrix is Hermitian.

        In a Hermitian matrix element i,j is the complex conjugate of
        element j,i.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy import I
        >>> from sympy.abc import x
        >>> a = Matrix([[1, I], [-I, 1]])
        >>> a
        Matrix([
        [ 1, I],
        [-I, 1]])
        >>> a.is_hermitian
        True
        >>> a[0, 0] = 2*I
        >>> a.is_hermitian
        False
        >>> a[0, 0] = x
        >>> a.is_hermitian
        >>> a[0, 1] = a[1, 0]*I
        >>> a.is_hermitian
        False
        """
        if not self.is_square:
            return False

        return self._eval_is_matrix_hermitian(_utilities_simplify)

    @property
    def is_Identity(self) -> FuzzyBool:
        if not self.is_square:
            return False
        return self._eval_is_Identity()

    @property
    def is_lower_hessenberg(self):
        r"""Checks if the matrix is in the lower-Hessenberg form.

        The lower hessenberg matrix has zero entries
        above the first superdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 0, 0], [5, 2, 3, 0], [3, 4, 3, 7], [5, 6, 1, 1]])
        >>> a
        Matrix([
        [1, 2, 0, 0],
        [5, 2, 3, 0],
        [3, 4, 3, 7],
        [5, 6, 1, 1]])
        >>> a.is_lower_hessenberg
        True

        See Also
        ========

        is_upper_hessenberg
        is_lower
        """
        return self._eval_is_lower_hessenberg()

    @property
    def is_lower(self):
        """Check if matrix is a lower triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_lower
        True

        >>> m = Matrix(4, 3, [0, 0, 0, 2, 0, 0, 1, 4, 0, 6, 6, 5])
        >>> m
        Matrix([
        [0, 0, 0],
        [2, 0, 0],
        [1, 4, 0],
        [6, 6, 5]])
        >>> m.is_lower
        True

        >>> from sympy.abc import x, y
        >>> m = Matrix(2, 2, [x**2 + y, y**2 + x, 0, x + y])
        >>> m
        Matrix([
        [x**2 + y, x + y**2],
        [       0,    x + y]])
        >>> m.is_lower
        False

        See Also
        ========

        is_upper
        is_diagonal
        is_lower_hessenberg
        """
        return self._eval_is_lower()

    @property
    def is_square(self):
        """Checks if a matrix is square.

        A matrix is square if the number of rows equals the number of columns.
        The empty matrix is square by definition, since the number of rows and
        the number of columns are both zero.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> b = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> c = Matrix([])
        >>> a.is_square
        False
        >>> b.is_square
        True
        >>> c.is_square
        True
        """
        return self.rows == self.cols

    def is_symbolic(self):
        """Checks if any elements contain Symbols.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.is_symbolic()
        True

        """
        return self._eval_is_symbolic()

    def is_symmetric(self, simplify=True):
        """Check if matrix is symmetric matrix,
        that is square matrix and is equal to its transpose.

        By default, simplifications occur before testing symmetry.
        They can be skipped using 'simplify=False'; while speeding things a bit,
        this may however induce false negatives.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [0, 1, 1, 2])
        >>> m
        Matrix([
        [0, 1],
        [1, 2]])
        >>> m.is_symmetric()
        True

        >>> m = Matrix(2, 2, [0, 1, 2, 0])
        >>> m
        Matrix([
        [0, 1],
        [2, 0]])
        >>> m.is_symmetric()
        False

        >>> m = Matrix(2, 3, [0, 0, 0, 0, 0, 0])
        >>> m
        Matrix([
        [0, 0, 0],
        [0, 0, 0]])
        >>> m.is_symmetric()
        False

        >>> from sympy.abc import x, y
        >>> m = Matrix(3, 3, [1, x**2 + 2*x + 1, y, (x + 1)**2, 2, 0, y, 0, 3])
        >>> m
        Matrix([
        [         1, x**2 + 2*x + 1, y],
        [(x + 1)**2,              2, 0],
        [         y,              0, 3]])
        >>> m.is_symmetric()
        True

        If the matrix is already simplified, you may speed-up is_symmetric()
        test by using 'simplify=False'.

        >>> bool(m.is_symmetric(simplify=False))
        False
        >>> m1 = m.expand()
        >>> m1.is_symmetric(simplify=False)
        True
        """
        simpfunc = simplify
        if not isfunction(simplify):
            simpfunc = _utilities_simplify if simplify else lambda x: x

        if not self.is_square:
            return False

        return self._eval_is_symmetric(simpfunc)

    @property
    def is_upper_hessenberg(self):
        """Checks if the matrix is the upper-Hessenberg form.

        The upper hessenberg matrix has zero entries
        below the first subdiagonal.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[1, 4, 2, 3], [3, 4, 1, 7], [0, 2, 3, 4], [0, 0, 1, 3]])
        >>> a
        Matrix([
        [1, 4, 2, 3],
        [3, 4, 1, 7],
        [0, 2, 3, 4],
        [0, 0, 1, 3]])
        >>> a.is_upper_hessenberg
        True

        See Also
        ========

        is_lower_hessenberg
        is_upper
        """
        return self._eval_is_upper_hessenberg()

    @property
    def is_upper(self):
        """Check if matrix is an upper triangular matrix. True can be returned
        even if the matrix is not square.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, [1, 0, 0, 1])
        >>> m
        Matrix([
        [1, 0],
        [0, 1]])
        >>> m.is_upper
        True

        >>> m = Matrix(4, 3, [5, 1, 9, 0, 4, 6, 0, 0, 5, 0, 0, 0])
        >>> m
        Matrix([
        [5, 1, 9],
        [0, 4, 6],
        [0, 0, 5],
        [0, 0, 0]])
        >>> m.is_upper
        True

        >>> m = Matrix(2, 3, [4, 2, 5, 6, 1, 1])
        >>> m
        Matrix([
        [4, 2, 5],
        [6, 1, 1]])
        >>> m.is_upper
        False

        See Also
        ========

        is_lower
        is_diagonal
        is_upper_hessenberg
        """
        return self._eval_is_upper()

    @property
    def is_zero_matrix(self):
        """Checks if a matrix is a zero matrix.

        A matrix is zero if every element is zero.  A matrix need not be square
        to be considered zero.  The empty matrix is zero by the principle of
        vacuous truth.  For a matrix that may or may not be zero (e.g.
        contains a symbol), this will be None

        Examples
        ========

        >>> from sympy import Matrix, zeros
        >>> from sympy.abc import x
        >>> a = Matrix([[0, 0], [0, 0]])
        >>> b = zeros(3, 4)
        >>> c = Matrix([[0, 1], [0, 0]])
        >>> d = Matrix([])
        >>> e = Matrix([[x, 0], [0, 0]])
        >>> a.is_zero_matrix
        True
        >>> b.is_zero_matrix
        True
        >>> c.is_zero_matrix
        False
        >>> d.is_zero_matrix
        True
        >>> e.is_zero_matrix
        """
        return self._eval_is_zero_matrix()

    def values(self):
        """Return non-zero values of self.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> m.values()
        [1, 2, 3]

        See Also
        ========

        iter_values
        tolist
        flat
        """
        return self._eval_values()

    def iter_values(self):
        """
        Iterate over non-zero values of self.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> list(m.iter_values())
        [1, 2, 3]

        See Also
        ========

        values
        """
        return self._eval_iter_values()

    def iter_items(self):
        """Iterate over indices and values of nonzero items.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 1], [2, 3]])
        >>> list(m.iter_items())
        [((0, 1), 1), ((1, 0), 2), ((1, 1), 3)]

        See Also
        ========

        iter_values
        todok
        """
        return self._eval_iter_items()

    def _eval_adjoint(self):
        return self.transpose().applyfunc(lambda x: x.adjoint())

    def _eval_applyfunc(self, f):
        cols = self.cols
        size = self.rows*self.cols

        dok = self.todok()
        valmap = {v: f(v) for v in dok.values()}

        if len(dok) < size and ((fzero := f(S.Zero)) is not S.Zero):
            out_flat = [fzero]*size
            for (i, j), v in dok.items():
                out_flat[i*cols + j] = valmap[v]
            out = self._new(self.rows, self.cols, out_flat)
        else:
            fdok = {ij: valmap[v] for ij, v in dok.items()}
            out = self.from_dok(self.rows, self.cols, fdok)

        return out

    def _eval_as_real_imag(self):  # type: ignore
        return (self.applyfunc(re), self.applyfunc(im))

    def _eval_conjugate(self):
        return self.applyfunc(lambda x: x.conjugate())

    def _eval_permute_cols(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[i, mapping[j]]

        return self._new(self.rows, self.cols, entry)

    def _eval_permute_rows(self, perm):
        # apply the permutation to a list
        mapping = list(perm)

        def entry(i, j):
            return self[mapping[i], j]

        return self._new(self.rows, self.cols, entry)

    def _eval_trace(self):
        return sum(self[i, i] for i in range(self.rows))

    def _eval_transpose(self):
        return self._new(self.cols, self.rows, lambda i, j: self[j, i])

    def adjoint(self):
        """Conjugate transpose or Hermitian conjugation."""
        return self._eval_adjoint()

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        return self._eval_applyfunc(f)

    def as_real_imag(self, deep=True, **hints):
        """Returns a tuple containing the (real, imaginary) part of matrix."""
        # XXX: Ignoring deep and hints...
        return self._eval_as_real_imag()

    def conjugate(self):
        """Return the by-element conjugation.

        Examples
        ========

        >>> from sympy import SparseMatrix, I
        >>> a = SparseMatrix(((1, 2 + I), (3, 4), (I, -I)))
        >>> a
        Matrix([
        [1, 2 + I],
        [3,     4],
        [I,    -I]])
        >>> a.C
        Matrix([
        [ 1, 2 - I],
        [ 3,     4],
        [-I,     I]])

        See Also
        ========

        transpose: Matrix transposition
        H: Hermite conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        return self._eval_conjugate()

    def doit(self, **hints):
        return self.applyfunc(lambda x: x.doit(**hints))

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """Apply evalf() to each element of self."""
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        return self.applyfunc(lambda i: i.evalf(n, **options))

    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
               mul=True, log=True, multinomial=True, basic=True, **hints):
        """Apply core.function.expand to each entry of the matrix.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import Matrix
        >>> Matrix(1, 1, [x*(x+1)])
        Matrix([[x*(x + 1)]])
        >>> _.expand()
        Matrix([[x**2 + x]])

        """
        return self.applyfunc(lambda x: x.expand(
            deep, modulus, power_base, power_exp, mul, log, multinomial, basic,
            **hints))

    @property
    def H(self):
        """Return Hermite conjugate.

        Examples
        ========

        >>> from sympy import Matrix, I
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m
        Matrix([
        [    0],
        [1 + I],
        [    2],
        [    3]])
        >>> m.H
        Matrix([[0, 1 - I, 2, 3]])

        See Also
        ========

        conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.D: Dirac conjugation
        """
        return self.adjoint()

    def permute(self, perm, orientation='rows', direction='forward'):
        r"""Permute the rows or columns of a matrix by the given list of
        swaps.

        Parameters
        ==========

        perm : Permutation, list, or list of lists
            A representation for the permutation.

            If it is ``Permutation``, it is used directly with some
            resizing with respect to the matrix size.

            If it is specified as list of lists,
            (e.g., ``[[0, 1], [0, 2]]``), then the permutation is formed
            from applying the product of cycles. The direction how the
            cyclic product is applied is described in below.

            If it is specified as a list, the list should represent
            an array form of a permutation. (e.g., ``[1, 2, 0]``) which
            would would form the swapping function
            `0 \mapsto 1, 1 \mapsto 2, 2\mapsto 0`.

        orientation : 'rows', 'cols'
            A flag to control whether to permute the rows or the columns

        direction : 'forward', 'backward'
            A flag to control whether to apply the permutations from
            the start of the list first, or from the back of the list
            first.

            For example, if the permutation specification is
            ``[[0, 1], [0, 2]]``,

            If the flag is set to ``'forward'``, the cycle would be
            formed as `0 \mapsto 2, 2 \mapsto 1, 1 \mapsto 0`.

            If the flag is set to ``'backward'``, the cycle would be
            formed as `0 \mapsto 1, 1 \mapsto 2, 2 \mapsto 0`.

            If the argument ``perm`` is not in a form of list of lists,
            this flag takes no effect.

        Examples
        ========

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.permute([[0, 1], [0, 2]], orientation='rows', direction='forward')
        Matrix([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])

        >>> from sympy import eye
        >>> M = eye(3)
        >>> M.permute([[0, 1], [0, 2]], orientation='rows', direction='backward')
        Matrix([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]])

        Notes
        =====

        If a bijective function
        `\sigma : \mathbb{N}_0 \rightarrow \mathbb{N}_0` denotes the
        permutation.

        If the matrix `A` is the matrix to permute, represented as
        a horizontal or a vertical stack of vectors:

        .. math::
            A =
            \begin{bmatrix}
            a_0 \\ a_1 \\ \vdots \\ a_{n-1}
            \end{bmatrix} =
            \begin{bmatrix}
            \alpha_0 & \alpha_1 & \cdots & \alpha_{n-1}
            \end{bmatrix}

        If the matrix `B` is the result, the permutation of matrix rows
        is defined as:

        .. math::
            B := \begin{bmatrix}
            a_{\sigma(0)} \\ a_{\sigma(1)} \\ \vdots \\ a_{\sigma(n-1)}
            \end{bmatrix}

        And the permutation of matrix columns is defined as:

        .. math::
            B := \begin{bmatrix}
            \alpha_{\sigma(0)} & \alpha_{\sigma(1)} &
            \cdots & \alpha_{\sigma(n-1)}
            \end{bmatrix}
        """
        from sympy.combinatorics import Permutation

        # allow british variants and `columns`
        if direction == 'forwards':
            direction = 'forward'
        if direction == 'backwards':
            direction = 'backward'
        if orientation == 'columns':
            orientation = 'cols'

        if direction not in ('forward', 'backward'):
            raise TypeError("direction='{}' is an invalid kwarg. "
                            "Try 'forward' or 'backward'".format(direction))
        if orientation not in ('rows', 'cols'):
            raise TypeError("orientation='{}' is an invalid kwarg. "
                            "Try 'rows' or 'cols'".format(orientation))

        if not isinstance(perm, (Permutation, Iterable)):
            raise ValueError(
                "{} must be a list, a list of lists, "
                "or a SymPy permutation object.".format(perm))

        # ensure all swaps are in range
        max_index = self.rows if orientation == 'rows' else self.cols
        if not all(0 <= t <= max_index for t in flatten(list(perm))):
            raise IndexError("`swap` indices out of range.")

        if perm and not isinstance(perm, Permutation) and \
            isinstance(perm[0], Iterable):
            if direction == 'forward':
                perm = list(reversed(perm))
            perm = Permutation(perm, size=max_index+1)
        else:
            perm = Permutation(perm, size=max_index+1)

        if orientation == 'rows':
            return self._eval_permute_rows(perm)
        if orientation == 'cols':
            return self._eval_permute_cols(perm)

    def permute_cols(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='cols', direction=direction)``

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='cols', direction=direction)

    def permute_rows(self, swaps, direction='forward'):
        """Alias for
        ``self.permute(swaps, orientation='rows', direction=direction)``

        See Also
        ========

        permute
        """
        return self.permute(swaps, orientation='rows', direction=direction)

    def refine(self, assumptions=True):
        """Apply refine to each element of the matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, Abs, sqrt, Q
        >>> x = Symbol('x')
        >>> Matrix([[Abs(x)**2, sqrt(x**2)],[sqrt(x**2), Abs(x)**2]])
        Matrix([
        [ Abs(x)**2, sqrt(x**2)],
        [sqrt(x**2),  Abs(x)**2]])
        >>> _.refine(Q.real(x))
        Matrix([
        [  x**2, Abs(x)],
        [Abs(x),   x**2]])

        """
        return self.applyfunc(lambda x: refine(x, assumptions))

    def replace(self, F, G, map=False, simultaneous=True, exact=None):
        """Replaces Function F in Matrix entries with Function G.

        Examples
        ========

        >>> from sympy import symbols, Function, Matrix
        >>> F, G = symbols('F, G', cls=Function)
        >>> M = Matrix(2, 2, lambda i, j: F(i+j)) ; M
        Matrix([
        [F(0), F(1)],
        [F(1), F(2)]])
        >>> N = M.replace(F,G)
        >>> N
        Matrix([
        [G(0), G(1)],
        [G(1), G(2)]])
        """
        kwargs = {'map': map, 'simultaneous': simultaneous, 'exact': exact}

        if map:

            d = {}
            def func(eij):
                eij, dij = eij.replace(F, G, **kwargs)
                d.update(dij)
                return eij

            M = self.applyfunc(func)
            return M, d

        else:
            return self.applyfunc(lambda i: i.replace(F, G, **kwargs))

    def rot90(self, k=1):
        """Rotates Matrix by 90 degrees

        Parameters
        ==========

        k : int
            Specifies how many times the matrix is rotated by 90 degrees
            (clockwise when positive, counter-clockwise when negative).

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> A = Matrix(2, 2, symbols('a:d'))
        >>> A
        Matrix([
        [a, b],
        [c, d]])

        Rotating the matrix clockwise one time:

        >>> A.rot90(1)
        Matrix([
        [c, a],
        [d, b]])

        Rotating the matrix anticlockwise two times:

        >>> A.rot90(-2)
        Matrix([
        [d, c],
        [b, a]])
        """

        mod = k%4
        if mod == 0:
            return self
        if mod == 1:
            return self[::-1, ::].T
        if mod == 2:
            return self[::-1, ::-1]
        if mod == 3:
            return self[::, ::-1].T

    def simplify(self, **kwargs):
        """Apply simplify to each element of the matrix.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, sin, cos
        >>> SparseMatrix(1, 1, [x*sin(y)**2 + x*cos(y)**2])
        Matrix([[x*sin(y)**2 + x*cos(y)**2]])
        >>> _.simplify()
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.simplify(**kwargs))

    def subs(self, *args, **kwargs):  # should mirror core.basic.subs
        """Return a new matrix with subs applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.subs(x, y)
        Matrix([[y]])
        >>> Matrix(_).subs(y, x)
        Matrix([[x]])
        """

        if len(args) == 1 and  not isinstance(args[0], (dict, set)) and iter(args[0]) and not is_sequence(args[0]):
            args = (list(args[0]),)

        return self.applyfunc(lambda x: x.subs(*args, **kwargs))

    def trace(self):
        """
        Returns the trace of a square matrix i.e. the sum of the
        diagonal elements.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.trace()
        5

        """
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        return self._eval_trace()

    def transpose(self):
        """
        Returns the transpose of the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.transpose()
        Matrix([
        [1, 3],
        [2, 4]])

        >>> from sympy import Matrix, I
        >>> m=Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m.transpose()
        Matrix([
        [    1, 3],
        [2 + I, 4]])
        >>> m.T == m.transpose()
        True

        See Also
        ========

        conjugate: By-element conjugation

        """
        return self._eval_transpose()

    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    @property
    def C(self):
        '''By-element conjugation'''
        return self.conjugate()

    def n(self, *args, **kwargs):
        """Apply evalf() to each element of self."""
        return self.evalf(*args, **kwargs)

    def xreplace(self, rule):  # should mirror core.basic.xreplace
        """Return a new matrix with xreplace applied to each entry.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import SparseMatrix, Matrix
        >>> SparseMatrix(1, 1, [x])
        Matrix([[x]])
        >>> _.xreplace({x: y})
        Matrix([[y]])
        >>> Matrix(_).xreplace({y: x})
        Matrix([[x]])
        """
        return self.applyfunc(lambda x: x.xreplace(rule))

    def _eval_simplify(self, **kwargs):
        # XXX: We can't use self.simplify here as mutable subclasses will
        # override simplify and have it return None
        return self.applyfunc(lambda x: x.simplify(**kwargs))

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        return self.applyfunc(lambda x: trigsimp(x, **opts))

    def upper_triangular(self, k=0):
        """Return the elements on and above the kth diagonal of a matrix.
        If k is not specified then simply returns upper-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.upper_triangular()
        Matrix([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]])

        >>> A.upper_triangular(2)
        Matrix([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

        >>> A.upper_triangular(-1)
        Matrix([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1]])

        """

        def entry(i, j):
            return self[i, j] if i + k <= j else self.zero

        return self._new(self.rows, self.cols, entry)

    def lower_triangular(self, k=0):
        """Return the elements on and below the kth diagonal of a matrix.
        If k is not specified then simply returns lower-triangular portion
        of a matrix

        Examples
        ========

        >>> from sympy import ones
        >>> A = ones(4)
        >>> A.lower_triangular()
        Matrix([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]])

        >>> A.lower_triangular(-2)
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]])

        >>> A.lower_triangular(1)
        Matrix([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]])

        """

        def entry(i, j):
            return self[i, j] if i + k >= j else self.zero

        return self._new(self.rows, self.cols, entry)

    def _eval_Abs(self):
        return self._new(self.rows, self.cols, lambda i, j: Abs(self[i, j]))

    def _eval_add(self, other):
        return self._new(self.rows, self.cols,
                         lambda i, j: self[i, j] + other[i, j])

    def _eval_matrix_mul(self, other):
        def entry(i, j):
            vec = [self[i,k]*other[k,j] for k in range(self.cols)]
            try:
                return Add(*vec)
            except (TypeError, SympifyError):
                # Some matrices don't work with `sum` or `Add`
                # They don't work with `sum` because `sum` tries to add `0`
                # Fall back to a safe way to multiply if the `Add` fails.
                return reduce(lambda a, b: a + b, vec)

        return self._new(self.rows, other.cols, entry)

    def _eval_matrix_mul_elementwise(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other[i,j])

    def _eval_matrix_rmul(self, other):
        def entry(i, j):
            return sum(other[i,k]*self[k,j] for k in range(other.cols))
        return self._new(other.rows, self.cols, entry)

    def _eval_pow_by_recursion(self, num):
        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion(num - 1)
        else:
            a = b = self._eval_pow_by_recursion(num // 2)

        return a.multiply(b)

    def _eval_pow_by_cayley(self, exp):
        from sympy.discrete.recurrences import linrec_coeffs
        row = self.shape[0]
        p = self.charpoly()

        coeffs = (-p).all_coeffs()[1:]
        coeffs = linrec_coeffs(coeffs, exp)
        new_mat = self.eye(row)
        ans = self.zeros(row)

        for i in range(row):
            ans += coeffs[i]*new_mat
            new_mat *= self

        return ans

    def _eval_pow_by_recursion_dotprodsimp(self, num, prevsimp=None):
        if prevsimp is None:
            prevsimp = [True]*len(self)

        if num == 1:
            return self

        if num % 2 == 1:
            a, b = self, self._eval_pow_by_recursion_dotprodsimp(num - 1,
                    prevsimp=prevsimp)
        else:
            a = b = self._eval_pow_by_recursion_dotprodsimp(num // 2,
                    prevsimp=prevsimp)

        m     = a.multiply(b, dotprodsimp=False)
        lenm  = len(m)
        elems = [None]*lenm

        for i in range(lenm):
            if prevsimp[i]:
                elems[i], prevsimp[i] = _dotprodsimp(m[i], withsimp=True)
            else:
                elems[i] = m[i]

        return m._new(m.rows, m.cols, elems)

    def _eval_scalar_mul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: self[i,j]*other)

    def _eval_scalar_rmul(self, other):
        return self._new(self.rows, self.cols, lambda i, j: other*self[i,j])

    def _eval_Mod(self, other):
        return self._new(self.rows, self.cols, lambda i, j: Mod(self[i, j], other))

    # Python arithmetic functions
    def __abs__(self):
        """Returns a new matrix with entry-wise absolute values."""
        return self._eval_Abs()

    @call_highest_priority('__radd__')
    def __add__(self, other):
        """Return self + other, raising ShapeError if shapes do not match."""

        other, T = _coerce_operand(self, other)

        if T != "is_matrix":
            return NotImplemented

        if self.shape != other.shape:
            raise ShapeError(f"Matrix size mismatch: {self.shape} + {other.shape}.")

        # Unify matrix types
        a, b = self, other
        if a.__class__ != classof(a, b):
            b, a = a, b

        return a._eval_add(b)

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * (self.one / other)

    @call_highest_priority('__rmatmul__')
    def __matmul__(self, other):
        self, other, T = _unify_with_other(self, other)

        if T != "is_matrix":
            return NotImplemented

        return self.__mul__(other)

    def __mod__(self, other):
        return self.applyfunc(lambda x: x % other)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """Return self*other where other is either a scalar or a matrix
        of compatible dimensions.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
        >>> 2*A == A*2 == Matrix([[2, 4, 6], [8, 10, 12]])
        True
        >>> B = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> A*B
        Matrix([
        [30, 36, 42],
        [66, 81, 96]])
        >>> B*A
        Traceback (most recent call last):
        ...
        ShapeError: Matrices size mismatch.
        >>>

        See Also
        ========

        matrix_multiply_elementwise
        """

        return self.multiply(other)

    def multiply(self, other, dotprodsimp=None):
        """Same as __mul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """

        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)

        self, other, T = _unify_with_other(self, other)

        if T == "possible_scalar":
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                return NotImplemented

        elif T == "is_matrix":

            if self.shape[1] != other.shape[0]:
                raise ShapeError(f"Matrix size mismatch: {self.shape} * {other.shape}.")

            m = self._eval_matrix_mul(other)

            if isimpbool:
                m = m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])

            return m

        else:
            return NotImplemented

    def multiply_elementwise(self, other):
        """Return the Hadamard product (elementwise product) of A and B

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix([[0, 1, 2], [3, 4, 5]])
        >>> B = Matrix([[1, 10, 100], [100, 10, 1]])
        >>> A.multiply_elementwise(B)
        Matrix([
        [  0, 10, 200],
        [300, 40,   5]])

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.cross
        sympy.matrices.matrixbase.MatrixBase.dot
        multiply
        """
        if self.shape != other.shape:
            raise ShapeError("Matrix shapes must agree {} != {}".format(self.shape, other.shape))

        return self._eval_matrix_mul_elementwise(other)

    def __neg__(self):
        return self._eval_scalar_mul(-1)

    @call_highest_priority('__rpow__')
    def __pow__(self, exp):
        """Return self**exp a scalar or symbol."""

        return self.pow(exp)


    def pow(self, exp, method=None):
        r"""Return self**exp a scalar or symbol.

        Parameters
        ==========

        method : multiply, mulsimp, jordan, cayley
            If multiply then it returns exponentiation using recursion.
            If jordan then Jordan form exponentiation will be used.
            If cayley then the exponentiation is done using Cayley-Hamilton
            theorem.
            If mulsimp then the exponentiation is done using recursion
            with dotprodsimp. This specifies whether intermediate term
            algebraic simplification is used during naive matrix power to
            control expression blowup and thus speed up calculation.
            If None, then it heuristically decides which method to use.

        """

        if method is not None and method not in ['multiply', 'mulsimp', 'jordan', 'cayley']:
            raise TypeError('No such method')
        if self.rows != self.cols:
            raise NonSquareMatrixError()
        a = self
        jordan_pow = getattr(a, '_matrix_pow_by_jordan_blocks', None)
        exp = sympify(exp)

        if exp.is_zero:
            return a._new(a.rows, a.cols, lambda i, j: int(i == j))
        if exp == 1:
            return a

        diagonal = getattr(a, 'is_diagonal', None)
        if diagonal is not None and diagonal():
            return a._new(a.rows, a.cols, lambda i, j: a[i,j]**exp if i == j else 0)

        if exp.is_Number and exp % 1 == 0:
            if a.rows == 1:
                return a._new([[a[0]**exp]])
            if exp < 0:
                exp = -exp
                a = a.inv()
        # When certain conditions are met,
        # Jordan block algorithm is faster than
        # computation by recursion.
        if method == 'jordan':
            try:
                return jordan_pow(exp)
            except MatrixError:
                if method == 'jordan':
                    raise

        elif method == 'cayley':
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("cayley method is only valid for integer powers")
            return a._eval_pow_by_cayley(exp)

        elif method == "mulsimp":
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("mulsimp method is only valid for integer powers")
            return a._eval_pow_by_recursion_dotprodsimp(exp)

        elif method == "multiply":
            if not exp.is_Number or exp % 1 != 0:
                raise ValueError("multiply method is only valid for integer powers")
            return a._eval_pow_by_recursion(exp)

        elif method is None and exp.is_Number and exp % 1 == 0:
            if exp.is_Float:
                exp = Integer(exp)
            # Decide heuristically which method to apply
            if a.rows == 2 and exp > 100000:
                return jordan_pow(exp)
            elif _get_intermediate_simp_bool(True, None):
                return a._eval_pow_by_recursion_dotprodsimp(exp)
            elif exp > 10000:
                return a._eval_pow_by_cayley(exp)
            else:
                return a._eval_pow_by_recursion(exp)

        if jordan_pow:
            try:
                return jordan_pow(exp)
            except NonInvertibleMatrixError:
                # Raised by jordan_pow on zero determinant matrix unless exp is
                # definitely known to be a non-negative integer.
                # Here we raise if n is definitely not a non-negative integer
                # but otherwise we can leave this as an unevaluated MatPow.
                if exp.is_integer is False or exp.is_nonnegative is False:
                    raise

        from sympy.matrices.expressions import MatPow
        return MatPow(a, exp)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self.__add__(other)

    @call_highest_priority('__matmul__')
    def __rmatmul__(self, other):
        self, other, T = _unify_with_other(self, other)

        if T != "is_matrix":
            return NotImplemented

        return self.__rmul__(other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self.rmultiply(other)

    def rmultiply(self, other, dotprodsimp=None):
        """Same as __rmul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        self, other, T = _unify_with_other(self, other)

        if T == "possible_scalar":
            try:
                return self._eval_scalar_rmul(other)
            except TypeError:
                return NotImplemented

        elif T == "is_matrix":
            if self.shape[0] != other.shape[1]:
                raise ShapeError("Matrix size mismatch.")

            m = self._eval_matrix_rmul(other)

            if isimpbool:
                return m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m])

            return m

        else:
            return NotImplemented

    @call_highest_priority('__sub__')
    def __rsub__(self, a):
        return (-self) + a

    @call_highest_priority('__rsub__')
    def __sub__(self, a):
        return self + (-a)

    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        return _det_bareiss(self, iszerofunc=iszerofunc)

    def _eval_det_berkowitz(self):
        return _det_berkowitz(self)

    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    def _eval_det_bird(self):
        return _det_bird(self)

    def _eval_det_laplace(self):
        return _det_laplace(self)

    def _eval_determinant(self): # for expressions.determinant.Determinant
        return _det(self)

    def adjugate(self, method="berkowitz"):
        return _adjugate(self, method=method)

    def charpoly(self, x='lambda', simplify=_utilities_simplify):
        return _charpoly(self, x=x, simplify=simplify)

    def cofactor(self, i, j, method="berkowitz"):
        return _cofactor(self, i, j, method=method)

    def cofactor_matrix(self, method="berkowitz"):
        return _cofactor_matrix(self, method=method)

    def det(self, method="bareiss", iszerofunc=None):
        return _det(self, method=method, iszerofunc=iszerofunc)

    def per(self):
        return _per(self)

    def minor(self, i, j, method="berkowitz"):
        return _minor(self, i, j, method=method)

    def minor_submatrix(self, i, j):
        return _minor_submatrix(self, i, j)

    _find_reasonable_pivot.__doc__       = _find_reasonable_pivot.__doc__
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__            = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__          = _det_berkowitz.__doc__
    _eval_det_bird.__doc__            = _det_bird.__doc__
    _eval_det_laplace.__doc__            = _det_laplace.__doc__
    _eval_det_lu.__doc__                 = _det_LU.__doc__
    _eval_determinant.__doc__            = _det.__doc__
    adjugate.__doc__                     = _adjugate.__doc__
    charpoly.__doc__                     = _charpoly.__doc__
    cofactor.__doc__                     = _cofactor.__doc__
    cofactor_matrix.__doc__              = _cofactor_matrix.__doc__
    det.__doc__                          = _det.__doc__
    per.__doc__                          = _per.__doc__
    minor.__doc__                        = _minor.__doc__
    minor_submatrix.__doc__              = _minor_submatrix.__doc__

    def echelon_form(self, iszerofunc=_iszero, simplify=False, with_pivots=False):
        return _echelon_form(self, iszerofunc=iszerofunc, simplify=simplify,
                with_pivots=with_pivots)

    @property
    def is_echelon(self):
        return _is_echelon(self)

    def rank(self, iszerofunc=_iszero, simplify=False):
        return _rank(self, iszerofunc=iszerofunc, simplify=simplify)

    def rref_rhs(self, rhs):
        """Return reduced row-echelon form of matrix, matrix showing
        rhs after reduction steps. ``rhs`` must have the same number
        of rows as ``self``.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> r1, r2 = symbols('r1 r2')
        >>> Matrix([[1, 1], [2, 1]]).rref_rhs(Matrix([r1, r2]))
        (Matrix([
        [1, 0],
        [0, 1]]), Matrix([
        [ -r1 + r2],
        [2*r1 - r2]]))
        """
        r, _ = _rref(self.hstack(self, self.eye(self.rows), rhs))
        return r[:, :self.cols], r[:, -rhs.cols:]

    def rref(self, iszerofunc=_iszero, simplify=False, pivots=True,
            normalize_last=True):
        return _rref(self, iszerofunc=iszerofunc, simplify=simplify,
            pivots=pivots, normalize_last=normalize_last)

    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__   = _is_echelon.__doc__
    rank.__doc__         = _rank.__doc__
    rref.__doc__         = _rref.__doc__

    def _normalize_op_args(self, op, col, k, col1, col2, error_str="col"):
        """Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed."""
        if op not in ["n->kn", "n<->m", "n->n+km"]:
            raise ValueError("Unknown {} operation '{}'. Valid col operations "
                             "are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))

        # define self_col according to error_str
        self_cols = self.cols if error_str == 'col' else self.rows

        # normalize and validate the arguments
        if op == "n->kn":
            col = col if col is not None else col1
            if col is None or k is None:
                raise ValueError("For a {0} operation 'n->kn' you must provide the "
                                 "kwargs `{0}` and `k`".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))

        elif op == "n<->m":
            # we need two cols to swap. It does not matter
            # how they were specified, so gather them together and
            # remove `None`
            cols = {col, k, col1, col2}.difference([None])
            if len(cols) > 2:
                # maybe the user left `k` by mistake?
                cols = {col, col1, col2}.difference([None])
            if len(cols) != 2:
                raise ValueError("For a {0} operation 'n<->m' you must provide the "
                                 "kwargs `{0}1` and `{0}2`".format(error_str))
            col1, col2 = cols
            if not 0 <= col1 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        elif op == "n->n+km":
            col = col1 if col is None else col
            col2 = col1 if col2 is None else col2
            if col is None or col2 is None or k is None:
                raise ValueError("For a {0} operation 'n->n+km' you must provide the "
                                 "kwargs `{0}`, `k`, and `{0}2`".format(error_str))
            if col == col2:
                raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must "
                                 "be different.".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))

        else:
            raise ValueError('invalid operation %s' % repr(op))

        return op, col, k, col1, col2

    def _eval_col_op_multiply_col_by_const(self, col, k):
        def entry(i, j):
            if j == col:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_swap(self, col1, col2):
        def entry(i, j):
            if j == col1:
                return self[i, col2]
            elif j == col2:
                return self[i, col1]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2):
        def entry(i, j):
            if j == col:
                return self[i, j] + k * self[i, col2]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_swap(self, row1, row2):
        def entry(i, j):
            if i == row1:
                return self[row2, j]
            elif i == row2:
                return self[row1, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_multiply_row_by_const(self, row, k):
        def entry(i, j):
            if i == row:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2):
        def entry(i, j):
            if i == row:
                return self[i, j] + k * self[row2, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def elementary_col_op(self, op="n->kn", col=None, k=None, col1=None, col2=None):
        """Performs the elementary column operation `op`.

        `op` may be one of

            * ``"n->kn"`` (column n goes to k*n)
            * ``"n<->m"`` (swap column n and column m)
            * ``"n->n+km"`` (column n goes to column n + k*column m)

        Parameters
        ==========

        op : string; the elementary row operation
        col : the column to apply the column operation
        k : the multiple to apply in the column operation
        col1 : one column of a column swap
        col2 : second column of a column swap or column "m" in the column operation
               "n->n+km"
        """

        op, col, k, col1, col2 = self._normalize_op_args(op, col, k, col1, col2, "col")

        # now that we've validated, we're all good to dispatch
        if op == "n->kn":
            return self._eval_col_op_multiply_col_by_const(col, k)
        if op == "n<->m":
            return self._eval_col_op_swap(col1, col2)
        if op == "n->n+km":
            return self._eval_col_op_add_multiple_to_other_col(col, k, col2)

    def elementary_row_op(self, op="n->kn", row=None, k=None, row1=None, row2=None):
        """Performs the elementary row operation `op`.

        `op` may be one of

            * ``"n->kn"`` (row n goes to k*n)
            * ``"n<->m"`` (swap row n and row m)
            * ``"n->n+km"`` (row n goes to row n + k*row m)

        Parameters
        ==========

        op : string; the elementary row operation
        row : the row to apply the row operation
        k : the multiple to apply in the row operation
        row1 : one row of a row swap
        row2 : second row of a row swap or row "m" in the row operation
               "n->n+km"
        """

        op, row, k, row1, row2 = self._normalize_op_args(op, row, k, row1, row2, "row")

        # now that we've validated, we're all good to dispatch
        if op == "n->kn":
            return self._eval_row_op_multiply_row_by_const(row, k)
        if op == "n<->m":
            return self._eval_row_op_swap(row1, row2)
        if op == "n->n+km":
            return self._eval_row_op_add_multiple_to_other_row(row, k, row2)

    def columnspace(self, simplify=False):
        return _columnspace(self, simplify=simplify)

    def nullspace(self, simplify=False, iszerofunc=_iszero):
        return _nullspace(self, simplify=simplify, iszerofunc=iszerofunc)

    def rowspace(self, simplify=False):
        return _rowspace(self, simplify=simplify)

    # This is a classmethod but is converted to such later in order to allow
    # assignment of __doc__ since that does not work for already wrapped
    # classmethods in Python 3.6.
    def orthogonalize(cls, *vecs, **kwargs):
        return _orthogonalize(cls, *vecs, **kwargs)

    columnspace.__doc__   = _columnspace.__doc__
    nullspace.__doc__     = _nullspace.__doc__
    rowspace.__doc__      = _rowspace.__doc__
    orthogonalize.__doc__ = _orthogonalize.__doc__

    orthogonalize         = classmethod(orthogonalize)  # type:ignore

    def eigenvals(self, error_when_incomplete=True, **flags):
        return _eigenvals(self, error_when_incomplete=error_when_incomplete, **flags)

    def eigenvects(self, error_when_incomplete=True, iszerofunc=_iszero, **flags):
        return _eigenvects(self, error_when_incomplete=error_when_incomplete,
                iszerofunc=iszerofunc, **flags)

    def is_diagonalizable(self, reals_only=False, **kwargs):
        return _is_diagonalizable(self, reals_only=reals_only, **kwargs)

    def diagonalize(self, reals_only=False, sort=False, normalize=False):
        return _diagonalize(self, reals_only=reals_only, sort=sort,
                normalize=normalize)

    def bidiagonalize(self, upper=True):
        return _bidiagonalize(self, upper=upper)

    def bidiagonal_decomposition(self, upper=True):
        return _bidiagonal_decomposition(self, upper=upper)

    @property
    def is_positive_definite(self):
        return _is_positive_definite(self)

    @property
    def is_positive_semidefinite(self):
        return _is_positive_semidefinite(self)

    @property
    def is_negative_definite(self):
        return _is_negative_definite(self)

    @property
    def is_negative_semidefinite(self):
        return _is_negative_semidefinite(self)

    @property
    def is_indefinite(self):
        return _is_indefinite(self)

    def jordan_form(self, calc_transform=True, **kwargs):
        return _jordan_form(self, calc_transform=calc_transform, **kwargs)

    def left_eigenvects(self, **flags):
        return _left_eigenvects(self, **flags)

    def singular_values(self):
        return _singular_values(self)

    eigenvals.__doc__                  = _eigenvals.__doc__
    eigenvects.__doc__                 = _eigenvects.__doc__
    is_diagonalizable.__doc__          = _is_diagonalizable.__doc__
    diagonalize.__doc__                = _diagonalize.__doc__
    is_positive_definite.__doc__       = _is_positive_definite.__doc__
    is_positive_semidefinite.__doc__   = _is_positive_semidefinite.__doc__
    is_negative_definite.__doc__       = _is_negative_definite.__doc__
    is_negative_semidefinite.__doc__   = _is_negative_semidefinite.__doc__
    is_indefinite.__doc__              = _is_indefinite.__doc__
    jordan_form.__doc__                = _jordan_form.__doc__
    left_eigenvects.__doc__            = _left_eigenvects.__doc__
    singular_values.__doc__            = _singular_values.__doc__
    bidiagonalize.__doc__              = _bidiagonalize.__doc__
    bidiagonal_decomposition.__doc__   = _bidiagonal_decomposition.__doc__

    def diff(self, *args, evaluate=True, **kwargs):
        """Calculate the derivative of each element in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
        # XXX this should be handled here rather than in Derivative
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        deriv = ArrayDerivative(self, *args, evaluate=evaluate)
        # XXX This can rather changed to always return immutable matrix
        if not isinstance(self, Basic) and evaluate:
            return deriv.as_mutable()
        return deriv

    def _eval_derivative(self, arg):
        return self.applyfunc(lambda x: x.diff(arg))

    def integrate(self, *args, **kwargs):
        """Integrate each element of the matrix.  ``args`` will
        be passed to the ``integrate`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.integrate((x, ))
        Matrix([
        [x**2/2, x*y],
        [     x,   0]])
        >>> M.integrate((x, 0, 2))
        Matrix([
        [2, 2*y],
        [2,   0]])

        See Also
        ========

        limit
        diff
        """
        return self.applyfunc(lambda x: x.integrate(*args, **kwargs))

    def jacobian(self, X):
        """Calculates the Jacobian matrix (derivative of a vector-valued function).

        Parameters
        ==========

        ``self`` : vector of expressions representing functions f_i(x_1, ..., x_n).
        X : set of x_i's in order, it can be a list or a Matrix

        Both ``self`` and X can be a row or a column matrix in any order
        (i.e., jacobian() should always work).

        Examples
        ========

        >>> from sympy import sin, cos, Matrix
        >>> from sympy.abc import rho, phi
        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
        >>> Y = Matrix([rho, phi])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0]])
        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)]])

        See Also
        ========

        hessian
        wronskian
        """
        from sympy.matrices.matrixbase import MatrixBase
        if not isinstance(X, MatrixBase):
            X = self._new(X)
        # Both X and ``self`` can be a row or a column matrix, so we need to make
        # sure all valid combinations work, but everything else fails:
        if self.shape[0] == 1:
            m = self.shape[1]
        elif self.shape[1] == 1:
            m = self.shape[0]
        else:
            raise TypeError("``self`` must be a row or a column matrix")
        if X.shape[0] == 1:
            n = X.shape[1]
        elif X.shape[1] == 1:
            n = X.shape[0]
        else:
            raise TypeError("X must be a row or a column matrix")

        # m is the number of functions and n is the number of variables
        # computing the Jacobian is now easy:
        return self._new(m, n, lambda j, i: self[j].diff(X[i]))

    def limit(self, *args):
        """Calculate the limit of each element in the matrix.
        ``args`` will be passed to the ``limit`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.limit(x, 2)
        Matrix([
        [2, y],
        [1, 0]])

        See Also
        ========

        integrate
        diff
        """
        return self.applyfunc(lambda x: x.limit(*args))

    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_utilities_simplify):
        return self.charpoly(x=x)

    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        """
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method."""
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method."""
        sign, minors = self.one, []

        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign

        return tuple(minors)

    def berkowitz(self):
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk

        if not self.is_square:
            raise NonSquareMatrixError()

        A, N = self, self.rows
        transforms = [0] * (N - 1)

        for n in range(N, 1, -1):
            T, k = zeros(n + 1, n), n - 1

            R, C = -A[k, :k], A[:k, k]
            A, a = A[:k, :k], -A[k, k]

            items = [C]

            for i in range(0, n - 2):
                items.append(A * items[i])

            for i, B in enumerate(items):
                items[i] = (R * B)[0, 0]

            items = [self.one, a] + items

            for i in range(n):
                T[i:, i] = items[:n - i + 1]

            transforms[k - 1] = T

        polys = [self._new([self.one, -A[0, 0]])]

        for i, T in enumerate(transforms):
            polys.append(T * polys[i])

        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method="berkowitz"):
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        return _det_bareiss(self)

    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

        See Also
        ========


        det
        berkowitz_det
        """
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        return self.jordan_block(size=n, eigenvalue=eigenval)

    def jordan_cells(self, calc_transformation=True):
        P, J = self.jordan_form()
        return P, J.get_diag_blocks()

    def minorEntry(self, i, j, method="berkowitz"):
        return self.minor(i, j, method=method)

    def minorMatrix(self, i, j):
        return self.minor_submatrix(i, j)

    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
        return self.permute_rows(perm, direction='backward')

    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
        return self.permute_rows(perm, direction='forward')

    @property
    def kind(self) -> MatrixKind:
        elem_kinds = {e.kind for e in self.flat()}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return MatrixKind(elemkind)

    def flat(self):
        """
        Returns a flat list of all elements in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> m = Matrix([[0, 2], [3, 4]])
        >>> m.flat()
        [0, 2, 3, 4]

        See Also
        ========

        tolist
        values
        """
        return [self[i, j] for i in range(self.rows) for j in range(self.cols)]

    def __array__(self, dtype=object, copy=None):
        if copy is not None and not copy:
            raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
        from .dense import matrix2numpy
        return matrix2numpy(self, dtype=dtype)

    def __len__(self):
        """Return the number of elements of ``self``.

        Implemented mainly so bool(Matrix()) == False.
        """
        return self.rows * self.cols

    def _matrix_pow_by_jordan_blocks(self, num):
        from sympy.matrices import diag, MutableMatrix

        def jordan_cell_power(jc, n):
            N = jc.shape[0]
            l = jc[0,0]
            if l.is_zero:
                if N == 1 and n.is_nonnegative:
                    jc[0,0] = l**n
                elif not (n.is_integer and n.is_nonnegative):
                    raise NonInvertibleMatrixError("Non-invertible matrix can only be raised to a nonnegative integer")
                else:
                    for i in range(N):
                        jc[0,i] = KroneckerDelta(i, n)
            else:
                for i in range(N):
                    bn = binomial(n, i)
                    if isinstance(bn, binomial):
                        bn = bn._eval_expand_func()
                    jc[0,i] = l**(n-i)*bn
            for i in range(N):
                for j in range(1, N-i):
                    jc[j,i+j] = jc [j-1,i+j-1]

        P, J = self.jordan_form()
        jordan_cells = J.get_diag_blocks()
        # Make sure jordan_cells matrices are mutable:
        jordan_cells = [MutableMatrix(j) for j in jordan_cells]
        for j in jordan_cells:
            jordan_cell_power(j, num)
        return self._new(P.multiply(diag(*jordan_cells))
                .multiply(P.inv()))

    def __str__(self):
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        return "Matrix(%s)" % str(self.tolist())

    def _format_str(self, printer=None):
        if not printer:
            printer = StrPrinter()
        # Handle zero dimensions:
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        if self.rows == 1:
            return "Matrix([%s])" % self.table(printer, rowsep=',\n')
        return "Matrix([\n%s])" % self.table(printer, rowsep=',\n')

    @classmethod
    def irregular(cls, ntop, *matrices, **kwargs):
        """Return a matrix filled by the given matrices which
        are listed in order of appearance from left to right, top to
        bottom as they first appear in the matrix. They must fill the
        matrix completely.

        Examples
        ========

        >>> from sympy import ones, Matrix
        >>> Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,
        ...   ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7)
        Matrix([
            [1, 2, 2, 2, 3, 3],
            [1, 2, 2, 2, 3, 3],
            [4, 2, 2, 2, 5, 5],
            [6, 6, 7, 7, 5, 5]])
        """
        ntop = as_int(ntop)
        # make sure we are working with explicit matrices
        b = [i.as_explicit() if hasattr(i, 'as_explicit') else i
            for i in matrices]
        q = list(range(len(b)))
        dat = [i.rows for i in b]
        active = [q.pop(0) for _ in range(ntop)]
        cols = sum(b[i].cols for i in active)
        rows = []
        while any(dat):
            r = []
            for a, j in enumerate(active):
                r.extend(b[j][-dat[j], :])
                dat[j] -= 1
                if dat[j] == 0 and q:
                    active[a] = q.pop(0)
            if len(r) != cols:
                raise ValueError(filldedent('''
                    Matrices provided do not appear to fill
                    the space completely.'''))
            rows.append(r)
        return cls._new(rows)

    @classmethod
    def _handle_ndarray(cls, arg):
        # NumPy array or matrix or some other object that implements
        # __array__. So let's first use this method to get a
        # numpy.array() and then make a Python list out of it.
        arr = arg.__array__()
        if len(arr.shape) == 2:
            rows, cols = arr.shape[0], arr.shape[1]
            flat_list = [cls._sympify(i) for i in arr.ravel()]
            return rows, cols, flat_list
        elif len(arr.shape) == 1:
            flat_list = [cls._sympify(i) for i in arr]
            return arr.shape[0], 1, flat_list
        else:
            raise NotImplementedError(
                "SymPy supports just 1D and 2D matrices")

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        """Return the number of rows, cols and flat matrix elements.

        Examples
        ========

        >>> from sympy import Matrix, I

        Matrix can be constructed as follows:

        * from a nested list of iterables

        >>> Matrix( ((1, 2+I), (3, 4)) )
        Matrix([
        [1, 2 + I],
        [3,     4]])

        * from un-nested iterable (interpreted as a column)

        >>> Matrix( [1, 2] )
        Matrix([
        [1],
        [2]])

        * from un-nested iterable with dimensions

        >>> Matrix(1, 2, [1, 2] )
        Matrix([[1, 2]])

        * from no arguments (a 0 x 0 matrix)

        >>> Matrix()
        Matrix(0, 0, [])

        * from a rule

        >>> Matrix(2, 2, lambda i, j: i/(j + 1) )
        Matrix([
        [0,   0],
        [1, 1/2]])

        See Also
        ========
        irregular - filling a matrix with irregular blocks
        """
        from sympy.matrices import SparseMatrix
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.matrices.expressions.blockmatrix import BlockMatrix

        flat_list = None

        if len(args) == 1:
            # Matrix(SparseMatrix(...))
            if isinstance(args[0], SparseMatrix):
                return args[0].rows, args[0].cols, flatten(args[0].tolist())

            # Matrix(Matrix(...))
            elif isinstance(args[0], MatrixBase):
                return args[0].rows, args[0].cols, args[0].flat()

            # Matrix(MatrixSymbol('X', 2, 2))
            elif isinstance(args[0], Basic) and args[0].is_Matrix:
                return args[0].rows, args[0].cols, args[0].as_explicit().flat()

            elif isinstance(args[0], mp.matrix):
                M = args[0]
                flat_list = [cls._sympify(x) for x in M]
                return M.rows, M.cols, flat_list

            # Matrix(numpy.ones((2, 2)))
            elif hasattr(args[0], "__array__"):
                return cls._handle_ndarray(args[0])

            # Matrix([1, 2, 3]) or Matrix([[1, 2], [3, 4]])
            elif is_sequence(args[0]) \
                    and not isinstance(args[0], DeferredVector):
                dat = list(args[0])
                ismat = lambda i: isinstance(i, MatrixBase) and (
                    evaluate or isinstance(i, (BlockMatrix, MatrixSymbol)))
                raw = lambda i: is_sequence(i) and not ismat(i)
                evaluate = kwargs.get('evaluate', True)


                if evaluate:

                    def make_explicit(x):
                        """make Block and Symbol explicit"""
                        if isinstance(x, BlockMatrix):
                            return x.as_explicit()
                        elif isinstance(x, MatrixSymbol) and all(_.is_Integer for _ in x.shape):
                            return x.as_explicit()
                        else:
                            return x

                    def make_explicit_row(row):
                        # Could be list or could be list of lists
                        if isinstance(row, (list, tuple)):
                            return [make_explicit(x) for x in row]
                        else:
                            return make_explicit(row)

                    if isinstance(dat, (list, tuple)):
                        dat = [make_explicit_row(row) for row in dat]

                if len(dat) == 0:
                    rows = cols = 0
                    flat_list = []
                elif all(raw(i) for i in dat) and len(dat[0]) == 0:
                    if not all(len(i) == 0 for i in dat):
                        raise ValueError('mismatched dimensions')
                    rows = len(dat)
                    cols = 0
                    flat_list = []
                elif not any(raw(i) or ismat(i) for i in dat):
                    # a column as a list of values
                    flat_list = [cls._sympify(i) for i in dat]
                    rows = len(flat_list)
                    cols = 1 if rows else 0
                elif evaluate and all(ismat(i) for i in dat):
                    # a column as a list of matrices
                    ncol = {i.cols for i in dat if any(i.shape)}
                    if ncol:
                        if len(ncol) != 1:
                            raise ValueError('mismatched dimensions')
                        flat_list = [_ for i in dat for r in i.tolist() for _ in r]
                        cols = ncol.pop()
                        rows = len(flat_list)//cols
                    else:
                        rows = cols = 0
                        flat_list = []
                elif evaluate and any(ismat(i) for i in dat):
                    ncol = set()
                    flat_list = []
                    for i in dat:
                        if ismat(i):
                            flat_list.extend(
                                [k for j in i.tolist() for k in j])
                            if any(i.shape):
                                ncol.add(i.cols)
                        elif raw(i):
                            if i:
                                ncol.add(len(i))
                                flat_list.extend([cls._sympify(ij) for ij in i])
                        else:
                            ncol.add(1)
                            flat_list.append(i)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                    cols = ncol.pop()
                    rows = len(flat_list)//cols
                else:
                    # list of lists; each sublist is a logical row
                    # which might consist of many rows if the values in
                    # the row are matrices
                    flat_list = []
                    ncol = set()
                    rows = cols = 0
                    for row in dat:
                        if not is_sequence(row) and \
                                not getattr(row, 'is_Matrix', False):
                            raise ValueError('expecting list of lists')

                        if hasattr(row, '__array__'):
                            if 0 in row.shape:
                                continue

                        if evaluate and all(ismat(i) for i in row):
                            r, c, flatT = cls._handle_creation_inputs(
                                [i.T for i in row])
                            T = reshape(flatT, [c])
                            flat = \
                                [T[i][j] for j in range(c) for i in range(r)]
                            r, c = c, r
                        else:
                            r = 1
                            if getattr(row, 'is_Matrix', False):
                                c = 1
                                flat = [row]
                            else:
                                c = len(row)
                                flat = [cls._sympify(i) for i in row]
                        ncol.add(c)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                        flat_list.extend(flat)
                        rows += r
                    cols = ncol.pop() if ncol else 0

        elif len(args) == 3:
            rows = as_int(args[0])
            cols = as_int(args[1])

            if rows < 0 or cols < 0:
                raise ValueError("Cannot create a {} x {} matrix. "
                                 "Both dimensions must be positive".format(rows, cols))

            # Matrix(2, 2, lambda i, j: i+j)
            if len(args) == 3 and isinstance(args[2], Callable):
                op = args[2]
                flat_list = []
                for i in range(rows):
                    flat_list.extend(
                        [cls._sympify(op(cls._sympify(i), cls._sympify(j)))
                         for j in range(cols)])

            # Matrix(2, 2, [1, 2, 3, 4])
            elif len(args) == 3 and is_sequence(args[2]):
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError(
                        'List length should be equal to rows*columns')
                flat_list = [cls._sympify(i) for i in flat_list]


        # Matrix()
        elif len(args) == 0:
            # Empty Matrix
            rows = cols = 0
            flat_list = []

        if flat_list is None:
            raise TypeError(filldedent('''
                Data type not understood; expecting list of lists
                or lists of values.'''))

        return rows, cols, flat_list

    def _setitem(self, key, value):
        """Helper to set value at location given by key.

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        from .dense import Matrix

        is_slice = isinstance(key, slice)
        i, j = key = self.key2ij(key)
        is_mat = isinstance(value, MatrixBase)
        if isinstance(i, slice) or isinstance(j, slice):
            if is_mat:
                self.copyin_matrix(key, value)
                return
            if not isinstance(value, Expr) and is_sequence(value):
                self.copyin_list(key, value)
                return
            raise ValueError('unexpected value: %s' % value)
        else:
            if (not is_mat and
                    not isinstance(value, Basic) and is_sequence(value)):
                value = Matrix(value)
                is_mat = True
            if is_mat:
                if is_slice:
                    key = (slice(*divmod(i, self.cols)),
                           slice(*divmod(j, self.cols)))
                else:
                    key = (slice(i, i + value.rows),
                           slice(j, j + value.cols))
                self.copyin_matrix(key, value)
            else:
                return i, j, self._sympify(value)
            return

    def add(self, b):
        """Return self + b."""
        return self + b

    def condition_number(self):
        """Returns the condition number of a matrix.

        This is the maximum singular value divided by the minimum singular value

        Examples
        ========

        >>> from sympy import Matrix, S
        >>> A = Matrix([[1, 0, 0], [0, 10, 0], [0, 0, S.One/10]])
        >>> A.condition_number()
        100

        See Also
        ========

        singular_values
        """

        if not self:
            return self.zero
        singularvalues = self.singular_values()
        return Max(*singularvalues) / Min(*singularvalues)

    def copy(self):
        """
        Returns the copy of a matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> A = Matrix(2, 2, [1, 2, 3, 4])
        >>> A.copy()
        Matrix([
        [1, 2],
        [3, 4]])

        """
        return self._new(self.rows, self.cols, self.flat())

    def cross(self, b):
        r"""
        Return the cross product of ``self`` and ``b`` relaxing the condition
        of compatible dimensions: if each has 3 elements, a matrix of the
        same type and shape as ``self`` will be returned. If ``b`` has the same
        shape as ``self`` then common identities for the cross product (like
        `a \times b = - b \times a`) will hold.

        Parameters
        ==========
            b : 3x1 or 1x3 Matrix

        See Also
        ========

        dot
        hat
        vee
        multiply
        multiply_elementwise
        """
        from sympy.matrices.expressions.matexpr import MatrixExpr

        if not isinstance(b, (MatrixBase, MatrixExpr)):
            raise TypeError(
                "{} must be a Matrix, not {}.".format(b, type(b)))

        if not (self.rows * self.cols == b.rows * b.cols == 3):
            raise ShapeError("Dimensions incorrect for cross product: %s x %s" %
                             ((self.rows, self.cols), (b.rows, b.cols)))
        else:
            return self._new(self.rows, self.cols, (
                (self[1] * b[2] - self[2] * b[1]),
                (self[2] * b[0] - self[0] * b[2]),
                (self[0] * b[1] - self[1] * b[0])))

    def hat(self):
        r"""
        Return the skew-symmetric matrix representing the cross product,
        so that ``self.hat() * b`` is equivalent to  ``self.cross(b)``.

        Examples
        ========

        Calling ``hat`` creates a skew-symmetric 3x3 Matrix from a 3x1 Matrix:

        >>> from sympy import Matrix
        >>> a = Matrix([1, 2, 3])
        >>> a.hat()
        Matrix([
        [ 0, -3,  2],
        [ 3,  0, -1],
        [-2,  1,  0]])

        Multiplying it with another 3x1 Matrix calculates the cross product:

        >>> b = Matrix([3, 2, 1])
        >>> a.hat() * b
        Matrix([
        [-4],
        [ 8],
        [-4]])

        Which is equivalent to calling the ``cross`` method:

        >>> a.cross(b)
        Matrix([
        [-4],
        [ 8],
        [-4]])

        See Also
        ========

        dot
        cross
        vee
        multiply
        multiply_elementwise
        """

        if self.shape != (3, 1):
            raise ShapeError("Dimensions incorrect, expected (3, 1), got " +
                             str(self.shape))
        else:
            x, y, z = self
            return self._new(3, 3, (
                 0, -z,  y,
                 z,  0, -x,
                -y,  x,  0))

    def vee(self):
        r"""
        Return a 3x1 vector from a skew-symmetric matrix representing the cross product,
        so that ``self * b`` is equivalent to  ``self.vee().cross(b)``.

        Examples
        ========

        Calling ``vee`` creates a vector from a skew-symmetric Matrix:

        >>> from sympy import Matrix
        >>> A = Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        >>> a = A.vee()
        >>> a
        Matrix([
        [1],
        [2],
        [3]])

        Calculating the matrix product of the original matrix with a vector
        is equivalent to a cross product:

        >>> b = Matrix([3, 2, 1])
        >>> A * b
        Matrix([
        [-4],
        [ 8],
        [-4]])

        >>> a.cross(b)
        Matrix([
        [-4],
        [ 8],
        [-4]])

        ``vee`` can also be used to retrieve angular velocity expressions.
        Defining a rotation matrix:

        >>> from sympy import rot_ccw_axis3, trigsimp
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> theta = dynamicsymbols('theta')
        >>> R = rot_ccw_axis3(theta)
        >>> R
        Matrix([
        [cos(theta(t)), -sin(theta(t)), 0],
        [sin(theta(t)),  cos(theta(t)), 0],
        [            0,              0, 1]])

        We can retrieve the angular velocity:

        >>> Omega = R.T * R.diff()
        >>> Omega = trigsimp(Omega)
        >>> Omega.vee()
        Matrix([
        [                      0],
        [                      0],
        [Derivative(theta(t), t)]])

        See Also
        ========

        dot
        cross
        hat
        multiply
        multiply_elementwise
        """

        if self.shape != (3, 3):
            raise ShapeError("Dimensions incorrect, expected (3, 3), got " +
                             str(self.shape))
        elif not self.is_anti_symmetric():
            raise ValueError("Matrix is not skew-symmetric")
        else:
            return self._new(3, 1, (
                 self[2, 1],
                 self[0, 2],
                 self[1, 0]))

    @property
    def D(self):
        """Return Dirac conjugate (if ``self.rows == 4``).

        Examples
        ========

        >>> from sympy import Matrix, I, eye
        >>> m = Matrix((0, 1 + I, 2, 3))
        >>> m.D
        Matrix([[0, 1 - I, -2, -3]])
        >>> m = (eye(4) + I*eye(4))
        >>> m[0, 3] = 2
        >>> m.D
        Matrix([
        [1 - I,     0,      0,      0],
        [    0, 1 - I,      0,      0],
        [    0,     0, -1 + I,      0],
        [    2,     0,      0, -1 + I]])

        If the matrix does not have 4 rows an AttributeError will be raised
        because this property is only defined for matrices with 4 rows.

        >>> Matrix(eye(2)).D
        Traceback (most recent call last):
        ...
        AttributeError: Matrix has no attribute D.

        See Also
        ========

        sympy.matrices.matrixbase.MatrixBase.conjugate: By-element conjugation
        sympy.matrices.matrixbase.MatrixBase.H: Hermite conjugation
        """
        from sympy.physics.matrices import mgamma
        if self.rows != 4:
            # In Python 3.2, properties can only return an AttributeError
            # so we can't raise a ShapeError -- see commit which added the
            # first line of this inline comment. Also, there is no need
            # for a message since MatrixBase will raise the AttributeError
            raise AttributeError
        return self.H * mgamma(0)

    def dot(self, b, hermitian=None, conjugate_convention=None):
        """Return the dot or inner product of two vectors of equal length.
        Here ``self`` must be a ``Matrix`` of size 1 x n or n x 1, and ``b``
        must be either a matrix of size 1 x n, n x 1, or a list/tuple of length n.
        A scalar is returned.

        By default, ``dot`` does not conjugate ``self`` or ``b``, even if there are
        complex entries. Set ``hermitian=True`` (and optionally a ``conjugate_convention``)
        to compute the hermitian inner product.

        Possible kwargs are ``hermitian`` and ``conjugate_convention``.

        If ``conjugate_convention`` is ``"left"``, ``"math"`` or ``"maths"``,
        the conjugate of the first vector (``self``) is used.  If ``"right"``
        or ``"physics"`` is specified, the conjugate of the second vector ``b`` is used.

        Examples
        ========

        >>> from sympy import Matrix
        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> v = Matrix([1, 1, 1])
        >>> M.row(0).dot(v)
        6
        >>> M.col(0).dot(v)
        12
        >>> v = [3, 2, 1]
        >>> M.row(0).dot(v)
        10

        >>> from sympy import I
        >>> q = Matrix([1*I, 1*I, 1*I])
        >>> q.dot(q, hermitian=False)
        -3

        >>> q.dot(q, hermitian=True)
        3

        >>> q1 = Matrix([1, 1, 1*I])
        >>> q.dot(q1, hermitian=True, conjugate_convention="maths")
        1 - 2*I
        >>> q.dot(q1, hermitian=True, conjugate_convention="physics")
        1 + 2*I


        See Also
        ========

        cross
        multiply
        multiply_elementwise
        """
        from .dense import Matrix

        if not isinstance(b, MatrixBase):
            if is_sequence(b):
                if len(b) != self.cols and len(b) != self.rows:
                    raise ShapeError(
                        "Dimensions incorrect for dot product: %s, %s" % (
                            self.shape, len(b)))
                return self.dot(Matrix(b))
            else:
                raise TypeError(
                    "`b` must be an ordered iterable or Matrix, not %s." %
                    type(b))

        if (1 not in self.shape) or (1 not in b.shape):
            raise ShapeError
        if len(self) != len(b):
            raise ShapeError(
                "Dimensions incorrect for dot product: %s, %s" % (self.shape, b.shape))

        mat = self
        n = len(mat)
        if mat.shape != (1, n):
            mat = mat.reshape(1, n)
        if b.shape != (n, 1):
            b = b.reshape(n, 1)

        # Now ``mat`` is a row vector and ``b`` is a column vector.

        # If it so happens that only conjugate_convention is passed
        # then automatically set hermitian to True. If only hermitian
        # is true but no conjugate_convention is not passed then
        # automatically set it to ``"maths"``

        if conjugate_convention is not None and hermitian is None:
            hermitian = True
        if hermitian and conjugate_convention is None:
            conjugate_convention = "maths"

        if hermitian == True:
            if conjugate_convention in ("maths", "left", "math"):
                mat = mat.conjugate()
            elif conjugate_convention in ("physics", "right"):
                b = b.conjugate()
            else:
                raise ValueError("Unknown conjugate_convention was entered."
                                 " conjugate_convention must be one of the"
                                 " following: math, maths, left, physics or right.")
        return (mat * b)[0]

    def dual(self):
        """Returns the dual of a matrix.

        A dual of a matrix is:

        ``(1/2)*levicivita(i, j, k, l)*M(k, l)`` summed over indices `k` and `l`

        Since the levicivita method is anti_symmetric for any pairwise
        exchange of indices, the dual of a symmetric matrix is the zero
        matrix. Strictly speaking the dual defined here assumes that the
        'matrix' `M` is a contravariant anti_symmetric second rank tensor,
        so that the dual is a covariant second rank tensor.

        """
        from sympy.matrices import zeros

        M, n = self[:, :], self.rows
        work = zeros(n)
        if self.is_symmetric():
            return work

        for i in range(1, n):
            for j in range(1, n):
                acum = 0
                for k in range(1, n):
                    acum += LeviCivita(i, j, 0, k) * M[0, k]
                work[i, j] = acum
                work[j, i] = -acum

        for l in range(1, n):
            acum = 0
            for a in range(1, n):
                for b in range(1, n):
                    acum += LeviCivita(0, l, a, b) * M[a, b]
            acum /= 2
            work[0, l] = -acum
            work[l, 0] = acum

        return work

    def _eval_matrix_exp_jblock(self):
        """A helper function to compute an exponential of a Jordan block
        matrix

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([[exp(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_exp_jblock()
        Matrix([
        [exp(lamda), exp(lamda), exp(lamda)/2],
        [         0, exp(lamda),   exp(lamda)],
        [         0,          0,   exp(lamda)]])

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Matrix_function#Jordan_decomposition
        """
        size = self.rows
        l = self[0, 0]
        exp_l = exp(l)

        bands = {i: exp_l / factorial(i) for i in range(size)}

        from .sparsetools import banded
        return self.__class__(banded(size, bands))


    def analytic_func(self, f, x):
        """
        Computes f(A) where A is a Square Matrix
        and f is an analytic function.

        Examples
        ========

        >>> from sympy import Symbol, Matrix, S, log

        >>> x = Symbol('x')
        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> f = log(x)
        >>> m.analytic_func(f, x)
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Parameters
        ==========

        f : Expr
            Analytic Function
        x : Symbol
            parameter of f

        """

        f, x = _sympify(f), _sympify(x)
        if not self.is_square:
            raise NonSquareMatrixError
        if not x.is_symbol:
            raise ValueError("{} must be a symbol.".format(x))
        if x not in f.free_symbols:
            raise ValueError(
                "{} must be a parameter of {}.".format(x, f))
        if x in self.free_symbols:
            raise ValueError(
                "{} must not be a parameter of {}.".format(x, self))

        eigen = self.eigenvals()
        max_mul = max(eigen.values())
        derivative = {}
        dd = f
        for i in range(max_mul - 1):
            dd = diff(dd, x)
            derivative[i + 1] = dd
        n = self.shape[0]
        r = self.zeros(n)
        f_val = self.zeros(n, 1)
        row = 0

        for i in eigen:
            mul = eigen[i]
            f_val[row] = f.subs(x, i)
            if f_val[row].is_number and not f_val[row].is_complex:
                raise ValueError(
                    "Cannot evaluate the function because the "
                    "function {} is not analytic at the given "
                    "eigenvalue {}".format(f, f_val[row]))
            val = 1
            for a in range(n):
                r[row, a] = val
                val *= i
            if mul > 1:
                coe = [1 for ii in range(n)]
                deri = 1
                while mul > 1:
                    row = row + 1
                    mul -= 1
                    d_i = derivative[deri].subs(x, i)
                    if d_i.is_number and not d_i.is_complex:
                        raise ValueError(
                            "Cannot evaluate the function because the "
                            "derivative {} is not analytic at the given "
                            "eigenvalue {}".format(derivative[deri], d_i))
                    f_val[row] = d_i
                    for a in range(n):
                        if a - deri + 1 <= 0:
                            r[row, a] = 0
                            coe[a] = 0
                            continue
                        coe[a] = coe[a]*(a - deri + 1)
                        r[row, a] = coe[a]*pow(i, a - deri)
                    deri += 1
            row += 1
        c = r.solve(f_val)
        ans = self.zeros(n)
        pre = self.eye(n)
        for i in range(n):
            ans = ans + c[i]*pre
            pre *= self
        return ans


    def exp(self):
        """Return the exponential of a square matrix.

        Examples
        ========

        >>> from sympy import Symbol, Matrix

        >>> t = Symbol('t')
        >>> m = Matrix([[0, 1], [-1, 0]]) * t
        >>> m.exp()
        Matrix([
        [    exp(I*t)/2 + exp(-I*t)/2, -I*exp(I*t)/2 + I*exp(-I*t)/2],
        [I*exp(I*t)/2 - I*exp(-I*t)/2,      exp(I*t)/2 + exp(-I*t)/2]])
        """
        if not self.is_square:
            raise NonSquareMatrixError(
                "Exponentiation is valid only for square matrices")
        try:
            P, J = self.jordan_form()
            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError(
                "Exponentiation is implemented only for matrices for which the Jordan normal form can be computed")

        blocks = [cell._eval_matrix_exp_jblock() for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)
        # n = self.rows
        ret = P.multiply(eJ, dotprodsimp=None).multiply(P.inv(), dotprodsimp=None)
        if all(value.is_real for value in self.values()):
            return type(self)(re(ret))
        else:
            return type(self)(ret)

    def _eval_matrix_log_jblock(self):
        """Helper function to compute logarithm of a jordan block.

        Examples
        ========

        >>> from sympy import Symbol, Matrix
        >>> l = Symbol('lamda')

        A trivial example of 1*1 Jordan block:

        >>> m = Matrix.jordan_block(1, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([[log(lamda)]])

        An example of 3*3 Jordan block:

        >>> m = Matrix.jordan_block(3, l)
        >>> m._eval_matrix_log_jblock()
        Matrix([
        [log(lamda),    1/lamda, -1/(2*lamda**2)],
        [         0, log(lamda),         1/lamda],
        [         0,          0,      log(lamda)]])
        """
        size = self.rows
        l = self[0, 0]

        if l.is_zero:
            raise MatrixError(
                'Could not take logarithm or reciprocal for the given '
                'eigenvalue {}'.format(l))

        bands = {0: log(l)}
        for i in range(1, size):
            bands[i] = -((-l) ** -i) / i

        from .sparsetools import banded
        return self.__class__(banded(size, bands))

    def log(self, simplify=cancel):
        """Return the logarithm of a square matrix.

        Parameters
        ==========

        simplify : function, bool
            The function to simplify the result with.

            Default is ``cancel``, which is effective to reduce the
            expression growing for taking reciprocals and inverses for
            symbolic matrices.

        Examples
        ========

        >>> from sympy import S, Matrix

        Examples for positive-definite matrices:

        >>> m = Matrix([[1, 1], [0, 1]])
        >>> m.log()
        Matrix([
        [0, 1],
        [0, 0]])

        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])
        >>> m.log()
        Matrix([
        [     0, log(2)],
        [log(2),      0]])

        Examples for non positive-definite matrices:

        >>> m = Matrix([[S(3)/4, S(5)/4], [S(5)/4, S(3)/4]])
        >>> m.log()
        Matrix([
        [         I*pi/2, log(2) - I*pi/2],
        [log(2) - I*pi/2,          I*pi/2]])

        >>> m = Matrix(
        ...     [[0, 0, 0, 1],
        ...      [0, 0, 1, 0],
        ...      [0, 1, 0, 0],
        ...      [1, 0, 0, 0]])
        >>> m.log()
        Matrix([
        [ I*pi/2,       0,       0, -I*pi/2],
        [      0,  I*pi/2, -I*pi/2,       0],
        [      0, -I*pi/2,  I*pi/2,       0],
        [-I*pi/2,       0,       0,  I*pi/2]])
        """
        if not self.is_square:
            raise NonSquareMatrixError(
                "Logarithm is valid only for square matrices")

        try:
            if simplify:
                P, J = simplify(self).jordan_form()
            else:
                P, J = self.jordan_form()

            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError(
                "Logarithm is implemented only for matrices for which "
                "the Jordan normal form can be computed")

        blocks = [
            cell._eval_matrix_log_jblock()
            for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)

        if simplify:
            ret = simplify(P * eJ * simplify(P.inv()))
            ret = self.__class__(ret)
        else:
            ret = P * eJ * P.inv()

        return ret

    def is_nilpotent(self):
        """Checks if a matrix is nilpotent.

        A matrix B is nilpotent if for some integer k, B**k is
        a zero matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> a = Matrix([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        True

        >>> a = Matrix([[1, 0, 1], [1, 0, 0], [1, 1, 0]])
        >>> a.is_nilpotent()
        False
        """
        if not self:
            return True
        if not self.is_square:
            raise NonSquareMatrixError(
                "Nilpotency is valid only for square matrices")
        x = uniquely_named_symbol('x', self, modify=lambda s: '_' + s)
        p = self.charpoly(x)
        if p.args[0] == x ** self.rows:
            return True
        return False

    def key2bounds(self, keys):
        """Converts a key with potentially mixed types of keys (integer and slice)
        into a tuple of ranges and raises an error if any index is out of ``self``'s
        range.

        See Also
        ========

        key2ij
        """
        islice, jslice = [isinstance(k, slice) for k in keys]
        if islice:
            if not self.rows:
                rlo = rhi = 0
            else:
                rlo, rhi = keys[0].indices(self.rows)[:2]
        else:
            rlo = a2idx(keys[0], self.rows)
            rhi = rlo + 1
        if jslice:
            if not self.cols:
                clo = chi = 0
            else:
                clo, chi = keys[1].indices(self.cols)[:2]
        else:
            clo = a2idx(keys[1], self.cols)
            chi = clo + 1
        return rlo, rhi, clo, chi

    def key2ij(self, key):
        """Converts key into canonical form, converting integers or indexable
        items into valid integers for ``self``'s range or returning slices
        unchanged.

        See Also
        ========

        key2bounds
        """
        if is_sequence(key):
            if not len(key) == 2:
                raise TypeError('key must be a sequence of length 2')
            return [a2idx(i, n) if not isinstance(i, slice) else i
                    for i, n in zip(key, self.shape)]
        elif isinstance(key, slice):
            return key.indices(len(self))[:2]
        else:
            return divmod(a2idx(key, len(self)), self.cols)

    def normalized(self, iszerofunc=_iszero):
        """Return the normalized version of ``self``.

        Parameters
        ==========

        iszerofunc : Function, optional
            A function to determine whether ``self`` is a zero vector.
            The default ``_iszero`` tests to see if each element is
            exactly zero.

        Returns
        =======

        Matrix
            Normalized vector form of ``self``.
            It has the same length as a unit vector. However, a zero vector
            will be returned for a vector with norm 0.

        Raises
        ======

        ShapeError
            If the matrix is not in a vector form.

        See Also
        ========

        norm
        """
        if self.rows != 1 and self.cols != 1:
            raise ShapeError("A Matrix must be a vector to normalize.")
        norm = self.norm()
        if iszerofunc(norm):
            out = self.zeros(self.rows, self.cols)
        else:
            out = self.applyfunc(lambda i: i / norm)
        return out

    def norm(self, ord=None):
        """Return the Norm of a Matrix or Vector.

        In the simplest case this is the geometric size of the vector
        Other norms can be specified by the ord parameter


        =====  ============================  ==========================
        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                - does not exist
        inf    maximum row sum               max(abs(x))
        -inf   --                            min(abs(x))
        1      maximum column sum            as below
        -1     --                            as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  - does not exist              sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

        Examples
        ========

        >>> from sympy import Matrix, Symbol, trigsimp, cos, sin, oo
        >>> x = Symbol('x', real=True)
        >>> v = Matrix([cos(x), sin(x)])
        >>> trigsimp( v.norm() )
        1
        >>> v.norm(10)
        (sin(x)**10 + cos(x)**10)**(1/10)
        >>> A = Matrix([[1, 1], [1, 1]])
        >>> A.norm(1) # maximum sum of absolute values of A is 2
        2
        >>> A.norm(2) # Spectral norm (max of |Ax|/|x| under 2-vector-norm)
        2
        >>> A.norm(-2) # Inverse spectral norm (smallest singular value)
        0
        >>> A.norm() # Frobenius Norm
        2
        >>> A.norm(oo) # Infinity Norm
        2
        >>> Matrix([1, -2]).norm(oo)
        2
        >>> Matrix([-1, 2]).norm(-oo)
        1

        See Also
        ========

        normalized
        """
        # Row or Column Vector Norms
        vals = list(self.values()) or [0]
        if S.One in self.shape:
            if ord in (2, None):  # Common case sqrt(<x, x>)
                return sqrt(Add(*(abs(i) ** 2 for i in vals)))

            elif ord == 1:  # sum(abs(x))
                return Add(*(abs(i) for i in vals))

            elif ord is S.Infinity:  # max(abs(x))
                return Max(*[abs(i) for i in vals])

            elif ord is S.NegativeInfinity:  # min(abs(x))
                return Min(*[abs(i) for i in vals])

            # Otherwise generalize the 2-norm, Sum(x_i**ord)**(1/ord)
            # Note that while useful this is not mathematically a norm
            try:
                return Pow(Add(*(abs(i) ** ord for i in vals)), S.One / ord)
            except (NotImplementedError, TypeError):
                raise ValueError("Expected order to be Number, Symbol, oo")

        # Matrix Norms
        else:
            if ord == 1:  # Maximum column sum
                m = self.applyfunc(abs)
                return Max(*[sum(m.col(i)) for i in range(m.cols)])

            elif ord == 2:  # Spectral Norm
                # Maximum singular value
                return Max(*self.singular_values())

            elif ord == -2:
                # Minimum singular value
                return Min(*self.singular_values())

            elif ord is S.Infinity:   # Infinity Norm - Maximum row sum
                m = self.applyfunc(abs)
                return Max(*[sum(m.row(i)) for i in range(m.rows)])

            elif (ord is None or isinstance(ord,
                                            str) and ord.lower() in
                ['f', 'fro', 'frobenius', 'vector']):
                # Reshape as vector and send back to norm function
                return self.vec().norm(ord=2)

            else:
                raise NotImplementedError("Matrix Norms under development")

    def print_nonzero(self, symb="X"):
        """Shows location of non-zero entries for fast shape lookup.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> m = Matrix(2, 3, lambda i, j: i*3+j)
        >>> m
        Matrix([
        [0, 1, 2],
        [3, 4, 5]])
        >>> m.print_nonzero()
        [ XX]
        [XXX]
        >>> m = eye(4)
        >>> m.print_nonzero("x")
        [x   ]
        [ x  ]
        [  x ]
        [   x]

        """
        s = []
        for i in range(self.rows):
            line = []
            for j in range(self.cols):
                if self[i, j] == 0:
                    line.append(" ")
                else:
                    line.append(str(symb))
            s.append("[%s]" % ''.join(line))
        print('\n'.join(s))

    def project(self, v):
        """Return the projection of ``self`` onto the line containing ``v``.

        Examples
        ========

        >>> from sympy import Matrix, S, sqrt
        >>> V = Matrix([sqrt(3)/2, S.Half])
        >>> x = Matrix([[1, 0]])
        >>> V.project(x)
        Matrix([[sqrt(3)/2, 0]])
        >>> V.project(-x)
        Matrix([[sqrt(3)/2, 0]])
        """
        return v * (self.dot(v) / v.dot(v))

    def table(self, printer, rowstart='[', rowend=']', rowsep='\n',
              colsep=', ', align='right'):
        r"""
        String form of Matrix as a table.

        ``printer`` is the printer to use for on the elements (generally
        something like StrPrinter())

        ``rowstart`` is the string used to start each row (by default '[').

        ``rowend`` is the string used to end each row (by default ']').

        ``rowsep`` is the string used to separate rows (by default a newline).

        ``colsep`` is the string used to separate columns (by default ', ').

        ``align`` defines how the elements are aligned. Must be one of 'left',
        'right', or 'center'.  You can also use '<', '>', and '^' to mean the
        same thing, respectively.

        This is used by the string printer for Matrix.

        Examples
        ========

        >>> from sympy import Matrix, StrPrinter
        >>> M = Matrix([[1, 2], [-33, 4]])
        >>> printer = StrPrinter()
        >>> M.table(printer)
        '[  1, 2]\n[-33, 4]'
        >>> print(M.table(printer))
        [  1, 2]
        [-33, 4]
        >>> print(M.table(printer, rowsep=',\n'))
        [  1, 2],
        [-33, 4]
        >>> print('[%s]' % M.table(printer, rowsep=',\n'))
        [[  1, 2],
        [-33, 4]]
        >>> print(M.table(printer, colsep=' '))
        [  1 2]
        [-33 4]
        >>> print(M.table(printer, align='center'))
        [ 1 , 2]
        [-33, 4]
        >>> print(M.table(printer, rowstart='{', rowend='}'))
        {  1, 2}
        {-33, 4}
        """
        # Handle zero dimensions:
        if S.Zero in self.shape:
            return '[]'
        # Build table of string representations of the elements
        res = []
        # Track per-column max lengths for pretty alignment
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                s = printer._print(self[i, j])
                res[-1].append(s)
                maxlen[j] = max(len(s), maxlen[j])
        # Patch strings together
        align = {
            'left': 'ljust',
            'right': 'rjust',
            'center': 'center',
            '<': 'ljust',
            '>': 'rjust',
            '^': 'center',
        }[align]
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                row[j] = getattr(elem, align)(maxlen[j])
            res[i] = rowstart + colsep.join(row) + rowend
        return rowsep.join(res)

    def rank_decomposition(self, iszerofunc=_iszero, simplify=False):
        return _rank_decomposition(self, iszerofunc=iszerofunc,
                simplify=simplify)

    def cholesky(self, hermitian=True):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LDLdecomposition(self, hermitian=True):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LUdecomposition(self, iszerofunc=_iszero, simpfunc=None,
            rankcheck=False):
        return _LUdecomposition(self, iszerofunc=iszerofunc, simpfunc=simpfunc,
                rankcheck=rankcheck)

    def LUdecomposition_Simple(self, iszerofunc=_iszero, simpfunc=None,
            rankcheck=False):
        return _LUdecomposition_Simple(self, iszerofunc=iszerofunc,
                simpfunc=simpfunc, rankcheck=rankcheck)

    def LUdecompositionFF(self):
        return _LUdecompositionFF(self)

    def singular_value_decomposition(self):
        return _singular_value_decomposition(self)

    def QRdecomposition(self):
        return _QRdecomposition(self)

    def upper_hessenberg_decomposition(self):
        return _upper_hessenberg_decomposition(self)

    def diagonal_solve(self, rhs):
        return _diagonal_solve(self, rhs)

    def lower_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def upper_triangular_solve(self, rhs):
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def cholesky_solve(self, rhs):
        return _cholesky_solve(self, rhs)

    def LDLsolve(self, rhs):
        return _LDLsolve(self, rhs)

    def LUsolve(self, rhs, iszerofunc=_iszero):
        return _LUsolve(self, rhs, iszerofunc=iszerofunc)

    def QRsolve(self, b):
        return _QRsolve(self, b)

    def gauss_jordan_solve(self, B, freevar=False):
        return _gauss_jordan_solve(self, B, freevar=freevar)

    def pinv_solve(self, B, arbitrary_matrix=None):
        return _pinv_solve(self, B, arbitrary_matrix=arbitrary_matrix)

    def cramer_solve(self, rhs, det_method="laplace"):
        return _cramer_solve(self, rhs, det_method=det_method)

    def solve(self, rhs, method='GJ'):
        return _solve(self, rhs, method=method)

    def solve_least_squares(self, rhs, method='CH'):
        return _solve_least_squares(self, rhs, method=method)

    def pinv(self, method='RD'):
        return _pinv(self, method=method)

    def inverse_ADJ(self, iszerofunc=_iszero):
        return _inv_ADJ(self, iszerofunc=iszerofunc)

    def inverse_BLOCK(self, iszerofunc=_iszero):
        return _inv_block(self, iszerofunc=iszerofunc)

    def inverse_GE(self, iszerofunc=_iszero):
        return _inv_GE(self, iszerofunc=iszerofunc)

    def inverse_LU(self, iszerofunc=_iszero):
        return _inv_LU(self, iszerofunc=iszerofunc)

    def inverse_CH(self, iszerofunc=_iszero):
        return _inv_CH(self, iszerofunc=iszerofunc)

    def inverse_LDL(self, iszerofunc=_iszero):
        return _inv_LDL(self, iszerofunc=iszerofunc)

    def inverse_QR(self, iszerofunc=_iszero):
        return _inv_QR(self, iszerofunc=iszerofunc)

    def inv(self, method=None, iszerofunc=_iszero, try_block_diag=False):
        return _inv(self, method=method, iszerofunc=iszerofunc,
                try_block_diag=try_block_diag)

    def connected_components(self):
        return _connected_components(self)

    def connected_components_decomposition(self):
        return _connected_components_decomposition(self)

    def strongly_connected_components(self):
        return _strongly_connected_components(self)

    def strongly_connected_components_decomposition(self, lower=True):
        return _strongly_connected_components_decomposition(self, lower=lower)

    _sage_ = Basic._sage_

    rank_decomposition.__doc__     = _rank_decomposition.__doc__
    cholesky.__doc__               = _cholesky.__doc__
    LDLdecomposition.__doc__       = _LDLdecomposition.__doc__
    LUdecomposition.__doc__        = _LUdecomposition.__doc__
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    LUdecompositionFF.__doc__      = _LUdecompositionFF.__doc__
    singular_value_decomposition.__doc__ = _singular_value_decomposition.__doc__
    QRdecomposition.__doc__        = _QRdecomposition.__doc__
    upper_hessenberg_decomposition.__doc__ = _upper_hessenberg_decomposition.__doc__

    diagonal_solve.__doc__         = _diagonal_solve.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    cholesky_solve.__doc__         = _cholesky_solve.__doc__
    LDLsolve.__doc__               = _LDLsolve.__doc__
    LUsolve.__doc__                = _LUsolve.__doc__
    QRsolve.__doc__                = _QRsolve.__doc__
    gauss_jordan_solve.__doc__     = _gauss_jordan_solve.__doc__
    pinv_solve.__doc__             = _pinv_solve.__doc__
    cramer_solve.__doc__           = _cramer_solve.__doc__
    solve.__doc__                  = _solve.__doc__
    solve_least_squares.__doc__    = _solve_least_squares.__doc__

    pinv.__doc__                   = _pinv.__doc__
    inverse_ADJ.__doc__            = _inv_ADJ.__doc__
    inverse_GE.__doc__             = _inv_GE.__doc__
    inverse_LU.__doc__             = _inv_LU.__doc__
    inverse_CH.__doc__             = _inv_CH.__doc__
    inverse_LDL.__doc__            = _inv_LDL.__doc__
    inverse_QR.__doc__             = _inv_QR.__doc__
    inverse_BLOCK.__doc__          = _inv_block.__doc__
    inv.__doc__                    = _inv.__doc__

    connected_components.__doc__   = _connected_components.__doc__
    connected_components_decomposition.__doc__ = \
        _connected_components_decomposition.__doc__
    strongly_connected_components.__doc__   = \
        _strongly_connected_components.__doc__
    strongly_connected_components_decomposition.__doc__ = \
        _strongly_connected_components_decomposition.__doc__


def _convert_matrix(typ, mat):
    """Convert mat to a Matrix of type typ."""
    from sympy.matrices.matrixbase import MatrixBase
    if getattr(mat, "is_Matrix", False) and not isinstance(mat, MatrixBase):
        # This is needed for interop between Matrix and the redundant matrix
        # mixin types like _MinimalMatrix etc. If anyone should happen to be
        # using those then this keeps them working. Really _MinimalMatrix etc
        # should be deprecated and removed though.
        return typ(*mat.shape, list(mat))
    else:
        return typ(mat)


def _has_matrix_shape(other):
    shape = getattr(other, 'shape', None)
    if shape is None:
        return False
    return isinstance(shape, tuple) and len(shape) == 2


def _has_rows_cols(other):
    return hasattr(other, 'rows') and hasattr(other, 'cols')


def _coerce_operand(self, other):
    """Convert other to a Matrix, or check for possible scalar."""

    INVALID = None, 'invalid_type'

    # Disallow mixing Matrix and Array
    if isinstance(other, NDimArray):
        return INVALID

    is_Matrix = getattr(other, 'is_Matrix', None)

    # Return a Matrix as-is
    if is_Matrix:
        return other, 'is_matrix'

    # Try to convert numpy array, mpmath matrix etc.
    if is_Matrix is None:
        if _has_matrix_shape(other) or _has_rows_cols(other):
            return _convert_matrix(type(self), other), 'is_matrix'

    # Could be a scalar but only if not iterable...
    if not isinstance(other, Iterable):
        return other, 'possible_scalar'

    return INVALID


def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.

    Currently the strategy is that immutability is contagious.

    Examples
    ========

    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.matrixbase import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    priority_A = getattr(A, '_class_priority', None)
    priority_B = getattr(B, '_class_priority', None)
    if None not in (priority_A, priority_B):
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__

    try:
        import numpy
    except ImportError:
        pass
    else:
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__

    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))


def _unify_with_other(self, other):
    """Unify self and other into a single matrix type, or check for scalar."""
    other, T = _coerce_operand(self, other)

    if T == "is_matrix":
        typ = classof(self, other)
        if typ != self.__class__:
            self = _convert_matrix(typ, self)
        if typ != other.__class__:
            other = _convert_matrix(typ, other)

    return self, other, T


def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    if not isinstance(j, int):
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError("Invalid index a[%r]" % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    return int(j)


class DeferredVector(Symbol, NotIterable): # type: ignore
    """A vector whose components are deferred (e.g. for use with lambdify).

    Examples
    ========

    >>> from sympy import DeferredVector, lambdify
    >>> X = DeferredVector( 'X' )
    >>> X
    X
    >>> expr = (X[0] + 2, X[2] + 3)
    >>> func = lambdify( X, expr)
    >>> func( [1, 2, 3] )
    (3, 6)
    """

    def __getitem__(self, i):
        if i == -0:
            i = 0
        if i < 0:
            raise IndexError('DeferredVector index out of range')
        component_name = '%s[%d]' % (self.name, i)
        return Symbol(component_name)

    def __str__(self):
        return sstr(self)

    def __repr__(self):
        return "DeferredVector('%s')" % self.name
<|endoftext|>from sympy.core.containers import Dict
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int, filldedent

from .sparse import MutableSparseMatrix as SparseMatrix


def _doktocsr(dok):
    """Converts a sparse matrix to Compressed Sparse Row (CSR) format.

    Parameters
    ==========

    A : contains non-zero elements sorted by key (row, column)
    JA : JA[i] is the column corresponding to A[i]
    IA : IA[i] contains the index in A for the first non-zero element
        of row[i]. Thus IA[i+1] - IA[i] gives number of non-zero
        elements row[i]. The length of IA is always 1 more than the
        number of rows in the matrix.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _doktocsr
    >>> from sympy import SparseMatrix, diag
    >>> m = SparseMatrix(diag(1, 2, 3))
    >>> m[2, 0] = -1
    >>> _doktocsr(m)
    [[1, 2, -1, 3], [0, 1, 0, 2], [0, 1, 2, 4], [3, 3]]

    """
    row, JA, A = [list(i) for i in zip(*dok.row_list())]
    IA = [0]*((row[0] if row else 0) + 1)
    for i, r in enumerate(row):
        IA.extend([i]*(r - row[i - 1]))  # if i = 0 nothing is extended
    IA.extend([len(A)]*(dok.rows - len(IA) + 1))
    shape = [dok.rows, dok.cols]
    return [A, JA, IA, shape]


def _csrtodok(csr):
    """Converts a CSR representation to DOK representation.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _csrtodok
    >>> _csrtodok([[5, 8, 3, 6], [0, 1, 2, 1], [0, 0, 2, 3, 4], [4, 3]])
    Matrix([
    [0, 0, 0],
    [5, 8, 0],
    [0, 0, 3],
    [0, 6, 0]])

    """
    smat = {}
    A, JA, IA, shape = csr
    for i in range(len(IA) - 1):
        indices = slice(IA[i], IA[i + 1])
        for l, m in zip(A[indices], JA[indices]):
            smat[i, m] = l
    return SparseMatrix(*shape, smat)


def banded(*args, **kwargs):
    """Returns a SparseMatrix from the given dictionary describing
    the diagonals of the matrix. The keys are positive for upper
    diagonals and negative for those below the main diagonal. The
    values may be:

    * expressions or single-argument functions,

    * lists or tuples of values,

    * matrices

    Unless dimensions are given, the size of the returned matrix will
    be large enough to contain the largest non-zero value provided.

    kwargs
    ======

    rows : rows of the resulting matrix; computed if
           not given.

    cols : columns of the resulting matrix; computed if
           not given.

    Examples
    ========

    >>> from sympy import banded, ones, Matrix
    >>> from sympy.abc import x

    If explicit values are given in tuples,
    the matrix will autosize to contain all values, otherwise
    a single value is filled onto the entire diagonal:

    >>> banded({1: (1, 2, 3), -1: (4, 5, 6), 0: x})
    Matrix([
    [x, 1, 0, 0],
    [4, x, 2, 0],
    [0, 5, x, 3],
    [0, 0, 6, x]])

    A function accepting a single argument can be used to fill the
    diagonal as a function of diagonal index (which starts at 0).
    The size (or shape) of the matrix must be given to obtain more
    than a 1x1 matrix:

    >>> s = lambda d: (1 + d)**2
    >>> banded(5, {0: s, 2: s, -2: 2})
    Matrix([
    [1, 0, 1,  0,  0],
    [0, 4, 0,  4,  0],
    [2, 0, 9,  0,  9],
    [0, 2, 0, 16,  0],
    [0, 0, 2,  0, 25]])

    The diagonal of matrices placed on a diagonal will coincide
    with the indicated diagonal:

    >>> vert = Matrix([1, 2, 3])
    >>> banded({0: vert}, cols=3)
    Matrix([
    [1, 0, 0],
    [2, 1, 0],
    [3, 2, 1],
    [0, 3, 2],
    [0, 0, 3]])

    >>> banded(4, {0: ones(2)})
    Matrix([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]])

    Errors are raised if the designated size will not hold
    all values an integral number of times. Here, the rows
    are designated as odd (but an even number is required to
    hold the off-diagonal 2x2 ones):

    >>> banded({0: 2, 1: ones(2)}, rows=5)
    Traceback (most recent call last):
    ...
    ValueError:
    sequence does not fit an integral number of times in the matrix

    And here, an even number of rows is given...but the square
    matrix has an even number of columns, too. As we saw
    in the previous example, an odd number is required:

    >>> banded(4, {0: 2, 1: ones(2)})  # trying to make 4x4 and cols must be odd
    Traceback (most recent call last):
    ...
    ValueError:
    sequence does not fit an integral number of times in the matrix

    A way around having to count rows is to enclosing matrix elements
    in a tuple and indicate the desired number of them to the right:

    >>> banded({0: 2, 2: (ones(2),)*3})
    Matrix([
    [2, 0, 1, 1, 0, 0, 0, 0],
    [0, 2, 1, 1, 0, 0, 0, 0],
    [0, 0, 2, 0, 1, 1, 0, 0],
    [0, 0, 0, 2, 1, 1, 0, 0],
    [0, 0, 0, 0, 2, 0, 1, 1],
    [0, 0, 0, 0, 0, 2, 1, 1]])

    An error will be raised if more than one value
    is written to a given entry. Here, the ones overlap
    with the main diagonal if they are placed on the
    first diagonal:

    >>> banded({0: (2,)*5, 1: (ones(2),)*3})
    Traceback (most recent call last):
    ...
    ValueError: collision at (1, 1)

    By placing a 0 at the bottom left of the 2x2 matrix of
    ones, the collision is avoided:

    >>> u2 = Matrix([
    ... [1, 1],
    ... [0, 1]])
    >>> banded({0: [2]*5, 1: [u2]*3})
    Matrix([
    [2, 1, 1, 0, 0, 0, 0],
    [0, 2, 1, 0, 0, 0, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 0, 0, 2, 1, 0, 0],
    [0, 0, 0, 0, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 1]])
    """
    try:
        if len(args) not in (1, 2, 3):
            raise TypeError
        if not isinstance(args[-1], (dict, Dict)):
            raise TypeError
        if len(args) == 1:
            rows = kwargs.get('rows', None)
            cols = kwargs.get('cols', None)
            if rows is not None:
                rows = as_int(rows)
            if cols is not None:
                cols = as_int(cols)
        elif len(args) == 2:
            rows = cols = as_int(args[0])
        else:
            rows, cols = map(as_int, args[:2])
        # fails with ValueError if any keys are not ints
        _ = all(as_int(k) for k in args[-1])
    except (ValueError, TypeError):
        raise TypeError(filldedent(
            '''unrecognized input to banded:
            expecting [[row,] col,] {int: value}'''))
    def rc(d):
        # return row,col coord of diagonal start
        r = -d if d < 0 else 0
        c = 0 if r else d
        return r, c
    smat = {}
    undone = []
    tba = Dummy()
    # first handle objects with size
    for d, v in args[-1].items():
        r, c = rc(d)
        # note: only list and tuple are recognized since this
        # will allow other Basic objects like Tuple
        # into the matrix if so desired
        if isinstance(v, (list, tuple)):
            extra = 0
            for i, vi in enumerate(v):
                i += extra
                if is_sequence(vi):
                    vi = SparseMatrix(vi)
                    smat[r + i, c + i] = vi
                    extra += min(vi.shape) - 1
                else:
                    smat[r + i, c + i] = vi
        elif is_sequence(v):
            v = SparseMatrix(v)
            rv, cv = v.shape
            if rows and cols:
                nr, xr = divmod(rows - r, rv)
                nc, xc = divmod(cols - c, cv)
                x = xr or xc
                do = min(nr, nc)
            elif rows:
                do, x = divmod(rows - r, rv)
            elif cols:
                do, x = divmod(cols - c, cv)
            else:
                do = 1
                x = 0
            if x:
                raise ValueError(filldedent('''
                    sequence does not fit an integral number of times
                    in the matrix'''))
            j = min(v.shape)
            for i in range(do):
                smat[r, c] = v
                r += j
                c += j
        elif v:
            smat[r, c] = tba
            undone.append((d, v))
    s = SparseMatrix(None, smat)  # to expand matrices
    smat = s.todok()
    # check for dim errors here
    if rows is not None and rows < s.rows:
        raise ValueError('Designated rows %s < needed %s' % (rows, s.rows))
    if cols is not None and cols < s.cols:
        raise ValueError('Designated cols %s < needed %s' % (cols, s.cols))
    if rows is cols is None:
        rows = s.rows
        cols = s.cols
    elif rows is not None and cols is None:
        cols = max(rows, s.cols)
    elif cols is not None and rows is None:
        rows = max(cols, s.rows)
    def update(i, j, v):
        # update smat and make sure there are
        # no collisions
        if v:
            if (i, j) in smat and smat[i, j] not in (tba, v):
                raise ValueError('collision at %s' % ((i, j),))
            smat[i, j] = v
    if undone:
        for d, vi in undone:
            r, c = rc(d)
            v = vi if callable(vi) else lambda _: vi
            i = 0
            while r + i < rows and c + i < cols:
                update(r + i, c + i, v(i))
                i += 1
    return SparseMatrix(rows, cols, smat)
<|endoftext|>from contextlib import contextmanager
from threading import local

from sympy.core.function import expand_mul


class DotProdSimpState(local):
    def __init__(self):
        self.state = None

_dotprodsimp_state = DotProdSimpState()

@contextmanager
def dotprodsimp(x):
    old = _dotprodsimp_state.state

    try:
        _dotprodsimp_state.state = x
        yield
    finally:
        _dotprodsimp_state.state = old


def _dotprodsimp(expr, withsimp=False):
    """Wrapper for simplify.dotprodsimp to avoid circular imports."""
    from sympy.simplify.simplify import dotprodsimp as dps
    return dps(expr, withsimp=withsimp)


def _get_intermediate_simp(deffunc=lambda x: x, offfunc=lambda x: x,
        onfunc=_dotprodsimp, dotprodsimp=None):
    """Support function for controlling intermediate simplification. Returns a
    simplification function according to the global setting of dotprodsimp
    operation.

    ``deffunc``     - Function to be used by default.
    ``offfunc``     - Function to be used if dotprodsimp has been turned off.
    ``onfunc``      - Function to be used if dotprodsimp has been turned on.
    ``dotprodsimp`` - True, False or None. Will be overridden by global
                      _dotprodsimp_state.state if that is not None.
    """

    if dotprodsimp is False or _dotprodsimp_state.state is False:
        return offfunc
    if dotprodsimp is True or _dotprodsimp_state.state is True:
        return onfunc

    return deffunc # None, None


def _get_intermediate_simp_bool(default=False, dotprodsimp=None):
    """Same as ``_get_intermediate_simp`` but returns bools instead of functions
    by default."""

    return _get_intermediate_simp(default, False, True, dotprodsimp)


def _iszero(x):
    """Returns True if x is zero."""
    return getattr(x, 'is_zero', None)


def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
    return expand_mul(x) == 0


def _simplify(expr):
    """ Wrapper to avoid circular imports. """
    from sympy.simplify.simplify import simplify
    return simplify(expr)
<|endoftext|>"""A module that handles matrices.

Includes functions for fast creating matrices like zero, one/eye, random
matrix, etc.
"""
from .exceptions import ShapeError, NonSquareMatrixError
from .kind import MatrixKind
from .dense import (
    GramSchmidt, casoratian, diag, eye, hessian, jordan_cell,
    list2numpy, matrix2numpy, matrix_multiply_elementwise, ones,
    randMatrix, rot_axis1, rot_axis2, rot_axis3, rot_ccw_axis1,
    rot_ccw_axis2, rot_ccw_axis3, rot_givens,
    symarray, wronskian, zeros)
from .dense import MutableDenseMatrix
from .matrixbase import DeferredVector, MatrixBase

MutableMatrix = MutableDenseMatrix
Matrix = MutableMatrix

from .sparse import MutableSparseMatrix
from .sparsetools import banded
from .immutable import ImmutableDenseMatrix, ImmutableSparseMatrix

ImmutableMatrix = ImmutableDenseMatrix
SparseMatrix = MutableSparseMatrix

from .expressions import (
    MatrixSlice, BlockDiagMatrix, BlockMatrix, FunctionMatrix, Identity,
    Inverse, MatAdd, MatMul, MatPow, MatrixExpr, MatrixSymbol, Trace,
    Transpose, ZeroMatrix, OneMatrix, blockcut, block_collapse, matrix_symbols, Adjoint,
    hadamard_product, HadamardProduct, HadamardPower, Determinant, det,
    diagonalize_vector, DiagMatrix, DiagonalMatrix, DiagonalOf, trace,
    DotProduct, kronecker_product, KroneckerProduct,
    PermutationMatrix, MatrixPermute, MatrixSet, Permanent, per)

from .utilities import dotprodsimp

__all__ = [
    'ShapeError', 'NonSquareMatrixError', 'MatrixKind',

    'GramSchmidt', 'casoratian', 'diag', 'eye', 'hessian', 'jordan_cell',
    'list2numpy', 'matrix2numpy', 'matrix_multiply_elementwise', 'ones',
    'randMatrix', 'rot_axis1', 'rot_axis2', 'rot_axis3', 'symarray',
    'wronskian', 'zeros', 'rot_ccw_axis1', 'rot_ccw_axis2', 'rot_ccw_axis3',
    'rot_givens',

    'MutableDenseMatrix',

    'DeferredVector', 'MatrixBase',

    'Matrix', 'MutableMatrix',

    'MutableSparseMatrix',

    'banded',

    'ImmutableDenseMatrix', 'ImmutableSparseMatrix',

    'ImmutableMatrix', 'SparseMatrix',

    'MatrixSlice', 'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix',
    'Identity', 'Inverse', 'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr',
    'MatrixSymbol', 'Trace', 'Transpose', 'ZeroMatrix', 'OneMatrix',
    'blockcut', 'block_collapse', 'matrix_symbols', 'Adjoint',
    'hadamard_product', 'HadamardProduct', 'HadamardPower', 'Determinant',
    'det', 'diagonalize_vector', 'DiagMatrix', 'DiagonalMatrix',
    'DiagonalOf', 'trace', 'DotProduct', 'kronecker_product',
    'KroneckerProduct', 'PermutationMatrix', 'MatrixPermute', 'MatrixSet',
    'Permanent', 'per',

    'dotprodsimp',
]
<|endoftext|>from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols

from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero


def _diagonal_solve(M, rhs):
    """Solves ``Ax = B`` efficiently, where A is a diagonal Matrix,
    with non-zero diagonal entries.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.diagonal_solve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_diagonal():
        raise TypeError("Matrix should be diagonal")
    if rhs.rows != M.rows:
        raise TypeError("Size mismatch")

    return M._new(
        rhs.rows, rhs.cols, lambda i, j: rhs[i, j] / M[i, i])


def _lower_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in range(M.rows):
            if M[i, i] == 0:
                raise TypeError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i))) / M[i, i])

    return M._new(X)

def _lower_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i > j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    for j in range(rhs.cols):
        for i in range(rhs.rows):
            for u, v in rows[i]:
                X[i, j] -= v*X[u, j]

            X[i, j] = dps(X[i, j] / M[i, i])

    return M._new(X)


def _upper_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:
        raise TypeError("Matrix is not upper triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in reversed(range(M.rows)):
            if M[i, i] == 0:
                raise ValueError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i + 1, M.rows))) / M[i, i])

    return M._new(X)

def _upper_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:
        raise TypeError("Matrix is not upper triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i < j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    for j in range(rhs.cols):
        for i in reversed(range(rhs.rows)):
            for u, v in reversed(rows[i]):
                X[i, j] -= v*X[u, j]

            X[i, j] = dps(X[i, j] / M[i, i])

    return M._new(X)


def _cholesky_solve(M, rhs):
    """Solves ``Ax = B`` using Cholesky decomposition,
    for a general square non-singular matrix.
    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')

    hermitian = True
    reform    = False

    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True

    if reform or _fuzzy_positive_definite(M) is False:
        H         = M.H
        M         = H.multiply(M)
        rhs       = H.multiply(rhs)
        hermitian = not M.is_symmetric()

    L = M.cholesky(hermitian=hermitian)
    Y = L.lower_triangular_solve(rhs)

    if hermitian:
        return (L.H).upper_triangular_solve(Y)
    else:
        return (L.T).upper_triangular_solve(Y)


def _LDLsolve(M, rhs):
    """Solves ``Ax = B`` using LDL decomposition,
    for a general square and non-singular matrix.

    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.LDLsolve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')

    hermitian = True
    reform    = False

    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True

    if reform or _fuzzy_positive_definite(M) is False:
        H         = M.H
        M         = H.multiply(M)
        rhs       = H.multiply(rhs)
        hermitian = not M.is_symmetric()

    L, D = M.LDLdecomposition(hermitian=hermitian)
    Y    = L.lower_triangular_solve(rhs)
    Z    = D.diagonal_solve(Y)

    if hermitian:
        return (L.H).upper_triangular_solve(Z)
    else:
        return (L.T).upper_triangular_solve(Z)


def _LUsolve(M, rhs, iszerofunc=_iszero):
    """Solve the linear system ``Ax = rhs`` for ``x`` where ``A = M``.

    This is for symbolic matrices, for real or complex ones use
    mpmath.lu_solve or mpmath.qr_solve.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    QRsolve
    pinv_solve
    LUdecomposition
    cramer_solve
    """

    if rhs.rows != M.rows:
        raise ShapeError(
            "``M`` and ``rhs`` must have the same number of rows.")

    m = M.rows
    n = M.cols

    if m < n:
        raise NotImplementedError("Underdetermined systems not supported.")

    try:
        A, perm = M.LUdecomposition_Simple(
            iszerofunc=iszerofunc, rankcheck=True)
    except ValueError:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    dps = _get_intermediate_simp()
    b   = rhs.permute_rows(perm).as_mutable()

    # forward substitution, all diag entries are scaled to 1
    for i in range(m):
        for j in range(min(i, n)):
            scale = A[i, j]
            b.zip_row_op(i, j, lambda x, y: dps(x - scale * y))

    # consistency check for overdetermined systems
    if m > n:
        for i in range(n, m):
            for j in range(b.cols):
                if not iszerofunc(b[i, j]):
                    raise ValueError("The system is inconsistent.")

        b = b[0:n, :]   # truncate zero rows if consistent

    # backward substitution
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            scale = A[i, j]
            b.zip_row_op(i, j, lambda x, y: dps(x - scale * y))

        scale = A[i, i]
        b.row_op(i, lambda x, _: dps(scale**-1 * x))

    return rhs.__class__(b)


def _QRsolve(M, b):
    """Solve the linear system ``Ax = b``.

    ``M`` is the matrix ``A``, the method argument is the vector
    ``b``.  The method returns the solution vector ``x``.  If ``b`` is a
    matrix, the system is solved for each column of ``b`` and the
    return value is a matrix of the same shape as ``b``.

    This method is slower (approximately by a factor of 2) but
    more stable for floating-point arithmetic than the LUsolve method.
    However, LUsolve usually uses an exact arithmetic, so you do not need
    to use QRsolve.

    This is mainly for educational purposes and symbolic matrices, for real
    (or complex) matrices use mpmath.qr_solve.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    pinv_solve
    QRdecomposition
    cramer_solve
    """

    dps  = _get_intermediate_simp(expand_mul, expand_mul)
    Q, R = M.QRdecomposition()
    y    = Q.T * b

    # back substitution to solve R*x = y:
    # We build up the result "backwards" in the vector 'x' and reverse it
    # only in the end.
    x = []
    n = R.rows

    for j in range(n - 1, -1, -1):
        tmp = y[j, :]

        for k in range(j + 1, n):
            tmp -= R[j, k] * x[n - 1 - k]

        tmp = dps(tmp)

        x.append(tmp / R[j, j])

    return M.vstack(*x[::-1])


def _gauss_jordan_solve(M, B, freevar=False):
    """
    Solves ``Ax = B`` using Gauss Jordan elimination.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned. If infinite solutions exist, it will
    be returned parametrically. If no solutions exist, It will throw
    ValueError.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.

    freevar : boolean, optional
        Flag, when set to `True` will return the indices of the free
        variables in the solutions (column Matrix), for a system that is
        undetermined (e.g. A has more columns than rows), for which
        infinite solutions are possible, in terms of arbitrary
        values of free variables. Default `False`.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    params : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        parameters. These arbitrary parameters are returned as params
        Matrix.

    free_var_index : List, optional
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        values of free variables. Then the indices of the free variables
        in the solutions (column Matrix) are returned by free_var_index,
        if the flag `freevar` is set to `True`.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> taus_zeroes = { tau:0 for tau in params }
    >>> sol_unique = sol.xreplace(taus_zeroes)
    >>> sol_unique
        Matrix([
    [2],
    [0],
    [5],
    [0]])


    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> B = Matrix([3, 6, 9])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-1],
    [ 2],
    [ 0]])
    >>> params
    Matrix(0, 1, [])

    >>> A = Matrix([[2, -7], [-1, 4]])
    >>> B = Matrix([[-21, 3], [12, -2]])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [0, -2],
    [3, -1]])
    >>> params
    Matrix(0, 2, [])


    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params, freevars = A.gauss_jordan_solve(B, freevar=True)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> freevars
    [1, 3]


    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_elimination

    """

    from sympy.matrices import Matrix, zeros

    cls      = M.__class__
    aug      = M.hstack(M.copy(), B.copy())
    B_cols   = B.cols
    row, col = aug[:, :-B_cols].shape

    # solve by reduced row echelon form
    A, pivots = aug.rref(simplify=True)
    A, v      = A[:, :-B_cols], A[:, -B_cols:]
    pivots    = list(filter(lambda p: p < col, pivots))
    rank      = len(pivots)

    # Get index of free symbols (free parameters)
    # non-pivots columns are free variables
    free_var_index = [c for c in range(A.cols) if c not in pivots]

    # Bring to block form
    permutation = Matrix(pivots + free_var_index).T

    # check for existence of solutions
    # rank of aug Matrix should be equal to rank of coefficient matrix
    if not v[rank:, :].is_zero_matrix:
        raise ValueError("Linear system has no solution")

    # Free parameters
    # what are current unnumbered free symbol names?
    name = uniquely_named_symbol('tau', [aug],
            compare=lambda i: str(i).rstrip('1234567890'),
            modify=lambda s: '_' + s).name
    gen  = numbered_symbols(name)
    tau  = Matrix([next(gen) for k in range((col - rank)*B_cols)]).reshape(
            col - rank, B_cols)

    # Full parametric solution
    V        = A[:rank, free_var_index]
    vt       = v[:rank, :]
    free_sol = tau.vstack(vt - V * tau, tau)

    # Undo permutation
    sol = zeros(col, B_cols)

    for k in range(col):
        sol[permutation[k], :] = free_sol[k,:]

    sol, tau = cls(sol), cls(tau)

    if freevar:
        return sol, tau, free_var_index
    else:
        return sol, tau


def _pinv_solve(M, B, arbitrary_matrix=None):
    """Solve ``Ax = B`` using the Moore-Penrose pseudoinverse.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned.  If infinite solutions exist, one will
    be returned based on the value of arbitrary_matrix.  If no solutions
    exist, the least-squares solution is returned.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.
    arbitrary_matrix : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of an arbitrary
        matrix.  This parameter may be set to a specific matrix to use
        for that purpose; if so, it must be the same shape as x, with as
        many rows as matrix A has columns, and as many columns as matrix
        B.  If left as None, an appropriate matrix containing dummy
        symbols in the form of ``wn_m`` will be used, with n and m being
        row and column position of each symbol.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> B = Matrix([7, 8])
    >>> A.pinv_solve(B)
    Matrix([
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 - 55/18],
    [-_w0_0/3 + 2*_w1_0/3 - _w2_0/3 + 1/9],
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 + 59/18]])
    >>> A.pinv_solve(B, arbitrary_matrix=Matrix([0, 0, 0]))
    Matrix([
    [-55/18],
    [   1/9],
    [ 59/18]])

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    Notes
    =====

    This may return either exact solutions or least squares solutions.
    To determine which, check ``A * A.pinv() * B == B``.  It will be
    True if exact solutions exist, and False if only a least-squares
    solution exists.  Be aware that the left hand side of that equation
    may need to be simplified to correctly compare to the right hand
    side.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse#Obtaining_all_solutions_of_a_linear_system

    """

    from sympy.matrices import eye

    A      = M
    A_pinv = M.pinv()

    if arbitrary_matrix is None:
        rows, cols       = A.cols, B.cols
        w                = symbols('w:{}_:{}'.format(rows, cols), cls=Dummy)
        arbitrary_matrix = M.__class__(cols, rows, w).T

    return A_pinv.multiply(B) + (eye(A.cols) -
            A_pinv.multiply(A)).multiply(arbitrary_matrix)


def _cramer_solve(M, rhs, det_method="laplace"):
    """Solves system of linear equations using Cramer's rule.

    This method is relatively inefficient compared to other methods.
    However it only uses a single division, assuming a division-free determinant
    method is provided. This is helpful to minimize the chance of divide-by-zero
    cases in symbolic solutions to linear systems.

    Parameters
    ==========
    M : Matrix
        The matrix representing the left hand side of the equation.
    rhs : Matrix
        The matrix representing the right hand side of the equation.
    det_method : str or callable
        The method to use to calculate the determinant of the matrix.
        The default is ``'laplace'``.  If a callable is passed, it should take a
        single argument, the matrix, and return the determinant of the matrix.

    Returns
    =======
    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[0, -6, 1], [0, -6, -1], [-5, -2, 3]])
    >>> B = Matrix([[-30, -9], [-18, -27], [-26, 46]])
    >>> x = A.cramer_solve(B)
    >>> x
    Matrix([
    [ 0, -5],
    [ 4,  3],
    [-6,  9]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cramer%27s_rule#Explicit_formulas_for_small_systems

    """
    from .dense import zeros

    def entry(i, j):
        return rhs[i, sol] if j == col else M[i, j]

    if det_method == "bird":
        from .determinant import _det_bird
        det = _det_bird
    elif det_method == "laplace":
        from .determinant import _det_laplace
        det = _det_laplace
    elif isinstance(det_method, str):
        det = lambda matrix: matrix.det(method=det_method)
    else:
        det = det_method
    det_M = det(M)
    x = zeros(*rhs.shape)
    for sol in range(rhs.shape[1]):
        for col in range(rhs.shape[0]):
            x[col, sol] = det(M.__class__(*M.shape, entry)) / det_M
    return M.__class__(x)


def _solve(M, rhs, method='GJ'):
    """Solves linear equation where the unique solution exists.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string, optional
        If set to ``'GJ'`` or ``'GE'``, the Gauss-Jordan elimination will be
        used, which is implemented in the routine ``gauss_jordan_solve``.

        If set to ``'LU'``, ``LUsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        If set to ``'CRAMER'``, ``cramer_solve`` routine will be used.

        It also supports the methods available for special linear systems

        For positive definite systems:

        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        To use a different method and to compute the solution via the
        inverse, use a method defined in the .inv() docstring.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Raises
    ======

    ValueError
        If there is not a unique solution then a ``ValueError`` will be
        raised.

        If ``M`` is not square, a ``ValueError`` and a different routine
        for solving the system will be suggested.
    """

    if method in ('GJ', 'GE'):
        try:
            soln, param = M.gauss_jordan_solve(rhs)

            if param:
                raise NonInvertibleMatrixError("Matrix det == 0; not invertible. "
                "Try ``M.gauss_jordan_solve(rhs)`` to obtain a parametric solution.")

        except ValueError:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return soln

    elif method == 'LU':
        return M.LUsolve(rhs)
    elif method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    elif method == 'CRAMER':
        return M.cramer_solve(rhs)
    else:
        return M.inv(method=method).multiply(rhs)


def _solve_least_squares(M, rhs, method='CH'):
    """Return the least-square fit to the data.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string or boolean, optional
        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        Otherwise, the conjugate of ``M`` will be used to create a system
        of equations that is passed to ``solve`` along with the hint
        defined by ``method``.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Examples
    ========

    >>> from sympy import Matrix, ones
    >>> A = Matrix([1, 2, 3])
    >>> B = Matrix([2, 3, 4])
    >>> S = Matrix(A.row_join(B))
    >>> S
    Matrix([
    [1, 2],
    [2, 3],
    [3, 4]])

    If each line of S represent coefficients of Ax + By
    and x and y are [2, 3] then S*xy is:

    >>> r = S*Matrix([2, 3]); r
    Matrix([
    [ 8],
    [13],
    [18]])

    But let's add 1 to the middle value and then solve for the
    least-squares value of xy:

    >>> xy = S.solve_least_squares(Matrix([8, 14, 18])); xy
    Matrix([
    [ 5/3],
    [10/3]])

    The error is given by S*xy - r:

    >>> S*xy - r
    Matrix([
    [1/3],
    [1/3],
    [1/3]])
    >>> _.norm().n(2)
    0.58

    If a different xy is used, the norm will be higher:

    >>> xy += ones(2, 1)/10
    >>> (S*xy - r).norm().n(2)
    1.5

    """

    if method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    else:
        t = M.H
        return (t * M).solve(t * rhs, method=method)
<|endoftext|>from collections.abc import Callable

from sympy.core.containers import Dict
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int

from .matrixbase import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix

from .utilities import _iszero

from .decompositions import (
    _liupc, _row_structure_symbolic_cholesky, _cholesky_sparse,
    _LDLdecomposition_sparse)

from .solvers import (
    _lower_triangular_solve_sparse, _upper_triangular_solve_sparse)


class SparseRepMatrix(RepMatrix):
    """
    A sparse matrix (a matrix with a large number of zero elements).

    Examples
    ========

    >>> from sympy import SparseMatrix, ones
    >>> SparseMatrix(2, 2, range(4))
    Matrix([
    [0, 1],
    [2, 3]])
    >>> SparseMatrix(2, 2, {(1, 1): 2})
    Matrix([
    [0, 0],
    [0, 2]])

    A SparseMatrix can be instantiated from a ragged list of lists:

    >>> SparseMatrix([[1, 2, 3], [1, 2], [1]])
    Matrix([
    [1, 2, 3],
    [1, 2, 0],
    [1, 0, 0]])

    For safety, one may include the expected size and then an error
    will be raised if the indices of any element are out of range or
    (for a flat list) if the total number of elements does not match
    the expected shape:

    >>> SparseMatrix(2, 2, [1, 2])
    Traceback (most recent call last):
    ...
    ValueError: List length (2) != rows*columns (4)

    Here, an error is not raised because the list is not flat and no
    element is out of range:

    >>> SparseMatrix(2, 2, [[1, 2]])
    Matrix([
    [1, 2],
    [0, 0]])

    But adding another element to the first (and only) row will cause
    an error to be raised:

    >>> SparseMatrix(2, 2, [[1, 2, 3]])
    Traceback (most recent call last):
    ...
    ValueError: The location (0, 2) is out of designated range: (1, 1)

    To autosize the matrix, pass None for rows:

    >>> SparseMatrix(None, [[1, 2, 3]])
    Matrix([[1, 2, 3]])
    >>> SparseMatrix(None, {(1, 1): 1, (3, 3): 3})
    Matrix([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 3]])

    Values that are themselves a Matrix are automatically expanded:

    >>> SparseMatrix(4, 4, {(1, 1): ones(2)})
    Matrix([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]])

    A ValueError is raised if the expanding matrix tries to overwrite
    a different element already present:

    >>> SparseMatrix(3, 3, {(0, 0): ones(2), (1, 1): 2})
    Traceback (most recent call last):
    ...
    ValueError: collision at (1, 1)

    See Also
    ========
    DenseMatrix
    MutableSparseMatrix
    ImmutableSparseMatrix
    """

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], MatrixBase):
            rows = args[0].rows
            cols = args[0].cols
            smat = args[0].todok()
            return rows, cols, smat

        smat = {}
        # autosizing
        if len(args) == 2 and args[0] is None:
            args = [None, None, args[1]]

        if len(args) == 3:
            r, c = args[:2]
            if r is c is None:
                rows = cols = None
            elif None in (r, c):
                raise ValueError(
                    'Pass rows=None and no cols for autosizing.')
            else:
                rows, cols = as_int(args[0]), as_int(args[1])

            if isinstance(args[2], Callable):
                op = args[2]

                if None in (rows, cols):
                    raise ValueError(
                        "{} and {} must be integers for this "
                        "specification.".format(rows, cols))

                row_indices = [cls._sympify(i) for i in range(rows)]
                col_indices = [cls._sympify(j) for j in range(cols)]

                for i in row_indices:
                    for j in col_indices:
                        value = cls._sympify(op(i, j))
                        if value != cls.zero:
                            smat[i, j] = value

                return rows, cols, smat

            elif isinstance(args[2], (dict, Dict)):
                def update(i, j, v):
                    # update smat and make sure there are no collisions
                    if v:
                        if (i, j) in smat and v != smat[i, j]:
                            raise ValueError(
                                "There is a collision at {} for {} and {}."
                                .format((i, j), v, smat[i, j])
                            )
                        smat[i, j] = v

                # manual copy, copy.deepcopy() doesn't work
                for (r, c), v in args[2].items():
                    if isinstance(v, MatrixBase):
                        for (i, j), vv in v.todok().items():
                            update(r + i, c + j, vv)
                    elif isinstance(v, (list, tuple)):
                        _, _, smat = cls._handle_creation_inputs(v, **kwargs)
                        for i, j in smat:
                            update(r + i, c + j, smat[i, j])
                    else:
                        v = cls._sympify(v)
                        update(r, c, cls._sympify(v))

            elif is_sequence(args[2]):
                flat = not any(is_sequence(i) for i in args[2])
                if not flat:
                    _, _, smat = \
                        cls._handle_creation_inputs(args[2], **kwargs)
                else:
                    flat_list = args[2]
                    if len(flat_list) != rows * cols:
                        raise ValueError(
                            "The length of the flat list ({}) does not "
                            "match the specified size ({} * {})."
                            .format(len(flat_list), rows, cols)
                        )

                    for i in range(rows):
                        for j in range(cols):
                            value = flat_list[i*cols + j]
                            value = cls._sympify(value)
                            if value != cls.zero:
                                smat[i, j] = value

            if rows is None:  # autosizing
                keys = smat.keys()
                rows = max(r for r, _ in keys) + 1 if keys else 0
                cols = max(c for _, c in keys) + 1 if keys else 0

            else:
                for i, j in smat.keys():
                    if i and i >= rows or j and j >= cols:
                        raise ValueError(
                            "The location {} is out of the designated range"
                            "[{}, {}]x[{}, {}]"
                            .format((i, j), 0, rows - 1, 0, cols - 1)
                        )

            return rows, cols, smat

        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            # list of values or lists
            v = args[0]
            c = 0
            for i, row in enumerate(v):
                if not isinstance(row, (list, tuple)):
                    row = [row]
                for j, vv in enumerate(row):
                    if vv != cls.zero:
                        smat[i, j] = cls._sympify(vv)
                c = max(c, len(row))
            rows = len(v) if c else 0
            cols = c
            return rows, cols, smat

        else:
            # handle full matrix forms with _handle_creation_inputs
            rows, cols, mat = super()._handle_creation_inputs(*args)
            for i in range(rows):
                for j in range(cols):
                    value = mat[cols*i + j]
                    if value != cls.zero:
                        smat[i, j] = value

            return rows, cols, smat

    @property
    def _smat(self):

        sympy_deprecation_warning(
            """
            The private _smat attribute of SparseMatrix is deprecated. Use the
            .todok() method instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-private-matrix-attributes"
        )

        return self.todok()

    def _eval_inverse(self, **kwargs):
        return self.inv(method=kwargs.get('method', 'LDL'),
                        iszerofunc=kwargs.get('iszerofunc', _iszero),
                        try_block_diag=kwargs.get('try_block_diag', False))

    def applyfunc(self, f):
        """Apply a function to each element of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> m = SparseMatrix(2, 2, lambda i, j: i*2+j)
        >>> m
        Matrix([
        [0, 1],
        [2, 3]])
        >>> m.applyfunc(lambda i: 2*i)
        Matrix([
        [0, 2],
        [4, 6]])

        """
        if not callable(f):
            raise TypeError("`f` must be callable.")

        # XXX: This only applies the function to the nonzero elements of the
        # matrix so is inconsistent with DenseMatrix.applyfunc e.g.
        #   zeros(2, 2).applyfunc(lambda x: x + 1)
        dok = {}
        for k, v in self.todok().items():
            fv = f(v)
            if fv != 0:
                dok[k] = fv

        return self._new(self.rows, self.cols, dok)

    def as_immutable(self):
        """Returns an Immutable version of this Matrix."""
        from .immutable import ImmutableSparseMatrix
        return ImmutableSparseMatrix(self)

    def as_mutable(self):
        """Returns a mutable version of this matrix.

        Examples
        ========

        >>> from sympy import ImmutableMatrix
        >>> X = ImmutableMatrix([[1, 2], [3, 4]])
        >>> Y = X.as_mutable()
        >>> Y[1, 1] = 5 # Can set values in Y
        >>> Y
        Matrix([
        [1, 2],
        [3, 5]])
        """
        return MutableSparseMatrix(self)

    def col_list(self):
        """Returns a column-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a=SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.CL
        [(0, 0, 1), (1, 0, 3), (0, 1, 2), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.row_list
        """
        return [tuple(k + (self[k],)) for k in sorted(self.todok().keys(), key=lambda k: list(reversed(k)))]

    def nnz(self):
        """Returns the number of non-zero elements in Matrix."""
        return len(self.todok())

    def row_list(self):
        """Returns a row-sorted list of non-zero elements of the matrix.

        Examples
        ========

        >>> from sympy import SparseMatrix
        >>> a = SparseMatrix(((1, 2), (3, 4)))
        >>> a
        Matrix([
        [1, 2],
        [3, 4]])
        >>> a.RL
        [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]

        See Also
        ========

        sympy.matrices.sparse.SparseMatrix.col_list
        """
        return [tuple(k + (self[k],)) for k in
            sorted(self.todok().keys(), key=list)]

    def scalar_multiply(self, scalar):
        "Scalar element-wise multiplication"
        return scalar * self

    def solve_least_squares(self, rhs, method='LDL'):
        """Return the least-square fit to the data.

        By default the cholesky_solve routine is used (method='CH'); other
        methods of matrix inversion can be used. To find out which are
        available, see the docstring of the .inv() method.

        Examples
        ========

        >>> from sympy import SparseMatrix, Matrix, ones
        >>> A = Matrix([1, 2, 3])
        >>> B = Matrix([2, 3, 4])
        >>> S = SparseMatrix(A.row_join(B))
        >>> S
        Matrix([
        [1, 2],
        [2, 3],
        [3, 4]])

        If each line of S represent coefficients of Ax + By
        and x and y are [2, 3] then S*xy is:

        >>> r = S*Matrix([2, 3]); r
        Matrix([
        [ 8],
        [13],
        [18]])

        But let's add 1 to the middle value and then solve for the
        least-squares value of xy:

        >>> xy = S.solve_least_squares(Matrix([8, 14, 18])); xy
        Matrix([
        [ 5/3],
        [10/3]])

        The error is given by S*xy - r:

        >>> S*xy - r
        Matrix([
        [1/3],
        [1/3],
        [1/3]])
        >>> _.norm().n(2)
        0.58

        If a different xy is used, the norm will be higher:

        >>> xy += ones(2, 1)/10
        >>> (S*xy - r).norm().n(2)
        1.5

        """
        t = self.T
        return (t*self).inv(method=method)*t*rhs

    def solve(self, rhs, method='LDL'):
        """Return solution to self*soln = rhs using given inversion method.

        For a list of possible inversion methods, see the .inv() docstring.
        """
        if not self.is_square:
            if self.rows < self.cols:
                raise ValueError('Under-determined system.')
            elif self.rows > self.cols:
                raise ValueError('For over-determined system, M, having '
                    'more rows than columns, try M.solve_least_squares(rhs).')
        else:
            return self.inv(method=method).multiply(rhs)

    RL = property(row_list, None, None, "Alternate faster representation")
    CL = property(col_list, None, None, "Alternate faster representation")

    def liupc(self):
        return _liupc(self)

    def row_structure_symbolic_cholesky(self):
        return _row_structure_symbolic_cholesky(self)

    def cholesky(self, hermitian=True):
        return _cholesky_sparse(self, hermitian=hermitian)

    def LDLdecomposition(self, hermitian=True):
        return _LDLdecomposition_sparse(self, hermitian=hermitian)

    def lower_triangular_solve(self, rhs):
        return _lower_triangular_solve_sparse(self, rhs)

    def upper_triangular_solve(self, rhs):
        return _upper_triangular_solve_sparse(self, rhs)

    liupc.__doc__                           = _liupc.__doc__
    row_structure_symbolic_cholesky.__doc__ = _row_structure_symbolic_cholesky.__doc__
    cholesky.__doc__                        = _cholesky_sparse.__doc__
    LDLdecomposition.__doc__                = _LDLdecomposition_sparse.__doc__
    lower_triangular_solve.__doc__          = lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__          = upper_triangular_solve.__doc__


class MutableSparseMatrix(SparseRepMatrix, MutableRepMatrix):

    @classmethod
    def _new(cls, *args, **kwargs):
        rows, cols, smat = cls._handle_creation_inputs(*args, **kwargs)

        rep = cls._smat_to_DomainMatrix(rows, cols, smat)

        return cls._fromrep(rep)


SparseMatrix = MutableSparseMatrix
<|endoftext|>from types import FunctionType

from sympy.core.cache import cacheit
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.ddm import DDM

from .exceptions import NonSquareMatrixError
from .utilities import (
    _get_intermediate_simp, _get_intermediate_simp_bool,
    _iszero, _is_zero_after_expand_mul, _dotprodsimp, _simplify)


def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):
    """ Find the lowest index of an item in ``col`` that is
    suitable for a pivot.  If ``col`` consists only of
    Floats, the pivot with the largest norm is returned.
    Otherwise, the first element where ``iszerofunc`` returns
    False is used.  If ``iszerofunc`` does not return false,
    items are simplified and retested until a suitable
    pivot is found.

    Returns a 4-tuple
        (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    where pivot_offset is the index of the pivot, pivot_val is
    the (possibly simplified) value of the pivot, assumed_nonzero
    is True if an assumption that the pivot was non-zero
    was made without being proved, and newly_determined are
    elements that were simplified during the process of pivot
    finding."""

    newly_determined = []
    col = list(col)
    # a column that contains a mix of floats and integers
    # but at least one float is considered a numerical
    # column, and so we do partial pivoting
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        col_abs = [abs(x) for x in col]
        max_value = max(col_abs)
        if iszerofunc(max_value):
            # just because iszerofunc returned True, doesn't
            # mean the value is numerically zero.  Make sure
            # to replace all entries with numerical zeros
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            return (None, None, False, newly_determined)
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    # PASS 1 (iszerofunc directly)
    possible_zeros = []
    for i, x in enumerate(col):
        is_zero = iszerofunc(x)
        # is someone wrote a custom iszerofunc, it may return
        # BooleanFalse or BooleanTrue instead of True or False,
        # so use == for comparison instead of `is`
        if is_zero == False:
            # we found something that is definitely not zero
            return (i, x, False, newly_determined)
        possible_zeros.append(is_zero)

    # by this point, we've found no certain non-zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 2 (iszerofunc after simplify)
    # we haven't found any for-sure non-zeros, so
    # go through the elements iszerofunc couldn't
    # make a determination about and opportunistically
    # simplify to see if we find something
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        simped = simpfunc(x)
        is_zero = iszerofunc(simped)
        if is_zero in (True, False):
            newly_determined.append((i, simped))
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero

    # after simplifying, some things that were recognized
    # as zeros might be zeros
    if all(possible_zeros):
        # if everything is definitely zero, we have
        # no pivot
        return (None, None, False, newly_determined)

    # PASS 3 (.equals(0))
    # some expressions fail to simplify to zero, but
    # ``.equals(0)`` evaluates to True.  As a last-ditch
    # attempt, apply ``.equals`` to these expressions
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        if x.equals(S.Zero):
            # ``.iszero`` may return False with
            # an implicit assumption (e.g., ``x.equals(0)``
            # when ``x`` is a symbol), so only treat it
            # as proved when ``.equals(0)`` returns True
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        return (None, None, False, newly_determined)

    # at this point there is nothing that could definitely
    # be a pivot.  To maintain compatibility with existing
    # behavior, we'll assume that an illdetermined thing is
    # non-zero.  We should probably raise a warning in this case
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)


def _find_reasonable_pivot_naive(col, iszerofunc=_iszero, simpfunc=None):
    """
    Helper that computes the pivot value and location from a
    sequence of contiguous matrix column elements. As a side effect
    of the pivot search, this function may simplify some of the elements
    of the input column. A list of these simplified entries and their
    indices are also returned.
    This function mimics the behavior of _find_reasonable_pivot(),
    but does less work trying to determine if an indeterminate candidate
    pivot simplifies to zero. This more naive approach can be much faster,
    with the trade-off that it may erroneously return a pivot that is zero.

    ``col`` is a sequence of contiguous column entries to be searched for
    a suitable pivot.
    ``iszerofunc`` is a callable that returns a Boolean that indicates
    if its input is zero, or None if no such determination can be made.
    ``simpfunc`` is a callable that simplifies its input. It must return
    its input if it does not simplify its input. Passing in
    ``simpfunc=None`` indicates that the pivot search should not attempt
    to simplify any candidate pivots.

    Returns a 4-tuple:
    (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    ``pivot_offset`` is the sequence index of the pivot.
    ``pivot_val`` is the value of the pivot.
    pivot_val and col[pivot_index] are equivalent, but will be different
    when col[pivot_index] was simplified during the pivot search.
    ``assumed_nonzero`` is a boolean indicating if the pivot cannot be
    guaranteed to be zero. If assumed_nonzero is true, then the pivot
    may or may not be non-zero. If assumed_nonzero is false, then
    the pivot is non-zero.
    ``newly_determined`` is a list of index-value pairs of pivot candidates
    that were simplified during the pivot search.
    """

    # indeterminates holds the index-value pairs of each pivot candidate
    # that is neither zero or non-zero, as determined by iszerofunc().
    # If iszerofunc() indicates that a candidate pivot is guaranteed
    # non-zero, or that every candidate pivot is zero then the contents
    # of indeterminates are unused.
    # Otherwise, the only viable candidate pivots are symbolic.
    # In this case, indeterminates will have at least one entry,
    # and all but the first entry are ignored when simpfunc is None.
    indeterminates = []
    for i, col_val in enumerate(col):
