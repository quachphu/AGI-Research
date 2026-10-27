    symmetries of the Riemann tensor; this is an abuse of notation,
    since the shape ``[2, 2]`` corresponds usually to the irreducible
    representation characterized by the monoterm symmetries and by the
    cyclic symmetry.
    """
    from sympy.combinatorics import Permutation

    def tableau2bsgs(a):
        if len(a) == 1:
            # antisymmetric vector
            n = a[0]
            bsgs = get_symmetric_group_sgs(n, 1)
        else:
            if all(x == 1 for x in a):
                # symmetric vector
                n = len(a)
                bsgs = get_symmetric_group_sgs(n)
            elif a == [2, 2]:
                bsgs = riemann_bsgs
            else:
                raise NotImplementedError
        return bsgs

    if not args:
        return TensorSymmetry(Tuple(), Tuple(Permutation(1)))

    if len(args) == 2 and isinstance(args[1][0], Permutation):
        return TensorSymmetry(args)
    base, sgs = tableau2bsgs(args[0])
    for a in args[1:]:
        basex, sgsx = tableau2bsgs(a)
        base, sgs = bsgs_direct_product(base, sgs, basex, sgsx)
    return TensorSymmetry(Tuple(base, sgs))

@deprecated(
    "TensorType is deprecated. Use tensor_heads() instead.",
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensortype",
)
class TensorType(Basic):
    """
    Class of tensor types. Deprecated, use tensor_heads() instead.

    Parameters
    ==========

    index_types : list of ``TensorIndexType`` of the tensor indices
    symmetry : ``TensorSymmetry`` of the tensor

    Attributes
    ==========

    ``index_types``
    ``symmetry``
    ``types`` : list of ``TensorIndexType`` without repetitions
    """
    is_commutative = False

    def __new__(cls, index_types, symmetry, **kw_args):
        assert symmetry.rank == len(index_types)
        obj = Basic.__new__(cls, Tuple(*index_types), symmetry, **kw_args)
        return obj

    @property
    def index_types(self):
        return self.args[0]

    @property
    def symmetry(self):
        return self.args[1]

    @property
    def types(self):
        return sorted(set(self.index_types), key=lambda x: x.name)

    def __str__(self):
        return 'TensorType(%s)' % ([str(x) for x in self.index_types])

    def __call__(self, s, comm=0):
        """
        Return a TensorHead object or a list of TensorHead objects.

        Parameters
        ==========

        s : name or string of names.

        comm : Commutation group.

        see ``_TensorManager.set_comm``
        """
        if isinstance(s, str):
            names = [x.name for x in symbols(s, seq=True)]
        else:
            raise ValueError('expecting a string')
        if len(names) == 1:
            return TensorHead(names[0], self.index_types, self.symmetry, comm)
        else:
            return [TensorHead(name, self.index_types, self.symmetry, comm) for name in names]


@deprecated(
    """
    The tensorhead() function is deprecated. Use tensor_heads() instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorhead",
)
def tensorhead(name, typ, sym=None, comm=0):
    """
    Function generating tensorhead(s). This method is deprecated,
    use TensorHead constructor or tensor_heads() instead.

    Parameters
    ==========

    name : name or sequence of names (as in ``symbols``)

    typ :  index types

    sym :  same as ``*args`` in ``tensorsymmetry``

    comm : commutation group number
    see ``_TensorManager.set_comm``
    """
    if sym is None:
        sym = [[1] for i in range(len(typ))]
    with ignore_warnings(SymPyDeprecationWarning):
        sym = tensorsymmetry(*sym)
    return TensorHead(name, typ, sym, comm)


class TensorHead(Basic):
    """
    Tensor head of the tensor.

    Parameters
    ==========

    name : name of the tensor
    index_types : list of TensorIndexType
    symmetry : TensorSymmetry of the tensor
    comm : commutation group number

    Attributes
    ==========

    ``name``
    ``index_types``
    ``rank`` : total number of indices
    ``symmetry``
    ``comm`` : commutation group

    Notes
    =====

    Similar to ``symbols`` multiple TensorHeads can be created using
    ``tensorhead(s, typ, sym=None, comm=0)`` function, where ``s``
    is the string of names and ``sym`` is the monoterm tensor symmetry
    (see ``tensorsymmetry``).

    A ``TensorHead`` belongs to a commutation group, defined by a
    symbol on number ``comm`` (see ``_TensorManager.set_comm``);
    tensors in a commutation group have the same commutation properties;
    by default ``comm`` is ``0``, the group of the commuting tensors.

    Examples
    ========

    Define a fully antisymmetric tensor of rank 2:

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> asym2 = TensorSymmetry.fully_symmetric(-2)
    >>> A = TensorHead('A', [Lorentz, Lorentz], asym2)

    Examples with ndarray values, the components data assigned to the
    ``TensorHead`` object are assumed to be in a fully-contravariant
    representation. In case it is necessary to assign components data which
    represents the values of a non-fully covariant tensor, see the other
    examples.

    >>> from sympy.tensor.tensor import tensor_indices
    >>> from sympy import diag
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i0, i1 = tensor_indices('i0:2', Lorentz)

    Specify a replacement dictionary to keep track of the arrays to use for
    replacements in the tensorial expression. The ``TensorIndexType`` is
    associated to the metric used for contractions (in fully covariant form):

    >>> repl = {Lorentz: diag(1, -1, -1, -1)}

    Let's see some examples of working with components with the electromagnetic
    tensor:

    >>> from sympy import symbols
    >>> Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
    >>> c = symbols('c', positive=True)

    Let's define `F`, an antisymmetric tensor:

    >>> F = TensorHead('F', [Lorentz, Lorentz], asym2)

    Let's update the dictionary to contain the matrix to use in the
    replacements:

    >>> repl.update({F(-i0, -i1): [
    ... [0, Ex/c, Ey/c, Ez/c],
    ... [-Ex/c, 0, -Bz, By],
    ... [-Ey/c, Bz, 0, -Bx],
    ... [-Ez/c, -By, Bx, 0]]})

    Now it is possible to retrieve the contravariant form of the Electromagnetic
    tensor:

    >>> F(i0, i1).replace_with_arrays(repl, [i0, i1])
    [[0, -E_x/c, -E_y/c, -E_z/c], [E_x/c, 0, -B_z, B_y], [E_y/c, B_z, 0, -B_x], [E_z/c, -B_y, B_x, 0]]

    and the mixed contravariant-covariant form:

    >>> F(i0, -i1).replace_with_arrays(repl, [i0, -i1])
    [[0, E_x/c, E_y/c, E_z/c], [E_x/c, 0, B_z, -B_y], [E_y/c, -B_z, 0, B_x], [E_z/c, B_y, -B_x, 0]]

    Energy-momentum of a particle may be represented as:

    >>> from sympy import symbols
    >>> P = TensorHead('P', [Lorentz], TensorSymmetry.no_symmetry(1))
    >>> E, px, py, pz = symbols('E p_x p_y p_z', positive=True)
    >>> repl.update({P(i0): [E, px, py, pz]})

    The contravariant and covariant components are, respectively:

    >>> P(i0).replace_with_arrays(repl, [i0])
    [E, p_x, p_y, p_z]
    >>> P(-i0).replace_with_arrays(repl, [-i0])
    [E, -p_x, -p_y, -p_z]

    The contraction of a 1-index tensor by itself:

    >>> expr = P(i0)*P(-i0)
    >>> expr.replace_with_arrays(repl, [])
    E**2 - p_x**2 - p_y**2 - p_z**2
    """
    is_commutative = False

    def __new__(cls, name, index_types, symmetry=None, comm=0):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError("invalid name")

        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)

        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), symmetry, sympify(comm))
        return obj

    @property
    def name(self):
        return self.args[0].name

    @property
    def index_types(self):
        return list(self.args[1])

    @property
    def symmetry(self):
        return self.args[2]

    @property
    def comm(self):
        return TensorManager.comm_symbols2i(self.args[3])

    @property
    def rank(self):
        return len(self.index_types)

    def __lt__(self, other):
        return (self.name, self.index_types) < (other.name, other.index_types)

    def commutes_with(self, other):
        """
        Returns ``0`` if ``self`` and ``other`` commute, ``1`` if they anticommute.

        Returns ``None`` if ``self`` and ``other`` neither commute nor anticommute.
        """
        r = TensorManager.get_comm(self.comm, other.comm)
        return r

    def _print(self):
        return '%s(%s)' %(self.name, ','.join([str(x) for x in self.index_types]))

    def __call__(self, *indices, **kw_args):
        """
        Returns a tensor with indices.

        Explanation
        ===========

        There is a special behavior in case of indices denoted by ``True``,
        they are considered auto-matrix indices, their slots are automatically
        filled, and confer to the tensor the behavior of a matrix or vector
        upon multiplication with another tensor containing auto-matrix indices
        of the same ``TensorIndexType``. This means indices get summed over the
        same way as in matrix multiplication. For matrix behavior, define two
        auto-matrix indices, for vector behavior define just one.

        Indices can also be strings, in which case the attribute
        ``index_types`` is used to convert them to proper ``TensorIndex``.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, TensorHead
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b = tensor_indices('a,b', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
        >>> t = A(a, -b)
        >>> t
        A(a, -b)

        """

        updated_indices = []
        for idx, typ in zip(indices, self.index_types):
            if isinstance(idx, str):
                idx = idx.strip().replace(" ", "")
                if idx.startswith('-'):
                    updated_indices.append(TensorIndex(idx[1:], typ,
                                           is_up=False))
                else:
                    updated_indices.append(TensorIndex(idx, typ))
            else:
                updated_indices.append(idx)

        updated_indices += indices[len(updated_indices):]

        tensor = Tensor(self, updated_indices, **kw_args)
        return tensor.doit()

    # Everything below this line is deprecated

    def __pow__(self, other):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError("No power on abstract tensors.")
            from .array import tensorproduct, tensorcontraction
            metrics = [_.data for _ in self.index_types]

            marray = self.data
            marraydim = marray.rank()
            for metric in metrics:
                marray = tensorproduct(marray, metric, marray)
                marray = tensorcontraction(marray, (0, marraydim), (marraydim+1, marraydim+2))

            return marray ** (other * S.Half)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]

    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data.__iter__()

    def _components_data_full_destroy(self):
        """
        EXPERIMENTAL: do not rely on this API method.

        Destroy components data associated to the ``TensorHead`` object, this
        checks for attached components data, and destroys components data too.
        """
        # do not garbage collect Kronecker tensor (it should be done by
        # ``TensorIndexType`` garbage collection)
        deprecate_data()
        if self.name == "KD":
            return

        # the data attached to a tensor must be deleted only by the TensorHead
        # destructor. If the TensorHead is deleted, it means that there are no
        # more instances of that tensor anywhere.
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]


def tensor_heads(s, index_types, symmetry=None, comm=0):
    """
    Returns a sequence of TensorHeads from a string `s`
    """
    if isinstance(s, str):
        names = [x.name for x in symbols(s, seq=True)]
    else:
        raise ValueError('expecting a string')

    thlist = [TensorHead(name, index_types, symmetry, comm) for name in names]
    if len(thlist) == 1:
        return thlist[0]
    return thlist


class TensExpr(Expr, ABC):
    """
    Abstract base class for tensor expressions

    Notes
    =====

    A tensor expression is an expression formed by tensors;
    currently the sums of tensors are distributed.

    A ``TensExpr`` can be a ``TensAdd`` or a ``TensMul``.

    ``TensMul`` objects are formed by products of component tensors,
    and include a coefficient, which is a SymPy expression.


    In the internal representation contracted indices are represented
    by ``(ipos1, ipos2, icomp1, icomp2)``, where ``icomp1`` is the position
    of the component tensor with contravariant index, ``ipos1`` is the
    slot which the index occupies in that component tensor.

    Contracted indices are therefore nameless in the internal representation.
    """

    _op_priority = 12.0
    is_commutative = False

    def __neg__(self):
        return self*S.NegativeOne

    def __abs__(self):
        raise NotImplementedError

    def __add__(self, other):
        return TensAdd(self, other).doit(deep=False)

    def __radd__(self, other):
        return TensAdd(other, self).doit(deep=False)

    def __sub__(self, other):
        return TensAdd(self, -other).doit(deep=False)

    def __rsub__(self, other):
        return TensAdd(other, -self).doit(deep=False)

    def __mul__(self, other):
        """
        Multiply two tensors using Einstein summation convention.

        Explanation
        ===========

        If the two tensors have an index in common, one contravariant
        and the other covariant, in their product the indices are summed

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t1 = p(m0)
        >>> t2 = q(-m0)
        >>> t1*t2
        p(L_0)*q(-L_0)
        """
        return TensMul(self, other).doit(deep=False)

    def __rmul__(self, other):
        return TensMul(other, self).doit(deep=False)

    def __truediv__(self, other):
        other = _sympify(other)
        if isinstance(other, TensExpr):
            raise ValueError('cannot divide by a tensor')
        return TensMul(self, S.One/other).doit(deep=False)

    def __rtruediv__(self, other):
        raise ValueError('cannot divide by a tensor')

    def __pow__(self, other):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError("No power without ndarray data.")
            from .array import tensorproduct, tensorcontraction
            free = self.free
            marray = self.data
            mdim = marray.rank()
            for metric in free:
                marray = tensorcontraction(
                    tensorproduct(
                    marray,
                    metric[0].tensor_index_type.data,
                    marray),
                    (0, mdim), (mdim+1, mdim+2)
                )
            return marray ** (other * S.Half)

    def __rpow__(self, other):
        raise NotImplementedError

    @property
    @abstractmethod
    def nocoeff(self):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def coeff(self):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_indices(self):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_free_indices(self) -> list[TensorIndex]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        raise NotImplementedError("abstract method")

    def fun_eval(self, *index_tuples):
        deprecate_fun_eval()
        return self.substitute_indices(*index_tuples)

    def get_matrix(self):
        """
        DEPRECATED: do not use.

        Returns ndarray components data as a matrix, if components data are
        available and ndarray dimension does not exceed 2.
        """
        from sympy.matrices.dense import Matrix
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if 0 < self.rank <= 2:
                rows = self.data.shape[0]
                columns = self.data.shape[1] if self.rank == 2 else 1
                if self.rank == 2:
                    mat_list = [] * rows
                    for i in range(rows):
                        mat_list.append([])
                        for j in range(columns):
                            mat_list[i].append(self[i, j])
                else:
                    mat_list = [None] * rows
                    for i in range(rows):
                        mat_list[i] = self[i]
                return Matrix(mat_list)
            else:
                raise NotImplementedError(
                    "missing multidimensional reduction to matrix.")

    @staticmethod
    def _get_indices_permutation(indices1, indices2):
        return [indices1.index(i) for i in indices2]

    def _get_free_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_free_indices_set())
        return indset

    def _get_dummy_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_dummy_indices_set())
        return indset

    def _get_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_indices_set())
        return indset

    @property
    def _iterate_dummy_indices(self):
        dummy_set = self._get_dummy_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in dummy_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    @property
    def _iterate_free_indices(self):
        free_set = self._get_free_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in free_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    @property
    def _iterate_indices(self):
        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    @staticmethod
    def _contract_and_permute_with_metric(metric, array, pos, dim):
        # TODO: add possibility of metric after (spinors)
        from .array import tensorcontraction, tensorproduct, permutedims

        array = tensorcontraction(tensorproduct(metric, array), (1, 2+pos))
        permu = list(range(dim))
        permu[0], permu[pos] = permu[pos], permu[0]
        return permutedims(array, permu)

    @staticmethod
    def _match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict):
        from .array import permutedims

        index_types1 = [i.tensor_index_type for i in free_ind1]

        # Check if variance of indices needs to be fixed:
        pos2up = []
        pos2down = []
        free2remaining = free_ind2[:]
        for pos1, index1 in enumerate(free_ind1):
            if index1 in free2remaining:
                pos2 = free2remaining.index(index1)
                free2remaining[pos2] = None
                continue
            if -index1 in free2remaining:
                pos2 = free2remaining.index(-index1)
                free2remaining[pos2] = None
                free_ind2[pos2] = index1
                if index1.is_up:
                    pos2up.append(pos2)
                else:
                    pos2down.append(pos2)
            else:
                index2 = free2remaining[pos1]
                if index2 is None:
                    raise ValueError("incompatible indices: %s and %s" % (free_ind1, free_ind2))
                free2remaining[pos1] = None
                free_ind2[pos1] = index1
                if index1.is_up ^ index2.is_up:
                    if index1.is_up:
                        pos2up.append(pos1)
                    else:
                        pos2down.append(pos1)

        if len(set(free_ind1) & set(free_ind2)) < len(free_ind1):
            raise ValueError("incompatible indices: %s and %s" % (free_ind1, free_ind2))

        # Raise indices:
        for pos in pos2up:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError("No metric provided to lower index")
            metric = replacement_dict[index_type_pos]
            metric_inverse = _TensorDataLazyEvaluator.inverse_matrix(metric)
            array = TensExpr._contract_and_permute_with_metric(metric_inverse, array, pos, len(free_ind1))
        # Lower indices:
        for pos in pos2down:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError("No metric provided to lower index")
            metric = replacement_dict[index_type_pos]
            array = TensExpr._contract_and_permute_with_metric(metric, array, pos, len(free_ind1))

        if free_ind1:
            permutation = TensExpr._get_indices_permutation(free_ind2, free_ind1)
            array = permutedims(array, permutation)

        if hasattr(array, "rank") and array.rank() == 0:
            array = array[()]

        return free_ind2, array

    def replace_with_arrays(self, replacement_dict, indices=None):
        """
        Replace the tensorial expressions with arrays. The final array will
        correspond to the N-dimensional array with indices arranged according
        to ``indices``.

        Parameters
        ==========

        replacement_dict
            dictionary containing the replacement rules for tensors.
        indices
            the index order with respect to which the array is read. The
            original index order will be used if no value is passed.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
        >>> from sympy.tensor.tensor import TensorHead
        >>> from sympy import symbols, diag

        >>> L = TensorIndexType("L")
        >>> i, j = tensor_indices("i j", L)
        >>> A = TensorHead("A", [L])
        >>> A(i).replace_with_arrays({A(i): [1, 2]}, [i])
        [1, 2]

        Since 'indices' is optional, we can also call replace_with_arrays by
        this way if no specific index order is needed:

        >>> A(i).replace_with_arrays({A(i): [1, 2]})
        [1, 2]

        >>> expr = A(i)*A(j)
        >>> expr.replace_with_arrays({A(i): [1, 2]})
        [[1, 2], [2, 4]]

        For contractions, specify the metric of the ``TensorIndexType``, which
        in this case is ``L``, in its covariant form:

        >>> expr = A(i)*A(-i)
        >>> expr.replace_with_arrays({A(i): [1, 2], L: diag(1, -1)})
        -3

        Symmetrization of an array:

        >>> H = TensorHead("H", [L, L])
        >>> a, b, c, d = symbols("a b c d")
        >>> expr = H(i, j)/2 + H(j, i)/2
        >>> expr.replace_with_arrays({H(i, j): [[a, b], [c, d]]})
        [[a, b/2 + c/2], [b/2 + c/2, d]]

        Anti-symmetrization of an array:

        >>> expr = H(i, j)/2 - H(j, i)/2
        >>> repl = {H(i, j): [[a, b], [c, d]]}
        >>> expr.replace_with_arrays(repl)
        [[0, b/2 - c/2], [-b/2 + c/2, 0]]

        The same expression can be read as the transpose by inverting ``i`` and
        ``j``:

        >>> expr.replace_with_arrays(repl, [j, i])
        [[0, -b/2 + c/2], [b/2 - c/2, 0]]
        """
        from .array import Array

        indices = indices or []
        remap = {k.args[0] if k.is_up else -k.args[0]: k for k in self.get_free_indices()}
        for i, index in enumerate(indices):
            if isinstance(index, (Symbol, Mul)):
                if index in remap:
                    indices[i] = remap[index]
                else:
                    indices[i] = -remap[-index]

        replacement_dict = {tensor: Array(array) for tensor, array in replacement_dict.items()}

        # Check dimensions of replaced arrays:
        for tensor, array in replacement_dict.items():
            if isinstance(tensor, TensorIndexType):
                expected_shape = [tensor.dim for i in range(2)]
            else:
                expected_shape = [index_type.dim for index_type in tensor.index_types]
            if len(expected_shape) != array.rank() or (not all(dim1 == dim2 if
                dim1.is_number else True for dim1, dim2 in zip(expected_shape,
                array.shape))):
                raise ValueError("shapes for tensor %s expected to be %s, "\
                    "replacement array shape is %s" % (tensor, expected_shape,
                    array.shape))

        ret_indices, array = self._extract_data(replacement_dict)

        last_indices, array = self._match_indices_with_other_tensor(array, indices, ret_indices, replacement_dict)
        return array

    def _check_add_Sum(self, expr, index_symbols):
        from sympy.concrete.summations import Sum
        indices = self.get_indices()
        dum = self.dum
        sum_indices = [ (index_symbols[i], 0,
            indices[i].tensor_index_type.dim-1) for i, j in dum]
        if sum_indices:
            expr = Sum(expr, *sum_indices)
        return expr

    def _expand_partial_derivative(self):
        # simply delegate the _expand_partial_derivative() to
        # its arguments to expand a possibly found PartialDerivative
        return self.func(*[
                    a._expand_partial_derivative()
                    if isinstance(a, TensExpr) else a
                    for a in self.args])

    def _matches_simple(self, expr, repl_dict=None, old=False):
        """
        Matches assuming there are no wild objects in self.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if not isinstance(expr, TensExpr):
            if len(self.get_free_indices()) > 0:
                #self has indices, but expr does not.
                return None
        elif set(self.get_free_indices()) != set(expr.get_free_indices()):
                #If there are no wilds and the free indices are not the same, they cannot match.
                return None

        if canon_bp(self - expr) == S.Zero:
            return repl_dict
        else:
            return None


class TensAdd(TensExpr, AssocOp):
    """
    Sum of tensors.

    Parameters
    ==========

    free_args : list of the free indices

    Attributes
    ==========

    ``args`` : tuple of addends
    ``rank`` : rank of the tensor
    ``free_args`` : list of the free indices in sorted order

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_heads, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> a, b = tensor_indices('a,b', Lorentz)
    >>> p, q = tensor_heads('p,q', [Lorentz])
    >>> t = p(a) + q(a); t
    p(a) + q(a)

    Examples with components data added to the tensor expression:

    >>> from sympy import symbols, diag
    >>> x, y, z, t = symbols("x y z t")
    >>> repl = {}
    >>> repl[Lorentz] = diag(1, -1, -1, -1)
    >>> repl[p(a)] = [1, 2, 3, 4]
    >>> repl[q(a)] = [x, y, z, t]

    The following are: 2**2 - 3**2 - 2**2 - 7**2 ==> -58

    >>> expr = p(a) + q(a)
    >>> expr.replace_with_arrays(repl, [a])
    [x + 1, y + 2, z + 3, t + 4]
    """

    def __new__(cls, *args, **kw_args):
        args = [_sympify(x) for x in args if x]
        args = TensAdd._tensAdd_flatten(args)
        args.sort(key=default_sort_key)
        if not args:
            return S.Zero
        if len(args) == 1:
            return args[0]

        return Basic.__new__(cls, *args, **kw_args)

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    def get_free_indices(self) -> list[TensorIndex]:
        return self.free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        newargs = [arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args]
        return self.func(*newargs)

    @memoize_property
    def rank(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].rank
        else:
            return 0

    @memoize_property
    def free_args(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].free_args
        else:
            return []

    @memoize_property
    def free_indices(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].get_free_indices()
        else:
            return set()

    def doit(self, **hints) -> Expr:
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args # type: ignore

        # if any of the args are zero (after doit), drop them. Otherwise, _tensAdd_check will complain about non-matching indices, even though the TensAdd is correctly formed.
        args = [arg for arg in args if arg != S.Zero]

        if len(args) == 0:
            return S.Zero
        elif len(args) == 1:
            return args[0]

        # now check that all addends have the same indices:
        TensAdd._tensAdd_check(args)

        # Collect terms appearing more than once, differing by their coefficients:
        args = TensAdd._tensAdd_collect_terms(args)

        # collect canonicalized terms
        def sort_key(t):
            if not isinstance(t, TensExpr):
                return [], [], []
            if hasattr(t, "_index_structure") and hasattr(t, "components"):
                x = get_index_structure(t)
                return t.components, x.free, x.dum
            return [], [], []
        args.sort(key=sort_key)

        if not args:
            return S.Zero
        # it there is only a component tensor return it
        if len(args) == 1:
            return args[0]

        obj = self.func(*args)
        return obj

    @staticmethod
    def _tensAdd_flatten(args):
        # flatten TensAdd, coerce terms which are not tensors to tensors
        a = []
        for x in args:
            if isinstance(x, (Add, TensAdd)):
                a.extend(list(x.args))
            else:
                a.append(x)
        args = [x for x in a if x.coeff]
        return args

    @staticmethod
    def _tensAdd_check(args):
        # check that all addends have the same free indices

        def get_indices_set(x: Expr) -> set[TensorIndex]:
            if isinstance(x, TensExpr):
                return set(x.get_free_indices())
            return set()

        indices0 = get_indices_set(args[0])
        list_indices = [get_indices_set(arg) for arg in args[1:]]
        if not all(x == indices0 for x in list_indices):
            raise ValueError('all tensors must have the same indices')

    @staticmethod
    def _tensAdd_collect_terms(args):
        # collect TensMul terms differing at most by their coefficient
        terms_dict = defaultdict(list)
        scalars = S.Zero
        if isinstance(args[0], TensExpr):
            free_indices = set(args[0].get_free_indices())
        else:
            free_indices = set()

        for arg in args:
            if not isinstance(arg, TensExpr):
                if free_indices != set():
                    raise ValueError("wrong valence")
                scalars += arg
                continue
            if free_indices != set(arg.get_free_indices()):
                raise ValueError("wrong valence")
            # TODO: what is the part which is not a coeff?
            # needs an implementation similar to .as_coeff_Mul()
            terms_dict[arg.nocoeff].append(arg.coeff)

        new_args = [TensMul(Add(*coeff), t).doit(deep=False) for t, coeff in terms_dict.items() if Add(*coeff) != 0]
        if isinstance(scalars, Add):
            new_args = list(scalars.args) + new_args
        elif scalars != 0:
            new_args = [scalars] + new_args
        return new_args

    def get_indices(self):
        indices = []
        for arg in self.args:
            indices.extend([i for i in get_indices(arg) if i not in indices])
        return indices


    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        index_tuples = list(zip(free_args, indices))
        a = [x.func(*x.substitute_indices(*index_tuples).args) for x in self.args]
        res = TensAdd(*a).doit(deep=False)
        return res

    def canon_bp(self):
        """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.
        """
        expr = self.expand()
        if isinstance(expr, self.func):
            args = [canon_bp(x) for x in expr.args]
            res = TensAdd(*args).doit(deep=False)
            return res
        else:
            return canon_bp(expr)

    def equals(self, other):
        other = _sympify(other)
        if isinstance(other, TensMul) and other.coeff == 0:
            return all(x.coeff == 0 for x in self.args)
        if isinstance(other, TensExpr):
            if self.rank != other.rank:
                return False
        if isinstance(other, TensAdd):
            if set(self.args) != set(other.args):
                return False
            else:
                return True
        t = self - other
        if not isinstance(t, TensExpr):
            return t == 0
        else:
            if isinstance(t, TensMul):
                return t.coeff == 0
            else:
                return all(x.coeff == 0 for x in t.args)

    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def contract_delta(self, delta):
        args = [x.contract_delta(delta) if isinstance(x, TensExpr) else x for x in self.args]
        t = TensAdd(*args).doit(deep=False)
        return canon_bp(t)

    def contract_metric(self, g):
        """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g :  metric

        contract_all : if True, eliminate all ``g`` which are contracted

        Notes
        =====

        see the ``TensorIndexType`` docstring for the contraction conventions
        """

        args = [contract_metric(x, g) for x in self.args]
        t = TensAdd(*args).doit(deep=False)
        return canon_bp(t)

    def substitute_indices(self, *index_tuples):
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        return TensAdd(*new_args).doit(deep=False)

    def _print(self):
        a = []
        args = self.args
        for x in args:
            a.append(str(x))
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _extract_data(self, replacement_dict):
        from sympy.tensor.array import Array, permutedims
        args_indices, arrays = zip(*[
            arg._extract_data(replacement_dict) if
            isinstance(arg, TensExpr) else ([], arg) for arg in self.args
        ])
        arrays = [Array(i) for i in arrays]
        ref_indices = args_indices[0]
        for i in range(1, len(args_indices)):
            indices = args_indices[i]
            array = arrays[i]
            permutation = TensMul._get_indices_permutation(indices, ref_indices)
            arrays[i] = permutedims(array, permutation)
        return ref_indices, sum(arrays, Array.zeros(*array.shape))

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self.expand()]

    @data.setter
    def data(self, data):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]

    def __iter__(self):
        deprecate_data()
        if not self.data:
            raise ValueError("No iteration on abstract tensors")
        return self.data.flatten().__iter__()

    def _eval_rewrite_as_Indexed(self, *args, **kwargs):
        return Add.fromiter(args)

    def _eval_partial_derivative(self, s):
        # Evaluation like Add
        list_addends = []
        for a in self.args:
            if isinstance(a, TensExpr):
                list_addends.append(a._eval_partial_derivative(s))
            # do not call diff if s is no symbol
            elif s._diff_wrt:
                list_addends.append(a._eval_derivative(s))

        return self.func(*list_addends).doit(deep=False)

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if not isinstance(expr, TensAdd):
            return None

        if len(_get_wilds(self)) == 0:
            return self._matches_simple(expr, repl_dict, old)

        def siftkey(arg):
            wildatoms = _get_wilds(arg)
            wildatom_types = sift(wildatoms, type)
            if len(wildatoms) == 0:
                return "nonwild"
            elif WildTensor in wildatom_types.keys():
                for w in wildatom_types["WildTensor"]:
                    if len(w.get_indices()) == 0:
                        return "indexless_wildtensor"
                return "wildtensor"
            else:
                return "otherwild"

        query_sifted = sift(self.args, siftkey)
        expr_sifted = sift(expr.args, siftkey)

        #First try to match the terms without WildTensors
        matched_e_tensors = [] #Used to make sure that the same tensor in expr is not matched with more than one tensor in self.
        for q_tensor in query_sifted["nonwild"]:
            matched_this_q = False
            for e_tensor in expr_sifted["nonwild"]:
                if e_tensor in matched_e_tensors:
                    continue

                m = q_tensor.matches(e_tensor, repl_dict=repl_dict, old=old)
                if m is None:
                    continue
                else:
                    matched_this_q = True
                    repl_dict.update(m)
                    matched_e_tensors.append(e_tensor)
                    break

            if not matched_this_q:
                return None

        remaining_e_tensors = [t for t in expr_sifted["nonwild"] if t not in matched_e_tensors]
        for w in query_sifted["otherwild"]:
            for e in remaining_e_tensors:
                m = w.matches(e)
                if m is not None:
                    matched_e_tensors.append(e)
                    if w in repl_dict.keys():
                        repl_dict[w] += m.pop(w)
                    repl_dict.update(m)

        remaining_e_tensors = [t for t in expr_sifted["nonwild"] if t not in matched_e_tensors]
        for w in query_sifted["wildtensor"]:
            for e in remaining_e_tensors:
                m = w.matches(e)
                if m is not None:
                    matched_e_tensors.append(e)
                    if w.component in repl_dict.keys():
                        repl_dict[w.component] += m.pop(w.component)
                    repl_dict.update(m)

        remaining_e_tensors = [t for t in expr_sifted["nonwild"] if t not in matched_e_tensors]
        for w in query_sifted["indexless_wildtensor"]:
            for e in remaining_e_tensors:
                m = w.matches(e)
                if m is not None:
                    matched_e_tensors.append(e)
                    if w.component in repl_dict.keys():
                        repl_dict[w.component] += m.pop(w.component)
                    repl_dict.update(m)

        remaining_e_tensors = [t for t in expr_sifted["nonwild"] if t not in matched_e_tensors]
        if len(remaining_e_tensors) > 0:
            return None
        else:
            return repl_dict


class Tensor(TensExpr):
    """
    Base tensor class, i.e. this represents a tensor, the single unit to be
    put into an expression.

    Explanation
    ===========

    This object is usually created from a ``TensorHead``, by attaching indices
    to it. Indices preceded by a minus sign are considered contravariant,
    otherwise covariant.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
    >>> Lorentz = TensorIndexType("Lorentz", dummy_name="L")
    >>> mu, nu = tensor_indices('mu nu', Lorentz)
    >>> A = TensorHead("A", [Lorentz, Lorentz])
    >>> A(mu, -nu)
    A(mu, -nu)
    >>> A(mu, -mu)
    A(L_0, -L_0)

    It is also possible to use symbols instead of inidices (appropriate indices
    are then generated automatically).

    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> A(x, mu)
    A(x, mu)
    >>> A(x, -x)
    A(L_0, -L_0)

    """

    is_commutative = False

    _index_structure: _IndexStructure
    args: tuple[TensorHead, Tuple]

    def __new__(cls, tensor_head, indices, *, is_canon_bp=False, **kw_args):
        indices = cls._parse_indices(tensor_head, indices)
        obj = Basic.__new__(cls, tensor_head, Tuple(*indices), **kw_args)
        obj._index_structure = _IndexStructure.from_indices(*indices)
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        obj._coeff = S.One
        obj._nocoeff = obj
        obj._component = tensor_head
        obj._components = [tensor_head]
        if tensor_head.rank != len(indices):
            raise ValueError("wrong number of indices")
        obj.is_canon_bp = is_canon_bp
        obj._index_map = Tensor._build_index_map(indices, obj._index_structure)
        return obj

    @property
    def free(self):
        return self._free

    @property
    def dum(self):
        return self._dum

    @property
    def ext_rank(self):
        return self._ext_rank

    @property
    def coeff(self):
        return self._coeff

    @property
    def nocoeff(self):
        return self._nocoeff

    @property
    def component(self):
        return self._component

    @property
    def components(self):
        return self._components

    @property
    def head(self):
        return self.args[0]

    @property
    def indices(self):
        return self.args[1]

    @property
    def free_indices(self):
        return set(self._index_structure.get_free_indices())

    @property
    def index_types(self):
        return self.head.index_types

    @property
    def rank(self):
        return len(self.free_indices)

    @staticmethod
    def _build_index_map(indices, index_structure):
        index_map = {}
        for idx in indices:
            index_map[idx] = (indices.index(idx),)
        return index_map

    def doit(self, **hints):
        args, indices, free, dum = TensMul._tensMul_contract_indices([self])
        return args[0]

    @staticmethod
    def _parse_indices(tensor_head, indices):
        if not isinstance(indices, (tuple, list, Tuple)):
            raise TypeError("indices should be an array, got %s" % type(indices))
        indices = list(indices)
        for i, index in enumerate(indices):
            if isinstance(index, Symbol):
                indices[i] = TensorIndex(index, tensor_head.index_types[i], True)
            elif isinstance(index, Mul):
                c, e = index.as_coeff_Mul()
                if c == -1 and isinstance(e, Symbol):
                    indices[i] = TensorIndex(e, tensor_head.index_types[i], False)
                else:
                    raise ValueError("index not understood: %s" % index)
            elif not isinstance(index, TensorIndex):
                raise TypeError("wrong type for index: %s is %s" % (index, type(index)))
        return indices

    def _set_new_index_structure(self, im, is_canon_bp=False):
        indices = im.get_indices()
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        if len(indices) != self.ext_rank:
            raise ValueError("indices length mismatch")
        return self.func(self.args[0], indices, is_canon_bp=is_canon_bp).doit()

    def _get_free_indices_set(self):
        return {i[0] for i in self._index_structure.free}

    def _get_dummy_indices_set(self):
        dummy_pos = set(itertools.chain(*self._index_structure.dum))
        return {idx for i, idx in enumerate(self.args[1]) if i in dummy_pos}

    def _get_indices_set(self):
        return set(self.args[1].args)

    @property
    def free_in_args(self):
        return [(ind, pos, 0) for ind, pos in self.free]

    @property
    def dum_in_args(self):
        return [(p1, p2, 0, 0) for p1, p2 in self.dum]

    @property
    def free_args(self):
        return sorted([x[0] for x in self.free])

    def commutes_with(self, other):
        """
        :param other:
        :return:
            0  commute
            1  anticommute
            None  neither commute nor anticommute
        """
        if not isinstance(other, TensExpr):
            return 0
        elif isinstance(other, Tensor):
            return self.component.commutes_with(other.component)
        return NotImplementedError

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns the tensor corresponding to the permutation ``g``.

        For further details, see the method in ``TIDS`` with the same name.
        """
        return perm2tensor(self, g, is_canon_bp)

    def canon_bp(self):
        if self.is_canon_bp:
            return self
        expr = self.expand()
        g, dummies, msym = expr._index_structure.indices_canon_args()
        v = components_canon_args([expr.component])
        can = canonicalize(g, dummies, msym, *v)
        if can == 0:
            return S.Zero
        tensor = self.perm2tensor(can, True)
        return tensor

    def split(self):
        return [self]

    def sorted_components(self):
        return self

    def get_indices(self) -> list[TensorIndex]:
        """
        Get a list of indices, corresponding to those of the tensor.
        """
        return list(self.args[1])

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Get a list of free indices, corresponding to those of the tensor.
        """
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        # TODO: this could be optimized by only swapping the indices
        # instead of visiting the whole expression tree:
        return self.xreplace(repl)

    def as_base_exp(self):
        return self, S.One

    def substitute_indices(self, *index_tuples):
        """
        Return a tensor with free indices substituted according to ``index_tuples``.

        ``index_types`` list of tuples ``(old_index, new_index)``.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> i, j, k, l = tensor_indices('i,j,k,l', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(i, k)*B(-k, -j); t
        A(i, L_0)*B(-L_0, -j)
        >>> t.substitute_indices((i, k),(-j, l))
        A(k, L_0)*B(-L_0, l)
        """
        indices = []
        for index in self.indices:
            for ind_old, ind_new in index_tuples:
                if (index.name == ind_old.name and index.tensor_index_type ==
                                                   ind_old.tensor_index_type):
                    if index.is_up == ind_old.is_up:
                        indices.append(ind_new)
                    else:
                        indices.append(-ind_new)
                    break
            else:
                indices.append(index)
        return self.head(*indices)

    def _get_symmetrized_forms(self):
        """
        Return a list giving all possible permutations of self that are allowed by its symmetries.
        """
        comp = self.component
        gens = comp.symmetry.generators
        rank = comp.rank

        old_perms = None
        new_perms = {self}
        while new_perms != old_perms:
            old_perms = new_perms.copy()
            for tens in old_perms:
                for gen in gens:
                    inds = tens.get_indices()
                    per = [gen.apply(i) for i in range(0,rank)]
                    sign = (-1)**(gen.apply(rank) - rank)
                    ind_map = dict(zip(inds, [inds[i] for i in per]))
                    new_perms.add( sign * tens._replace_indices(ind_map) )

        return new_perms

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        #simple checks
        if self == expr:
            return repl_dict
        if not isinstance(expr, Tensor):
            return None
        if self.head != expr.head:
            return None

        #Now consider all index symmetries of expr, and see if any of them allow a match.
        for new_expr in expr._get_symmetrized_forms():
            m = self._matches(new_expr, repl_dict, old=old)
            if m is not None:
                repl_dict.update(m)
                return repl_dict

        return None

    def _matches(self, expr, repl_dict=None, old=False):
        """
        This does not account for index symmetries of expr
        """
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        #simple checks
        if self == expr:
            return repl_dict
        if not isinstance(expr, Tensor):
            return None
        if self.head != expr.head:
            return None

        s_indices = self.get_indices()
        e_indices = expr.get_indices()

        if len(s_indices) != len(e_indices):
            return None

        for i in range(len(s_indices)):
            s_ind = s_indices[i]
            m = s_ind.matches(e_indices[i])
            if m is None:
                return None
            elif -s_ind in repl_dict.keys() and -repl_dict[-s_ind] != m[s_ind]:
                return None
            else:
                repl_dict.update(m)

        return repl_dict

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        t = self.substitute_indices(*list(zip(free_args, indices)))

        # object is rebuilt in order to make sure that all contracted indices
        # get recognized as dummies, but only if there are contracted indices.
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        return t

    # TODO: put this into TensExpr?
    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data.__iter__()

    # TODO: put this into TensExpr?
    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def _extract_data(self, replacement_dict):
        from .array import Array
        for k, v in replacement_dict.items():
            if isinstance(k, Tensor) and k.args[0] == self.args[0]:
                other = k
                array = v
                break
        else:
            raise ValueError("%s not found in %s" % (self, replacement_dict))

        # TODO: inefficient, this should be done at root level only:
        replacement_dict = {k: Array(v) for k, v in replacement_dict.items()}
        array = Array(array)

        dum1 = self.dum
        dum2 = other.dum

        if len(dum2) > 0:
            for pair in dum2:
                # allow `dum2` if the contained values are also in `dum1`.
                if pair not in dum1:
                    raise NotImplementedError("%s with contractions is not implemented" % other)
            # Remove elements in `dum2` from `dum1`:
            dum1 = [pair for pair in dum1 if pair not in dum2]
        if len(dum1) > 0:
            indices1 = self.get_indices()
            indices2 = other.get_indices()
            repl = {}
            for p1, p2 in dum1:
                repl[indices2[p2]] = -indices2[p1]
                for pos in (p1, p2):
                    if indices1[pos].is_up ^ indices2[pos].is_up:
                        metric = replacement_dict[indices1[pos].tensor_index_type]
                        if indices1[pos].is_up:
                            metric = _TensorDataLazyEvaluator.inverse_matrix(metric)
                        array = self._contract_and_permute_with_metric(metric, array, pos, len(indices2))
            other = other.xreplace(repl).doit()
            array = _TensorDataLazyEvaluator.data_contract_dum([array], dum1, len(indices2))

        free_ind1 = self.get_free_indices()
        free_ind2 = other.get_free_indices()

        return self._match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        # TODO: check data compatibility with properties of tensor.
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]
            if self.metric in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self.metric]

    def _print(self):
        indices = [str(ind) for ind in self.indices]
        component = self.component
        if component.rank > 0:
            return ('%s(%s)' % (component.name, ', '.join(indices)))
        else:
            return ('%s' % component.name)

    def equals(self, other):
        if other == 0:
            return self.coeff == 0
        other = _sympify(other)
        if not isinstance(other, TensExpr):
            assert not self.components
            return S.One == other

        def _get_compar_comp(self):
            t = self.canon_bp()
            r = (t.coeff, tuple(t.components), \
                    tuple(sorted(t.free)), tuple(sorted(t.dum)))
            return r

        return _get_compar_comp(self) == _get_compar_comp(other)

    def contract_metric(self, g):
        # if metric is not the same, ignore this step:
        if self.component != g:
            return self
        # in case there are free components, do not perform anything:
        if len(self.free) != 0:
            return self

        #antisym = g.index_types[0].metric_antisym
        if g.symmetry == TensorSymmetry.fully_symmetric(-2):
            antisym = 1
        elif g.symmetry == TensorSymmetry.fully_symmetric(2):
            antisym = 0
        elif g.symmetry == TensorSymmetry.no_symmetry(2):
            antisym = None
        else:
            raise NotImplementedError
        sign = S.One
        typ = g.index_types[0]

        if not antisym:
            # g(i, -i)
            sign = sign*typ.dim
        else:
            # g(i, -i)
            sign = sign*typ.dim

            dp0, dp1 = self.dum[0]
            if dp0 < dp1:
                # g(i, -i) = -D with antisymmetric metric
                sign = -sign

        return sign

    def contract_delta(self, metric):
        return self.contract_metric(metric)

    def _eval_rewrite_as_Indexed(self, tens, indices, **kwargs):
        from sympy.tensor.indexed import Indexed
        # TODO: replace .args[0] with .name:
        index_symbols = [i.args[0] for i in self.get_indices()]
        expr = Indexed(tens.args[0], *index_symbols)
        return self._check_add_Sum(expr, index_symbols)

    def _eval_partial_derivative(self, s: Tensor) -> Expr:

        if not isinstance(s, Tensor):
            return S.Zero
        else:

            # @a_i/@a_k = delta_i^k
            # @a_i/@a^k = g_ij delta^j_k
            # @a^i/@a^k = delta^i_k
            # @a^i/@a_k = g^ij delta_j^k
            # TODO: if there is no metric present, the derivative should be zero?

            if self.head != s.head:
                return S.Zero

            # if heads are the same, provide delta and/or metric products
            # for every free index pair in the appropriate tensor
            # assumed that the free indices are in proper order
            # A contravariante index in the derivative becomes covariant
            # after performing the derivative and vice versa

            kronecker_delta_list = [1]

            # not guarantee a correct index order

            for (count, (iself, iother)) in enumerate(zip(self.get_free_indices(), s.get_free_indices())):
                if iself.tensor_index_type != iother.tensor_index_type:
                    raise ValueError("index types not compatible")
                else:
                    tensor_index_type = iself.tensor_index_type
                    tensor_metric = tensor_index_type.metric
                    dummy = TensorIndex("d_" + str(count), tensor_index_type,
                                        is_up=iself.is_up)
                    if iself.is_up == iother.is_up:
                        kroneckerdelta = tensor_index_type.delta(iself, -iother)
                    else:
                        kroneckerdelta = (
                            TensMul(tensor_metric(iself, dummy),
                                    tensor_index_type.delta(-dummy, -iother))
                        )
                    kronecker_delta_list.append(kroneckerdelta)
            return TensMul.fromiter(kronecker_delta_list).doit(deep=False)
            # doit necessary to rename dummy indices accordingly


class TensMul(TensExpr, AssocOp):
    """
    Product of tensors.

    Parameters
    ==========

    coeff : SymPy coefficient of the tensor
    args

    Attributes
    ==========

    ``components`` : list of ``TensorHead`` of the component tensors
    ``types`` : list of nonrepeated ``TensorIndexType``
    ``free`` : list of ``(ind, ipos, icomp)``, see Notes
    ``dum`` : list of ``(ipos1, ipos2, icomp1, icomp2)``, see Notes
    ``ext_rank`` : rank of the tensor counting the dummy indices
    ``rank`` : rank of the tensor
    ``coeff`` : SymPy coefficient of the tensor
    ``free_args`` : list of the free indices in sorted order
    ``is_canon_bp`` : ``True`` if the tensor in in canonical form

    Notes
    =====

    ``args[0]``   list of ``TensorHead`` of the component tensors.

    ``args[1]``   list of ``(ind, ipos, icomp)``
    where ``ind`` is a free index, ``ipos`` is the slot position
    of ``ind`` in the ``icomp``-th component tensor.

    ``args[2]`` list of tuples representing dummy indices.
    ``(ipos1, ipos2, icomp1, icomp2)`` indicates that the contravariant
    dummy index is the ``ipos1``-th slot position in the ``icomp1``-th
    component tensor; the corresponding covariant index is
    in the ``ipos2`` slot position in the ``icomp2``-th component tensor.

    """
    identity = S.One

    _index_structure: _IndexStructure

    def __new__(cls, *args, **kw_args):
        is_canon_bp = kw_args.get('is_canon_bp', False)
        args = list(map(_sympify, args))

        """
        If the internal dummy indices in one arg conflict with the free indices
        of the remaining args, we need to rename those internal dummy indices.
        """
        free = [get_free_indices(arg) for arg in args]
        free = set(itertools.chain(*free)) #flatten free
        newargs = []
        for arg in args:
            dum_this = set(get_dummy_indices(arg))
            dum_other = [get_dummy_indices(a) for a in newargs]
            dum_other = set(itertools.chain(*dum_other)) #flatten dum_other
            free_this = set(get_free_indices(arg))
            if len(dum_this.intersection(free)) > 0:
                exclude = free_this.union(free, dum_other)
                newarg = TensMul._dedupe_indices(arg, exclude)
            else:
                newarg = arg
            newargs.append(newarg)

        args = newargs

        # Flatten:
        args = [i for arg in args for i in (arg.args if isinstance(arg, (TensMul, Mul)) else [arg])]

        args, indices, free, dum = TensMul._tensMul_contract_indices(args, replace_indices=False)

        # Data for indices:
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)

        obj = TensExpr.__new__(cls, *args)
        obj._indices = indices
        obj._index_types = index_types.copy()
        obj._index_structure = index_structure
        obj._free = index_structure.free[:]
        obj._dum = index_structure.dum[:]
        obj._free_indices = {x[0] for x in obj.free}
        obj._rank = len(obj.free)
        obj._ext_rank = len(obj._index_structure.free) + 2*len(obj._index_structure.dum)
        obj._coeff = S.One
        obj._is_canon_bp = is_canon_bp
        return obj

    index_types = property(lambda self: self._index_types)
    free = property(lambda self: self._free)
    dum = property(lambda self: self._dum)
    free_indices = property(lambda self: self._free_indices)
    rank = property(lambda self: self._rank)
    ext_rank = property(lambda self: self._ext_rank)

    @staticmethod
    def _indices_to_free_dum(args_indices):
        free2pos1 = {}
        free2pos2 = {}
        dummy_data = []
        indices = []

        # Notation for positions (to better understand the code):
        # `pos1`: position in the `args`.
        # `pos2`: position in the indices.

        # Example:
        # A(i, j)*B(k, m, n)*C(p)
        # `pos1` of `n` is 1 because it's in `B` (second `args` of TensMul).
        # `pos2` of `n` is 4 because it's the fifth overall index.

        # Counter for the index position wrt the whole expression:
        pos2 = 0

        for pos1, arg_indices in enumerate(args_indices):

            for index in arg_indices:
                if not isinstance(index, TensorIndex):
                    raise TypeError("expected TensorIndex")
                if -index in free2pos1:
                    # Dummy index detected:
                    other_pos1 = free2pos1.pop(-index)
                    other_pos2 = free2pos2.pop(-index)
                    if index.is_up:
                        dummy_data.append((index, pos1, other_pos1, pos2, other_pos2))
                    else:
                        dummy_data.append((-index, other_pos1, pos1, other_pos2, pos2))
                    indices.append(index)
                elif index in free2pos1:
                    raise ValueError("Repeated index: %s" % index)
                else:
                    free2pos1[index] = pos1
                    free2pos2[index] = pos2
                    indices.append(index)
                pos2 += 1

        free = list(free2pos2.items())
        free_names = [i.name for i in free2pos2.keys()]

        dummy_data.sort(key=lambda x: x[3])
        return indices, free, free_names, dummy_data

    @staticmethod
    def _dummy_data_to_dum(dummy_data):
        return [(p2a, p2b) for (i, p1a, p1b, p2a, p2b) in dummy_data]

    @staticmethod
    def _tensMul_contract_indices(args, replace_indices=True):
        replacements = [{} for _ in args]

        #_index_order = all(_has_index_order(arg) for arg in args)

        args_indices = [get_indices(arg) for arg in args]
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)

        cdt = defaultdict(int)

        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd

        if replace_indices:
            for old_index, pos1cov, pos1contra, pos2cov, pos2contra in dummy_data:
                index_type = old_index.tensor_index_type
                while True:
                    dummy_name = dummy_name_gen(index_type)
                    if dummy_name not in free_names:
                        break
                dummy = old_index.func(dummy_name, index_type, *old_index.args[2:])
                replacements[pos1cov][old_index] = dummy
                replacements[pos1contra][-old_index] = -dummy
                indices[pos2cov] = dummy
                indices[pos2contra] = -dummy
            args = [
                arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg
                for arg, repl in zip(args, replacements)]

            """
            The order of indices might've changed due to the replacements (e.g. if one of the args is a TensAdd, replacing an index can change the sort order of the terms, thus changing the order of indices returned by its get_indices() method).
            To stay on the safe side, we calculate these quantities again.
            """
            args_indices = [get_indices(arg) for arg in args]
            indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)

        dum = TensMul._dummy_data_to_dum(dummy_data)
        return args, indices, free, dum

    @staticmethod
    def _get_components_from_args(args):
        """
        Get a list of ``Tensor`` objects having the same ``TIDS`` if multiplied
        by one another.
        """
        components = []
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            if isinstance(arg, TensAdd):
                continue
            components.extend(arg.components)
        return components

    @staticmethod
    def _rebuild_tensors_list(args, index_structure):
        indices = index_structure.get_indices()
        #tensors = [None for i in components]  # pre-allocate list
        ind_pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            prev_pos = ind_pos
            ind_pos += arg.ext_rank
            args[i] = Tensor(arg.component, indices[prev_pos:ind_pos])

    def doit(self, **hints):
        is_canon_bp = self._is_canon_bp
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]

            """
            There may now be conflicts between dummy indices of different args
            (each arg's doit method does not have any information about which
            dummy indices are already used in the other args), so we
            deduplicate them.
            """
            rule = dict(zip(self.args, args))
            rule = self._dedupe_indices_in_rule(rule)
            args = [rule[a] for a in self.args]

        else:
            args = self.args

        args = [arg for arg in args if arg != self.identity]

        # Extract non-tensor coefficients:
        coeff = reduce(lambda a, b: a*b, [arg for arg in args if not isinstance(arg, TensExpr)], S.One)
        args = [arg for arg in args if isinstance(arg, TensExpr)]

        if len(args) == 0:
            return coeff

        if coeff != self.identity:
            args = [coeff] + args
        if coeff == 0:
            return S.Zero

        if len(args) == 1:
            return args[0]

        args, indices, free, dum = TensMul._tensMul_contract_indices(args)

        # Data for indices:
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)

        obj = self.func(*args)
        obj._index_types = index_types
        obj._index_structure = index_structure
        obj._ext_rank = len(obj._index_structure.free) + 2*len(obj._index_structure.dum)
        obj._coeff = coeff
        obj._is_canon_bp = is_canon_bp
        return obj

    # TODO: this method should be private
    # TODO: should this method be renamed _from_components_free_dum ?
    @staticmethod
    def from_data(coeff, components, free, dum, **kw_args):
        return TensMul(coeff, *TensMul._get_tensors_from_components_free_dum(components, free, dum), **kw_args).doit(deep=False)

    @staticmethod
    def _get_tensors_from_components_free_dum(components, free, dum):
        """
        Get a list of ``Tensor`` objects by distributing ``free`` and ``dum`` indices on the ``components``.
        """
        index_structure = _IndexStructure.from_components_free_dum(components, free, dum)
        indices = index_structure.get_indices()
        tensors = [None for i in components]  # pre-allocate list

        # distribute indices on components to build a list of tensors:
        ind_pos = 0
        for i, component in enumerate(components):
            prev_pos = ind_pos
            ind_pos += component.rank
            tensors[i] = Tensor(component, indices[prev_pos:ind_pos])
        return tensors

    def _get_free_indices_set(self):
        return {i[0] for i in self.free}

    def _get_dummy_indices_set(self):
        dummy_pos = set(itertools.chain(*self.dum))
        return {idx for i, idx in enumerate(self._index_structure.get_indices()) if i in dummy_pos}

    def _get_position_offset_for_indices(self):
        arg_offset = [None for i in range(self.ext_rank)]
        counter = 0
        for arg in self.args:
            if not isinstance(arg, TensExpr):
                continue
            for j in range(arg.ext_rank):
                arg_offset[j + counter] = counter
            counter += arg.ext_rank
        return arg_offset

    @property
    def free_args(self):
        return sorted([x[0] for x in self.free])

    @property
    def components(self):
        return self._get_components_from_args(self.args)

    @property
    def free_in_args(self):
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        return [(ind, pos-arg_offset[pos], argpos[pos]) for (ind, pos) in self.free]

    @property
    def coeff(self):
        # return Mul.fromiter([c for c in self.args if not isinstance(c, TensExpr)])
        return self._coeff

    @property
    def nocoeff(self):
        return self.func(*self.args, 1/self.coeff).doit(deep=False)

    @property
    def dum_in_args(self):
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        return [(p1-arg_offset[p1], p2-arg_offset[p2], argpos[p1], argpos[p2]) for p1, p2 in self.dum]

    def equals(self, other):
        if other == 0:
            return self.coeff == 0
        other = _sympify(other)
        if not isinstance(other, TensExpr):
            assert not self.components
            return self.coeff == other

        return self.canon_bp() == other.canon_bp()

    def get_indices(self):
        """
        Returns the list of indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.
        The dummy indices are given a name which does not collide with
        the names of the free indices.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_indices()
        [L_0, -L_0, m2]
        """
        return self._indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Returns the list of free indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_free_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_free_indices()
        [m2]
        """
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        return self.func(*[arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args])

    def split(self):
        """
        Returns a list of tensors, whose product is ``self``.

        Explanation
        ===========

        Dummy indices contracted among different tensor components
        become free indices with the same name as the one used to
        represent the dummy indices.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(a,b)*B(-b,c)
        >>> t
        A(a, L_0)*B(-L_0, c)
        >>> t.split()
        [A(a, L_0), B(-L_0, c)]
        """
        if self.args == ():
            return [self]
        splitp = []
        res = 1
        for arg in self.args:
            if isinstance(arg, Tensor):
                splitp.append(res*arg)
                res = 1
            else:
                res *= arg
        return splitp

    def _eval_expand_mul(self, **hints):
        args1 = [arg.args if isinstance(arg, (Add, TensAdd)) else (arg,) for arg in self.args]
        return TensAdd(*[
            TensMul(*i).doit(deep=False) for i in itertools.product(*args1)]
        )

    def __neg__(self):
        return TensMul(S.NegativeOne, self, is_canon_bp=self._is_canon_bp).doit(deep=False)

    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def _get_args_for_traditional_printer(self):
        args = list(self.args)
        if self.coeff.could_extract_minus_sign():
            # expressions like "-A(a)"
            sign = "-"
            if args[0] == S.NegativeOne:
                args = args[1:]
            else:
                args[0] = -args[0]
        else:
            sign = ""
        return sign, args

    def _sort_args_for_sorted_components(self):
        """
        Returns the ``args`` sorted according to the components commutation
        properties.

        Explanation
        ===========

        The sorting is done taking into account the commutation group
        of the component tensors.
        """
        cv = [arg for arg in self.args if isinstance(arg, TensExpr)]
        sign = 1
        n = len(cv) - 1
        for i in range(n):
            for j in range(n, i, -1):
                c = cv[j-1].commutes_with(cv[j])
                # if `c` is `None`, it does neither commute nor anticommute, skip:
                if c not in (0, 1):
                    continue
                typ1 = sorted(set(cv[j-1].component.index_types), key=lambda x: x.name)
                typ2 = sorted(set(cv[j].component.index_types), key=lambda x: x.name)
                if (typ1, cv[j-1].component.name) > (typ2, cv[j].component.name):
                    cv[j-1], cv[j] = cv[j], cv[j-1]
                    # if `c` is 1, the anticommute, so change sign:
                    if c:
                        sign = -sign

        coeff = sign * self.coeff
        if coeff != 1:
            return [coeff] + cv
        return cv

    def sorted_components(self):
        """
        Returns a tensor product with sorted components.
        """
        return TensMul(*self._sort_args_for_sorted_components()).doit(deep=False)

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns the tensor corresponding to the permutation ``g``

        For further details, see the method in ``TIDS`` with the same name.
        """
        return perm2tensor(self, g, is_canon_bp=is_canon_bp)

    def canon_bp(self):
        """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        >>> t = A(m0,-m1)*A(m1,-m0)
        >>> t.canon_bp()
        -A(L_0, L_1)*A(-L_0, -L_1)
        >>> t = A(m0,-m1)*A(m1,-m2)*A(m2,-m0)
        >>> t.canon_bp()
        0
        """
        if self._is_canon_bp:
            return self
        expr = self.expand()
        if isinstance(expr, TensAdd):
            return expr.canon_bp()
        if not expr.components:
            return expr
        expr = expr.doit(deep=False) #make sure self.coeff is populated correctly
        t = expr.sorted_components()
        g, dummies, msym = t._index_structure.indices_canon_args()
        v = components_canon_args(t.components)
        can = canonicalize(g, dummies, msym, *v)
        if can == 0:
            return S.Zero
        tmul = t.perm2tensor(can, True)
        return tmul

    def contract_delta(self, delta):
        t = self.contract_metric(delta)
        return t

    def _get_indices_to_args_pos(self):
        """
        Get a dict mapping the index position to TensMul's argument number.
        """
        pos_map = {}
        pos_counter = 0
        for arg_i, arg in enumerate(self.args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            for i in range(arg.ext_rank):
                pos_map[pos_counter] = arg_i
                pos_counter += 1
        return pos_map

    def contract_metric(self, g):
        """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g : metric

        Notes
        =====

        See the ``TensorIndexType`` docstring for the contraction conventions.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m0)*q(m1)*g(-m0, -m1)
        >>> t.canon_bp()
        metric(L_0, L_1)*p(-L_0)*q(-L_1)
        >>> t.contract_metric(g).canon_bp()
        p(L_0)*q(-L_0)
        """
        expr = self.expand().doit(deep=False)
        if self != expr:
            expr = canon_bp(expr)
            return contract_metric(expr, g)
        pos_map = self._get_indices_to_args_pos()
        args = list(self.args)

        #antisym = g.index_types[0].metric_antisym
        if g.symmetry == TensorSymmetry.fully_symmetric(-2):
            antisym = 1
        elif g.symmetry == TensorSymmetry.fully_symmetric(2):
            antisym = 0
        elif g.symmetry == TensorSymmetry.no_symmetry(2):
            antisym = None
        else:
            raise NotImplementedError

        # list of positions of the metric ``g`` inside ``args``
        gpos = [i for i, x in enumerate(self.args) if isinstance(x, Tensor) and x.component == g]
        if not gpos:
            return self

        # Sign is either 1 or -1, to correct the sign after metric contraction
        # (for spinor indices).
        sign = 1
        dum = self.dum[:]
        free = self.free[:]
        elim = set()
        for gposx in gpos:
            if gposx in elim:
                continue
            free1 = [x for x in free if pos_map[x[1]] == gposx]
            dum1 = [x for x in dum if pos_map[x[0]] == gposx or pos_map[x[1]] == gposx]
            if not dum1:
                continue
            elim.add(gposx)
            # subs with the multiplication neutral element, that is, remove it:
            args[gposx] = 1
            if len(dum1) == 2:
                if not antisym:
                    dum10, dum11 = dum1
                    if pos_map[dum10[1]] == gposx:
                        # the index with pos p0 contravariant
                        p0 = dum10[0]
                    else:
                        # the index with pos p0 is covariant
                        p0 = dum10[1]
                    if pos_map[dum11[1]] == gposx:
                        # the index with pos p1 is contravariant
                        p1 = dum11[0]
                    else:
                        # the index with pos p1 is covariant
                        p1 = dum11[1]

                    dum.append((p0, p1))
                else:
                    dum10, dum11 = dum1
                    # change the sign to bring the indices of the metric to contravariant
                    # form; change the sign if dum10 has the metric index in position 0
                    if pos_map[dum10[1]] == gposx:
                        # the index with pos p0 is contravariant
                        p0 = dum10[0]
                        if dum10[1] == 1:
                            sign = -sign
                    else:
                        # the index with pos p0 is covariant
                        p0 = dum10[1]
                        if dum10[0] == 0:
                            sign = -sign
                    if pos_map[dum11[1]] == gposx:
                        # the index with pos p1 is contravariant
                        p1 = dum11[0]
                        sign = -sign
                    else:
                        # the index with pos p1 is covariant
                        p1 = dum11[1]

                    dum.append((p0, p1))

            elif len(dum1) == 1:
                if not antisym:
                    dp0, dp1 = dum1[0]
                    if pos_map[dp0] == pos_map[dp1]:
                        # g(i, -i)
                        typ = g.index_types[0]
                        sign = sign*typ.dim

                    else:
                        # g(i0, i1)*p(-i1)
                        if pos_map[dp0] == gposx:
                            p1 = dp1
                        else:
                            p1 = dp0

                        ind, p = free1[0]
                        free.append((ind, p1))
                else:
                    dp0, dp1 = dum1[0]
                    if pos_map[dp0] == pos_map[dp1]:
                        # g(i, -i)
                        typ = g.index_types[0]
                        sign = sign*typ.dim

                        if dp0 < dp1:
                            # g(i, -i) = -D with antisymmetric metric
                            sign = -sign
                    else:
                        # g(i0, i1)*p(-i1)
                        if pos_map[dp0] == gposx:
                            p1 = dp1
                            if dp0 == 0:
                                sign = -sign
                        else:
                            p1 = dp0
                        ind, p = free1[0]
                        free.append((ind, p1))
            dum = [x for x in dum if x not in dum1]
            free = [x for x in free if x not in free1]

        # shift positions:
        shift = 0
        shifts = [0]*len(args)
        for i in range(len(args)):
            if i in elim:
                shift += 2
                continue
            shifts[i] = shift
        free = [(ind, p - shifts[pos_map[p]]) for (ind, p) in free if pos_map[p] not in elim]
        dum = [(p0 - shifts[pos_map[p0]], p1 - shifts[pos_map[p1]]) for p0, p1 in dum if pos_map[p0] not in elim and pos_map[p1] not in elim]

        res = ( sign*TensMul(*args) ).doit(deep=False)
        if not isinstance(res, TensExpr):
            return res
        im = _IndexStructure.from_components_free_dum(res.components, free, dum)
        return res._set_new_index_structure(im)

    def _set_new_index_structure(self, im, is_canon_bp=False):
        indices = im.get_indices()
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        if len(indices) != self.ext_rank:
            raise ValueError("indices length mismatch")
        args = list(self.args)
        pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            ext_rank = arg.ext_rank
            args[i] = arg._set_indices(*indices[pos:pos+ext_rank])
            pos += ext_rank
        return TensMul(*args, is_canon_bp=is_canon_bp).doit(deep=False)

    @staticmethod
    def _index_replacement_for_contract_metric(args, free, dum):
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)

    def substitute_indices(self, *index_tuples):
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        return TensMul(*new_args).doit(deep=False)

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        t = self.substitute_indices(*list(zip(free_args, indices)))

        # object is rebuilt in order to make sure that all contracted indices
        # get recognized as dummies, but only if there are contracted indices.
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        return t

    def _extract_data(self, replacement_dict):
        args_indices, arrays = zip(*[arg._extract_data(replacement_dict) for arg in self.args if isinstance(arg, TensExpr)])
        coeff = reduce(operator.mul, [a for a in self.args if not isinstance(a, TensExpr)], S.One)
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)
        dum = TensMul._dummy_data_to_dum(dummy_data)
        ext_rank = self.ext_rank
        free.sort(key=lambda x: x[1])
        free_indices = [i[0] for i in free]
        return free_indices, coeff*_TensorDataLazyEvaluator.data_contract_dum(arrays, dum, ext_rank)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            dat = _tensor_data_substitution_dict[self.expand()]
        return dat

    @data.setter
    def data(self, data):
        deprecate_data()
        raise ValueError("Not possible to set component data to a tensor expression")

    @data.deleter
    def data(self):
        deprecate_data()
        raise ValueError("Not possible to delete component data to a tensor expression")

    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError("No iteration on abstract tensors")
            return self.data.__iter__()

    @staticmethod
    def _dedupe_indices(new, exclude):
        """
        exclude: set
        new: TensExpr

        If ``new`` has any dummy indices that are in ``exclude``, return a version
        of new with those indices replaced. If no replacements are needed,
        return None

        """
        exclude = set(exclude)
        dums_new = set(get_dummy_indices(new))
        free_new = set(get_free_indices(new))

        conflicts = dums_new.intersection(exclude)
        if len(conflicts) == 0:
            return None

        """
        ``exclude_for_gen`` is to be passed to ``_IndexStructure._get_generator_for_dummy_indices()``.
        Since the latter does not use the index position for anything, we just
        set it as ``None`` here.
        """
        exclude.update(dums_new)
        exclude.update(free_new)
        exclude_for_gen = [(i, None) for i in exclude]
        gen = _IndexStructure._get_generator_for_dummy_indices(exclude_for_gen)
        repl = {}
        for d in conflicts:
            if -d in repl.keys():
                continue
            newname = gen(d.tensor_index_type)
            new_d = d.func(newname, *d.args[1:])
            repl[d] = new_d
            repl[-d] = -new_d

        if len(repl) == 0:
            return None

        new_renamed = new._replace_indices(repl)
        return new_renamed

    def _dedupe_indices_in_rule(self, rule):
        """
        rule: dict

        This applies TensMul._dedupe_indices on all values of rule.

        """
        index_rules = {k:v for k,v in rule.items() if isinstance(k, TensorIndex)}
        other_rules = {k:v for k,v in rule.items() if k not in index_rules.keys()}
        exclude = set(self.get_indices())

        newrule = {}
        newrule.update(index_rules)
        exclude.update(index_rules.keys())
        exclude.update(index_rules.values())
        for old, new in other_rules.items():
            new_renamed = TensMul._dedupe_indices(new, exclude)
            if old == new or new_renamed is None:
                newrule[old] = new
            else:
                newrule[old] = new_renamed
                exclude.update(get_indices(new_renamed))
        return newrule

    def _eval_subs(self, old, new):
        """
        If new is an index which is already present in self as a dummy, the dummies in self should be renamed.
        """

        if not isinstance(new, TensorIndex):
            return None

        exclude = {new}
        self_renamed = self._dedupe_indices(self, exclude)
        if self_renamed is None:
            return None
        else:
            return self_renamed._subs(old, new).doit(deep=False)

    def _eval_rewrite_as_Indexed(self, *args, **kwargs):
        from sympy.concrete.summations import Sum
        index_symbols = [i.args[0] for i in self.get_indices()]
        args = [arg.args[0] if isinstance(arg, Sum) else arg for arg in args]
        expr = Mul.fromiter(args)
        return self._check_add_Sum(expr, index_symbols)

    def _eval_partial_derivative(self, s):
        # Evaluation like Mul
        terms = []
        for i, arg in enumerate(self.args):
            # checking whether some tensor instance is differentiated
            # or some other thing is necessary, but ugly
            if isinstance(arg, TensExpr):
                d = arg._eval_partial_derivative(s)
            else:
                # do not call diff is s is no symbol
                if s._diff_wrt:
                    d = arg._eval_derivative(s)
                else:
                    d = S.Zero
            if d:
                terms.append(TensMul.fromiter(self.args[:i] + (d,) + self.args[i + 1:]).doit(deep=False))
        return TensAdd.fromiter(terms).doit(deep=False)


    def _matches_commutative(self, expr, repl_dict=None, old=False):
        """
        Match assuming all tensors commute. But note that we are not assuming anything about their symmetry under index permutations.
        """
        #Take care of the various possible types for expr.
        if not isinstance(expr, TensMul):
            if isinstance(expr, (TensExpr, Expr)):
                expr = TensMul(expr)
            else:
                return None

        #The code that follows assumes expr is a TensMul

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

            #Make sure that none of the dummy indices in self, expr conflict with the values already present in repl_dict. This may happen due to automatic index relabelling when rem_query and rem_expr are formed later on in this function (it calls itself recursively).
            indices = [k for k in repl_dict.values() if isinstance(k ,TensorIndex)]

            def dedupe(expr):
                renamed = TensMul._dedupe_indices(expr, indices)
                if renamed is not None:
                    return renamed
                else:
                    return expr

            self = dedupe(self)
            expr = dedupe(expr)

        #Find the non-tensor part of expr. This need not be the same as expr.coeff when expr.doit() has not been called.
        expr_coeff = reduce(lambda a, b: a*b, [arg for arg in expr.args if not isinstance(arg, TensExpr)], S.One)

        # handle simple patterns
        if self == expr:
            return repl_dict

        if len(_get_wilds(self)) == 0:
            return self._matches_simple(expr, repl_dict, old)

        def siftkey(arg):
            if isinstance(arg, WildTensor):
                return "WildTensor"
            elif isinstance(arg, (Tensor, TensExpr)):
                return "Tensor"
            else:
                return "coeff"

        query_sifted = sift(self.args, siftkey)
        expr_sifted = sift(expr.args, siftkey)

        #Sanity checks
        if "coeff" in query_sifted.keys():
            if TensMul(*query_sifted["coeff"]).doit(deep=False) != self.coeff:
                raise NotImplementedError(f"Found something that we do not know to handle: {query_sifted['coeff']}")
        if "coeff" in expr_sifted.keys():
            if TensMul(*expr_sifted["coeff"]).doit(deep=False) != expr_coeff:
                raise NotImplementedError(f"Found something that we do not know to handle: {expr_sifted['coeff']}")

        query_tens_heads = {tuple(getattr(x, "components", [])) for x in query_sifted["Tensor"]} #We use getattr because, e.g. TensAdd does not have the 'components' attribute.
        expr_tens_heads = {tuple(getattr(x, "components", [])) for x in expr_sifted["Tensor"]}
        if not query_tens_heads.issubset(expr_tens_heads):
            #Some tensorheads in self are not present in the expr
            return None

        #Try to match all non-wild tensors of self with tensors that compose expr
        if len(query_sifted["Tensor"]) > 0:
            q_tensor = query_sifted["Tensor"][0]
            """
            We need to iterate over all possible symmetrized forms of q_tensor since the matches given by some of them may map dummy indices to free indices; the information about which indices are dummy/free will only be available later, when we are doing rem_q.matches(rem_e)
            """
            for q_tens in q_tensor._get_symmetrized_forms():
                for e in expr_sifted["Tensor"]:
                    if isinstance(q_tens, TensMul):
                        #q_tensor got a minus sign due to this permutation.
                        sign = -1
                    else:
                        sign = 1

                    """
                    _matches is used here since we are already iterating over index permutations of q_tensor. Also note that the sign is removed from q_tensor, and will later be put into rem_q.
                    """
                    m = (sign*q_tens)._matches(e)
                    if m is None:
                        continue

                    rem_query = self.func(sign, *[a for a in self.args if a != q_tensor]).doit(deep=False)
                    rem_expr = expr.func(*[a for a in expr.args if a != e]).doit(deep=False)
                    tmp_repl = {}
                    tmp_repl.update(repl_dict)
                    tmp_repl.update(m)
                    rem_m = rem_query.matches(rem_expr, repl_dict=tmp_repl)
                    if rem_m is not None:
                        #Check that contracted indices are not mapped to different indices.
                        internally_consistent = True
                        for k in rem_m.keys():
                            if isinstance(k,TensorIndex):
                                if -k in rem_m.keys() and rem_m[-k] != -rem_m[k]:
                                    internally_consistent = False
                                    break
                        if internally_consistent:
                            repl_dict.update(rem_m)
                            return repl_dict

            return None

        #Try to match WildTensor instances which have indices
        matched_e_tensors = []
        remaining_e_tensors = expr_sifted["Tensor"]
        indexless_wilds, wilds = sift(query_sifted["WildTensor"], lambda x: len(x.get_free_indices()) == 0, binary=True)

        for w in wilds:
            free_this_wild = set(w.get_free_indices())
            tensors_to_try = []
            for t in remaining_e_tensors:
                free = t.get_free_indices()
                shares_indices_with_wild = True
                for i in free:
                    if all(j.matches(i) is None for j in free_this_wild):
                        #The index i matches none of the indices in free_this_wild
                        shares_indices_with_wild = False
                if shares_indices_with_wild:
                    tensors_to_try.append(t)

            m = w.matches(TensMul(*tensors_to_try).doit(deep=False) )
            if m is None:
                return None
            else:
                for tens in tensors_to_try:
                    matched_e_tensors.append(tens)
                repl_dict.update(m)

        #Try to match indexless WildTensor instances
        remaining_e_tensors = [t for t in expr_sifted["Tensor"] if t not in matched_e_tensors]
        if len(indexless_wilds) > 0:
            #If there are any remaining tensors, match them with the indexless WildTensor
            m =  indexless_wilds[0].matches( TensMul(1,*remaining_e_tensors).doit(deep=False) )
            if m is None:
                return None
            else:
                repl_dict.update(m)
        elif len(remaining_e_tensors) > 0:
            return None

        #Try to match the non-tensorial coefficient
        m = self.coeff.matches(expr_coeff, old=old)
        if m is None:
            return None
        else:
            repl_dict.update(m)

        return repl_dict

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        commute = all(arg.component.comm == 0 for arg in expr.args if isinstance(arg, Tensor))
        if commute:
            return self._matches_commutative(expr, repl_dict, old)
        else:
            raise NotImplementedError("Tensor matching not implemented for non-commuting tensors")

class TensorElement(TensExpr):
    """
    Tensor with evaluated components.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> i, j, k = symbols("i j k")
    >>> A = TensorHead("A", [L, L], TensorSymmetry.fully_symmetric(2))
    >>> A(i, j).get_free_indices()
    [i, j]

    If we want to set component ``i`` to a specific value, use the
    ``TensorElement`` class:

    >>> from sympy.tensor.tensor import TensorElement
    >>> te = TensorElement(A(i, j), {i: 2})

    As index ``i`` has been accessed (``{i: 2}`` is the evaluation of its 3rd
    element), the free indices will only contain ``j``:

    >>> te.get_free_indices()
    [j]
    """

    def __new__(cls, expr, index_map):
        if not isinstance(expr, Tensor):
            # remap
            if not isinstance(expr, TensExpr):
                raise TypeError("%s is not a tensor expression" % expr)
            return expr.func(*[TensorElement(arg, index_map) for arg in expr.args])
        expr_free_indices = expr.get_free_indices()
        name_translation = {i.args[0]: i for i in expr_free_indices}
        index_map = {name_translation.get(index, index): value for index, value in index_map.items()}
        index_map = {index: value for index, value in index_map.items() if index in expr_free_indices}
        if len(index_map) == 0:
            return expr
        free_indices = [i for i in expr_free_indices if i not in index_map.keys()]
        index_map = Dict(index_map)
        obj = TensExpr.__new__(cls, expr, index_map)
        obj._free_indices = free_indices
        return obj

    @property
    def free(self):
        return [(index, i) for i, index in enumerate(self.get_free_indices())]

    @property
    def dum(self):
        # TODO: inherit dummies from expr
        return []

    @property
    def expr(self):
        return self._args[0]

    @property
    def index_map(self):
        return self._args[1]

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    def get_free_indices(self):
        return self._free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        # TODO: can be improved:
        return self.xreplace(repl)

    def get_indices(self):
        return self.get_free_indices()

    def _extract_data(self, replacement_dict):
        ret_indices, array = self.expr._extract_data(replacement_dict)
        index_map = self.index_map
        slice_tuple = tuple(index_map.get(i, slice(None)) for i in ret_indices)
        ret_indices = [i for i in ret_indices if i not in index_map]
        array = array.__getitem__(slice_tuple)
        return ret_indices, array


class WildTensorHead(TensorHead):
    """
    A wild object that is used to create ``WildTensor`` instances

    Explanation
    ===========

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)

    A WildTensorHead can be created without specifying a ``TensorIndexType``

    >>> W = WildTensorHead("W")

    Calling it with a ``TensorIndex`` creates a ``WildTensor`` instance.

    >>> type(W(p))
    <class 'sympy.tensor.tensor.WildTensor'>

    The ``TensorIndexType`` is automatically detected from the index that is passed

    >>> W(p).component
    W(R3)

    Calling it with no indices returns an object that can match tensors with any number of indices.

    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    If you want to ignore the order of indices while matching, pass ``unordered_indices=True``.

    >>> U = WildTensorHead("U", unordered_indices=True)
    >>> W(p,q).matches(Q(q,p))
    >>> U(p,q).matches(Q(q,p))
    {U(R3,R3): _WildTensExpr(Q(q, p))}

    Parameters
    ==========
    name : name of the tensor
    unordered_indices : whether the order of the indices matters for matching
        (default: False)

    See also
    ========
    ``WildTensor``
    ``TensorHead``

    """
    def __new__(cls, name, index_types=None, symmetry=None, comm=0,  unordered_indices=False):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError("invalid name")

        if index_types is None:
            index_types = []

        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)

        if symmetry != TensorSymmetry.no_symmetry(len(index_types)):
            raise NotImplementedError("Wild matching based on symmetry is not implemented.")

        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), sympify(symmetry), sympify(comm), sympify(unordered_indices))

        return obj

    @property
    def unordered_indices(self):
        return self.args[4]

    def __call__(self, *indices, **kwargs):
        tensor = WildTensor(self, indices, **kwargs)
        return tensor.doit()


class WildTensor(Tensor):
    """
    A wild object which matches ``Tensor`` instances

    Explanation
    ===========
    This is instantiated by attaching indices to a ``WildTensorHead`` instance.

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])

    Matching also takes the indices into account
    >>> W(p).matches(K(p))
    {W(R3): _WildTensExpr(K(p))}
    >>> W(p).matches(K(q))
    >>> W(p).matches(K(-p))

    If you want to match objects with any number of indices, just use a ``WildTensor`` with no indices.
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    See Also
    ========
    ``WildTensorHead``
    ``Tensor``

    """
    def __new__(cls, tensor_head, indices, **kw_args):
        is_canon_bp = kw_args.pop("is_canon_bp", False)

        if tensor_head.func == TensorHead:
            """
            If someone tried to call WildTensor by supplying a TensorHead (not a WildTensorHead), return a normal tensor instead. This is helpful when using subs on an expression to replace occurrences of a WildTensorHead with a TensorHead.
            """
            return Tensor(tensor_head, indices, is_canon_bp=is_canon_bp, **kw_args)
        elif tensor_head.func == _WildTensExpr:
            return tensor_head(*indices)

        indices = cls._parse_indices(tensor_head, indices)
        index_types = [ind.tensor_index_type for ind in indices]
        tensor_head = tensor_head.func(
            tensor_head.name,
            index_types,
            symmetry=None,
            comm=tensor_head.comm,
            unordered_indices=tensor_head.unordered_indices,
            )

        obj = Basic.__new__(cls, tensor_head, Tuple(*indices))
        obj.name = tensor_head.name
        obj._index_structure = _IndexStructure.from_indices(*indices)
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        obj._coeff = S.One
        obj._nocoeff = obj
        obj._component = tensor_head
        obj._components = [tensor_head]
        if tensor_head.rank != len(indices):
            raise ValueError("wrong number of indices")
        obj.is_canon_bp = is_canon_bp
        obj._index_map = obj._build_index_map(indices, obj._index_structure)

        return obj


    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, TensExpr) and expr != S(1):
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if len(self.indices) > 0:
            if not hasattr(expr, "get_free_indices"):
                return None
            expr_indices = expr.get_free_indices()
            if len(expr_indices) != len(self.indices):
                return None
            if self._component.unordered_indices:
                m = self._match_indices_ignoring_order(expr)
                if m is None:
                    return None
                else:
                    repl_dict.update(m)
            else:
                for i in range(len(expr_indices)):
                    m = self.indices[i].matches(expr_indices[i])
                    if m is None:
                        return None
                    else:
                        repl_dict.update(m)

            repl_dict[self.component] = _WildTensExpr(expr)
        else:
            #If no indices were passed to the WildTensor, it may match tensors with any number of indices.
            repl_dict[self] = expr

        return repl_dict

    def _match_indices_ignoring_order(self, expr, repl_dict=None, old=False):
        """
        Helper method for matches. Checks if the indices of self and expr
        match disregarding index ordering.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        def siftkey(ind):
            if isinstance(ind, WildTensorIndex):
                if ind.ignore_updown:
                    return "wild, updown"
                else:
                    return "wild"
            else:
                return "nonwild"

        indices_sifted = sift(self.indices, siftkey)

        matched_indices = []
        expr_indices_remaining = expr.get_indices()
        for ind in indices_sifted["nonwild"]:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                if e_ind in matched_indices:
                    continue
                m = ind.matches(e_ind)
                if m is not None:
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None

        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
        for ind in indices_sifted["wild"]:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                m = ind.matches(e_ind)
                if m is not None:
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None

        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
        for ind in indices_sifted["wild, updown"]:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                m = ind.matches(e_ind)
                if m is not None:
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None

        if len(matched_indices) < len(self.indices):
            return None
        else:
            return repl_dict

class WildTensorIndex(TensorIndex):
    """
    A wild object that matches TensorIndex instances.

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorIndex, TensorIndexType, WildTensorIndex
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex("p", R3)

    By default, covariant indices only match with covariant indices (and
    similarly for contravariant)

    >>> q = WildTensorIndex("q", R3)
    >>> (q).matches(p)
    {q: p}
    >>> (q).matches(-p)

    If you want matching to ignore whether the index is co/contra-variant, set
    ignore_updown=True

    >>> r = WildTensorIndex("r", R3, ignore_updown=True)
    >>> (r).matches(-p)
    {r: -p}
    >>> (r).matches(p)
    {r: p}

    Parameters
    ==========
    name : name of the index (string), or ``True`` if you want it to be
        automatically assigned
    tensor_index_type : ``TensorIndexType`` of the index
    is_up :  flag for contravariant index (is_up=True by default)
    ignore_updown : bool, Whether this should match both co- and contra-variant
        indices (default:False)
    """
    def __new__(cls, name, tensor_index_type, is_up=True, ignore_updown=False):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        elif name is True:
            name = "_i{}".format(len(tensor_index_type._autogenerated))
            name_symbol = Symbol(name)
            tensor_index_type._autogenerated.append(name_symbol)
        else:
            raise ValueError("invalid name")

        is_up = sympify(is_up)
        ignore_updown = sympify(ignore_updown)
        return Basic.__new__(cls, name_symbol, tensor_index_type, is_up, ignore_updown)

    @property
    def ignore_updown(self):
        return self.args[3]

    def __neg__(self):
        t1 = WildTensorIndex(self.name, self.tensor_index_type,
                (not self.is_up), self.ignore_updown)
        return t1

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, TensorIndex):
            return None
        if self.tensor_index_type != expr.tensor_index_type:
            return None
        if not self.ignore_updown:
            if self.is_up != expr.is_up:
                return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        repl_dict[self] = expr
        return repl_dict


class _WildTensExpr(Basic):
    """
    INTERNAL USE ONLY

    This is an object that helps with replacement of WildTensors in expressions.
    When this object is set as the tensor_head of a WildTensor, it replaces the
    WildTensor by a TensExpr (passed when initializing this object).

    Examples
    ========
    >>> from sympy.tensor.tensor import WildTensorHead, TensorIndex, TensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> print( ( K(p) ).replace( W(p), W(q)*W(-q)*W(p) ) )
    K(R_0)*K(-R_0)*K(p)

    """
    def __init__(self, expr):
        if not isinstance(expr, TensExpr):
            raise TypeError("_WildTensExpr expects a TensExpr as argument")
        self.expr = expr

    def __call__(self, *indices):
        return self.expr._replace_indices(dict(zip(self.expr.get_free_indices(), indices)))

    def __neg__(self):
        return self.func(self.expr*S.NegativeOne)

    def __abs__(self):
        raise NotImplementedError

    def __add__(self, other):
        if other.func != self.func:
            raise TypeError(f"Cannot add {self.func} to {other.func}")
        return self.func(self.expr+other.expr)

    def __radd__(self, other):
        if other.func != self.func:
            raise TypeError(f"Cannot add {self.func} to {other.func}")
        return self.func(other.expr+self.expr)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError


def canon_bp(p):
    """
    Butler-Portugal canonicalization. See ``tensor_can.py`` from the
    combinatorics module for the details.
    """
    if isinstance(p, TensExpr):
        return p.canon_bp()
    return p


def tensor_mul(*a):
    """
    product of tensors
    """
    if not a:
        return TensMul.from_data(S.One, [], [], [])
    t = a[0]
    for tx in a[1:]:
        t = t*tx
    return t


def riemann_cyclic_replace(t_r):
    """
    replace Riemann tensor with an equivalent expression

    ``R(m,n,p,q) -> 2/3*R(m,n,p,q) - 1/3*R(m,q,n,p) + 1/3*R(m,p,n,q)``

    """
    free = sorted(t_r.free, key=lambda x: x[1])
    m, n, p, q = [x[0] for x in free]
    t0 = t_r*Rational(2, 3)
    t1 = -t_r.substitute_indices((m,m),(n,q),(p,n),(q,p))*Rational(1, 3)
    t2 = t_r.substitute_indices((m,m),(n,p),(p,n),(q,q))*Rational(1, 3)
    t3 = t0 + t1 + t2
    return t3

def riemann_cyclic(t2):
    """
    Replace each Riemann tensor with an equivalent expression
    satisfying the cyclic identity.

    This trick is discussed in the reference guide to Cadabra.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, riemann_cyclic, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i, j, k, l = tensor_indices('i,j,k,l', Lorentz)
    >>> R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    >>> t = R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    >>> riemann_cyclic(t)
    0
    """
    t2 = t2.expand()
    if isinstance(t2, (TensMul, Tensor)):
        args = [t2]
    else:
        args = t2.args
    a1 = [x.split() for x in args]
    a2 = [[riemann_cyclic_replace(tx) for tx in y] for y in a1]
    a3 = [tensor_mul(*v) for v in a2]
    t3 = TensAdd(*a3).doit(deep=False)
    if not t3:
        return t3
    else:
        return canon_bp(t3)


def get_lines(ex, index_type):
    """
    Returns ``(lines, traces, rest)`` for an index type,
    where ``lines`` is the list of list of positions of a matrix line,
    ``traces`` is the list of list of traced matrix lines,
    ``rest`` is the rest of the elements of the tensor.
    """
    def _join_lines(a):
        i = 0
        while i < len(a):
            x = a[i]
            xend = x[-1]
            xstart = x[0]
            hit = True
            while hit:
                hit = False
                for j in range(i + 1, len(a)):
                    if j >= len(a):
                        break
                    if a[j][0] == xend:
                        hit = True
                        x.extend(a[j][1:])
                        xend = x[-1]
                        a.pop(j)
                        continue
                    if a[j][0] == xstart:
                        hit = True
                        a[i] = reversed(a[j][1:]) + x
                        x = a[i]
                        xstart = a[i][0]
                        a.pop(j)
                        continue
                    if a[j][-1] == xend:
                        hit = True
                        x.extend(reversed(a[j][:-1]))
                        xend = x[-1]
                        a.pop(j)
                        continue
                    if a[j][-1] == xstart:
                        hit = True
                        a[i] = a[j][:-1] + x
                        x = a[i]
                        xstart = x[0]
                        a.pop(j)
                        continue
            i += 1
        return a

    arguments = ex.args
    dt = {}
    for c in ex.args:
        if not isinstance(c, TensExpr):
            continue
        if c in dt:
            continue
        index_types = c.index_types
        a = []
        for i in range(len(index_types)):
            if index_types[i] is index_type:
                a.append(i)
        if len(a) > 2:
            raise ValueError('at most two indices of type %s allowed' % index_type)
        if len(a) == 2:
            dt[c] = a
    #dum = ex.dum
    lines = []
    traces = []
    traces1 = []
    #indices_to_args_pos = ex._get_indices_to_args_pos()
    # TODO: add a dum_to_components_map ?
    for p0, p1, c0, c1 in ex.dum_in_args:
        if arguments[c0] not in dt:
            continue
        if c0 == c1:
            traces.append([c0])
            continue
        ta0 = dt[arguments[c0]]
        ta1 = dt[arguments[c1]]
        if p0 not in ta0:
            continue
        if ta0.index(p0) == ta1.index(p1):
            # case gamma(i,s0,-s1) in c0, gamma(j,-s0,s2) in c1;
            # to deal with this case one could add to the position
            # a flag for transposition;
            # one could write [(c0, False), (c1, True)]
            raise NotImplementedError
        # if p0 == ta0[1] then G in pos c0 is mult on the right by G in c1
        # if p0 == ta0[0] then G in pos c1 is mult on the right by G in c0
        ta0 = dt[arguments[c0]]
        b0, b1 = (c0, c1) if p0 == ta0[1]  else (c1, c0)
        lines1 = lines.copy()
        for line in lines:
            if line[-1] == b0:
                if line[0] == b1:
                    n = line.index(min(line))
                    traces1.append(line)
                    traces.append(line[n:] + line[:n])
                else:
                    line.append(b1)
                break
            elif line[0] == b1:
                line.insert(0, b0)
                break
        else:
            lines1.append([b0, b1])

        lines = [x for x in lines1 if x not in traces1]
        lines = _join_lines(lines)
    rest = []
    for line in lines:
        for y in line:
            rest.append(y)
    for line in traces:
        for y in line:
            rest.append(y)
    rest = [x for x in range(len(arguments)) if x not in rest]

    return lines, traces, rest


def get_free_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    return t.get_free_indices()


def get_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    return t.get_indices()

def get_dummy_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    inds = t.get_indices()
    free = t.get_free_indices()
    return [i for i in inds if i not in free]

def get_index_structure(t):
    if isinstance(t, TensExpr):
        return t._index_structure
    return _IndexStructure([], [], [], [])


def get_coeff(t):
    if isinstance(t, Tensor):
        return S.One
    if isinstance(t, TensMul):
        return t.coeff
    if isinstance(t, TensExpr):
        raise ValueError("no coefficient associated to this tensor expression")
    return t

def contract_metric(t, g):
    if isinstance(t, TensExpr):
        return t.contract_metric(g)
    return t

def perm2tensor(t, g, is_canon_bp=False):
    """
    Returns the tensor corresponding to the permutation ``g``

    For further details, see the method in ``TIDS`` with the same name.
    """
    if not isinstance(t, TensExpr):
        return t
    elif isinstance(t, (Tensor, TensMul)):
        nim = get_index_structure(t).perm2tensor(g, is_canon_bp=is_canon_bp)
        res = t._set_new_index_structure(nim, is_canon_bp=is_canon_bp)
        if g[-1] != len(g) - 1:
            return -res

        return res
    raise NotImplementedError()


def substitute_indices(t, *index_tuples):
    if not isinstance(t, TensExpr):
        return t
    return t.substitute_indices(*index_tuples)


def _get_wilds(expr):
    return list(expr.atoms(Wild, WildFunction, WildTensor, WildTensorIndex, WildTensorHead))


def get_postprocessor(cls):
    def _postprocessor(expr):
        tens_class = {Mul: TensMul, Add: TensAdd}[cls]
        if any(isinstance(a, TensExpr) for a in expr.args):
            return tens_class(*expr.args)
        else:
            return expr

    return _postprocessor

Basic._constructor_postprocessor_mapping[TensExpr] = {
    "Mul": [get_postprocessor(Mul)],
}
<|endoftext|>from collections.abc import Iterable
from functools import singledispatch

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.parameters import global_parameters


class TensorProduct(Expr):
    """
    Generic class for tensor products.
    """
    is_number = False

    def __new__(cls, *args, **kwargs):
        from sympy.tensor.array import NDimArray, tensorproduct, Array
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.strategies import flatten

        args = [sympify(arg) for arg in args]
        evaluate = kwargs.get("evaluate", global_parameters.evaluate)

        if not evaluate:
            obj = Expr.__new__(cls, *args)
            return obj

        arrays = []
        other = []
        scalar = S.One
        for arg in args:
            if isinstance(arg, (Iterable, MatrixBase, NDimArray)):
                arrays.append(Array(arg))
            elif isinstance(arg, (MatrixExpr,)):
                other.append(arg)
            else:
                scalar *= arg

        coeff = scalar*tensorproduct(*arrays)
        if len(other) == 0:
            return coeff
        if coeff != 1:
            newargs = [coeff] + other
        else:
            newargs = other
        obj = Expr.__new__(cls, *newargs, **kwargs)
        return flatten(obj)

    def rank(self):
        return len(self.shape)

    def _get_args_shapes(self):
        from sympy.tensor.array import Array
        return [i.shape if hasattr(i, "shape") else Array(i).shape for i in self.args]

    @property
    def shape(self):
        shape_list = self._get_args_shapes()
        return sum(shape_list, ())

    def __getitem__(self, index):
        index = iter(index)
        return Mul.fromiter(
            arg.__getitem__(tuple(next(index) for i in shp))
            for arg, shp in zip(self.args, self._get_args_shapes())
        )


@singledispatch
def shape(expr):
    """
    Return the shape of the *expr* as a tuple. *expr* should represent
    suitable object such as matrix or array.

    Parameters
    ==========

    expr : SymPy object having ``MatrixKind`` or ``ArrayKind``.

    Raises
    ======

    NoShapeError : Raised when object with wrong kind is passed.

    Examples
    ========

    This function returns the shape of any object representing matrix or array.

    >>> from sympy import shape, Array, ImmutableDenseMatrix, Integral
    >>> from sympy.abc import x
    >>> A = Array([1, 2])
    >>> shape(A)
    (2,)
    >>> shape(Integral(A, x))
    (2,)
    >>> M = ImmutableDenseMatrix([1, 2])
    >>> shape(M)
    (2, 1)
    >>> shape(Integral(M, x))
    (2, 1)

    You can support new type by dispatching.

    >>> from sympy import Expr
    >>> class NewExpr(Expr):
    ...     pass
    >>> @shape.register(NewExpr)
    ... def _(expr):
    ...     return shape(expr.args[0])
    >>> shape(NewExpr(M))
    (2, 1)

    If unsuitable expression is passed, ``NoShapeError()`` will be raised.

    >>> shape(Integral(x, x))
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: Integral(x, x)

    Notes
    =====

    Array-like classes (such as ``Matrix`` or ``NDimArray``) has ``shape``
    property which returns its shape, but it cannot be used for non-array
    classes containing array. This function returns the shape of any
    registered object representing array.

    """
    if hasattr(expr, "shape"):
        return expr.shape
    raise NoShapeError(
        "%s does not have shape, or its type is not registered to shape()." % expr)


class NoShapeError(Exception):
    """
    Raised when ``shape()`` is called on non-array object.

    This error can be imported from ``sympy.tensor.functions``.

    Examples
    ========

    >>> from sympy import shape
    >>> from sympy.abc import x
    >>> shape(x)
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: x
    """
    pass
<|endoftext|>from sympy.tensor.array.ndim_array import NDimArray


class MutableNDimArray(NDimArray):

    def as_immutable(self):
        raise NotImplementedError("abstract method")

    def as_mutable(self):
        return self

    def _sympy_(self):
        return self.as_immutable()
<|endoftext|>import itertools
from collections.abc import Iterable

from sympy.core._print_helpers import Printable
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.core.sympify import _sympify

from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray


def _arrayfy(a):
    from sympy.matrices import MatrixBase

    if isinstance(a, NDimArray):
        return a
    if isinstance(a, (MatrixBase, list, tuple, Tuple)):
        return ImmutableDenseNDimArray(a)
    return a


def tensorproduct(*args):
    """
    Tensor product among scalars or array-like objects.

    The equivalent operator for array expressions is ``ArrayTensorProduct``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.tensor.array import tensorproduct, Array
    >>> from sympy.abc import x, y, z, t
    >>> A = Array([[1, 2], [3, 4]])
    >>> B = Array([x, y])
    >>> tensorproduct(A, B)
    [[[x, y], [2*x, 2*y]], [[3*x, 3*y], [4*x, 4*y]]]
    >>> tensorproduct(A, x)
    [[x, 2*x], [3*x, 4*x]]
    >>> tensorproduct(A, B, B)
    [[[[x**2, x*y], [x*y, y**2]], [[2*x**2, 2*x*y], [2*x*y, 2*y**2]]], [[[3*x**2, 3*x*y], [3*x*y, 3*y**2]], [[4*x**2, 4*x*y], [4*x*y, 4*y**2]]]]

    Applying this function on two matrices will result in a rank 4 array.

    >>> from sympy import Matrix, eye
    >>> m = Matrix([[x, y], [z, t]])
    >>> p = tensorproduct(eye(3), m)
    >>> p
    [[[[x, y], [z, t]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[x, y], [z, t]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[x, y], [z, t]]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayTensorProduct

    """
    from sympy.tensor.array import SparseNDimArray, ImmutableSparseNDimArray

    if len(args) == 0:
        return S.One
    if len(args) == 1:
        return _arrayfy(args[0])
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if any(isinstance(arg, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)) for arg in args):
        return ArrayTensorProduct(*args)
    if len(args) > 2:
        return tensorproduct(tensorproduct(args[0], args[1]), *args[2:])

    # length of args is 2:
    a, b = map(_arrayfy, args)

    if not isinstance(a, NDimArray) or not isinstance(b, NDimArray):
        return a*b

    if isinstance(a, SparseNDimArray) and isinstance(b, SparseNDimArray):
        lp = len(b)
        new_array = {k1*lp + k2: v1*v2 for k1, v1 in a._sparse_array.items() for k2, v2 in b._sparse_array.items()}
        return ImmutableSparseNDimArray(new_array, a.shape + b.shape)

    product_list = [i*j for i in Flatten(a) for j in Flatten(b)]
    return ImmutableDenseNDimArray(product_list, a.shape + b.shape)


def _util_contraction_diagonal(array, *contraction_or_diagonal_axes):
    array = _arrayfy(array)

    # Verify contraction_axes:
    taken_dims = set()
    for axes_group in contraction_or_diagonal_axes:
        if not isinstance(axes_group, Iterable):
            raise ValueError("collections of contraction/diagonal axes expected")

        dim = array.shape[axes_group[0]]

        for d in axes_group:
            if d in taken_dims:
                raise ValueError("dimension specified more than once")
            if dim != array.shape[d]:
                raise ValueError("cannot contract or diagonalize between axes of different dimension")
            taken_dims.add(d)

    rank = array.rank()

    remaining_shape = [dim for i, dim in enumerate(array.shape) if i not in taken_dims]
    cum_shape = [0]*rank
    _cumul = 1
    for i in range(rank):
        cum_shape[rank - i - 1] = _cumul
        _cumul *= int(array.shape[rank - i - 1])

    # DEFINITION: by absolute position it is meant the position along the one
    # dimensional array containing all the tensor components.

    # Possible future work on this module: move computation of absolute
    # positions to a class method.

    # Determine absolute positions of the uncontracted indices:
    remaining_indices = [[cum_shape[i]*j for j in range(array.shape[i])]
                         for i in range(rank) if i not in taken_dims]

    # Determine absolute positions of the contracted indices:
    summed_deltas = []
    for axes_group in contraction_or_diagonal_axes:
        lidx = []
        for js in range(array.shape[axes_group[0]]):
            lidx.append(sum(cum_shape[ig] * js for ig in axes_group))
        summed_deltas.append(lidx)

    return array, remaining_indices, remaining_shape, summed_deltas


def tensorcontraction(array, *contraction_axes):
    """
    Contraction of an array-like object on the specified axes.

    The equivalent operator for array expressions is ``ArrayContraction``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy import Array, tensorcontraction
    >>> from sympy import Matrix, eye
    >>> tensorcontraction(eye(3), (0, 1))
    3
    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensorcontraction(A, (0, 2))
    [21, 30]

    Matrix multiplication may be emulated with a proper combination of
    ``tensorcontraction`` and ``tensorproduct``

    >>> from sympy import tensorproduct
    >>> from sympy.abc import a,b,c,d,e,f,g,h
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> m2 = Matrix([[e, f], [g, h]])
    >>> p = tensorproduct(m1, m2)
    >>> p
    [[[[a*e, a*f], [a*g, a*h]], [[b*e, b*f], [b*g, b*h]]], [[[c*e, c*f], [c*g, c*h]], [[d*e, d*f], [d*g, d*h]]]]
    >>> tensorcontraction(p, (1, 2))
    [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    >>> m1*m2
    Matrix([
    [a*e + b*g, a*f + b*h],
    [c*e + d*g, c*f + d*h]])

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayContraction

    """
    from sympy.tensor.array.expressions.array_expressions import _array_contraction
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_contraction(array, *contraction_axes)

    array, remaining_indices, remaining_shape, summed_deltas = _util_contraction_diagonal(array, *contraction_axes)

    # Compute the contracted array:
    #
    # 1. external for loops on all uncontracted indices.
    #    Uncontracted indices are determined by the combinatorial product of
    #    the absolute positions of the remaining indices.
    # 2. internal loop on all contracted indices.
    #    It sums the values of the absolute contracted index and the absolute
    #    uncontracted index for the external loop.
    contracted_array = []
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = S.Zero
        for sum_to_index in itertools.product(*summed_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum += array[idx]

        contracted_array.append(isum)

    if len(remaining_indices) == 0:
        assert len(contracted_array) == 1
        return contracted_array[0]

    return type(array)(contracted_array, remaining_shape)


def tensordiagonal(array, *diagonal_axes):
    """
    Diagonalization of an array-like object on the specified axes.

    This is equivalent to multiplying the expression by Kronecker deltas
    uniting the axes.

    The diagonal indices are put at the end of the axes.

    The equivalent operator for array expressions is ``ArrayDiagonal``, which
    can be used to keep the expression unevaluated.

    Examples
    ========

    ``tensordiagonal`` acting on a 2-dimensional array by axes 0 and 1 is
    equivalent to the diagonal of the matrix:

    >>> from sympy import Array, tensordiagonal
    >>> from sympy import Matrix, eye
    >>> tensordiagonal(eye(3), (0, 1))
    [1, 1, 1]

    >>> from sympy.abc import a,b,c,d
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> tensordiagonal(m1, [0, 1])
    [a, d]

    In case of higher dimensional arrays, the diagonalized out dimensions
    are appended removed and appended as a single dimension at the end:

    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensordiagonal(A, (0, 2))
    [[0, 7, 14], [3, 10, 17]]
    >>> from sympy import permutedims
    >>> tensordiagonal(A, (0, 2)) == permutedims(Array([A[0, :, 0], A[1, :, 1], A[2, :, 2]]), [1, 0])
    True

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayDiagonal

    """
    if any(len(i) <= 1 for i in diagonal_axes):
        raise ValueError("need at least two axes to diagonalize")

    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, _array_diagonal
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_diagonal(array, *diagonal_axes)

    ArrayDiagonal._validate(array, *diagonal_axes)

    array, remaining_indices, remaining_shape, diagonal_deltas = _util_contraction_diagonal(array, *diagonal_axes)

    # Compute the diagonalized array:
    #
    # 1. external for loops on all undiagonalized indices.
    #    Undiagonalized indices are determined by the combinatorial product of
    #    the absolute positions of the remaining indices.
    # 2. internal loop on all diagonal indices.
    #    It appends the values of the absolute diagonalized index and the absolute
    #    undiagonalized index for the external loop.
    diagonalized_array = []
    diagonal_shape = [len(i) for i in diagonal_deltas]
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = []
        for sum_to_index in itertools.product(*diagonal_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum.append(array[idx])

        isum = type(array)(isum).reshape(*diagonal_shape)
        diagonalized_array.append(isum)

    return type(array)(diagonalized_array, remaining_shape + diagonal_shape)


def derive_by_array(expr, dx):
    r"""
    Derivative by arrays. Supports both arrays and scalars.

    The equivalent operator for array expressions is ``array_derive``.

    Explanation
    ===========

    Given the array `A_{i_1, \ldots, i_N}` and the array `X_{j_1, \ldots, j_M}`
    this function will return a new array `B` defined by

    `B_{j_1,\ldots,j_M,i_1,\ldots,i_N} := \frac{\partial A_{i_1,\ldots,i_N}}{\partial X_{j_1,\ldots,j_M}}`

    Examples
    ========

    >>> from sympy import derive_by_array
    >>> from sympy.abc import x, y, z, t
    >>> from sympy import cos
    >>> derive_by_array(cos(x*t), x)
    -t*sin(t*x)
    >>> derive_by_array(cos(x*t), [x, y, z, t])
    [-t*sin(t*x), 0, 0, -x*sin(t*x)]
    >>> derive_by_array([x, y**2*z], [[x, y], [z, t]])
    [[[1, 0], [0, 2*y*z]], [[0, y**2], [0, 0]]]

    """
    from sympy.matrices import MatrixBase
    from sympy.tensor.array import SparseNDimArray
    array_types = (Iterable, MatrixBase, NDimArray)

    if isinstance(dx, array_types):
        dx = ImmutableDenseNDimArray(dx)
        for i in dx:
            if not i._diff_wrt:
                raise ValueError("cannot derive by this array")

    if isinstance(expr, array_types):
        if isinstance(expr, NDimArray):
            expr = expr.as_immutable()
        else:
            expr = ImmutableDenseNDimArray(expr)

        if isinstance(dx, array_types):
            if isinstance(expr, SparseNDimArray):
                lp = len(expr)
                new_array = {k + i*lp: v
                             for i, x in enumerate(Flatten(dx))
                             for k, v in expr.diff(x)._sparse_array.items()}
            else:
                new_array = [[y.diff(x) for y in Flatten(expr)] for x in Flatten(dx)]
            return type(expr)(new_array, dx.shape + expr.shape)
        else:
            return expr.diff(dx)
    else:
        expr = _sympify(expr)
        if isinstance(dx, array_types):
            return ImmutableDenseNDimArray([expr.diff(i) for i in Flatten(dx)], dx.shape)
        else:
            dx = _sympify(dx)
            return diff(expr, dx)


def permutedims(expr, perm=None, index_order_old=None, index_order_new=None):
    """
    Permutes the indices of an array.

    Parameter specifies the permutation of the indices.

    The equivalent operator for array expressions is ``PermuteDims``, which can
    be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.abc import x, y, z, t
    >>> from sympy import sin
    >>> from sympy import Array, permutedims
    >>> a = Array([[x, y, z], [t, sin(x), 0]])
    >>> a
    [[x, y, z], [t, sin(x), 0]]
    >>> permutedims(a, (1, 0))
    [[x, t], [y, sin(x)], [z, 0]]

    If the array is of second order, ``transpose`` can be used:

    >>> from sympy import transpose
    >>> transpose(a)
    [[x, t], [y, sin(x)], [z, 0]]

    Examples on higher dimensions:

    >>> b = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> permutedims(b, (2, 1, 0))
    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    >>> permutedims(b, (1, 2, 0))
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    An alternative way to specify the same permutations as in the previous
    lines involves passing the *old* and *new* indices, either as a list or as
    a string:

    >>> permutedims(b, index_order_old="cba", index_order_new="abc")
    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    >>> permutedims(b, index_order_old="cab", index_order_new="abc")
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    ``Permutation`` objects are also allowed:

    >>> from sympy.combinatorics import Permutation
    >>> permutedims(b, Permutation([1, 2, 0]))
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.PermuteDims

    """
    from sympy.tensor.array import SparseNDimArray

    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _permute_dims
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.tensor.array.expressions import PermuteDims
    from sympy.tensor.array.expressions.array_expressions import get_rank
    perm = PermuteDims._get_permutation_from_arguments(perm, index_order_old, index_order_new, get_rank(expr))
    if isinstance(expr, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _permute_dims(expr, perm)

    if not isinstance(expr, NDimArray):
        expr = ImmutableDenseNDimArray(expr)

    from sympy.combinatorics import Permutation
    if not isinstance(perm, Permutation):
        perm = Permutation(list(perm))

    if perm.size != expr.rank():
        raise ValueError("wrong permutation size")

    # Get the inverse permutation:
    iperm = ~perm
    new_shape = perm(expr.shape)

    if isinstance(expr, SparseNDimArray):
        return type(expr)({tuple(perm(expr._get_tuple_index(k))): v
                           for k, v in expr._sparse_array.items()}, new_shape)

    indices_span = perm([range(i) for i in expr.shape])

    new_array = [None]*len(expr)
    for i, idx in enumerate(itertools.product(*indices_span)):
        t = iperm(idx)
        new_array[i] = expr[t]

    return type(expr)(new_array, new_shape)


class Flatten(Printable):
    """
    Flatten an iterable object to a list in a lazy-evaluation way.

    Notes
    =====

    This class is an iterator with which the memory cost can be economised.
    Optimisation has been considered to ameliorate the performance for some
    specific data types like DenseNDimArray and SparseNDimArray.

    Examples
    ========

    >>> from sympy.tensor.array.arrayop import Flatten
    >>> from sympy.tensor.array import Array
    >>> A = Array(range(6)).reshape(2, 3)
    >>> Flatten(A)
    Flatten([[0, 1, 2], [3, 4, 5]])
    >>> [i for i in Flatten(A)]
    [0, 1, 2, 3, 4, 5]
    """
    def __init__(self, iterable):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import NDimArray

        if not isinstance(iterable, (Iterable, MatrixBase)):
            raise NotImplementedError("Data type not yet supported")

        if isinstance(iterable, list):
            iterable = NDimArray(iterable)

        self._iter = iterable
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        from sympy.matrices.matrixbase import MatrixBase

        if len(self._iter) > self._idx:
            if isinstance(self._iter, DenseNDimArray):
                result = self._iter._array[self._idx]

            elif isinstance(self._iter, SparseNDimArray):
                if self._idx in self._iter._sparse_array:
                    result = self._iter._sparse_array[self._idx]
                else:
                    result = 0

            elif isinstance(self._iter, MatrixBase):
                result = self._iter[self._idx]

            elif hasattr(self._iter, '__next__'):
                result = next(self._iter)

            else:
                result = self._iter[self._idx]

        else:
            raise StopIteration

        self._idx += 1
        return result

    def next(self):
        return self.__next__()

    def _sympystr(self, printer):
        return type(self).__name__ + '(' + printer._print(self._iter) + ')'
<|endoftext|>from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.numbers import Integer
from sympy.matrices.matrixbase import MatrixBase
from .ndim_array import NDimArray
from .arrayop import derive_by_array
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.matrices.expressions.matexpr import _matrix_derivative


class ArrayDerivative(Derivative):

    is_scalar = False

    def __new__(cls, expr, *variables, **kwargs):
        obj = super().__new__(cls, expr, *variables, **kwargs)
        if isinstance(obj, ArrayDerivative):
            obj._shape = obj._get_shape()
        return obj

    def _get_shape(self):
        shape = ()
        for v, count in self.variable_count:
            if hasattr(v, "shape"):
                for i in range(count):
                    shape += v.shape
        if hasattr(self.expr, "shape"):
            shape += self.expr.shape
        return shape

    @property
    def shape(self):
        return self._shape

    @classmethod
    def _get_zero_with_shape_like(cls, expr):
        if isinstance(expr, (MatrixBase, NDimArray)):
            return expr.zeros(*expr.shape)
        elif isinstance(expr, MatrixExpr):
            return ZeroMatrix(*expr.shape)
        else:
            raise RuntimeError("Unable to determine shape of array-derivative.")

    @staticmethod
    def _call_derive_scalar_by_matrix(expr: Expr, v: MatrixBase) -> Expr:
        return v.applyfunc(lambda x: expr.diff(x))

    @staticmethod
    def _call_derive_scalar_by_matexpr(expr: Expr, v: MatrixExpr) -> Expr:
        if expr.has(v):
            return _matrix_derivative(expr, v)
        else:
            return ZeroMatrix(*v.shape)

    @staticmethod
    def _call_derive_scalar_by_array(expr: Expr, v: NDimArray) -> Expr:
        return v.applyfunc(lambda x: expr.diff(x))

    @staticmethod
    def _call_derive_matrix_by_scalar(expr: MatrixBase, v: Expr) -> Expr:
        return _matrix_derivative(expr, v)

    @staticmethod
    def _call_derive_matexpr_by_scalar(expr: MatrixExpr, v: Expr) -> Expr:
        return expr._eval_derivative(v)

    @staticmethod
    def _call_derive_array_by_scalar(expr: NDimArray, v: Expr) -> Expr:
        return expr.applyfunc(lambda x: x.diff(v))

    @staticmethod
    def _call_derive_default(expr: Expr, v: Expr) -> Expr | None:
        if expr.has(v):
            return _matrix_derivative(expr, v)
        else:
            return None

    @classmethod
    def _dispatch_eval_derivative_n_times(cls, expr, v, count):
        # Evaluate the derivative `n` times.  If
        # `_eval_derivative_n_times` is not overridden by the current
        # object, the default in `Basic` will call a loop over
        # `_eval_derivative`:

        if not isinstance(count, (int, Integer)) or ((count <= 0) == True):
            return None

        # TODO: this could be done with multiple-dispatching:
        if expr.is_scalar:
            if isinstance(v, MatrixBase):
                result = cls._call_derive_scalar_by_matrix(expr, v)
            elif isinstance(v, MatrixExpr):
                result = cls._call_derive_scalar_by_matexpr(expr, v)
            elif isinstance(v, NDimArray):
                result = cls._call_derive_scalar_by_array(expr, v)
            elif v.is_scalar:
                # scalar by scalar has a special
                return super()._dispatch_eval_derivative_n_times(expr, v, count)
            else:
                return None
        elif v.is_scalar:
            if isinstance(expr, MatrixBase):
                result = cls._call_derive_matrix_by_scalar(expr, v)
            elif isinstance(expr, MatrixExpr):
                result = cls._call_derive_matexpr_by_scalar(expr, v)
            elif isinstance(expr, NDimArray):
                result = cls._call_derive_array_by_scalar(expr, v)
            else:
                return None
        else:
            # Both `expr` and `v` are some array/matrix type:
            if isinstance(expr, MatrixBase) or isinstance(v, MatrixBase):
                result = derive_by_array(expr, v)
            elif isinstance(expr, MatrixExpr) and isinstance(v, MatrixExpr):
                result = cls._call_derive_default(expr, v)
            elif isinstance(expr, MatrixExpr) or isinstance(v, MatrixExpr):
                # if one expression is a symbolic matrix expression while the other isn't, don't evaluate:
                return None
            else:
                result = derive_by_array(expr, v)
        if result is None:
            return None
        if count == 1:
            return result
        else:
            return cls._dispatch_eval_derivative_n_times(result, v, count - 1)
<|endoftext|>import functools
from typing import List

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray, ArrayKind
from sympy.utilities.iterables import flatten


class DenseNDimArray(NDimArray):

    _array: List[Basic]

    def __new__(self, *args, **kwargs):
        return ImmutableDenseNDimArray(*args, **kwargs)

    @property
    def kind(self) -> ArrayKind:
        return ArrayKind._union(self._array)

    def __getitem__(self, index):
        """
        Allows to get items from N-dim array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([0, 1, 2, 3], (2, 2))
        >>> a
        [[0, 1], [2, 3]]
        >>> a[0, 0]
        0
        >>> a[1, 1]
        3
        >>> a[0]
        [0, 1]
        >>> a[1]
        [2, 3]


        Symbolic index:

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        Replace `i` and `j` to get element `(1, 1)`:

        >>> a[i, j].subs({i: 1, j: 1})
        3

        """
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex

        index = self._check_index_for_getitem(index)

        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            sl_factors, eindices = self._get_slice_data_for_array_access(index)
            array = [self._array[self._parse_index(i)] for i in eindices]
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._array[index]

    @classmethod
    def zeros(cls, *shape):
        list_length = functools.reduce(lambda x, y: x*y, shape, S.One)
        return cls._new(([0]*list_length,), shape)

    def tomatrix(self):
        """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        """
        from sympy.matrices import Matrix

        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')

        return Matrix(self.shape[0], self.shape[1], self._array)

    def reshape(self, *newshape):
        """
        Returns MutableDenseNDimArray instance with new shape. Elements number
        must be        suitable to new shape. The only argument of method sets
        new shape.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
        >>> a.shape
        (2, 3)
        >>> a
        [[1, 2, 3], [4, 5, 6]]
        >>> b = a.reshape(3, 2)
        >>> b.shape
        (3, 2)
        >>> b
        [[1, 2], [3, 4], [5, 6]]

        """
        new_total_size = functools.reduce(lambda x,y: x*y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError('Expecting reshape size to %d but got prod(%s) = %d' % (
                self._loop_size, str(newshape), new_total_size))

        # there is no `.func` as this class does not subtype `Basic`:
        return type(self)(self._array, newshape)


class ImmutableDenseNDimArray(DenseNDimArray, ImmutableNDimArray): # type: ignore
    def __new__(cls, iterable, shape=None, **kwargs):
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        shape = Tuple(*map(_sympify, shape))
        cls._check_special_bounds(flat_list, shape)
        flat_list = flatten(flat_list)
        flat_list = Tuple(*flat_list)
        self = Basic.__new__(cls, flat_list, shape, **kwargs)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape, 1)
        return self

    def __setitem__(self, index, value):
        raise TypeError('immutable N-dim array')

    def as_mutable(self):
        return MutableDenseNDimArray(self)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import simplify
        return self.applyfunc(simplify)

class MutableDenseNDimArray(DenseNDimArray, MutableNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        flat_list = flatten(flat_list)
        self = object.__new__(cls)
        self._shape = shape
        self._array = list(flat_list)
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)
        return self

    def __setitem__(self, index, value):
        """Allows to set items to MutableDenseNDimArray.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2,  2)
        >>> a[0,0] = 1
        >>> a[1,1] = 1
        >>> a
        [[1, 0], [0, 1]]

        """
        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            value, eindices, slice_offsets = self._get_slice_data_for_array_assignment(index, value)
            for i in eindices:
                other_i = [ind - j for ind, j in zip(i, slice_offsets) if j is not None]
                self._array[self._parse_index(i)] = value[other_i]
        else:
            index = self._parse_index(index)
            self._setter_iterable_check(value)
            value = _sympify(value)
            self._array[index] = value

    def as_immutable(self):
        return ImmutableDenseNDimArray(self)

    @property
    def free_symbols(self):
        return {i for j in self._array for i in j.free_symbols}
<|endoftext|>import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer


class ArrayComprehension(Basic):
    """
    Generate a list comprehension.

    Explanation
    ===========

    If there is a symbolic dimension, for example, say [i for i in range(1, N)] where
    N is a Symbol, then the expression will not be expanded to an array. Otherwise,
    calling the doit() function will launch the expansion.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehension
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
    >>> a
    ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
    >>> a.doit()
    [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
    >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k))
    >>> b.doit()
    ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k))
    """
    def __new__(cls, function, *symbols, **assumptions):
        if any(len(l) != 3 or None for l in symbols):
            raise ValueError('ArrayComprehension requires values lower and upper bound'
                              ' for the expression')
        arglist = [sympify(function)]
        arglist.extend(cls._check_limits_validity(function, symbols))
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args[1:]
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        return obj

    @property
    def function(self):
        """The function applied across limits.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.function
        10*i + j
        """
        return self._args[0]

    @property
    def limits(self):
        """
        The list of limits that will be applied while expanding the array.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.limits
        ((i, 1, 4), (j, 1, 3))
        """
        return self._limits

    @property
    def free_symbols(self):
        """
        The set of the free_symbols in the array.
        Variables appeared in the bounds are supposed to be excluded
        from the free symbol set.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.free_symbols
        set()
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.free_symbols
        {k}
        """
        expr_free_sym = self.function.free_symbols
        for var, inf, sup in self._limits:
            expr_free_sym.discard(var)
            curr_free_syms = inf.free_symbols.union(sup.free_symbols)
            expr_free_sym = expr_free_sym.union(curr_free_syms)
        return expr_free_sym

    @property
    def variables(self):
        """The tuples of the variables in the limits.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.variables
        [i, j]
        """
        return [l[0] for l in self._limits]

    @property
    def bound_symbols(self):
        """The list of dummy variables.

        Note
        ====

        Note that all variables are dummy variables since a limit without
        lower bound or upper bound is not accepted.
        """
        return [l[0] for l in self._limits if len(l) != 1]

    @property
    def shape(self):
        """
        The shape of the expanded array, which may have symbols.

        Note
        ====

        Both the lower and the upper bounds are included while
        calculating the shape.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.shape
        (4, 3)
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.shape
        (4, k + 3)
        """
        return self._shape

    @property
    def is_shape_numeric(self):
        """
        Test if the array is shape-numeric which means there is no symbolic
        dimension.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.is_shape_numeric
        True
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.is_shape_numeric
        False
        """
        for _, inf, sup in self._limits:
            if Basic(inf, sup).atoms(Symbol):
                return False
        return True

    def rank(self):
        """The rank of the expanded array.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.rank()
        2
        """
        return self._rank

    def __len__(self):
        """
        The length of the expanded array which means the number
        of elements in the array.

        Raises
        ======

        ValueError : When the length of the array is symbolic

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> len(a)
        12
        """
        if self._loop_size.free_symbols:
            raise ValueError('Symbolic length is not supported')
        return self._loop_size

    @classmethod
    def _check_limits_validity(cls, function, limits):
        #limits = sympify(limits)
        new_limits = []
        for var, inf, sup in limits:
            var = _sympify(var)
            inf = _sympify(inf)
            #since this is stored as an argument, it should be
            #a Tuple
            if isinstance(sup, list):
                sup = Tuple(*sup)
            else:
                sup = _sympify(sup)
            new_limits.append(Tuple(var, inf, sup))
            if any((not isinstance(i, Expr)) or i.atoms(Symbol, Integer) != i.atoms()
                                                                for i in [inf, sup]):
                raise TypeError('Bounds should be an Expression(combination of Integer and Symbol)')
            if (inf > sup) == True:
                raise ValueError('Lower bound should be inferior to upper bound')
            if var in inf.free_symbols or var in sup.free_symbols:
                raise ValueError('Variable should not be part of its bounds')
        return new_limits

    @classmethod
    def _calculate_shape_from_limits(cls, limits):
        return tuple([sup - inf + 1 for _, inf, sup in limits])

    @classmethod
    def _calculate_loop_size(cls, shape):
        if not shape:
            return 0
        loop_size = 1
        for l in shape:
            loop_size = loop_size * l

        return loop_size

    def doit(self, **hints):
        if not self.is_shape_numeric:
            return self

        return self._expand_array()

    def _expand_array(self):
        res = []
        for values in itertools.product(*[range(inf, sup+1)
                                        for var, inf, sup
                                        in self._limits]):
            res.append(self._get_element(values))

        return ImmutableDenseNDimArray(res, self.shape)

    def _get_element(self, values):
        temp = self.function
        for var, val in zip(self.variables, values):
            temp = temp.subs(var, val)
        return temp

    def tolist(self):
        """Transform the expanded array to a list.

        Raises
        ======

        ValueError : When there is a symbolic dimension

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.tolist()
        [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
        """
        if self.is_shape_numeric:
            return self._expand_array().tolist()

        raise ValueError("A symbolic array cannot be expanded to a list")

    def tomatrix(self):
        """Transform the expanded array to a matrix.

        Raises
        ======

        ValueError : When there is a symbolic dimension
        ValueError : When the rank of the expanded array is not equal to 2

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.tomatrix()
        Matrix([
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33],
        [41, 42, 43]])
        """
        from sympy.matrices import Matrix

        if not self.is_shape_numeric:
            raise ValueError("A symbolic array cannot be expanded to a matrix")
        if self._rank != 2:
            raise ValueError('Dimensions must be of size of 2')

        return Matrix(self._expand_array().tomatrix())


def isLambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

class ArrayComprehensionMap(ArrayComprehension):
    '''
    A subclass of ArrayComprehension dedicated to map external function lambda.

    Notes
    =====

    Only the lambda function is considered.
    At most one argument in lambda function is accepted in order to avoid ambiguity
    in value assignment.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehensionMap
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehensionMap(lambda: 1, (i, 1, 4))
    >>> a.doit()
    [1, 1, 1, 1]
    >>> b = ArrayComprehensionMap(lambda a: a+1, (j, 1, 4))
    >>> b.doit()
    [2, 3, 4, 5]

    '''
    def __new__(cls, function, *symbols, **assumptions):
        if any(len(l) != 3 or None for l in symbols):
            raise ValueError('ArrayComprehension requires values lower and upper bound'
                              ' for the expression')

        if not isLambda(function):
            raise ValueError('Data type not supported')

        arglist = cls._check_limits_validity(function, symbols)
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        obj._lambda = function
        return obj

    @property
    def func(self):
        class _(ArrayComprehensionMap):
            def __new__(cls, *args, **kwargs):
                return ArrayComprehensionMap(self._lambda, *args, **kwargs)
        return _

    def _get_element(self, values):
        temp = self._lambda
        if self._lambda.__code__.co_argcount == 0:
            temp = temp()
        elif self._lambda.__code__.co_argcount == 1:
            temp = temp(functools.reduce(lambda a, b: a*b, values))
        return temp
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.utilities.iterables import flatten

import functools

class SparseNDimArray(NDimArray):

    def __new__(self, *args, **kwargs):
        return ImmutableSparseNDimArray(*args, **kwargs)

    def __getitem__(self, index):
        """
        Get an element from a sparse N-dim array.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray(range(4), (2, 2))
        >>> a
        [[0, 1], [2, 3]]
        >>> a[0, 0]
        0
        >>> a[1, 1]
        3
        >>> a[0]
        [0, 1]
        >>> a[1]
        [2, 3]

        Symbolic indexing:

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        Replace `i` and `j` to get element `(0, 0)`:

        >>> a[i, j].subs({i: 0, j: 0})
        0

        """
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex

        index = self._check_index_for_getitem(index)

        # `index` is a tuple with one or more slices:
        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            sl_factors, eindices = self._get_slice_data_for_array_access(index)
            array = [self._sparse_array.get(self._parse_index(i), S.Zero) for i in eindices]
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            index = self._parse_index(index)
            return self._sparse_array.get(index, S.Zero)

    @classmethod
    def zeros(cls, *shape):
        """
        Return a sparse N-dim array of zeros.
        """
        return cls({}, shape)

    def tomatrix(self):
        """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        """
        from sympy.matrices import SparseMatrix
        if self.rank() != 2:
            raise ValueError('Dimensions must be of size of 2')

        mat_sparse = {}
        for key, value in self._sparse_array.items():
            mat_sparse[self._get_tuple_index(key)] = value

        return SparseMatrix(self.shape[0], self.shape[1], mat_sparse)

    def reshape(self, *newshape):
        new_total_size = functools.reduce(lambda x,y: x*y, newshape)
        if new_total_size != self._loop_size:
            raise ValueError("Invalid reshape parameters " + newshape)

        return type(self)(self._sparse_array, newshape)

class ImmutableSparseNDimArray(SparseNDimArray, ImmutableNDimArray): # type: ignore

    def __new__(cls, iterable=None, shape=None, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        shape = Tuple(*map(_sympify, shape))
        cls._check_special_bounds(flat_list, shape)
        loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)

        # Sparse array:
        if isinstance(flat_list, (dict, Dict)):
            sparse_array = Dict(flat_list)
        else:
            sparse_array = {}
            for i, el in enumerate(flatten(flat_list)):
                if el != 0:
                    sparse_array[i] = _sympify(el)

        sparse_array = Dict(sparse_array)

        self = Basic.__new__(cls, sparse_array, shape, **kwargs)
        self._shape = shape
        self._rank = len(shape)
        self._loop_size = loop_size
        self._sparse_array = sparse_array

        return self

    def __setitem__(self, index, value):
        raise TypeError("immutable N-dim array")

    def as_mutable(self):
        return MutableSparseNDimArray(self)


class MutableSparseNDimArray(MutableNDimArray, SparseNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        self = object.__new__(cls)
        self._shape = shape
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)

        # Sparse array:
        if isinstance(flat_list, (dict, Dict)):
            self._sparse_array = dict(flat_list)
            return self

        self._sparse_array = {}

        for i, el in enumerate(flatten(flat_list)):
            if el != 0:
                self._sparse_array[i] = _sympify(el)

        return self

    def __setitem__(self, index, value):
        """Allows to set items to MutableDenseNDimArray.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray.zeros(2, 2)
        >>> a[0, 0] = 1
        >>> a[1, 1] = 1
        >>> a
        [[1, 0], [0, 1]]
        """
        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            value, eindices, slice_offsets = self._get_slice_data_for_array_assignment(index, value)
            for i in eindices:
                other_i = [ind - j for ind, j in zip(i, slice_offsets) if j is not None]
                other_value = value[other_i]
                complete_index = self._parse_index(i)
                if other_value != 0:
                    self._sparse_array[complete_index] = other_value
                elif complete_index in self._sparse_array:
                    self._sparse_array.pop(complete_index)
        else:
            index = self._parse_index(index)
            value = _sympify(value)
            if value == 0 and index in self._sparse_array:
                self._sparse_array.pop(index)
            else:
                self._sparse_array[index] = value

    def as_immutable(self):
        return ImmutableSparseNDimArray(self)

    @property
    def free_symbols(self):
        return {i for j in self._sparse_array.values() for i in j.free_symbols}
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable

import itertools
from collections.abc import Iterable


class ArrayKind(Kind):
    """
    Kind for N-dimensional array in SymPy.

    This kind represents the multidimensional array that algebraic
    operations are defined. Basic class for this kind is ``NDimArray``,
    but any expression representing the array can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the array contains only numbers.

    Examples
    ========

    Any instance of array class has ``ArrayKind``.

    >>> from sympy import NDimArray
    >>> NDimArray([1,2,3]).kind
    ArrayKind(NumberKind)

    Although expressions representing an array may be not instance of
    array class, it will have ``ArrayKind`` as well.

    >>> from sympy import Integral
    >>> from sympy.tensor.array import NDimArray
    >>> from sympy.abc import x
    >>> intA = Integral(NDimArray([1,2,3]), x)
    >>> isinstance(intA, NDimArray)
    False
    >>> intA.kind
    ArrayKind(NumberKind)

    Use ``isinstance()`` to check for ``ArrayKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.

    >>> from sympy.tensor.array import ArrayKind
    >>> from sympy.core import NumberKind
    >>> boolA = NDimArray([True, False])
    >>> isinstance(boolA.kind, ArrayKind)
    True
    >>> boolA.kind is ArrayKind(NumberKind)
    False

    See Also
    ========

    shape : Function to return the shape of objects with ``MatrixKind``.

    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "ArrayKind(%s)" % self.element_kind

    @classmethod
    def _union(cls, kinds) -> 'ArrayKind':
        elem_kinds = {e.kind for e in kinds}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return ArrayKind(elemkind)


class NDimArray(Printable):
    """N-dimensional array.

    Examples
    ========

    Create an N-dim array of zeros:

    >>> from sympy import MutableDenseNDimArray
    >>> a = MutableDenseNDimArray.zeros(2, 3, 4)
    >>> a
    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    Create an N-dim array from a list;

    >>> a = MutableDenseNDimArray([[2, 3], [4, 5]])
    >>> a
    [[2, 3], [4, 5]]

    >>> b = MutableDenseNDimArray([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    >>> b
    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]

    Create an N-dim array from a flat list with dimension shape:

    >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
    >>> a
    [[1, 2, 3], [4, 5, 6]]

    Create an N-dim array from a matrix:

    >>> from sympy import Matrix
    >>> a = Matrix([[1,2],[3,4]])
    >>> a
    Matrix([
    [1, 2],
    [3, 4]])
    >>> b = MutableDenseNDimArray(a)
    >>> b
    [[1, 2], [3, 4]]

    Arithmetic operations on N-dim arrays

    >>> a = MutableDenseNDimArray([1, 1, 1, 1], (2, 2))
    >>> b = MutableDenseNDimArray([4, 4, 4, 4], (2, 2))
    >>> c = a + b
    >>> c
    [[5, 5], [5, 5]]
    >>> a - b
    [[-3, -3], [-3, -3]]

    """

    _diff_wrt = True
    is_scalar = False

    def __new__(cls, iterable, shape=None, **kwargs):
        from sympy.tensor.array import ImmutableDenseNDimArray
        return ImmutableDenseNDimArray(iterable, shape, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError("A subclass of NDimArray should implement __getitem__")

    def _parse_index(self, index):
        if isinstance(index, (SYMPY_INTS, Integer)):
            if index >= self._loop_size:
                raise ValueError("Only a tuple index is accepted")
            return index

        if self._loop_size == 0:
            raise ValueError("Index not valid with an empty array")

        if len(index) != self._rank:
            raise ValueError('Wrong number of array axes')

        real_index = 0
        # check if input index can exist in current indexing
        for i in range(self._rank):
            if (index[i] >= self.shape[i]) or (index[i] < -self.shape[i]):
                raise ValueError('Index ' + str(index) + ' out of border')
            if index[i] < 0:
                real_index += 1
            real_index = real_index*self.shape[i] + index[i]

        return real_index

    def _get_tuple_index(self, integer_index):
        index = []
        for sh in reversed(self.shape):
            index.append(integer_index % sh)
            integer_index //= sh
        index.reverse()
        return tuple(index)

    def _check_symbolic_index(self, index):
        # Check if any index is symbolic:
        tuple_index = (index if isinstance(index, tuple) else (index,))
        if any((isinstance(i, Expr) and (not i.is_number)) for i in tuple_index):
            for i, nth_dim in zip(tuple_index, self.shape):
                if ((i < 0) == True) or ((i >= nth_dim) == True):
                    raise ValueError("index out of range")
            from sympy.tensor import Indexed
            return Indexed(self, *tuple_index)
        return None

    def _setter_iterable_check(self, value):
        from sympy.matrices.matrixbase import MatrixBase
        if isinstance(value, (Iterable, MatrixBase, NDimArray)):
            raise NotImplementedError

    @classmethod
    def _scan_iterable_shape(cls, iterable):
        def f(pointer):
            if not isinstance(pointer, Iterable):
                return [pointer], ()

            if len(pointer) == 0:
                return [], (0,)

            result = []
            elems, shapes = zip(*[f(i) for i in pointer])
            if len(set(shapes)) != 1:
                raise ValueError("could not determine shape unambiguously")
            for i in elems:
                result.extend(i)
            return result, (len(shapes),)+shapes[0]

        return f(iterable)

    @classmethod
    def _handle_ndarray_creation_inputs(cls, iterable=None, shape=None, **kwargs):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray

        if shape is None:
            if iterable is None:
                shape = ()
                iterable = ()
            # Construction of a sparse array from a sparse array
            elif isinstance(iterable, SparseNDimArray):
                return iterable._shape, iterable._sparse_array

            # Construct N-dim array from another N-dim array:
            elif isinstance(iterable, NDimArray):
                shape = iterable.shape

            # Construct N-dim array from an iterable (numpy arrays included):
            elif isinstance(iterable, Iterable):
                iterable, shape = cls._scan_iterable_shape(iterable)

            # Construct N-dim array from a Matrix:
            elif isinstance(iterable, MatrixBase):
                shape = iterable.shape

            else:
                shape = ()
                iterable = (iterable,)

        if isinstance(iterable, (Dict, dict)) and shape is not None:
            new_dict = iterable.copy()
            for k in new_dict:
                if isinstance(k, (tuple, Tuple)):
                    new_key = 0
                    for i, idx in enumerate(k):
                        new_key = new_key * shape[i] + idx
                    iterable[new_key] = iterable[k]
                    del iterable[k]

        if isinstance(shape, (SYMPY_INTS, Integer)):
            shape = (shape,)

        if not all(isinstance(dim, (SYMPY_INTS, Integer)) for dim in shape):
            raise TypeError("Shape should contain integers only.")

        return tuple(shape), iterable

    def __len__(self):
        """Overload common function len(). Returns number of elements in array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        >>> len(a)
        9

        """
        return self._loop_size

    @property
    def shape(self):
        """
        Returns array shape (dimension).

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a.shape
        (3, 3)

        """
        return self._shape

    def rank(self):
        """
        Returns rank of array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3,4,5,6,3)
        >>> a.rank()
        5

        """
        return self._rank

    def diff(self, *args, **kwargs):
        """
        Calculate the derivative of each element in the array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> from sympy.abc import x, y
        >>> M = ImmutableDenseNDimArray([[x, y], [1, x*y]])
        >>> M.diff(x)
        [[1, 0], [0, y]]

        """
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        kwargs.setdefault('evaluate', True)
        return ArrayDerivative(self.as_immutable(), *args, **kwargs)

    def _eval_derivative(self, base):
        # Types are (base: scalar, self: array)
        return self.applyfunc(lambda x: base.diff(x))

    def _eval_derivative_n_times(self, s, n):
        return Basic._eval_derivative_n_times(self, s, n)

    def applyfunc(self, f):
        """Apply a function to each element of the N-dim array.

        Examples
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> m = ImmutableDenseNDimArray([i*2+j for i in range(2) for j in range(2)], (2, 2))
        >>> m
        [[0, 1], [2, 3]]
        >>> m.applyfunc(lambda i: 2*i)
        [[0, 2], [4, 6]]
        """
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray) and f(S.Zero) == 0:
            return type(self)({k: f(v) for k, v in self._sparse_array.items() if f(v) != 0}, self.shape)

        return type(self)(map(f, Flatten(self)), self.shape)

    def _sympystr(self, printer):
        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return "["+", ".join([printer._print(self[self._get_tuple_index(e)]) for e in range(i, j)])+"]"

            sh //= shape_left[0]
            return "[" + ", ".join([f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh) for e in range(shape_left[0])]) + "]" # + "\n"*len(shape_left)

        if self.rank() == 0:
            return printer._print(self[()])
        if 0 in self.shape:
            return f"{self.__class__.__name__}([], {self.shape})"
        return f(self._loop_size, self.shape, 0, self._loop_size)

    def tolist(self):
        """
        Converting MutableDenseNDimArray to one-dim list

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))
        >>> a
        [[1, 2], [3, 4]]
        >>> b = a.tolist()
        >>> b
        [[1, 2], [3, 4]]
        """

        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return [self[self._get_tuple_index(e)] for e in range(i, j)]
            result = []
            sh //= shape_left[0]
            for e in range(shape_left[0]):
                result.append(f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh))
            return result

        return f(self._loop_size, self.shape, 0, self._loop_size)

    def __add__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NDimArray):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i+j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NDimArray):
            return NotImplemented

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i-j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [i*other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        other = sympify(other)
        if isinstance(self, SparseNDimArray):
            if other.is_zero:
                return type(self)({}, self.shape)
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [other*i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __truediv__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected")

        other = sympify(other)
        if isinstance(self, SparseNDimArray) and other != S.Zero:
            return type(self)({k: v/other for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [i/other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        raise NotImplementedError('unsupported operation on NDimArray')

    def __neg__(self):
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray):
            return type(self)({k: -v for (k, v) in self._sparse_array.items()}, self.shape)

        result_list = [-i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __iter__(self):
        def iterator():
            if self._shape:
                for i in range(self._shape[0]):
                    yield self[i]
            else:
                yield self[()]

        return iterator()

    def __eq__(self, other):
        """
        NDimArray instances can be compared to each other.
        Instances equal if they have same shape and data.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2, 3)
        >>> b = MutableDenseNDimArray.zeros(2, 3)
        >>> a == b
        True
        >>> c = a.reshape(3, 2)
        >>> c == b
        False
        >>> a[0,0] = 1
        >>> b[0,0] = 2
        >>> a == b
        False
        """
        from sympy.tensor.array import SparseNDimArray
        if not isinstance(other, NDimArray):
            return False

        if not self.shape == other.shape:
            return False

        if isinstance(self, SparseNDimArray) and isinstance(other, SparseNDimArray):
            return dict(self._sparse_array) == dict(other._sparse_array)

        return list(self) == list(other)

    def __ne__(self, other):
        return not self == other

    def _eval_transpose(self):
        if self.rank() != 2:
            raise ValueError("array rank not 2")
        from .arrayop import permutedims
        return permutedims(self, (1, 0))

    def transpose(self):
        return self._eval_transpose()

    def _eval_conjugate(self):
        from sympy.tensor.array.arrayop import Flatten

        return self.func([i.conjugate() for i in Flatten(self)], self.shape)

    def conjugate(self):
        return self._eval_conjugate()

    def _eval_adjoint(self):
        return self.transpose().conjugate()

    def adjoint(self):
        return self._eval_adjoint()

    def _slice_expand(self, s, dim):
        if not isinstance(s, slice):
                return (s,)
        start, stop, step = s.indices(dim)
        return [start + i*step for i in range((stop-start)//step)]

    def _get_slice_data_for_array_access(self, index):
        sl_factors = [self._slice_expand(i, dim) for (i, dim) in zip(index, self.shape)]
        eindices = itertools.product(*sl_factors)
        return sl_factors, eindices

    def _get_slice_data_for_array_assignment(self, index, value):
        if not isinstance(value, NDimArray):
            value = type(self)(value)
        sl_factors, eindices = self._get_slice_data_for_array_access(index)
        slice_offsets = [min(i) if isinstance(i, list) else None for i in sl_factors]
        # TODO: add checks for dimensions for `value`?
        return value, eindices, slice_offsets

    @classmethod
    def _check_special_bounds(cls, flat_list, shape):
        if shape == () and len(flat_list) != 1:
            raise ValueError("arrays without shape need one scalar value")
        if shape == (0,) and len(flat_list) > 0:
            raise ValueError("if array shape is (0,) there cannot be elements")

    def _check_index_for_getitem(self, index):
        if isinstance(index, (SYMPY_INTS, Integer, slice)):
            index = (index,)

        if len(index) < self.rank():
            index = tuple(index) + \
                          tuple(slice(None) for i in range(len(index), self.rank()))

        if len(index) > self.rank():
            raise ValueError('Dimension of index greater than rank of array')

        return index


class ImmutableNDimArray(NDimArray, Basic):
    _op_priority = 11.0

    def __hash__(self):
        return Basic.__hash__(self)

    def as_immutable(self):
        return self

    def as_mutable(self):
        raise NotImplementedError("abstract method")
<|endoftext|>import operator
from functools import reduce, singledispatch

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.matrices.expressions.hadamard import HadamardProduct
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import (MatrixExpr, MatrixSymbol)
from sympy.matrices.expressions.special import Identity, OneMatrix
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.tensor.array.expressions.array_expressions import (
    _ArrayExpr, ZeroArray, ArraySymbol, ArrayTensorProduct, ArrayAdd,
    PermuteDims, ArrayDiagonal, ArrayElementwiseApplyFunc, get_rank,
    get_shape, ArrayContraction, _array_tensor_product, _array_contraction,
    _array_diagonal, _array_add, _permute_dims, Reshape)
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array


@singledispatch
def array_derive(expr, x):
    """
    Derivatives (gradients) for array expressions.
    """
    raise NotImplementedError(f"not implemented for type {type(expr)}")


@array_derive.register(Expr)
