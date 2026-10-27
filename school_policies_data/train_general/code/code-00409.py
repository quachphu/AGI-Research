class WicksTheoremDoesNotApply(SecondQuantizationError):
    pass


class Dagger(Expr):
    """
    Hermitian conjugate of creation/annihilation operators.

    Examples
    ========

    >>> from sympy import I
    >>> from sympy.physics.secondquant import Dagger, B, Bd
    >>> Dagger(2*I)
    -2*I
    >>> Dagger(B(0))
    CreateBoson(0)
    >>> Dagger(Bd(0))
    AnnihilateBoson(0)

    """

    def __new__(cls, arg):
        arg = sympify(arg)
        r = cls.eval(arg)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, arg)
        return obj

    @classmethod
    def eval(cls, arg):
        """
        Evaluates the Dagger instance.

        Examples
        ========

        >>> from sympy import I
        >>> from sympy.physics.secondquant import Dagger, B, Bd
        >>> Dagger(2*I)
        -2*I
        >>> Dagger(B(0))
        CreateBoson(0)
        >>> Dagger(Bd(0))
        AnnihilateBoson(0)

        The eval() method is called automatically.

        """
        dagger = getattr(arg, '_dagger_', None)
        if dagger is not None:
            return dagger()
        if isinstance(arg, Symbol) and arg.is_commutative:
            return conjugate(arg)
        if isinstance(arg, Basic):
            if arg.is_Add:
                return Add(*tuple(map(Dagger, arg.args)))
            if arg.is_Mul:
                return Mul(*tuple(map(Dagger, reversed(arg.args))))
            if arg.is_Number:
                return arg
            if arg.is_Pow:
                return Pow(Dagger(arg.args[0]), arg.args[1])
            if arg == I:
                return -arg
        if isinstance(arg, Function):
            if all(a.is_commutative for a in arg.args):
                return arg.func(*[Dagger(a) for a in arg.args])
        else:
            return None

    def _dagger_(self):
        return self.args[0]


class TensorSymbol(Expr):

    is_commutative = True


class AntiSymmetricTensor(TensorSymbol):
    """Stores upper and lower indices in separate Tuple's.

    Each group of indices is assumed to be antisymmetric.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import AntiSymmetricTensor
    >>> i, j = symbols('i j', below_fermi=True)
    >>> a, b = symbols('a b', above_fermi=True)
    >>> AntiSymmetricTensor('v', (a, i), (b, j))
    AntiSymmetricTensor(v, (a, i), (b, j))
    >>> AntiSymmetricTensor('v', (i, a), (b, j))
    -AntiSymmetricTensor(v, (a, i), (b, j))

    As you can see, the indices are automatically sorted to a canonical form.

    """

    def __new__(cls, symbol, upper, lower):

        try:
            upper, signu = _sort_anticommuting_fermions(
                upper, key=_sqkey_index)
            lower, signl = _sort_anticommuting_fermions(
                lower, key=_sqkey_index)

        except ViolationOfPauliPrinciple:
            return S.Zero

        symbol = sympify(symbol)
        upper = Tuple(*upper)
        lower = Tuple(*lower)

        if (signu + signl) % 2:
            return -TensorSymbol.__new__(cls, symbol, upper, lower)
        else:

            return TensorSymbol.__new__(cls, symbol, upper, lower)

    def _latex(self, printer):
        return "{%s^{%s}_{%s}}" % (
            self.symbol,
            "".join([ printer._print(i) for i in self.args[1]]),
            "".join([ printer._print(i) for i in self.args[2]])
        )

    @property
    def symbol(self):
        """
        Returns the symbol of the tensor.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).symbol
        v

        """
        return self.args[0]

    @property
    def upper(self):
        """
        Returns the upper indices.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).upper
        (a, i)


        """
        return self.args[1]

    @property
    def lower(self):
        """
        Returns the lower indices.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import AntiSymmetricTensor
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> AntiSymmetricTensor('v', (a, i), (b, j))
        AntiSymmetricTensor(v, (a, i), (b, j))
        >>> AntiSymmetricTensor('v', (a, i), (b, j)).lower
        (b, j)

        """
        return self.args[2]

    def __str__(self):
        return "%s(%s,%s)" % self.args


class SqOperator(Expr):
    """
    Base class for Second Quantization operators.
    """

    op_symbol = 'sq'

    is_commutative = False

    def __new__(cls, k):
        obj = Basic.__new__(cls, sympify(k))
        return obj

    @property
    def state(self):
        """
        Returns the state index related to this operator.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F, Fd, B, Bd
        >>> p = Symbol('p')
        >>> F(p).state
        p
        >>> Fd(p).state
        p
        >>> B(p).state
        p
        >>> Bd(p).state
        p

        """
        return self.args[0]

    @property
    def is_symbolic(self):
        """
        Returns True if the state is a symbol (as opposed to a number).

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> p = Symbol('p')
        >>> F(p).is_symbolic
        True
        >>> F(1).is_symbolic
        False

        """
        if self.state.is_Integer:
            return False
        else:
            return True

    def __repr__(self):
        return NotImplemented

    def __str__(self):
        return "%s(%r)" % (self.op_symbol, self.state)

    def apply_operator(self, state):
        """
        Applies an operator to itself.
        """
        raise NotImplementedError('implement apply_operator in a subclass')


class BosonicOperator(SqOperator):
    pass


class Annihilator(SqOperator):
    pass


class Creator(SqOperator):
    pass


class AnnihilateBoson(BosonicOperator, Annihilator):
    """
    Bosonic annihilation operator.

    Examples
    ========

    >>> from sympy.physics.secondquant import B
    >>> from sympy.abc import x
    >>> B(x)
    AnnihilateBoson(x)
    """

    op_symbol = 'b'

    def _dagger_(self):
        return CreateBoson(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, BKet
        >>> from sympy.abc import x, y, n
        >>> B(x).apply_operator(y)
        y*AnnihilateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))

        """
        if not self.is_symbolic and isinstance(state, FockStateKet):
            element = self.state
            amp = sqrt(state[element])
            return amp*state.down(element)
        else:
            return Mul(self, state)

    def __repr__(self):
        return "AnnihilateBoson(%s)" % self.state

    def _latex(self, printer):
        if self.state is S.Zero:
            return "b_{0}"
        else:
            return "b_{%s}" % printer._print(self.state)

class CreateBoson(BosonicOperator, Creator):
    """
    Bosonic creation operator.
    """

    op_symbol = 'b+'

    def _dagger_(self):
        return AnnihilateBoson(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if not self.is_symbolic and isinstance(state, FockStateKet):
            element = self.state
            amp = sqrt(state[element] + 1)
            return amp*state.up(element)
        else:
            return Mul(self, state)

    def __repr__(self):
        return "CreateBoson(%s)" % self.state

    def _latex(self, printer):
        if self.state is S.Zero:
            return "{b^\\dagger_{0}}"
        else:
            return "{b^\\dagger_{%s}}" % printer._print(self.state)

B = AnnihilateBoson
Bd = CreateBoson


class FermionicOperator(SqOperator):

    @property
    def is_restricted(self):
        """
        Is this FermionicOperator restricted with respect to fermi level?

        Returns
        =======

        1  : restricted to orbits above fermi
        0  : no restriction
        -1 : restricted to orbits below fermi

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F, Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_restricted
        1
        >>> Fd(a).is_restricted
        1
        >>> F(i).is_restricted
        -1
        >>> Fd(i).is_restricted
        -1
        >>> F(p).is_restricted
        0
        >>> Fd(p).is_restricted
        0

        """
        ass = self.args[0].assumptions0
        if ass.get("below_fermi"):
            return -1
        if ass.get("above_fermi"):
            return 1
        return 0

    @property
    def is_above_fermi(self):
        """
        Does the index of this FermionicOperator allow values above fermi?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_above_fermi
        True
        >>> F(i).is_above_fermi
        False
        >>> F(p).is_above_fermi
        True

        Note
        ====

        The same applies to creation operators Fd

        """
        return not self.args[0].assumptions0.get("below_fermi")

    @property
    def is_below_fermi(self):
        """
        Does the index of this FermionicOperator allow values below fermi?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_below_fermi
        False
        >>> F(i).is_below_fermi
        True
        >>> F(p).is_below_fermi
        True

        The same applies to creation operators Fd

        """
        return not self.args[0].assumptions0.get("above_fermi")

    @property
    def is_only_below_fermi(self):
        """
        Is the index of this FermionicOperator restricted to values below fermi?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_only_below_fermi
        False
        >>> F(i).is_only_below_fermi
        True
        >>> F(p).is_only_below_fermi
        False

        The same applies to creation operators Fd
        """
        return self.is_below_fermi and not self.is_above_fermi

    @property
    def is_only_above_fermi(self):
        """
        Is the index of this FermionicOperator restricted to values above fermi?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_only_above_fermi
        True
        >>> F(i).is_only_above_fermi
        False
        >>> F(p).is_only_above_fermi
        False

        The same applies to creation operators Fd
        """
        return self.is_above_fermi and not self.is_below_fermi

    def _sortkey(self):
        h = hash(self)
        label = str(self.args[0])

        if self.is_only_q_creator:
            return 1, label, h
        if self.is_only_q_annihilator:
            return 4, label, h
        if isinstance(self, Annihilator):
            return 3, label, h
        if isinstance(self, Creator):
            return 2, label, h


class AnnihilateFermion(FermionicOperator, Annihilator):
    """
    Fermionic annihilation operator.
    """

    op_symbol = 'f'

    def _dagger_(self):
        return CreateFermion(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if isinstance(state, FockStateFermionKet):
            element = self.state
            return state.down(element)

        elif isinstance(state, Mul):
            c_part, nc_part = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                element = self.state
                return Mul(*(c_part + [nc_part[0].down(element)] + nc_part[1:]))
            else:
                return Mul(self, state)

        else:
            return Mul(self, state)

    @property
    def is_q_creator(self):
        """
        Can we create a quasi-particle?  (create hole or create particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_q_creator
        0
        >>> F(i).is_q_creator
        -1
        >>> F(p).is_q_creator
        -1

        """
        if self.is_below_fermi:
            return -1
        return 0

    @property
    def is_q_annihilator(self):
        """
        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=1)
        >>> i = Symbol('i', below_fermi=1)
        >>> p = Symbol('p')

        >>> F(a).is_q_annihilator
        1
        >>> F(i).is_q_annihilator
        0
        >>> F(p).is_q_annihilator
        1

        """
        if self.is_above_fermi:
            return 1
        return 0

    @property
    def is_only_q_creator(self):
        """
        Always create a quasi-particle?  (create hole or create particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_only_q_creator
        False
        >>> F(i).is_only_q_creator
        True
        >>> F(p).is_only_q_creator
        False

        """
        return self.is_only_below_fermi

    @property
    def is_only_q_annihilator(self):
        """
        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import F
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> F(a).is_only_q_annihilator
        True
        >>> F(i).is_only_q_annihilator
        False
        >>> F(p).is_only_q_annihilator
        False

        """
        return self.is_only_above_fermi

    def __repr__(self):
        return "AnnihilateFermion(%s)" % self.state

    def _latex(self, printer):
        if self.state is S.Zero:
            return "a_{0}"
        else:
            return "a_{%s}" % printer._print(self.state)


class CreateFermion(FermionicOperator, Creator):
    """
    Fermionic creation operator.
    """

    op_symbol = 'f+'

    def _dagger_(self):
        return AnnihilateFermion(self.state)

    def apply_operator(self, state):
        """
        Apply state to self if self is not symbolic and state is a FockStateKet, else
        multiply self by state.

        Examples
        ========

        >>> from sympy.physics.secondquant import B, Dagger, BKet
        >>> from sympy.abc import x, y, n
        >>> Dagger(B(x)).apply_operator(y)
        y*CreateBoson(x)
        >>> B(0).apply_operator(BKet((n,)))
        sqrt(n)*FockStateBosonKet((n - 1,))
        """
        if isinstance(state, FockStateFermionKet):
            element = self.state
            return state.up(element)

        elif isinstance(state, Mul):
            c_part, nc_part = state.args_cnc()
            if isinstance(nc_part[0], FockStateFermionKet):
                element = self.state
                return Mul(*(c_part + [nc_part[0].up(element)] + nc_part[1:]))

        return Mul(self, state)

    @property
    def is_q_creator(self):
        """
        Can we create a quasi-particle?  (create hole or create particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_creator
        1
        >>> Fd(i).is_q_creator
        0
        >>> Fd(p).is_q_creator
        1

        """
        if self.is_above_fermi:
            return 1
        return 0

    @property
    def is_q_annihilator(self):
        """
        Can we destroy a quasi-particle?  (annihilate hole or annihilate particle)
        If so, would that be above or below the fermi surface?

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=1)
        >>> i = Symbol('i', below_fermi=1)
        >>> p = Symbol('p')

        >>> Fd(a).is_q_annihilator
        0
        >>> Fd(i).is_q_annihilator
        -1
        >>> Fd(p).is_q_annihilator
        -1

        """
        if self.is_below_fermi:
            return -1
        return 0

    @property
    def is_only_q_creator(self):
        """
        Always create a quasi-particle?  (create hole or create particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_only_q_creator
        True
        >>> Fd(i).is_only_q_creator
        False
        >>> Fd(p).is_only_q_creator
        False

        """
        return self.is_only_above_fermi

    @property
    def is_only_q_annihilator(self):
        """
        Always destroy a quasi-particle?  (annihilate hole or annihilate particle)

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import Fd
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> Fd(a).is_only_q_annihilator
        False
        >>> Fd(i).is_only_q_annihilator
        True
        >>> Fd(p).is_only_q_annihilator
        False

        """
        return self.is_only_below_fermi

    def __repr__(self):
        return "CreateFermion(%s)" % self.state

    def _latex(self, printer):
        if self.state is S.Zero:
            return "{a^\\dagger_{0}}"
        else:
            return "{a^\\dagger_{%s}}" % printer._print(self.state)

Fd = CreateFermion
F = AnnihilateFermion


class FockState(Expr):
    """
    Many particle Fock state with a sequence of occupation numbers.

    Anywhere you can have a FockState, you can also have S.Zero.
    All code must check for this!

    Base class to represent FockStates.
    """
    is_commutative = False

    def __new__(cls, occupations):
        """
        occupations is a list with two possible meanings:

        - For bosons it is a list of occupation numbers.
          Element i is the number of particles in state i.

        - For fermions it is a list of occupied orbits.
          Element 0 is the state that was occupied first, element i
          is the i'th occupied state.
        """
        occupations = list(map(sympify, occupations))
        obj = Basic.__new__(cls, Tuple(*occupations))
        return obj

    def __getitem__(self, i):
        i = int(i)
        return self.args[0][i]

    def __repr__(self):
        return ("FockState(%r)") % (self.args)

    def __str__(self):
        return "%s%r%s" % (getattr(self, 'lbracket', ""), self._labels(), getattr(self, 'rbracket', ""))

    def _labels(self):
        return self.args[0]

    def __len__(self):
        return len(self.args[0])

    def _latex(self, printer):
        return "%s%s%s" % (getattr(self, 'lbracket_latex', ""), printer._print(self._labels()), getattr(self, 'rbracket_latex', ""))


class BosonState(FockState):
    """
    Base class for FockStateBoson(Ket/Bra).
    """

    def up(self, i):
        """
        Performs the action of a creation operator.

        Examples
        ========

        >>> from sympy.physics.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        FockStateBosonBra((1, 2))
        >>> b.up(1)
        FockStateBosonBra((1, 3))
        """
        i = int(i)
        new_occs = list(self.args[0])
        new_occs[i] = new_occs[i] + S.One
        return self.__class__(new_occs)

    def down(self, i):
        """
        Performs the action of an annihilation operator.

        Examples
        ========

        >>> from sympy.physics.secondquant import BBra
        >>> b = BBra([1, 2])
        >>> b
        FockStateBosonBra((1, 2))
        >>> b.down(1)
        FockStateBosonBra((1, 1))
        """
        i = int(i)
        new_occs = list(self.args[0])
        if new_occs[i] == S.Zero:
            return S.Zero
        else:
            new_occs[i] = new_occs[i] - S.One
            return self.__class__(new_occs)


class FermionState(FockState):
    """
    Base class for FockStateFermion(Ket/Bra).
    """

    fermi_level = 0

    def __new__(cls, occupations, fermi_level=0):
        occupations = list(map(sympify, occupations))
        if len(occupations) > 1:
            try:
                (occupations, sign) = _sort_anticommuting_fermions(
                    occupations, key=_sqkey_index)
            except ViolationOfPauliPrinciple:
                return S.Zero
        else:
            sign = 0

        cls.fermi_level = fermi_level

        if cls._count_holes(occupations) > fermi_level:
            return S.Zero

        if sign % 2:
            return S.NegativeOne*FockState.__new__(cls, occupations)
        else:
            return FockState.__new__(cls, occupations)

    def up(self, i):
        """
        Performs the action of a creation operator.

        Explanation
        ===========

        If below fermi we try to remove a hole,
        if above fermi we try to create a particle.

        If general index p we return ``Kronecker(p,i)*self``
        where ``i`` is a new symbol with restriction above or below.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import FKet
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        >>> FKet([]).up(a)
        FockStateFermionKet((a,))

        A creator acting on vacuum below fermi vanishes

        >>> FKet([]).up(i)
        0


        """
        present = i in self.args[0]

        if self._only_above_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        elif self._only_below_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero
        else:
            if present:
                hole = Dummy("i", below_fermi=True)
                return KroneckerDelta(i, hole)*self._remove_orbit(i)
            else:
                particle = Dummy("a", above_fermi=True)
                return KroneckerDelta(i, particle)*self._add_orbit(i)

    def down(self, i):
        """
        Performs the action of an annihilation operator.

        Explanation
        ===========

        If below fermi we try to create a hole,
        If above fermi we try to remove a particle.

        If general index p we return ``Kronecker(p,i)*self``
        where ``i`` is a new symbol with restriction above or below.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.secondquant import FKet
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')

        An annihilator acting on vacuum above fermi vanishes

        >>> FKet([]).down(a)
        0

        Also below fermi, it vanishes, unless we specify a fermi level > 0

        >>> FKet([]).down(i)
        0
        >>> FKet([],4).down(i)
        FockStateFermionKet((i,))

        """
        present = i in self.args[0]

        if self._only_above_fermi(i):
            if present:
                return self._remove_orbit(i)
            else:
                return S.Zero

        elif self._only_below_fermi(i):
            if present:
                return S.Zero
            else:
                return self._add_orbit(i)
        else:
            if present:
                hole = Dummy("i", below_fermi=True)
                return KroneckerDelta(i, hole)*self._add_orbit(i)
            else:
                particle = Dummy("a", above_fermi=True)
                return KroneckerDelta(i, particle)*self._remove_orbit(i)

    @classmethod
    def _only_below_fermi(cls, i):
        """
        Tests if given orbit is only below fermi surface.

        If nothing can be concluded we return a conservative False.
        """
        if i.is_number:
            return i <= cls.fermi_level
        if i.assumptions0.get('below_fermi'):
            return True
        return False

    @classmethod
    def _only_above_fermi(cls, i):
        """
        Tests if given orbit is only above fermi surface.

        If fermi level has not been set we return True.
        If nothing can be concluded we return a conservative False.
        """
        if i.is_number:
            return i > cls.fermi_level
        if i.assumptions0.get('above_fermi'):
            return True
        return not cls.fermi_level

    def _remove_orbit(self, i):
        """
        Removes particle/fills hole in orbit i. No input tests performed here.
        """
        new_occs = list(self.args[0])
        pos = new_occs.index(i)
        del new_occs[pos]
        if (pos) % 2:
            return S.NegativeOne*self.__class__(new_occs, self.fermi_level)
        else:
            return self.__class__(new_occs, self.fermi_level)

    def _add_orbit(self, i):
        """
        Adds particle/creates hole in orbit i. No input tests performed here.
        """
        return self.__class__((i,) + self.args[0], self.fermi_level)

    @classmethod
    def _count_holes(cls, occupations):
        """
        Returns the number of identified hole states in occupations list.
        """
        return len([i for i in occupations if cls._only_below_fermi(i)])

    def _negate_holes(self, occupations):
        """
        Returns the occupations list where states below the fermi level have negative labels.

        For symbolic state labels, no sign is included.
        """
        return tuple([-i if self._only_below_fermi(i) and i.is_number else i for i in occupations])

    def __repr__(self):
        if self.fermi_level:
            return "FockStateKet(%r, fermi_level=%s)" % (self.args[0], self.fermi_level)
        else:
            return "FockStateKet(%r)" % (self.args[0],)

    def _labels(self):
        return self._negate_holes(self.args[0])


class FockStateKet(FockState):
    """
    Representation of a ket.
    """
    lbracket = '|'
    rbracket = '>'
    lbracket_latex = r'\left|'
    rbracket_latex = r'\right\rangle'


class FockStateBra(FockState):
    """
    Representation of a bra.
    """
    lbracket = '<'
    rbracket = '|'
    lbracket_latex = r'\left\langle'
    rbracket_latex = r'\right|'

    def __mul__(self, other):
        if isinstance(other, FockStateKet):
            return InnerProduct(self, other)
        else:
            return Expr.__mul__(self, other)


class FockStateBosonKet(BosonState, FockStateKet):
    """
    Many particle Fock state with a sequence of occupation numbers.

    Occupation numbers can be any integer >= 0.

    Examples
    ========

    >>> from sympy.physics.secondquant import BKet
    >>> BKet([1, 2])
    FockStateBosonKet((1, 2))
    """
    def _dagger_(self):
        return FockStateBosonBra(*self.args)


class FockStateBosonBra(BosonState, FockStateBra):
    """
    Describes a collection of BosonBra particles.

    Examples
    ========

    >>> from sympy.physics.secondquant import BBra
    >>> BBra([1, 2])
    FockStateBosonBra((1, 2))
    """
    def _dagger_(self):
        return FockStateBosonKet(*self.args)


class FockStateFermionKet(FermionState, FockStateKet):
    """
    Many-particle Fock state with a sequence of occupied orbits.

    Explanation
    ===========

    Each state can only have one particle, so we choose to store a list of
    occupied orbits rather than a tuple with occupation numbers (zeros and ones).

    states below fermi level are holes, and are represented by negative labels
    in the occupation list.

    For symbolic state labels, the fermi_level caps the number of allowed hole-
    states.

    Examples
    ========

    >>> from sympy.physics.secondquant import FKet
    >>> FKet([1, 2])
    FockStateFermionKet((1, 2))
    """
    def _dagger_(self):
        return FockStateFermionBra(*self.args)


class FockStateFermionBra(FermionState, FockStateBra):
    """
    See Also
    ========

    FockStateFermionKet

    Examples
    ========

    >>> from sympy.physics.secondquant import FBra
    >>> FBra([1, 2])
    FockStateFermionBra((1, 2))
    """
    def _dagger_(self):
        return FockStateFermionKet(*self.args)

BBra = FockStateBosonBra
BKet = FockStateBosonKet
FBra = FockStateFermionBra
FKet = FockStateFermionKet


def _apply_Mul(m):
    """
    Take a Mul instance with operators and apply them to states.

    Explanation
    ===========

    This method applies all operators with integer state labels
    to the actual states.  For symbolic state labels, nothing is done.
    When inner products of FockStates are encountered (like <a|b>),
    they are converted to instances of InnerProduct.

    This does not currently work on double inner products like,
    <a|b><c|d>.

    If the argument is not a Mul, it is simply returned as is.
    """
    if not isinstance(m, Mul):
        return m
    c_part, nc_part = m.args_cnc()
    n_nc = len(nc_part)
    if n_nc in (0, 1):
        return m
    else:
        last = nc_part[-1]
        next_to_last = nc_part[-2]
        if isinstance(last, FockStateKet):
            if isinstance(next_to_last, SqOperator):
                if next_to_last.is_symbolic:
                    return m
                else:
                    result = next_to_last.apply_operator(last)
                    if result == 0:
                        return S.Zero
                    else:
                        return _apply_Mul(Mul(*(c_part + nc_part[:-2] + [result])))
            elif isinstance(next_to_last, Pow):
                if isinstance(next_to_last.base, SqOperator) and \
                        next_to_last.exp.is_Integer:
                    if next_to_last.base.is_symbolic:
                        return m
                    else:
                        result = last
                        for i in range(next_to_last.exp):
                            result = next_to_last.base.apply_operator(result)
                            if result == 0:
                                break
                        if result == 0:
                            return S.Zero
                        else:
                            return _apply_Mul(Mul(*(c_part + nc_part[:-2] + [result])))
                else:
                    return m
            elif isinstance(next_to_last, FockStateBra):
                result = InnerProduct(next_to_last, last)
                if result == 0:
                    return S.Zero
                else:
                    return _apply_Mul(Mul(*(c_part + nc_part[:-2] + [result])))
            else:
                return m
        else:
            return m


def apply_operators(e):
    """
    Take a SymPy expression with operators and states and apply the operators.

    Examples
    ========

    >>> from sympy.physics.secondquant import apply_operators
    >>> from sympy import sympify
    >>> apply_operators(sympify(3)+4)
    7
    """
    e = e.expand()
    muls = e.atoms(Mul)
    subs_list = [(m, _apply_Mul(m)) for m in iter(muls)]
    return e.subs(subs_list)


class InnerProduct(Basic):
    """
    An unevaluated inner product between a bra and ket.

    Explanation
    ===========

    Currently this class just reduces things to a product of
    Kronecker Deltas.  In the future, we could introduce abstract
    states like ``|a>`` and ``|b>``, and leave the inner product unevaluated as
    ``<a|b>``.

    """
    is_commutative = True

    def __new__(cls, bra, ket):
        if not isinstance(bra, FockStateBra):
            raise TypeError("must be a bra")
        if not isinstance(ket, FockStateKet):
            raise TypeError("must be a ket")
        return cls.eval(bra, ket)

    @classmethod
    def eval(cls, bra, ket):
        result = S.One
        for i, j in zip(bra.args[0], ket.args[0]):
            result *= KroneckerDelta(i, j)
            if result == 0:
                break
        return result

    @property
    def bra(self):
        """Returns the bra part of the state"""
        return self.args[0]

    @property
    def ket(self):
        """Returns the ket part of the state"""
        return self.args[1]

    def __repr__(self):
        sbra = repr(self.bra)
        sket = repr(self.ket)
        return "%s|%s" % (sbra[:-1], sket[1:])

    def __str__(self):
        return self.__repr__()


def matrix_rep(op, basis):
    """
    Find the representation of an operator in a basis.

    Examples
    ========

    >>> from sympy.physics.secondquant import VarBosonicBasis, B, matrix_rep
    >>> b = VarBosonicBasis(5)
    >>> o = B(0)
    >>> matrix_rep(o, b)
    Matrix([
    [0, 1,       0,       0, 0],
    [0, 0, sqrt(2),       0, 0],
    [0, 0,       0, sqrt(3), 0],
    [0, 0,       0,       0, 2],
    [0, 0,       0,       0, 0]])
    """
    a = zeros(len(basis))
    for i in range(len(basis)):
        for j in range(len(basis)):
            a[i, j] = apply_operators(Dagger(basis[i])*op*basis[j])
    return a


class BosonicBasis:
    """
    Base class for a basis set of bosonic Fock states.
    """
    pass


class VarBosonicBasis:
    """
    A single state, variable particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import VarBosonicBasis
    >>> b = VarBosonicBasis(5)
    >>> b
    [FockState((0,)), FockState((1,)), FockState((2,)),
     FockState((3,)), FockState((4,))]
    """

    def __init__(self, n_max):
        self.n_max = n_max
        self._build_states()

    def _build_states(self):
        self.basis = []
        for i in range(self.n_max):
            self.basis.append(FockStateBosonKet([i]))
        self.n_basis = len(self.basis)

    def index(self, state):
        """
        Returns the index of state in basis.

        Examples
        ========

        >>> from sympy.physics.secondquant import VarBosonicBasis
        >>> b = VarBosonicBasis(3)
        >>> state = b.state(1)
        >>> b
        [FockState((0,)), FockState((1,)), FockState((2,))]
        >>> state
        FockStateBosonKet((1,))
        >>> b.index(state)
        1
        """
        return self.basis.index(state)

    def state(self, i):
        """
        The state of a single basis.

        Examples
        ========

        >>> from sympy.physics.secondquant import VarBosonicBasis
        >>> b = VarBosonicBasis(5)
        >>> b.state(3)
        FockStateBosonKet((3,))
        """
        return self.basis[i]

    def __getitem__(self, i):
        return self.state(i)

    def __len__(self):
        return len(self.basis)

    def __repr__(self):
        return repr(self.basis)


class FixedBosonicBasis(BosonicBasis):
    """
    Fixed particle number basis set.

    Examples
    ========

    >>> from sympy.physics.secondquant import FixedBosonicBasis
    >>> b = FixedBosonicBasis(2, 2)
    >>> state = b.state(1)
    >>> b
    [FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]
    >>> state
    FockStateBosonKet((1, 1))
    >>> b.index(state)
    1
    """
    def __init__(self, n_particles, n_levels):
        self.n_particles = n_particles
        self.n_levels = n_levels
        self._build_particle_locations()
        self._build_states()

    def _build_particle_locations(self):
        tup = ["i%i" % i for i in range(self.n_particles)]
        first_loop = "for i0 in range(%i)" % self.n_levels
        other_loops = ''
        for cur, prev in zip(tup[1:], tup):
            temp = "for %s in range(%s + 1) " % (cur, prev)
            other_loops = other_loops + temp
        tup_string = "(%s)" % ", ".join(tup)
        list_comp = "[%s %s %s]" % (tup_string, first_loop, other_loops)
        result = eval(list_comp)
        if self.n_particles == 1:
            result = [(item,) for item in result]
        self.particle_locations = result

    def _build_states(self):
        self.basis = []
        for tuple_of_indices in self.particle_locations:
            occ_numbers = self.n_levels*[0]
            for level in tuple_of_indices:
                occ_numbers[level] += 1
            self.basis.append(FockStateBosonKet(occ_numbers))
        self.n_basis = len(self.basis)

    def index(self, state):
        """Returns the index of state in basis.

        Examples
        ========

        >>> from sympy.physics.secondquant import FixedBosonicBasis
        >>> b = FixedBosonicBasis(2, 3)
        >>> b.index(b.state(3))
        3
        """
        return self.basis.index(state)

    def state(self, i):
        """Returns the state that lies at index i of the basis

        Examples
        ========

        >>> from sympy.physics.secondquant import FixedBosonicBasis
        >>> b = FixedBosonicBasis(2, 3)
        >>> b.state(3)
        FockStateBosonKet((1, 0, 1))
        """
        return self.basis[i]

    def __getitem__(self, i):
        return self.state(i)

    def __len__(self):
        return len(self.basis)

    def __repr__(self):
        return repr(self.basis)


class Commutator(Function):
    """
    The Commutator:  [A, B] = A*B - B*A

    The arguments are ordered according to .__cmp__()

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import Commutator
    >>> A, B = symbols('A,B', commutative=False)
    >>> Commutator(B, A)
    -Commutator(A, B)

    Evaluate the commutator with .doit()

    >>> comm = Commutator(A,B); comm
    Commutator(A, B)
    >>> comm.doit()
    A*B - B*A


    For two second quantization operators the commutator is evaluated
    immediately:

    >>> from sympy.physics.secondquant import Fd, F
    >>> a = symbols('a', above_fermi=True)
    >>> i = symbols('i', below_fermi=True)
    >>> p,q = symbols('p,q')

    >>> Commutator(Fd(a),Fd(i))
    2*NO(CreateFermion(a)*CreateFermion(i))

    But for more complicated expressions, the evaluation is triggered by
    a call to .doit()

    >>> comm = Commutator(Fd(p)*Fd(q),F(i)); comm
    Commutator(CreateFermion(p)*CreateFermion(q), AnnihilateFermion(i))
    >>> comm.doit(wicks=True)
    -KroneckerDelta(i, p)*CreateFermion(q) +
     KroneckerDelta(i, q)*CreateFermion(p)

    """

    is_commutative = False

    @classmethod
    def eval(cls, a, b):
        """
        The Commutator [A,B] is on canonical form if A < B.

        Examples
        ========

        >>> from sympy.physics.secondquant import Commutator, F, Fd
        >>> from sympy.abc import x
        >>> c1 = Commutator(F(x), Fd(x))
        >>> c2 = Commutator(Fd(x), F(x))
        >>> Commutator.eval(c1, c2)
        0
        """
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero

        #
        # [A+B,C]  ->  [A,C] + [B,C]
        #
        a = a.expand()
        if isinstance(a, Add):
            return Add(*[cls(term, b) for term in a.args])
        b = b.expand()
        if isinstance(b, Add):
            return Add(*[cls(a, term) for term in b.args])

        #
        # [xA,yB]  ->  xy*[A,B]
        #
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = list(ca) + list(cb)
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        #
        # single second quantization operators
        #
        if isinstance(a, BosonicOperator) and isinstance(b, BosonicOperator):
            if isinstance(b, CreateBoson) and isinstance(a, AnnihilateBoson):
                return KroneckerDelta(a.state, b.state)
            if isinstance(a, CreateBoson) and isinstance(b, AnnihilateBoson):
                return S.NegativeOne*KroneckerDelta(a.state, b.state)
            else:
                return S.Zero
        if isinstance(a, FermionicOperator) and isinstance(b, FermionicOperator):
            return wicks(a*b) - wicks(b*a)

        #
        # Canonical ordering of arguments
        #
        if a.sort_key() > b.sort_key():
            return S.NegativeOne*cls(b, a)

    def doit(self, **hints):
        """
        Enables the computation of complex expressions.

        Examples
        ========

        >>> from sympy.physics.secondquant import Commutator, F, Fd
        >>> from sympy import symbols
        >>> i, j = symbols('i,j', below_fermi=True)
        >>> a, b = symbols('a,b', above_fermi=True)
        >>> c = Commutator(Fd(a)*F(i),Fd(b)*F(j))
        >>> c.doit(wicks=True)
        0
        """
        a = self.args[0]
        b = self.args[1]

        if hints.get("wicks"):
            a = a.doit(**hints)
            b = b.doit(**hints)
            try:
                return wicks(a*b) - wicks(b*a)
            except ContractionAppliesOnlyToFermions:
                pass
            except WicksTheoremDoesNotApply:
                pass

        return (a*b - b*a).doit(**hints)

    def __repr__(self):
        return "Commutator(%s,%s)" % (self.args[0], self.args[1])

    def __str__(self):
        return "[%s,%s]" % (self.args[0], self.args[1])

    def _latex(self, printer):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg) for arg in self.args])


class NO(Expr):
    """
    This Object is used to represent normal ordering brackets.

    i.e.  {abcd}  sometimes written  :abcd:

    Explanation
    ===========

    Applying the function NO(arg) to an argument means that all operators in
    the argument will be assumed to anticommute, and have vanishing
    contractions.  This allows an immediate reordering to canonical form
    upon object creation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import NO, F, Fd
    >>> p,q = symbols('p,q')
    >>> NO(Fd(p)*F(q))
    NO(CreateFermion(p)*AnnihilateFermion(q))
    >>> NO(F(q)*Fd(p))
    -NO(CreateFermion(p)*AnnihilateFermion(q))


    Note
    ====

    If you want to generate a normal ordered equivalent of an expression, you
    should use the function wicks().  This class only indicates that all
    operators inside the brackets anticommute, and have vanishing contractions.
    Nothing more, nothing less.

    """
    is_commutative = False

    def __new__(cls, arg):
        """
        Use anticommutation to get canonical form of operators.

        Explanation
        ===========

        Employ associativity of normal ordered product: {ab{cd}} = {abcd}
        but note that {ab}{cd} /= {abcd}.

        We also employ distributivity: {ab + cd} = {ab} + {cd}.

        Canonical form also implies expand() {ab(c+d)} = {abc} + {abd}.

        """

        # {ab + cd} = {ab} + {cd}
        arg = sympify(arg)
        arg = arg.expand()
        if arg.is_Add:
            return Add(*[ cls(term) for term in arg.args])

        if arg.is_Mul:

            # take coefficient outside of normal ordering brackets
            c_part, seq = arg.args_cnc()
            if c_part:
                coeff = Mul(*c_part)
                if not seq:
                    return coeff
            else:
                coeff = S.One

            # {ab{cd}} = {abcd}
            newseq = []
            foundit = False
            for fac in seq:
                if isinstance(fac, NO):
                    newseq.extend(fac.args)
                    foundit = True
                else:
                    newseq.append(fac)
            if foundit:
                return coeff*cls(Mul(*newseq))

            # We assume that the user don't mix B and F operators
            if isinstance(seq[0], BosonicOperator):
                raise NotImplementedError

            try:
                newseq, sign = _sort_anticommuting_fermions(seq)
            except ViolationOfPauliPrinciple:
                return S.Zero

            if sign % 2:
                return (S.NegativeOne*coeff)*cls(Mul(*newseq))
            elif sign:
                return coeff*cls(Mul(*newseq))
            else:
                pass  # since sign==0, no permutations was necessary

            # if we couldn't do anything with Mul object, we just
            # mark it as normal ordered
            if coeff != S.One:
                return coeff*cls(Mul(*newseq))
            return Expr.__new__(cls, Mul(*newseq))

        if isinstance(arg, NO):
            return arg

        # if object was not Mul or Add, normal ordering does not apply
        return arg

    @property
    def has_q_creators(self):
        """
        Return 0 if the leftmost argument of the first argument is a not a
        q_creator, else 1 if it is above fermi or -1 if it is below fermi.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_creators
        1
        >>> NO(F(i)*F(a)).has_q_creators
        -1
        >>> NO(Fd(i)*F(a)).has_q_creators           #doctest: +SKIP
        0

        """
        return self.args[0].args[0].is_q_creator

    @property
    def has_q_annihilators(self):
        """
        Return 0 if the rightmost argument of the first argument is a not a
        q_annihilator, else 1 if it is above fermi or -1 if it is below fermi.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import NO, F, Fd

        >>> a = symbols('a', above_fermi=True)
        >>> i = symbols('i', below_fermi=True)
        >>> NO(Fd(a)*Fd(i)).has_q_annihilators
        -1
        >>> NO(F(i)*F(a)).has_q_annihilators
        1
        >>> NO(Fd(a)*F(i)).has_q_annihilators
        0

        """
        return self.args[0].args[-1].is_q_annihilator

    def doit(self, **hints):
        """
        Either removes the brackets or enables complex computations
        in its arguments.

        Examples
        ========

        >>> from sympy.physics.secondquant import NO, Fd, F
        >>> from textwrap import fill
        >>> from sympy import symbols, Dummy
        >>> p,q = symbols('p,q', cls=Dummy)
        >>> print(fill(str(NO(Fd(p)*F(q)).doit())))
        KroneckerDelta(_a, _p)*KroneckerDelta(_a,
        _q)*CreateFermion(_a)*AnnihilateFermion(_a) + KroneckerDelta(_a,
        _p)*KroneckerDelta(_i, _q)*CreateFermion(_a)*AnnihilateFermion(_i) -
        KroneckerDelta(_a, _q)*KroneckerDelta(_i,
        _p)*AnnihilateFermion(_a)*CreateFermion(_i) - KroneckerDelta(_i,
        _p)*KroneckerDelta(_i, _q)*AnnihilateFermion(_i)*CreateFermion(_i)
        """
        if hints.get("remove_brackets", True):
            return self._remove_brackets()
        else:
            return self.__new__(type(self), self.args[0].doit(**hints))

    def _remove_brackets(self):
        """
        Returns the sorted string without normal order brackets.

        The returned string have the property that no nonzero
        contractions exist.
        """

        # check if any creator is also an annihilator
        subslist = []
        for i in self.iter_q_creators():
            if self[i].is_q_annihilator:
                assume = self[i].state.assumptions0

                # only operators with a dummy index can be split in two terms
                if isinstance(self[i].state, Dummy):

                    # create indices with fermi restriction
                    assume.pop("above_fermi", None)
                    assume["below_fermi"] = True
                    below = Dummy('i', **assume)
                    assume.pop("below_fermi", None)
                    assume["above_fermi"] = True
                    above = Dummy('a', **assume)

                    cls = type(self[i])
                    split = (
                        self[i].__new__(cls, below)
                        * KroneckerDelta(below, self[i].state)
                        + self[i].__new__(cls, above)
                        * KroneckerDelta(above, self[i].state)
                    )
                    subslist.append((self[i], split))
                else:
                    raise SubstitutionOfAmbigousOperatorFailed(self[i])
        if subslist:
            result = NO(self.subs(subslist))
            if isinstance(result, Add):
                return Add(*[term.doit() for term in result.args])
        else:
            return self.args[0]

    def _expand_operators(self):
        """
        Returns a sum of NO objects that contain no ambiguous q-operators.

        Explanation
        ===========

        If an index q has range both above and below fermi, the operator F(q)
        is ambiguous in the sense that it can be both a q-creator and a q-annihilator.
        If q is dummy, it is assumed to be a summation variable and this method
        rewrites it into a sum of NO terms with unambiguous operators:

        {Fd(p)*F(q)} = {Fd(a)*F(b)} + {Fd(a)*F(i)} + {Fd(j)*F(b)} -{F(i)*Fd(j)}

        where a,b are above and i,j are below fermi level.
        """
        return NO(self._remove_brackets)

    def __getitem__(self, i):
        if isinstance(i, slice):
            indices = i.indices(len(self))
            return [self.args[0].args[i] for i in range(*indices)]
        else:
            return self.args[0].args[i]

    def __len__(self):
        return len(self.args[0].args)

    def iter_q_annihilators(self):
        """
        Iterates over the annihilation operators.

        Examples
        ========

        >>> from sympy import symbols
        >>> i, j = symbols('i j', below_fermi=True)
        >>> a, b = symbols('a b', above_fermi=True)
        >>> from sympy.physics.secondquant import NO, F, Fd
        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))

        >>> no.iter_q_creators()
        <generator object... at 0x...>
        >>> list(no.iter_q_creators())
        [0, 1]
        >>> list(no.iter_q_annihilators())
        [3, 2]

        """
        ops = self.args[0].args
        iter = range(len(ops) - 1, -1, -1)
        for i in iter:
            if ops[i].is_q_annihilator:
                yield i
            else:
                break

    def iter_q_creators(self):
        """
        Iterates over the creation operators.

        Examples
        ========

        >>> from sympy import symbols
        >>> i, j = symbols('i j', below_fermi=True)
        >>> a, b = symbols('a b', above_fermi=True)
        >>> from sympy.physics.secondquant import NO, F, Fd
        >>> no = NO(Fd(a)*F(i)*F(b)*Fd(j))

        >>> no.iter_q_creators()
        <generator object... at 0x...>
        >>> list(no.iter_q_creators())
        [0, 1]
        >>> list(no.iter_q_annihilators())
        [3, 2]

        """

        ops = self.args[0].args
        iter = range(0, len(ops))
        for i in iter:
            if ops[i].is_q_creator:
                yield i
            else:
                break

    def get_subNO(self, i):
        """
        Returns a NO() without FermionicOperator at index i.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.secondquant import F, NO
        >>> p, q, r = symbols('p,q,r')

        >>> NO(F(p)*F(q)*F(r)).get_subNO(1)
        NO(AnnihilateFermion(p)*AnnihilateFermion(r))

        """
        arg0 = self.args[0]  # it's a Mul by definition of how it's created
        mul = arg0._new_rawargs(*(arg0.args[:i] + arg0.args[i + 1:]))
        return NO(mul)

    def _latex(self, printer):
        return "\\left\\{%s\\right\\}" % printer._print(self.args[0])

    def __repr__(self):
        return "NO(%s)" % self.args[0]

    def __str__(self):
        return ":%s:" % self.args[0]


def contraction(a, b):
    """
    Calculates contraction of Fermionic operators a and b.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.secondquant import F, Fd, contraction
    >>> p, q = symbols('p,q')
    >>> a, b = symbols('a,b', above_fermi=True)
    >>> i, j = symbols('i,j', below_fermi=True)

    A contraction is non-zero only if a quasi-creator is to the right of a
    quasi-annihilator:

    >>> contraction(F(a),Fd(b))
    KroneckerDelta(a, b)
    >>> contraction(Fd(i),F(j))
    KroneckerDelta(i, j)

    For general indices a non-zero result restricts the indices to below/above
    the fermi surface:

    >>> contraction(Fd(p),F(q))
    KroneckerDelta(_i, q)*KroneckerDelta(p, q)
    >>> contraction(F(p),Fd(q))
    KroneckerDelta(_a, q)*KroneckerDelta(p, q)

    Two creators or two annihilators always vanishes:

    >>> contraction(F(p),F(q))
    0
    >>> contraction(Fd(p),Fd(q))
    0

    """
    if isinstance(b, FermionicOperator) and isinstance(a, FermionicOperator):
        if isinstance(a, AnnihilateFermion) and isinstance(b, CreateFermion):
            if b.state.assumptions0.get("below_fermi"):
                return S.Zero
            if a.state.assumptions0.get("below_fermi"):
                return S.Zero
            if b.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get("above_fermi"):
                return KroneckerDelta(a.state, b.state)

            return (KroneckerDelta(a.state, b.state)*
                    KroneckerDelta(b.state, Dummy('a', above_fermi=True)))
        if isinstance(b, AnnihilateFermion) and isinstance(a, CreateFermion):
            if b.state.assumptions0.get("above_fermi"):
                return S.Zero
            if a.state.assumptions0.get("above_fermi"):
                return S.Zero
            if b.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(a.state, b.state)
            if a.state.assumptions0.get("below_fermi"):
                return KroneckerDelta(a.state, b.state)

            return (KroneckerDelta(a.state, b.state)*
                    KroneckerDelta(b.state, Dummy('i', below_fermi=True)))

        # vanish if 2xAnnihilator or 2xCreator
        return S.Zero

    else:
        #not fermion operators
        t = ( isinstance(i, FermionicOperator) for i in (a, b) )
        raise ContractionAppliesOnlyToFermions(*t)


def _sqkey_operator(sq_operator):
    """Generates key for canonical sorting of SQ operators."""
    return sq_operator._sortkey()

def _sqkey_index(index):
    """Key for sorting of indices.

    particle < hole < general

    FIXME: This is a bottle-neck, can we do it faster?
    """
    h = hash(index)
    label = str(index)
    if isinstance(index, Dummy):
        if index.assumptions0.get('above_fermi'):
            return (20, label, h)
        elif index.assumptions0.get('below_fermi'):
            return (21, label, h)
        else:
            return (22, label, h)

    if index.assumptions0.get('above_fermi'):
        return (10, label, h)
    elif index.assumptions0.get('below_fermi'):
        return (11, label, h)
    else:
        return (12, label, h)



def _sort_anticommuting_fermions(string1, key=_sqkey_operator):
    """Sort fermionic operators to canonical order, assuming all pairs anticommute.

    Explanation
    ===========

    Uses a bidirectional bubble sort.  Items in string1 are not referenced
    so in principle they may be any comparable objects.   The sorting depends on the
    operators '>' and '=='.

    If the Pauli principle is violated, an exception is raised.

    Returns
    =======

    tuple (sorted_str, sign)

    sorted_str: list containing the sorted operators
    sign: int telling how many times the sign should be changed
          (if sign==0 the string was already sorted)
    """

    verified = False
    sign = 0
    rng = list(range(len(string1) - 1))
    rev = list(range(len(string1) - 3, -1, -1))

    keys = list(map(key, string1))
    key_val = dict(list(zip(keys, string1)))

    while not verified:
        verified = True
        for i in rng:
            left = keys[i]
            right = keys[i + 1]
            if left == right:
                raise ViolationOfPauliPrinciple([left, right])
            if left > right:
                verified = False
                keys[i:i + 2] = [right, left]
                sign = sign + 1
        if verified:
            break
        for i in rev:
            left = keys[i]
            right = keys[i + 1]
            if left == right:
                raise ViolationOfPauliPrinciple([left, right])
            if left > right:
                verified = False
                keys[i:i + 2] = [right, left]
                sign = sign + 1
    string1 = [ key_val[k] for k in keys ]
    return (string1, sign)


def evaluate_deltas(e):
    """
    We evaluate KroneckerDelta symbols in the expression assuming Einstein summation.

    Explanation
    ===========

    If one index is repeated it is summed over and in effect substituted with
    the other one. If both indices are repeated we substitute according to what
    is the preferred index.  this is determined by
    KroneckerDelta.preferred_index and KroneckerDelta.killable_index.

    In case there are no possible substitutions or if a substitution would
    imply a loss of information, nothing is done.

    In case an index appears in more than one KroneckerDelta, the resulting
    substitution depends on the order of the factors.  Since the ordering is platform
    dependent, the literal expression resulting from this function may be hard to
    predict.

    Examples
    ========

    We assume the following:

    >>> from sympy import symbols, Function, Dummy, KroneckerDelta
    >>> from sympy.physics.secondquant import evaluate_deltas
    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)
    >>> a,b = symbols('a b', above_fermi=True, cls=Dummy)
    >>> p,q = symbols('p q', cls=Dummy)
    >>> f = Function('f')
    >>> t = Function('t')

    The order of preference for these indices according to KroneckerDelta is
    (a, b, i, j, p, q).

    Trivial cases:

    >>> evaluate_deltas(KroneckerDelta(i,j)*f(i))       # d_ij f(i) -> f(j)
    f(_j)
    >>> evaluate_deltas(KroneckerDelta(i,j)*f(j))       # d_ij f(j) -> f(i)
    f(_i)
    >>> evaluate_deltas(KroneckerDelta(i,p)*f(p))       # d_ip f(p) -> f(i)
    f(_i)
    >>> evaluate_deltas(KroneckerDelta(q,p)*f(p))       # d_qp f(p) -> f(q)
    f(_q)
    >>> evaluate_deltas(KroneckerDelta(q,p)*f(q))       # d_qp f(q) -> f(p)
    f(_p)

    More interesting cases:

    >>> evaluate_deltas(KroneckerDelta(i,p)*t(a,i)*f(p,q))
    f(_i, _q)*t(_a, _i)
    >>> evaluate_deltas(KroneckerDelta(a,p)*t(a,i)*f(p,q))
    f(_a, _q)*t(_a, _i)
    >>> evaluate_deltas(KroneckerDelta(p,q)*f(p,q))
    f(_p, _p)

    Finally, here are some cases where nothing is done, because that would
    imply a loss of information:

    >>> evaluate_deltas(KroneckerDelta(i,p)*f(q))
    f(_q)*KroneckerDelta(_i, _p)
    >>> evaluate_deltas(KroneckerDelta(i,p)*f(i))
    f(_i)*KroneckerDelta(_i, _p)
    """

    # We treat Deltas only in mul objects
    # for general function objects we don't evaluate KroneckerDeltas in arguments,
    # but here we hard code exceptions to this rule
    accepted_functions = (
        Add,
    )
    if isinstance(e, accepted_functions):
        return e.func(*[evaluate_deltas(arg) for arg in e.args])

    elif isinstance(e, Mul):
        # find all occurrences of delta function and count each index present in
        # expression.
        deltas = []
        indices = {}
        for i in e.args:
            for s in i.free_symbols:
                if s in indices:
                    indices[s] += 1
                else:
                    indices[s] = 0  # geek counting simplifies logic below
            if isinstance(i, KroneckerDelta):
                deltas.append(i)

        for d in deltas:
            # If we do something, and there are more deltas, we should recurse
            # to treat the resulting expression properly
            if d.killable_index.is_Symbol and indices[d.killable_index]:
                e = e.subs(d.killable_index, d.preferred_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)
            elif (d.preferred_index.is_Symbol and indices[d.preferred_index]
                  and d.indices_contain_equal_information):
                e = e.subs(d.preferred_index, d.killable_index)
                if len(deltas) > 1:
                    return evaluate_deltas(e)
            else:
                pass

        return e
    # nothing to do, maybe we hit a Symbol or a number
    else:
        return e


def substitute_dummies(expr, new_indices=False, pretty_indices={}):
    """
    Collect terms by substitution of dummy variables.

    Explanation
    ===========

    This routine allows simplification of Add expressions containing terms
    which differ only due to dummy variables.

    The idea is to substitute all dummy variables consistently depending on
    the structure of the term.  For each term, we obtain a sequence of all
    dummy variables, where the order is determined by the index range, what
    factors the index belongs to and its position in each factor.  See
    _get_ordered_dummies() for more information about the sorting of dummies.
    The index sequence is then substituted consistently in each term.

    Examples
    ========

    >>> from sympy import symbols, Function, Dummy
    >>> from sympy.physics.secondquant import substitute_dummies
    >>> a,b,c,d = symbols('a b c d', above_fermi=True, cls=Dummy)
    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)
    >>> f = Function('f')

    >>> expr = f(a,b) + f(c,d); expr
    f(_a, _b) + f(_c, _d)

    Since a, b, c and d are equivalent summation indices, the expression can be
    simplified to a single term (for which the dummy indices are still summed over)

    >>> substitute_dummies(expr)
    2*f(_a, _b)


    Controlling output:

    By default the dummy symbols that are already present in the expression
    will be reused in a different permutation.  However, if new_indices=True,
    new dummies will be generated and inserted.  The keyword 'pretty_indices'
    can be used to control this generation of new symbols.

    By default the new dummies will be generated on the form i_1, i_2, a_1,
    etc.  If you supply a dictionary with key:value pairs in the form:

        { index_group: string_of_letters }

    The letters will be used as labels for the new dummy symbols.  The
    index_groups must be one of 'above', 'below' or 'general'.

    >>> expr = f(a,b,i,j)
    >>> my_dummies = { 'above':'st', 'below':'uv' }
    >>> substitute_dummies(expr, new_indices=True, pretty_indices=my_dummies)
    f(_s, _t, _u, _v)

    If we run out of letters, or if there is no keyword for some index_group
    the default dummy generator will be used as a fallback:

    >>> p,q = symbols('p q', cls=Dummy)  # general indices
    >>> expr = f(p,q)
    >>> substitute_dummies(expr, new_indices=True, pretty_indices=my_dummies)
    f(_p_0, _p_1)

    """

    # setup the replacing dummies
    if new_indices:
        letters_above = pretty_indices.get('above', "")
        letters_below = pretty_indices.get('below', "")
        letters_general = pretty_indices.get('general', "")
        len_above = len(letters_above)
        len_below = len(letters_below)
        len_general = len(letters_general)

        def _i(number):
            try:
                return letters_below[number]
            except IndexError:
                return 'i_' + str(number - len_below)

        def _a(number):
            try:
                return letters_above[number]
            except IndexError:
                return 'a_' + str(number - len_above)

        def _p(number):
            try:
                return letters_general[number]
            except IndexError:
                return 'p_' + str(number - len_general)

    aboves = []
    belows = []
    generals = []

    dummies = expr.atoms(Dummy)
    if not new_indices:
        dummies = sorted(dummies, key=default_sort_key)

    # generate lists with the dummies we will insert
    a = i = p = 0
    for d in dummies:
        assum = d.assumptions0

        if assum.get("above_fermi"):
            if new_indices:
                sym = _a(a)
                a += 1
            l1 = aboves
        elif assum.get("below_fermi"):
            if new_indices:
                sym = _i(i)
                i += 1
            l1 = belows
        else:
            if new_indices:
                sym = _p(p)
                p += 1
            l1 = generals

        if new_indices:
            l1.append(Dummy(sym, **assum))
        else:
            l1.append(d)

    expr = expr.expand()
    terms = Add.make_args(expr)
    new_terms = []
    for term in terms:
        i = iter(belows)
        a = iter(aboves)
        p = iter(generals)
        ordered = _get_ordered_dummies(term)
        subsdict = {}
        for d in ordered:
            if d.assumptions0.get('below_fermi'):
                subsdict[d] = next(i)
            elif d.assumptions0.get('above_fermi'):
                subsdict[d] = next(a)
            else:
                subsdict[d] = next(p)
        subslist = []
        final_subs = []
        for k, v in subsdict.items():
            if k == v:
                continue
            if v in subsdict:
                # We check if the sequence of substitutions end quickly.  In
                # that case, we can avoid temporary symbols if we ensure the
                # correct substitution order.
                if subsdict[v] in subsdict:
                    # (x, y) -> (y, x),  we need a temporary variable
                    x = Dummy('x')
                    subslist.append((k, x))
                    final_subs.append((x, v))
                else:
                    # (x, y) -> (y, a),  x->y must be done last
                    # but before temporary variables are resolved
                    final_subs.insert(0, (k, v))
            else:
                subslist.append((k, v))
        subslist.extend(final_subs)
        new_terms.append(term.subs(subslist))
    return Add(*new_terms)


class KeyPrinter(StrPrinter):
    """Printer for which only equal objects are equal in print"""
    def _print_Dummy(self, expr):
        return "(%s_%i)" % (expr.name, expr.dummy_index)


def __kprint(expr):
    p = KeyPrinter()
    return p.doprint(expr)


def _get_ordered_dummies(mul, verbose=False):
    """Returns all dummies in the mul sorted in canonical order.

    Explanation
    ===========

    The purpose of the canonical ordering is that dummies can be substituted
    consistently across terms with the result that equivalent terms can be
    simplified.

    It is not possible to determine if two terms are equivalent based solely on
    the dummy order.  However, a consistent substitution guided by the ordered
    dummies should lead to trivially (non-)equivalent terms, thereby revealing
    the equivalence.  This also means that if two terms have identical sequences of
    dummies, the (non-)equivalence should already be apparent.

    Strategy
    --------

    The canonical order is given by an arbitrary sorting rule.  A sort key
    is determined for each dummy as a tuple that depends on all factors where
    the index is present.  The dummies are thereby sorted according to the
    contraction structure of the term, instead of sorting based solely on the
    dummy symbol itself.

    After all dummies in the term has been assigned a key, we check for identical
    keys, i.e. unorderable dummies.  If any are found, we call a specialized
    method, _determine_ambiguous(), that will determine a unique order based
    on recursive calls to _get_ordered_dummies().

    Key description
    ---------------

    A high level description of the sort key:

        1. Range of the dummy index
        2. Relation to external (non-dummy) indices
        3. Position of the index in the first factor
        4. Position of the index in the second factor

    The sort key is a tuple with the following components:

        1. A single character indicating the range of the dummy (above, below
           or general.)
        2. A list of strings with fully masked string representations of all
           factors where the dummy is present.  By masked, we mean that dummies
           are represented by a symbol to indicate either below fermi, above or
           general.  No other information is displayed about the dummies at
           this point.  The list is sorted stringwise.
        3. An integer number indicating the position of the index, in the first
           factor as sorted in 2.
        4. An integer number indicating the position of the index, in the second
           factor as sorted in 2.

    If a factor is either of type AntiSymmetricTensor or SqOperator, the index
    position in items 3 and 4 is indicated as 'upper' or 'lower' only.
    (Creation operators are considered upper and annihilation operators lower.)

    If the masked factors are identical, the two factors cannot be ordered
    unambiguously in item 2.  In this case, items 3, 4 are left out.  If several
    indices are contracted between the unorderable factors, it will be handled by
    _determine_ambiguous()


    """
    # setup dicts to avoid repeated calculations in key()
    args = Mul.make_args(mul)
    fac_dum = { fac: fac.atoms(Dummy) for fac in args }
    fac_repr = { fac: __kprint(fac) for fac in args }
    all_dums = set().union(*fac_dum.values())
    mask = {}
    for d in all_dums:
        if d.assumptions0.get('below_fermi'):
            mask[d] = '0'
        elif d.assumptions0.get('above_fermi'):
            mask[d] = '1'
        else:
            mask[d] = '2'
    dum_repr = {d: __kprint(d) for d in all_dums}

    def _key(d):
        dumstruct = [ fac for fac in fac_dum if d in fac_dum[fac] ]
        other_dums = set().union(*[fac_dum[fac] for fac in dumstruct])
        fac = dumstruct[-1]
        if other_dums is fac_dum[fac]:
            other_dums = fac_dum[fac].copy()
        other_dums.remove(d)
        masked_facs = [ fac_repr[fac] for fac in dumstruct ]
        for d2 in other_dums:
            masked_facs = [ fac.replace(dum_repr[d2], mask[d2])
                    for fac in masked_facs ]
        all_masked = [ fac.replace(dum_repr[d], mask[d])
                       for fac in masked_facs ]
        masked_facs = dict(list(zip(dumstruct, masked_facs)))

        # dummies for which the ordering cannot be determined
        if has_dups(all_masked):
            all_masked.sort()
            return mask[d], tuple(all_masked)  # positions are ambiguous

        # sort factors according to fully masked strings
        keydict = dict(list(zip(dumstruct, all_masked)))
        dumstruct.sort(key=lambda x: keydict[x])
        all_masked.sort()

        pos_val = []
        for fac in dumstruct:
            if isinstance(fac, AntiSymmetricTensor):
                if d in fac.upper:
                    pos_val.append('u')
                if d in fac.lower:
                    pos_val.append('l')
            elif isinstance(fac, Creator):
                pos_val.append('u')
            elif isinstance(fac, Annihilator):
                pos_val.append('l')
            elif isinstance(fac, NO):
                ops = [ op for op in fac if op.has(d) ]
                for op in ops:
                    if isinstance(op, Creator):
                        pos_val.append('u')
                    else:
                        pos_val.append('l')
            else:
                # fallback to position in string representation
                facpos = -1
                while 1:
                    facpos = masked_facs[fac].find(dum_repr[d], facpos + 1)
                    if facpos == -1:
                        break
                    pos_val.append(facpos)
        return (mask[d], tuple(all_masked), pos_val[0], pos_val[-1])
    dumkey = dict(list(zip(all_dums, list(map(_key, all_dums)))))
    result = sorted(all_dums, key=lambda x: dumkey[x])
    if has_dups(iter(dumkey.values())):
        # We have ambiguities
        unordered = defaultdict(set)
        for d, k in dumkey.items():
            unordered[k].add(d)
        for k in [ k for k in unordered if len(unordered[k]) < 2 ]:
            del unordered[k]

        unordered = [ unordered[k] for k in sorted(unordered) ]
        result = _determine_ambiguous(mul, result, unordered)
    return result


def _determine_ambiguous(term, ordered, ambiguous_groups):
    # We encountered a term for which the dummy substitution is ambiguous.
    # This happens for terms with 2 or more contractions between factors that
    # cannot be uniquely ordered independent of summation indices.  For
    # example:
    #
    # Sum(p, q) v^{p, .}_{q, .}v^{q, .}_{p, .}
    #
    # Assuming that the indices represented by . are dummies with the
    # same range, the factors cannot be ordered, and there is no
    # way to determine a consistent ordering of p and q.
    #
    # The strategy employed here, is to relabel all unambiguous dummies with
    # non-dummy symbols and call _get_ordered_dummies again.  This procedure is
    # applied to the entire term so there is a possibility that
    # _determine_ambiguous() is called again from a deeper recursion level.

    # break recursion if there are no ordered dummies
    all_ambiguous = set()
    for dummies in ambiguous_groups:
        all_ambiguous |= dummies
    all_ordered = set(ordered) - all_ambiguous
    if not all_ordered:
        # FIXME: If we arrive here, there are no ordered dummies. A method to
        # handle this needs to be implemented.  In order to return something
        # useful nevertheless, we choose arbitrarily the first dummy and
        # determine the rest from this one.  This method is dependent on the
        # actual dummy labels which violates an assumption for the
        # canonicalization procedure.  A better implementation is needed.
        group = [ d for d in ordered if d in ambiguous_groups[0] ]
        d = group[0]
        all_ordered.add(d)
        ambiguous_groups[0].remove(d)

    stored_counter = _symbol_factory._counter
    subslist = []
    for d in [ d for d in ordered if d in all_ordered ]:
        nondum = _symbol_factory._next()
        subslist.append((d, nondum))
    newterm = term.subs(subslist)
    neworder = _get_ordered_dummies(newterm)
    _symbol_factory._set_counter(stored_counter)

    # update ordered list with new information
    for group in ambiguous_groups:
        ordered_group = [ d for d in neworder if d in group ]
        ordered_group.reverse()
        result = []
        for d in ordered:
            if d in group:
                result.append(ordered_group.pop())
            else:
                result.append(d)
        ordered = result
    return ordered


class _SymbolFactory:
    def __init__(self, label):
        self._counterVar = 0
        self._label = label

    def _set_counter(self, value):
        """
        Sets counter to value.
        """
        self._counterVar = value

    @property
    def _counter(self):
        """
        What counter is currently at.
        """
        return self._counterVar

    def _next(self):
        """
        Generates the next symbols and increments counter by 1.
        """
        s = Symbol("%s%i" % (self._label, self._counterVar))
        self._counterVar += 1
        return s
_symbol_factory = _SymbolFactory('_]"]_')  # most certainly a unique label


@cacheit
def _get_contractions(string1, keep_only_fully_contracted=False):
    """
    Returns Add-object with contracted terms.

    Uses recursion to find all contractions. -- Internal helper function --

    Will find nonzero contractions in string1 between indices given in
    leftrange and rightrange.

    """

    # Should we store current level of contraction?
    if keep_only_fully_contracted and string1:
        result = []
    else:
        result = [NO(Mul(*string1))]

    for i in range(len(string1) - 1):
        for j in range(i + 1, len(string1)):

            c = contraction(string1[i], string1[j])

            if c:
                sign = (j - i + 1) % 2
                if sign:
                    coeff = S.NegativeOne*c
                else:
                    coeff = c

                #
                #  Call next level of recursion
                #  ============================
                #
                # We now need to find more contractions among operators
                #
                # oplist = string1[:i]+ string1[i+1:j] + string1[j+1:]
                #
                # To prevent overcounting, we don't allow contractions
                # we have already encountered. i.e. contractions between
                #       string1[:i] <---> string1[i+1:j]
                # and   string1[:i] <---> string1[j+1:].
                #
                # This leaves the case:
                oplist = string1[i + 1:j] + string1[j + 1:]

                if oplist:

                    result.append(coeff*NO(
                        Mul(*string1[:i])*_get_contractions( oplist,
                            keep_only_fully_contracted=keep_only_fully_contracted)))

                else:
                    result.append(coeff*NO( Mul(*string1[:i])))

        if keep_only_fully_contracted:
            break   # next iteration over i leaves leftmost operator string1[0] uncontracted

    return Add(*result)


def wicks(e, **kw_args):
    """
    Returns the normal ordered equivalent of an expression using Wicks Theorem.

    Examples
    ========

    >>> from sympy import symbols, Dummy
    >>> from sympy.physics.secondquant import wicks, F, Fd
    >>> p, q, r = symbols('p,q,r')
    >>> wicks(Fd(p)*F(q))
    KroneckerDelta(_i, q)*KroneckerDelta(p, q) + NO(CreateFermion(p)*AnnihilateFermion(q))

    By default, the expression is expanded:

    >>> wicks(F(p)*(F(q)+F(r)))
    NO(AnnihilateFermion(p)*AnnihilateFermion(q)) + NO(AnnihilateFermion(p)*AnnihilateFermion(r))

    With the keyword 'keep_only_fully_contracted=True', only fully contracted
    terms are returned.

    By request, the result can be simplified in the following order:
     -- KroneckerDelta functions are evaluated
     -- Dummy variables are substituted consistently across terms

    >>> p, q, r = symbols('p q r', cls=Dummy)
    >>> wicks(Fd(p)*(F(q)+F(r)), keep_only_fully_contracted=True)
    KroneckerDelta(_i, _q)*KroneckerDelta(_p, _q) + KroneckerDelta(_i, _r)*KroneckerDelta(_p, _r)

    """

    if not e:
        return S.Zero

    opts = {
        'simplify_kronecker_deltas': False,
        'expand': True,
        'simplify_dummies': False,
        'keep_only_fully_contracted': False
    }
    opts.update(kw_args)

    # check if we are already normally ordered
    if isinstance(e, NO):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e
    elif isinstance(e, FermionicOperator):
        if opts['keep_only_fully_contracted']:
            return S.Zero
        else:
            return e

    # break up any NO-objects, and evaluate commutators
    e = e.doit(wicks=True)

    # make sure we have only one term to consider
    e = e.expand()
    if isinstance(e, Add):
        if opts['simplify_dummies']:
            return substitute_dummies(Add(*[ wicks(term, **kw_args) for term in e.args]))
        else:
            return Add(*[ wicks(term, **kw_args) for term in e.args])

    # For Mul-objects we can actually do something
    if isinstance(e, Mul):

        # we don't want to mess around with commuting part of Mul
        # so we factorize it out before starting recursion
        c_part = []
        string1 = []
        for factor in e.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                string1.append(factor)
        n = len(string1)

        # catch trivial cases
        if n == 0:
            result = e
        elif n == 1:
            if opts['keep_only_fully_contracted']:
                return S.Zero
            else:
                result = e

        else:  # non-trivial

            if isinstance(string1[0], BosonicOperator):
                raise NotImplementedError

            string1 = tuple(string1)

            # recursion over higher order contractions
            result = _get_contractions(string1,
                keep_only_fully_contracted=opts['keep_only_fully_contracted'] )
            result = Mul(*c_part)*result

        if opts['expand']:
            result = result.expand()
        if opts['simplify_kronecker_deltas']:
            result = evaluate_deltas(result)

        return result

    # there was nothing to do
    return e


class PermutationOperator(Expr):
    """
    Represents the index permutation operator P(ij).

    P(ij)*f(i)*g(j) = f(i)*g(j) - f(j)*g(i)
    """
    is_commutative = True

    def __new__(cls, i, j):
        i, j = sorted(map(sympify, (i, j)), key=default_sort_key)
        obj = Basic.__new__(cls, i, j)
        return obj

    def get_permuted(self, expr):
        """
        Returns -expr with permuted indices.

        Explanation
        ===========

        >>> from sympy import symbols, Function
        >>> from sympy.physics.secondquant import PermutationOperator
        >>> p,q = symbols('p,q')
        >>> f = Function('f')
        >>> PermutationOperator(p,q).get_permuted(f(p,q))
        -f(q, p)

        """
        i = self.args[0]
        j = self.args[1]
        if expr.has(i) and expr.has(j):
            tmp = Dummy()
            expr = expr.subs(i, tmp)
            expr = expr.subs(j, i)
            expr = expr.subs(tmp, j)
            return S.NegativeOne*expr
        else:
            return expr

    def _latex(self, printer):
        return "P(%s%s)" % tuple(printer._print(i) for i in self.args)


def simplify_index_permutations(expr, permutation_operators):
    """
    Performs simplification by introducing PermutationOperators where appropriate.

    Explanation
    ===========

    Schematically:
        [abij] - [abji] - [baij] + [baji] ->  P(ab)*P(ij)*[abij]

    permutation_operators is a list of PermutationOperators to consider.

    If permutation_operators=[P(ab),P(ij)] we will try to introduce the
    permutation operators P(ij) and P(ab) in the expression.  If there are other
    possible simplifications, we ignore them.

    >>> from sympy import symbols, Function
    >>> from sympy.physics.secondquant import simplify_index_permutations
    >>> from sympy.physics.secondquant import PermutationOperator
    >>> p,q,r,s = symbols('p,q,r,s')
    >>> f = Function('f')
    >>> g = Function('g')

    >>> expr = f(p)*g(q) - f(q)*g(p); expr
    f(p)*g(q) - f(q)*g(p)
    >>> simplify_index_permutations(expr,[PermutationOperator(p,q)])
    f(p)*g(q)*PermutationOperator(p, q)

    >>> PermutList = [PermutationOperator(p,q),PermutationOperator(r,s)]
    >>> expr = f(p,r)*g(q,s) - f(q,r)*g(p,s) + f(q,s)*g(p,r) - f(p,s)*g(q,r)
    >>> simplify_index_permutations(expr,PermutList)
    f(p, r)*g(q, s)*PermutationOperator(p, q)*PermutationOperator(r, s)

    """

    def _get_indices(expr, ind):
        """
        Collects indices recursively in predictable order.
        """
        result = []
        for arg in expr.args:
            if arg in ind:
                result.append(arg)
            else:
                if arg.args:
                    result.extend(_get_indices(arg, ind))
        return result

    def _choose_one_to_keep(a, b, ind):
        # we keep the one where indices in ind are in order ind[0] < ind[1]
        return min(a, b, key=lambda x: default_sort_key(_get_indices(x, ind)))

    expr = expr.expand()
    if isinstance(expr, Add):
        terms = set(expr.args)

        for P in permutation_operators:
            new_terms = set()
            on_hold = set()
            while terms:
                term = terms.pop()
                permuted = P.get_permuted(term)
                if permuted in terms | on_hold:
                    try:
                        terms.remove(permuted)
                    except KeyError:
                        on_hold.remove(permuted)
                    keep = _choose_one_to_keep(term, permuted, P.args)
                    new_terms.add(P*keep)
                else:

                    # Some terms must get a second chance because the permuted
                    # term may already have canonical dummy ordering.  Then
                    # substitute_dummies() does nothing.  However, the other
                    # term, if it exists, will be able to match with us.
                    permuted1 = permuted
                    permuted = substitute_dummies(permuted)
                    if permuted1 == permuted:
                        on_hold.add(term)
                    elif permuted in terms | on_hold:
                        try:
                            terms.remove(permuted)
                        except KeyError:
                            on_hold.remove(permuted)
                        keep = _choose_one_to_keep(term, permuted, P.args)
                        new_terms.add(P*keep)
                    else:
                        new_terms.add(term)
            terms = new_terms | on_hold
        return Add(*terms)
    return expr
<|endoftext|>"""
This module implements Pauli algebra by subclassing Symbol. Only algebraic
properties of Pauli matrices are used (we do not use the Matrix class).

See the documentation to the class Pauli for examples.

References
==========

.. [1] https://en.wikipedia.org/wiki/Pauli_matrices
"""

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.physics.quantum import TensorProduct

__all__ = ['evaluate_pauli_product']


def delta(i, j):
    """
    Returns 1 if ``i == j``, else 0.

    This is used in the multiplication of Pauli matrices.

    Examples
    ========

    >>> from sympy.physics.paulialgebra import delta
    >>> delta(1, 1)
    1
    >>> delta(2, 3)
    0
    """
    if i == j:
        return 1
    else:
        return 0


def epsilon(i, j, k):
    """
    Return 1 if i,j,k is equal to (1,2,3), (2,3,1), or (3,1,2);
    -1 if ``i``,``j``,``k`` is equal to (1,3,2), (3,2,1), or (2,1,3);
    else return 0.

    This is used in the multiplication of Pauli matrices.

    Examples
    ========

    >>> from sympy.physics.paulialgebra import epsilon
    >>> epsilon(1, 2, 3)
    1
    >>> epsilon(1, 3, 2)
    -1
    """
    if (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2)):
        return 1
    elif (i, j, k) in ((1, 3, 2), (3, 2, 1), (2, 1, 3)):
        return -1
    else:
        return 0


class Pauli(Symbol):
    """
    The class representing algebraic properties of Pauli matrices.

    Explanation
    ===========

    The symbol used to display the Pauli matrices can be changed with an
    optional parameter ``label="sigma"``. Pauli matrices with different
    ``label`` attributes cannot multiply together.

    If the left multiplication of symbol or number with Pauli matrix is needed,
    please use parentheses  to separate Pauli and symbolic multiplication
    (for example: 2*I*(Pauli(3)*Pauli(2))).

    Another variant is to use evaluate_pauli_product function to evaluate
    the product of Pauli matrices and other symbols (with commutative
    multiply rules).

    See Also
    ========

    evaluate_pauli_product

    Examples
    ========

    >>> from sympy.physics.paulialgebra import Pauli
    >>> Pauli(1)
    sigma1
    >>> Pauli(1)*Pauli(2)
    I*sigma3
    >>> Pauli(1)*Pauli(1)
    1
    >>> Pauli(3)**4
    1
    >>> Pauli(1)*Pauli(2)*Pauli(3)
    I

    >>> from sympy.physics.paulialgebra import Pauli
    >>> Pauli(1, label="tau")
    tau1
    >>> Pauli(1)*Pauli(2, label="tau")
    sigma1*tau2
    >>> Pauli(1, label="tau")*Pauli(2, label="tau")
    I*tau3

    >>> from sympy import I
    >>> I*(Pauli(2)*Pauli(3))
    -sigma1

    >>> from sympy.physics.paulialgebra import evaluate_pauli_product
    >>> f = I*Pauli(2)*Pauli(3)
    >>> f
    I*sigma2*sigma3
    >>> evaluate_pauli_product(f)
    -sigma1
    """

    __slots__ = ("i", "label")

    def __new__(cls, i, label="sigma"):
        if i not in [1, 2, 3]:
            raise IndexError("Invalid Pauli index")
        obj = Symbol.__new__(cls, "%s%d" %(label,i), commutative=False, hermitian=True)
        obj.i = i
        obj.label = label
        return obj

    def __getnewargs_ex__(self):
        return (self.i, self.label), {}

    def _hashable_content(self):
        return (self.i, self.label)

    # FIXME don't work for -I*Pauli(2)*Pauli(3)
    def __mul__(self, other):
        if isinstance(other, Pauli):
            j = self.i
            k = other.i
            jlab = self.label
            klab = other.label

            if jlab == klab:
                return delta(j, k) \
                    + I*epsilon(j, k, 1)*Pauli(1,jlab) \
                    + I*epsilon(j, k, 2)*Pauli(2,jlab) \
                    + I*epsilon(j, k, 3)*Pauli(3,jlab)
        return super().__mul__(other)

    def _eval_power(b, e):
        if e.is_Integer and e.is_positive:
            return super().__pow__(int(e) % 2)


def evaluate_pauli_product(arg):
    '''Help function to evaluate Pauli matrices product
    with symbolic objects.

    Parameters
    ==========

    arg: symbolic expression that contains Paulimatrices

    Examples
    ========

    >>> from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
    >>> from sympy import I
    >>> evaluate_pauli_product(I*Pauli(1)*Pauli(2))
    -sigma3

    >>> from sympy.abc import x
    >>> evaluate_pauli_product(x**2*Pauli(2)*Pauli(1))
    -I*x**2*sigma3
    '''
    start = arg
    end = arg

    if isinstance(arg, Pow) and isinstance(arg.args[0], Pauli):
        if arg.args[1].is_odd:
            return arg.args[0]
        else:
            return 1

    if isinstance(arg, Add):
        return Add(*[evaluate_pauli_product(part) for part in arg.args])

    if isinstance(arg, TensorProduct):
        return TensorProduct(*[evaluate_pauli_product(part) for part in arg.args])

    elif not(isinstance(arg, Mul)):
        return arg

    while not start == end or start == arg and end == arg:
        start = end

        tmp = start.as_coeff_mul()
        sigma_product = 1
        com_product = 1
        keeper = 1

        for el in tmp[1]:
            if isinstance(el, Pauli):
                sigma_product *= el
            elif not el.is_commutative:
                if isinstance(el, Pow) and isinstance(el.args[0], Pauli):
                    if el.args[1].is_odd:
                        sigma_product *= el.args[0]
                elif isinstance(el, TensorProduct):
                    keeper = keeper*sigma_product*\
                        TensorProduct(
                            *[evaluate_pauli_product(part) for part in el.args]
                        )
                    sigma_product = 1
                else:
                    keeper = keeper*sigma_product*el
                    sigma_product = 1
            else:
                com_product *= el
        end = tmp[0]*keeper*sigma_product*com_product
        if end == arg: break
    return end
<|endoftext|>from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.spherical_harmonics import Ynm


def R_nl(n, l, r, Z=1):
    """
    Returns the Hydrogen radial wavefunction R_{nl}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    r :
        Radial coordinate.
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import R_nl
    >>> from sympy.abc import r, Z
    >>> R_nl(1, 0, r, Z)
    2*sqrt(Z**3)*exp(-Z*r)
    >>> R_nl(2, 0, r, Z)
    sqrt(2)*(-Z*r + 2)*sqrt(Z**3)*exp(-Z*r/2)/4
    >>> R_nl(2, 1, r, Z)
    sqrt(6)*Z*r*sqrt(Z**3)*exp(-Z*r/2)/12

    For Hydrogen atom, you can just use the default value of Z=1:

    >>> R_nl(1, 0, r)
    2*exp(-r)
    >>> R_nl(2, 0, r)
    sqrt(2)*(2 - r)*exp(-r/2)/4
    >>> R_nl(3, 0, r)
    2*sqrt(3)*(2*r**2/9 - 2*r + 3)*exp(-r/3)/27

    For Silver atom, you would use Z=47:

    >>> R_nl(1, 0, r, Z=47)
    94*sqrt(47)*exp(-47*r)
    >>> R_nl(2, 0, r, Z=47)
    47*sqrt(94)*(2 - 47*r)*exp(-47*r/2)/4
    >>> R_nl(3, 0, r, Z=47)
    94*sqrt(141)*(4418*r**2/9 - 94*r + 3)*exp(-47*r/3)/27

    The normalization of the radial wavefunction is:

    >>> from sympy import integrate, oo
    >>> integrate(R_nl(1, 0, r)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 0, r)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 1, r)**2 * r**2, (r, 0, oo))
    1

    It holds for any atomic number:

    >>> integrate(R_nl(1, 0, r, Z=2)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 0, r, Z=3)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 1, r, Z=4)**2 * r**2, (r, 0, oo))
    1

    """
    # sympify arguments
    n, l, r, Z = map(S, [n, l, r, Z])
    # radial quantum number
    n_r = n - l - 1
    # rescaled "r"
    a = 1/Z  # Bohr radius
    r0 = 2 * r / (n * a)
    # normalization coefficient
    C = sqrt((S(2)/(n*a))**3 * factorial(n_r) / (2*n*factorial(n + l)))
    # This is an equivalent normalization coefficient, that can be found in
    # some books. Both coefficients seem to be the same fast:
    # C =  S(2)/n**2 * sqrt(1/a**3 * factorial(n_r) / (factorial(n+l)))
    return C * r0**l * assoc_laguerre(n_r, 2*l + 1, r0).expand() * exp(-r0/2)


def Psi_nlm(n, l, m, r, phi, theta, Z=1):
    """
    Returns the Hydrogen wave function psi_{nlm}. It's the product of
    the radial wavefunction R_{nl} and the spherical harmonic Y_{l}^{m}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    m : integer
        ``m`` is the Magnetic Quantum Number with values
        ranging from ``-l`` to ``l``.
    r :
        radial coordinate
    phi :
        azimuthal angle
    theta :
        polar angle
    Z :
        atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import Psi_nlm
    >>> from sympy import Symbol
    >>> r=Symbol("r", positive=True)
    >>> phi=Symbol("phi", real=True)
    >>> theta=Symbol("theta", real=True)
    >>> Z=Symbol("Z", positive=True, integer=True, nonzero=True)
    >>> Psi_nlm(1,0,0,r,phi,theta,Z)
    Z**(3/2)*exp(-Z*r)/sqrt(pi)
    >>> Psi_nlm(2,1,1,r,phi,theta,Z)
    -Z**(5/2)*r*exp(I*phi)*exp(-Z*r/2)*sin(theta)/(8*sqrt(pi))

    Integrating the absolute square of a hydrogen wavefunction psi_{nlm}
    over the whole space leads 1.

    The normalization of the hydrogen wavefunctions Psi_nlm is:

    >>> from sympy import integrate, conjugate, pi, oo, sin
    >>> wf=Psi_nlm(2,1,1,r,phi,theta,Z)
    >>> abs_sqrd=wf*conjugate(wf)
    >>> jacobi=r**2*sin(theta)
    >>> integrate(abs_sqrd*jacobi, (r,0,oo), (phi,0,2*pi), (theta,0,pi))
    1
    """

    # sympify arguments
    n, l, m, r, phi, theta, Z = map(S, [n, l, m, r, phi, theta, Z])
    # check if values for n,l,m make physically sense
    if n.is_integer and n < 1:
        raise ValueError("'n' must be positive integer")
    if l.is_integer and not (n > l):
        raise ValueError("'n' must be greater than 'l'")
    if m.is_integer and not (abs(m) <= l):
        raise ValueError("|'m'| must be less or equal 'l'")
    # return the hydrogen wave function
    return R_nl(n, l, r, Z)*Ynm(l, m, theta, phi).expand(func=True)


def E_nl(n, Z=1):
    """
    Returns the energy of the state (n, l) in Hartree atomic units.

    The energy does not depend on "l".

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl
    >>> from sympy.abc import n, Z
    >>> E_nl(n, Z)
    -Z**2/(2*n**2)
    >>> E_nl(1)
    -1/2
    >>> E_nl(2)
    -1/8
    >>> E_nl(3)
    -1/18
    >>> E_nl(3, 47)
    -2209/18

    """
    n, Z = S(n), S(Z)
    if n.is_integer and (n < 1):
        raise ValueError("'n' must be positive integer")
    return -Z**2/(2*n**2)


def E_nl_dirac(n, l, spin_up=True, Z=1, c=Float("137.035999037")):
    """
    Returns the relativistic energy of the state (n, l, spin) in Hartree atomic
    units.

    The energy is calculated from the Dirac equation. The rest mass energy is
    *not* included.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    spin_up :
        True if the electron spin is up (default), otherwise down
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)
    c :
        Speed of light in atomic units. Default value is 137.035999037,
        taken from https://arxiv.org/abs/1012.3627

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl_dirac
    >>> E_nl_dirac(1, 0)
    -0.500006656595360

    >>> E_nl_dirac(2, 0)
    -0.125002080189006
    >>> E_nl_dirac(2, 1)
    -0.125000416028342
    >>> E_nl_dirac(2, 1, False)
    -0.125002080189006

    >>> E_nl_dirac(3, 0)
    -0.0555562951740285
    >>> E_nl_dirac(3, 1)
    -0.0555558020932949
    >>> E_nl_dirac(3, 1, False)
    -0.0555562951740285
    >>> E_nl_dirac(3, 2)
    -0.0555556377366884
    >>> E_nl_dirac(3, 2, False)
    -0.0555558020932949

    """
    n, l, Z, c = map(S, [n, l, Z, c])
    if not (l >= 0):
        raise ValueError("'l' must be positive or zero")
    if not (n > l):
        raise ValueError("'n' must be greater than 'l'")
    if (l == 0 and spin_up is False):
        raise ValueError("Spin must be up for l==0.")
    # skappa is sign*kappa, where sign contains the correct sign
    if spin_up:
        skappa = -l - 1
    else:
        skappa = -l
    beta = sqrt(skappa**2 - Z**2/c**2)
    return c**2/sqrt(1 + Z**2/(n + skappa + beta)**2/c**2) - c**2
<|endoftext|>"""Known matrices related to physics"""

from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated


def msigma(i):
    r"""Returns a Pauli matrix `\sigma_i` with `i=1,2,3`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pauli_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import msigma
    >>> msigma(1)
    Matrix([
    [0, 1],
    [1, 0]])
    """
    if i == 1:
        mat = (
            (0, 1),
            (1, 0)
        )
    elif i == 2:
        mat = (
            (0, -I),
            (I, 0)
        )
    elif i == 3:
        mat = (
            (1, 0),
            (0, -1)
        )
    else:
        raise IndexError("Invalid Pauli index")
    return Matrix(mat)


def pat_matrix(m, dx, dy, dz):
    """Returns the Parallel Axis Theorem matrix to translate the inertia
    matrix a distance of `(dx, dy, dz)` for a body of mass m.

    Examples
    ========

    To translate a body having a mass of 2 units a distance of 1 unit along
    the `x`-axis we get:

    >>> from sympy.physics.matrices import pat_matrix
    >>> pat_matrix(2, 1, 0, 0)
    Matrix([
    [0, 0, 0],
    [0, 2, 0],
    [0, 0, 2]])

    """
    dxdy = -dx*dy
    dydz = -dy*dz
    dzdx = -dz*dx
    dxdx = dx**2
    dydy = dy**2
    dzdz = dz**2
    mat = ((dydy + dzdz, dxdy, dzdx),
           (dxdy, dxdx + dzdz, dydz),
           (dzdx, dydz, dydy + dxdx))
    return m*Matrix(mat)


def mgamma(mu, lower=False):
    r"""Returns a Dirac gamma matrix `\gamma^\mu` in the standard
    (Dirac) representation.

    Explanation
    ===========

    If you want `\gamma_\mu`, use ``gamma(mu, True)``.

    We use a convention:

    `\gamma^5 = i \cdot \gamma^0 \cdot \gamma^1 \cdot \gamma^2 \cdot \gamma^3`

    `\gamma_5 = i \cdot \gamma_0 \cdot \gamma_1 \cdot \gamma_2 \cdot \gamma_3 = - \gamma^5`

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import mgamma
    >>> mgamma(1)
    Matrix([
    [ 0,  0, 0, 1],
    [ 0,  0, 1, 0],
    [ 0, -1, 0, 0],
    [-1,  0, 0, 0]])
    """
    if mu not in (0, 1, 2, 3, 5):
        raise IndexError("Invalid Dirac index")
    if mu == 0:
        mat = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, -1)
        )
    elif mu == 1:
        mat = (
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (-1, 0, 0, 0)
        )
    elif mu == 2:
        mat = (
            (0, 0, 0, -I),
            (0, 0, I, 0),
            (0, I, 0, 0),
            (-I, 0, 0, 0)
        )
    elif mu == 3:
        mat = (
            (0, 0, 1, 0),
            (0, 0, 0, -1),
            (-1, 0, 0, 0),
            (0, 1, 0, 0)
        )
    elif mu == 5:
        mat = (
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 0, 0, 0),
            (0, 1, 0, 0)
        )
    m = Matrix(mat)
    if lower:
        if mu in (1, 2, 3, 5):
            m = -m
    return m

#Minkowski tensor using the convention (+,-,-,-) used in the Quantum Field
#Theory
minkowski_tensor = Matrix( (
    (1, 0, 0, 0),
    (0, -1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, 0, -1)
))


@deprecated(
    """
    The sympy.physics.matrices.mdft method is deprecated. Use
    sympy.DFT(n).as_explicit() instead.
    """,
    deprecated_since_version="1.9",
    active_deprecations_target="deprecated-physics-mdft",
)
def mdft(n):
    r"""
    .. deprecated:: 1.9

       Use DFT from sympy.matrices.expressions.fourier instead.

       To get identical behavior to ``mdft(n)``, use ``DFT(n).as_explicit()``.
    """
    from sympy.matrices.expressions.fourier import DFT
    return DFT(n).as_mutable()
<|endoftext|>from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar


def psi_n(n, x, m, omega):
    """
    Returns the wavefunction psi_{n} for the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        the "nodal" quantum number.  Corresponds to the number of nodes in the
        wavefunction.  ``n >= 0``
    x :
        x coordinate.
    m :
        Mass of the particle.
    omega :
        Angular frequency of the oscillator.

    Examples
    ========

    >>> from sympy.physics.qho_1d import psi_n
    >>> from sympy.abc import m, x, omega
    >>> psi_n(0, x, m, omega)
    (m*omega)**(1/4)*exp(-m*omega*x**2/(2*hbar))/(hbar**(1/4)*pi**(1/4))

    """

    # sympify arguments
    n, x, m, omega = map(S, [n, x, m, omega])
    nu = m * omega / hbar
    # normalization coefficient
    C = (nu/pi)**Rational(1, 4) * sqrt(1/(2**n*factorial(n)))

    return C * exp(-nu* x**2 /2) * hermite(n, sqrt(nu)*x)


def E_n(n, omega):
    """
    Returns the Energy of the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    omega :
        The harmonic oscillator angular frequency.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_n = hbar * omega*(n + 1/2)

    Examples
    ========

    >>> from sympy.physics.qho_1d import E_n
    >>> from sympy.abc import x, omega
    >>> E_n(x, omega)
    hbar*omega*(x + 1/2)
    """

    return hbar * omega * (n + S.Half)


def coherent_state(n, alpha):
    """
    Returns <n|alpha> for the coherent states of 1D harmonic oscillator.
    See https://en.wikipedia.org/wiki/Coherent_states

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    alpha :
        The eigen value of annihilation operator.
    """

    return exp(- Abs(alpha)**2/2)*(alpha**n)/sqrt(factorial(n))
<|endoftext|>r"""Activation dynamics for musclotendon models.

Musculotendon models are able to produce active force when they are activated,
which is when a chemical process has taken place within the muscle fibers
causing them to voluntarily contract. Biologically this chemical process (the
diffusion of :math:`\textrm{Ca}^{2+}` ions) is not the input in the system,
electrical signals from the nervous system are. These are termed excitations.
Activation dynamics, which relates the normalized excitation level to the
normalized activation level, can be modeled by the models present in this
module.

"""

from abc import ABC, abstractmethod
from functools import cached_property

from sympy.core.symbol import Symbol
from sympy.core.numbers import Float, Integer, Rational
from sympy.functions.elementary.hyperbolic import tanh
from sympy.matrices.dense import MutableDenseMatrix as Matrix, zeros
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics import dynamicsymbols


__all__ = [
    'ActivationBase',
    'FirstOrderActivationDeGroote2016',
    'ZerothOrderActivation',
]


class ActivationBase(ABC, _NamedMixin):
    """Abstract base class for all activation dynamics classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom activation dynamics types through
    subclassing.

    """

    def __init__(self, name):
        """Initializer for ``ActivationBase``."""
        self.name = str(name)

        # Symbols
        self._e = dynamicsymbols(f"e_{name}")
        self._a = dynamicsymbols(f"a_{name}")

    @classmethod
    @abstractmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants."""
        pass

    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
        return self._e

    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
        return self._e

    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
        return self._a

    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
        return self._a

    @property
    @abstractmethod
    def order(self):
        """Order of the (differential) equation governing activation."""
        pass

    @property
    @abstractmethod
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``constants`` can also be used to access the same attribute.

        """
        pass

    @property
    @abstractmethod
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        pass

    @property
    @abstractmethod
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        pass

    @abstractmethod
    def rhs(self):
        """

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        pass

    def __eq__(self, other):
        """Equality check for activation dynamics."""
        if type(self) != type(other):
            return False
        if self.name != other.name:
            return False
        return True

    def __repr__(self):
        """Default representation of activation dynamics."""
        return f'{self.__class__.__name__}({self.name!r})'


class ZerothOrderActivation(ActivationBase):
    """Simple zeroth-order activation dynamics mapping excitation to
    activation.

    Explanation
    ===========

    Zeroth-order activation dynamics are useful in instances where you want to
    reduce the complexity of your musculotendon dynamics as they simple map
    exictation to activation. As a result, no additional state equations are
    introduced to your system. They also remove a potential source of delay
    between the input and dynamics of your system as no (ordinary) differential
    equations are involved.

    """

    def __init__(self, name):
        """Initializer for ``ZerothOrderActivation``.

        Parameters
        ==========

        name : str
            The name identifier associated with the instance. Must be a string
            of length at least 1.

        """
        super().__init__(name)

        # Zeroth-order activation dynamics has activation equal excitation so
        # overwrite the symbol for activation with the excitation symbol.
        self._a = self._e

    @classmethod
    def with_defaults(cls, name):
        """Alternate constructor that provides recommended defaults for
        constants.

        Explanation
        ===========

        As this concrete class doesn't implement any constants associated with
        its dynamics, this ``classmethod`` simply creates a standard instance
        of ``ZerothOrderActivation``. An implementation is provided to ensure
        a consistent interface between all ``ActivationBase`` concrete classes.

        """
        return cls(name)

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        return 0

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property return an empty column ``Matrix`` with shape (0, 1).

        The alias ``x`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated state variables and so this
        property return an empty column ``Matrix`` with shape (0, 1).

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column ``Matrix`` with one entry, ``e``, and
        shape (1, 1).

        The alias ``r`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        Excitation is the only input in zeroth-order activation dynamics and so
        this property returns a column ``Matrix`` with one entry, ``e``, and
        shape (1, 1).

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column ``Matrix`` with shape (0, 1).

        The alias ``p`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        As zeroth-order activation dynamics simply maps excitation to
        activation, this class has no associated constants and so this property
        return an empty column ``Matrix`` with shape (0, 1).

        The alias ``constants`` can also be used to access the same attribute.

        """
        return zeros(0, 1)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
        return Matrix([])

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
        return zeros(0, 1)

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """
        return zeros(0, 1)


class FirstOrderActivationDeGroote2016(ActivationBase):
    r"""First-order activation dynamics based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the first-order activation dynamics equation for the rate of change
    of activation with respect to time as a function of excitation and
    activation.

    The function is defined by the equation:

    .. math::

        \frac{da}{dt} = \left(\frac{\frac{1}{2} + a0}{\tau_a \left(\frac{1}{2}
            + \frac{3a}{2}\right)} + \frac{\left(\frac{1}{2}
            + \frac{3a}{2}\right) \left(\frac{1}{2} - a0\right)}{\tau_d}\right)
            \left(e - a\right)

    where

    .. math::

        a0 = \frac{\tanh{\left(b \left(e - a\right) \right)}}{2}

    with constant values of :math:`tau_a = 0.015`, :math:`tau_d = 0.060`, and
    :math:`b = 10`.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    def __init__(self,
        name,
        activation_time_constant=None,
        deactivation_time_constant=None,
        smoothing_rate=None,
    ):
        """Initializer for ``FirstOrderActivationDeGroote2016``.

        Parameters
        ==========
        activation time constant : Symbol | Number | None
            The value of the activation time constant governing the delay
            between excitation and activation when excitation exceeds
            activation.
        deactivation time constant : Symbol | Number | None
            The value of the deactivation time constant governing the delay
            between excitation and activation when activation exceeds
            excitation.
        smoothing_rate : Symbol | Number | None
            The slope of the hyperbolic tangent function used to smooth between
            the switching of the equations where excitation exceed activation
            and where activation exceeds excitation. The recommended value to
            use is ``10``, but values between ``0.1`` and ``100`` can be used.

        """
        super().__init__(name)

        # Symbols
        self.activation_time_constant = activation_time_constant
        self.deactivation_time_constant = deactivation_time_constant
        self.smoothing_rate = smoothing_rate

    @classmethod
    def with_defaults(cls, name):
        r"""Alternate constructor that will use the published constants.

        Explanation
        ===========

        Returns an instance of ``FirstOrderActivationDeGroote2016`` using the
        three constant values specified in the original publication.

        These have the values:

        :math:`tau_a = 0.015`
        :math:`tau_d = 0.060`
        :math:`b = 10`

        """
        tau_a = Float('0.015')
        tau_d = Float('0.060')
        b = Float('10.0')
        return cls(name, tau_a, tau_d, b)

    @property
    def activation_time_constant(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ```tau_a`` can also be used to access the same attribute.

        """
        return self._tau_a

    @activation_time_constant.setter
    def activation_time_constant(self, tau_a):
        if hasattr(self, '_tau_a'):
            msg = (
                f'Can\'t set attribute `activation_time_constant` to '
                f'{repr(tau_a)} as it is immutable and already has value '
                f'{self._tau_a}.'
            )
            raise AttributeError(msg)
        self._tau_a = Symbol(f'tau_a_{self.name}') if tau_a is None else tau_a

    @property
    def tau_a(self):
        """Delay constant for activation.

        Explanation
        ===========

        The alias ``activation_time_constant`` can also be used to access the
        same attribute.

        """
        return self._tau_a

    @property
    def deactivation_time_constant(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``tau_d`` can also be used to access the same attribute.

        """
        return self._tau_d

    @deactivation_time_constant.setter
    def deactivation_time_constant(self, tau_d):
        if hasattr(self, '_tau_d'):
            msg = (
                f'Can\'t set attribute `deactivation_time_constant` to '
                f'{repr(tau_d)} as it is immutable and already has value '
                f'{self._tau_d}.'
            )
            raise AttributeError(msg)
        self._tau_d = Symbol(f'tau_d_{self.name}') if tau_d is None else tau_d

    @property
    def tau_d(self):
        """Delay constant for deactivation.

        Explanation
        ===========

        The alias ``deactivation_time_constant`` can also be used to access the
        same attribute.

        """
        return self._tau_d

    @property
    def smoothing_rate(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``b`` can also be used to access the same attribute.

        """
        return self._b

    @smoothing_rate.setter
    def smoothing_rate(self, b):
        if hasattr(self, '_b'):
            msg = (
                f'Can\'t set attribute `smoothing_rate` to {b!r} as it is '
                f'immutable and already has value {self._b!r}.'
            )
            raise AttributeError(msg)
        self._b = Symbol(f'b_{self.name}') if b is None else b

    @property
    def b(self):
        """Smoothing constant for the hyperbolic tangent term.

        Explanation
        ===========

        The alias ``smoothing_rate`` can also be used to access the same
        attribute.

        """
        return self._b

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        return 1

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        return Matrix([self._a])

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        return Matrix([self._a])

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        return Matrix([self._e])

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        Explanation
        ===========

        The alias ``p`` can also be used to access the same attribute.

        """
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
        constants = [self._tau_a, self._tau_d, self._b]
        symbolic_constants = [c for c in constants if not c.is_number]
        return Matrix(symbolic_constants) if symbolic_constants else zeros(0, 1)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        return Matrix([Integer(1)])

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        return Matrix([self._da_eqn])

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        """
        return Matrix([self._da_eqn])

    @cached_property
    def _da_eqn(self):
        HALF = Rational(1, 2)
        a0 = HALF * tanh(self._b * (self._e - self._a))
        a1 = (HALF + Rational(3, 2) * self._a)
        a2 = (HALF + a0) / (self._tau_a * a1)
        a3 = a1 * (HALF - a0) / self._tau_d
        activation_dynamics_equation = (a2 + a3) * (self._e - self._a)
        return activation_dynamics_equation

    def __eq__(self, other):
        """Equality check for ``FirstOrderActivationDeGroote2016``."""
        if type(self) != type(other):
            return False
        self_attrs = (self.name, self.tau_a, self.tau_d, self.b)
        other_attrs = (other.name, other.tau_a, other.tau_d, other.b)
        if self_attrs == other_attrs:
            return True
        return False

    def __repr__(self):
        """Representation of ``FirstOrderActivationDeGroote2016``."""
        return (
            f'{self.__class__.__name__}({self.name!r}, '
            f'activation_time_constant={self.tau_a!r}, '
            f'deactivation_time_constant={self.tau_d!r}, '
            f'smoothing_rate={self.b!r})'
        )
<|endoftext|>"""Biomechanics extension for SymPy.

Includes biomechanics-related constructs which allows users to extend multibody
models created using `sympy.physics.mechanics` into biomechanical or
musculoskeletal models involding musculotendons and activation dynamics.

"""

from .activation import (
   ActivationBase,
   FirstOrderActivationDeGroote2016,
   ZerothOrderActivation,
)
from .curve import (
   CharacteristicCurveCollection,
   CharacteristicCurveFunction,
   FiberForceLengthActiveDeGroote2016,
   FiberForceLengthPassiveDeGroote2016,
   FiberForceLengthPassiveInverseDeGroote2016,
   FiberForceVelocityDeGroote2016,
   FiberForceVelocityInverseDeGroote2016,
   TendonForceLengthDeGroote2016,
   TendonForceLengthInverseDeGroote2016,
)
from .musculotendon import (
   MusculotendonBase,
   MusculotendonDeGroote2016,
   MusculotendonFormulation,
)


__all__ = [
   # Musculotendon characteristic curve functions
   'CharacteristicCurveCollection',
   'CharacteristicCurveFunction',
   'FiberForceLengthActiveDeGroote2016',
   'FiberForceLengthPassiveDeGroote2016',
   'FiberForceLengthPassiveInverseDeGroote2016',
   'FiberForceVelocityDeGroote2016',
   'FiberForceVelocityInverseDeGroote2016',
   'TendonForceLengthDeGroote2016',
   'TendonForceLengthInverseDeGroote2016',

   # Activation dynamics classes
   'ActivationBase',
   'FirstOrderActivationDeGroote2016',
   'ZerothOrderActivation',

   # Musculotendon classes
   'MusculotendonBase',
   'MusculotendonDeGroote2016',
   'MusculotendonFormulation',
]
<|endoftext|>"""Implementations of musculotendon models.

Musculotendon models are a critical component of biomechanical models, one that
differentiates them from pure multibody systems. Musculotendon models produce a
force dependent on their level of activation, their length, and their
extension velocity. Length- and extension velocity-dependent force production
are governed by force-length and force-velocity characteristics.
These are normalized functions that are dependent on the musculotendon's state
and are specific to a given musculotendon model.

"""

from abc import abstractmethod
from enum import IntEnum, unique

from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix, diag, eye, zeros
from sympy.physics.biomechanics.activation import ActivationBase
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator
from sympy.physics.vector.functions import dynamicsymbols


__all__ = [
    'MusculotendonBase',
    'MusculotendonDeGroote2016',
    'MusculotendonFormulation',
]


@unique
class MusculotendonFormulation(IntEnum):
    """Enumeration of types of musculotendon dynamics formulations.

    Explanation
    ===========

    An (integer) enumeration is used as it allows for clearer selection of the
    different formulations of musculotendon dynamics.

    Members
    =======

    RIGID_TENDON : 0
        A rigid tendon model.
    FIBER_LENGTH_EXPLICIT : 1
        An explicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable.
    TENDON_FORCE_EXPLICIT : 2
        An explicit elastic tendon model with the tendon force (F_T) as the
        state variable.
    FIBER_LENGTH_IMPLICIT : 3
        An implicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable and the muscle fiber velocity as an additional input
        variable.
    TENDON_FORCE_IMPLICIT : 4
        An implicit elastic tendon model with the tendon force (F_T) as the
        state variable as the muscle fiber velocity as an additional input
        variable.

    """

    RIGID_TENDON = 0
    FIBER_LENGTH_EXPLICIT = 1
    TENDON_FORCE_EXPLICIT = 2
    FIBER_LENGTH_IMPLICIT = 3
    TENDON_FORCE_IMPLICIT = 4

    def __str__(self):
        """Returns a string representation of the enumeration value.

        Notes
        =====

        This hard coding is required due to an incompatibility between the
        ``IntEnum`` implementations in Python 3.10 and Python 3.11
        (https://github.com/python/cpython/issues/84247). From Python 3.11
        onwards, the ``__str__`` method uses ``int.__str__``, whereas prior it
        used ``Enum.__str__``. Once Python 3.11 becomes the minimum version
        supported by SymPy, this method override can be removed.

        """
        return str(self.value)


_DEFAULT_MUSCULOTENDON_FORMULATION = MusculotendonFormulation.RIGID_TENDON


class MusculotendonBase(ForceActuator, _NamedMixin):
    r"""Abstract base class for all musculotendon classes to inherit from.

    Explanation
    ===========

    A musculotendon generates a contractile force based on its activation,
    length, and shortening velocity. This abstract base class is to be inherited
    by all musculotendon subclasses that implement different characteristic
    musculotendon curves. Characteristic musculotendon curves are required for
    the tendon force-length, passive fiber force-length, active fiber force-
    length, and fiber force-velocity relationships.

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    """

    def __init__(
        self,
        name,
        pathway,
        activation_dynamics,
        *,
        musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
        tendon_slack_length=None,
        peak_isometric_force=None,
        optimal_fiber_length=None,
        maximal_fiber_velocity=None,
        optimal_pennation_angle=None,
        fiber_damping_coefficient=None,
        with_defaults=False,
    ):
        self.name = name

        # Supply a placeholder force to the super initializer, this will be
        # replaced later
        super().__init__(Symbol('F'), pathway)

        # Activation dynamics
        if not isinstance(activation_dynamics, ActivationBase):
            msg = (
                f'Can\'t set attribute `activation_dynamics` to '
                f'{activation_dynamics} as it must be of type '
                f'`ActivationBase`, not {type(activation_dynamics)}.'
            )
            raise TypeError(msg)
        self._activation_dynamics = activation_dynamics
        self._child_objects = (self._activation_dynamics, )

        # Constants
        if tendon_slack_length is not None:
            self._l_T_slack = tendon_slack_length
        else:
            self._l_T_slack = Symbol(f'l_T_slack_{self.name}')
        if peak_isometric_force is not None:
            self._F_M_max = peak_isometric_force
        else:
            self._F_M_max = Symbol(f'F_M_max_{self.name}')
        if optimal_fiber_length is not None:
            self._l_M_opt = optimal_fiber_length
        else:
            self._l_M_opt = Symbol(f'l_M_opt_{self.name}')
        if maximal_fiber_velocity is not None:
            self._v_M_max = maximal_fiber_velocity
        else:
            self._v_M_max = Symbol(f'v_M_max_{self.name}')
        if optimal_pennation_angle is not None:
            self._alpha_opt = optimal_pennation_angle
        else:
            self._alpha_opt = Symbol(f'alpha_opt_{self.name}')
        if fiber_damping_coefficient is not None:
            self._beta = fiber_damping_coefficient
        else:
            self._beta = Symbol(f'beta_{self.name}')

        # Musculotendon dynamics
        self._with_defaults = with_defaults
        if musculotendon_dynamics == MusculotendonFormulation.RIGID_TENDON:
            self._rigid_tendon_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.FIBER_LENGTH_EXPLICIT:
            self._fiber_length_explicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.TENDON_FORCE_EXPLICIT:
            self._tendon_force_explicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.FIBER_LENGTH_IMPLICIT:
            self._fiber_length_implicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.TENDON_FORCE_IMPLICIT:
            self._tendon_force_implicit_musculotendon_dynamics()
        else:
            msg = (
                f'Musculotendon dynamics {repr(musculotendon_dynamics)} '
                f'passed to `musculotendon_dynamics` was of type '
                f'{type(musculotendon_dynamics)}, must be '
                f'{MusculotendonFormulation}.'
            )
            raise TypeError(msg)
        self._musculotendon_dynamics = musculotendon_dynamics

        # Must override the placeholder value in `self._force` now that the
        # actual force has been calculated by
        # `self._<MUSCULOTENDON FORMULATION>_musculotendon_dynamics`.
        # Note that `self._force` assumes forces are expansile, musculotendon
        # forces are contractile hence the minus sign preceding `self._F_T`
        # (the tendon force).
        self._force = -self._F_T

    @classmethod
    def with_defaults(
        cls,
        name,
        pathway,
        activation_dynamics,
        *,
        musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
        tendon_slack_length=None,
        peak_isometric_force=None,
        optimal_fiber_length=None,
        maximal_fiber_velocity=Float('10.0'),
        optimal_pennation_angle=Float('0.0'),
        fiber_damping_coefficient=Float('0.1'),
    ):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the musculotendon class using recommended
        values for ``v_M_max``, ``alpha_opt``, and ``beta``. The values are:

            :math:`v^M_{max} = 10`
            :math:`\alpha_{opt} = 0`
            :math:`\beta = \frac{1}{10}`

        The musculotendon curves are also instantiated using the constants from
        the original publication.

        Parameters
        ==========

        name : str
            The name identifier associated with the musculotendon. This name is
            used as a suffix when automatically generated symbols are
            instantiated. It must be a string of nonzero length.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of a
            concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        activation_dynamics : ActivationBase
            The activation dynamics that will be modeled within the
            musculotendon. This must be an instance of a concrete subclass of
            ``ActivationBase``, e.g. ``FirstOrderActivationDeGroote2016``.
        musculotendon_dynamics : MusculotendonFormulation | int
            The formulation of musculotendon dynamics that should be used
            internally, i.e. rigid or elastic tendon model, the choice of
            musculotendon state etc. This must be a member of the integer
            enumeration ``MusculotendonFormulation`` or an integer that can be
            cast to a member. To use a rigid tendon formulation, set this to
            ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value
            ``0``, which will be cast to the enumeration member). There are four
            possible formulations for an elastic tendon model. To use an
            explicit formulation with the fiber length as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer
            value ``1``). To use an explicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT`` (or the integer
            value ``2``). To use an implicit formulation with the fiber length
            as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer
            value ``3``). To use an implicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT`` (or the integer
            value ``4``). The default is
            ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a
            rigid tendon formulation.
        tendon_slack_length : Expr | None
            The length of the tendon when the musculotendon is in its unloaded
            state. In a rigid tendon model the tendon length is the tendon slack
            length. In all musculotendon models, tendon slack length is used to
            normalize tendon length to give
            :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
        peak_isometric_force : Expr | None
            The maximum force that the muscle fiber can produce when it is
            undergoing an isometric contraction (no lengthening velocity). In
            all musculotendon models, peak isometric force is used to normalized
            tendon and muscle fiber force to give
            :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
        optimal_fiber_length : Expr | None
            The muscle fiber length at which the muscle fibers produce no
            passive force and their maximum active force. In all musculotendon
            models, optimal fiber length is used to normalize muscle fiber
            length to give :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
        maximal_fiber_velocity : Expr | None
            The fiber velocity at which, during muscle fiber shortening, the
            muscle fibers are unable to produce any active force. In all
            musculotendon models, maximal fiber velocity is used to normalize
            muscle fiber extension velocity to give
            :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
        optimal_pennation_angle : Expr | None
            The pennation angle when muscle fiber length equals the optimal
            fiber length.
        fiber_damping_coefficient : Expr | None
            The coefficient of damping to be used in the damping element in the
            muscle fiber model.

        """
        return cls(
            name,
            pathway,
            activation_dynamics=activation_dynamics,
            musculotendon_dynamics=musculotendon_dynamics,
            tendon_slack_length=tendon_slack_length,
            peak_isometric_force=peak_isometric_force,
            optimal_fiber_length=optimal_fiber_length,
            maximal_fiber_velocity=maximal_fiber_velocity,
            optimal_pennation_angle=optimal_pennation_angle,
            fiber_damping_coefficient=fiber_damping_coefficient,
            with_defaults=True,
        )

    @abstractmethod
    def curves(cls):
        """Return a ``CharacteristicCurveCollection`` of the curves related to
        the specific model."""
        pass

    @property
    def tendon_slack_length(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``l_T_slack`` can also be used to access the same attribute.

        """
        return self._l_T_slack

    @property
    def l_T_slack(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``tendon_slack_length`` can also be used to access the same
        attribute.

        """
        return self._l_T_slack

    @property
    def peak_isometric_force(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``F_M_max`` can also be used to access the same attribute.

        """
        return self._F_M_max

    @property
    def F_M_max(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``peak_isometric_force`` can also be used to access the same
        attribute.

        """
        return self._F_M_max

    @property
    def optimal_fiber_length(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``l_M_opt`` can also be used to access the same attribute.

        """
        return self._l_M_opt

    @property
    def l_M_opt(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``optimal_fiber_length`` can also be used to access the same
        attribute.

        """
        return self._l_M_opt

    @property
    def maximal_fiber_velocity(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``v_M_max`` can also be used to access the same attribute.

        """
        return self._v_M_max

    @property
    def v_M_max(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``maximal_fiber_velocity`` can also be used to access the same
        attribute.

        """
        return self._v_M_max

    @property
    def optimal_pennation_angle(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``alpha_opt`` can also be used to access the same attribute.

        """
        return self._alpha_opt

    @property
    def alpha_opt(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``optimal_pennation_angle`` can also be used to access the
        same attribute.

        """
        return self._alpha_opt

    @property
    def fiber_damping_coefficient(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``beta`` can also be used to access the same attribute.

        """
        return self._beta

    @property
    def beta(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``fiber_damping_coefficient`` can also be used to access the
        same attribute.

        """
        return self._beta

    @property
    def activation_dynamics(self):
        """Activation dynamics model governing this musculotendon's activation.

        Explanation
        ===========

        Returns the instance of a subclass of ``ActivationBase`` that governs
        the relationship between excitation and activation that is used to
        represent the activation dynamics of this musculotendon.

        """
        return self._activation_dynamics

    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._e

    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._e

    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._a

    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._a

    @property
    def musculotendon_dynamics(self):
        """The choice of rigid or type of elastic tendon musculotendon dynamics.

        Explanation
        ===========

        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.

        """
        return self._musculotendon_dynamics

    def _rigid_tendon_musculotendon_dynamics(self):
        """Rigid tendon musculotendon."""
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._l_T = self._l_T_slack
        self._l_T_tilde = Integer(1)
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        self._v_M = self._v_MT*(self._l_MT - self._l_T_slack)/self._l_M
        self._v_M_tilde = self._v_M/self._v_M_max
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
            self._fv_M = self.curves.fiber_force_velocity.with_defaults(self._v_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M = self.curves.fiber_force_velocity(self._v_M_tilde, *fv_M_constants)
        self._F_M_tilde = self.a*self._fl_M_act*self._fv_M + self._fl_M_pas + self._beta*self._v_M_tilde
        self._F_T_tilde = self._F_M_tilde
        self._F_M = self._F_M_tilde*self._F_M_max
        self._cos_alpha = cos(self._alpha_opt)
        self._F_T = self._F_M*self._cos_alpha

        # Containers
        self._state_vars = zeros(0, 1)
        self._input_vars = zeros(0, 1)
        self._state_eqns = zeros(0, 1)
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _fiber_length_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `l_M_tilde` as a state."""
        self._l_M_tilde = dynamicsymbols(f'l_M_tilde_{self.name}')
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._l_M = self._l_M_tilde*self._l_M_opt
        self._l_T = self._l_MT - sqrt(self._l_M**2 - (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_T_tilde = self._l_T/self._l_T_slack
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        self._F_T_tilde = self._fl_T
        self._F_T = self._F_T_tilde*self._F_M_max
        self._F_M = self._F_T/self._cos_alpha
        self._F_M_tilde = self._F_M/self._F_M_max
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        if self._with_defaults:
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        self._dl_M_tilde_dt = (self._v_M_max/self._l_M_opt)*self._v_M_tilde

        self._state_vars = Matrix([self._l_M_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dl_M_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _tendon_force_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `F_T_tilde` as a state."""
        self._F_T_tilde = dynamicsymbols(f'F_T_tilde_{self.name}')
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._fl_T = self._F_T_tilde
        if self._with_defaults:
            self._fl_T_inv = self.curves.tendon_force_length_inverse.with_defaults(self._fl_T)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T_inv = self.curves.tendon_force_length_inverse(self._fl_T, *fl_T_constants)
        self._l_T_tilde = self._fl_T_inv
        self._l_T = self._l_T_tilde*self._l_T_slack
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        if self._with_defaults:
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        self._F_T = self._F_T_tilde*self._F_M_max
        self._F_M = self._F_T/self._cos_alpha
        self._F_M_tilde = self._F_M/self._F_M_max
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        if self._with_defaults:
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        self._v_M_tilde = self._fv_M_inv
        self._v_M = self._v_M_tilde*self._v_M_max
        self._v_T = self._v_MT - (self._v_M/self._cos_alpha)
        self._v_T_tilde = self._v_T/self._l_T_slack
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
        else:
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
        self._dF_T_tilde_dt = self._fl_T.diff(dynamicsymbols._t).subs({self._l_T_tilde.diff(dynamicsymbols._t): self._v_T_tilde})

        self._state_vars = Matrix([self._F_T_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dF_T_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _fiber_length_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    def _tendon_force_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        state_vars = [self._state_vars]
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        return Matrix.vstack(*state_vars)

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        state_vars = [self._state_vars]
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        return Matrix.vstack(*state_vars)

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        input_vars = [self._input_vars]
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        return Matrix.vstack(*input_vars)

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        input_vars = [self._input_vars]
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        return Matrix.vstack(*input_vars)

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``p`` can also be used to access the same attribute.

        """
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        for child in self._child_objects:
            constants.append(child.constants)
        constants.append(self._curve_constants)
        return Matrix.vstack(*constants)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        for child in self._child_objects:
            constants.append(child.constants)
        constants.append(self._curve_constants)
        return Matrix.vstack(*constants)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
        M = [eye(len(self._state_vars))]
        for child in self._child_objects:
            M.append(child.M)
        return diag(*M)

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
        F = [self._state_eqns]
        for child in self._child_objects:
            F.append(child.F)
        return Matrix.vstack(*F)

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """
        is_explicit = (
            MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
            MusculotendonFormulation.TENDON_FORCE_EXPLICIT,
        )
        if self.musculotendon_dynamics is MusculotendonFormulation.RIGID_TENDON:
            child_rhs = [child.rhs() for child in self._child_objects]
            return Matrix.vstack(*child_rhs)
        elif self.musculotendon_dynamics in is_explicit:
            rhs = self._state_eqns
            child_rhs = [child.rhs() for child in self._child_objects]
            return Matrix.vstack(rhs, *child_rhs)
        return self.M.solve(self.F)

    def __repr__(self):
        """Returns a string representation to reinstantiate the model."""
        return (
            f'{self.__class__.__name__}({self.name!r}, '
            f'pathway={self.pathway!r}, '
            f'activation_dynamics={self.activation_dynamics!r}, '
            f'musculotendon_dynamics={self.musculotendon_dynamics}, '
            f'tendon_slack_length={self._l_T_slack!r}, '
            f'peak_isometric_force={self._F_M_max!r}, '
            f'optimal_fiber_length={self._l_M_opt!r}, '
            f'maximal_fiber_velocity={self._v_M_max!r}, '
            f'optimal_pennation_angle={self._alpha_opt!r}, '
            f'fiber_damping_coefficient={self._beta!r})'
        )

    def __str__(self):
        """Returns a string representation of the expression for musculotendon
        force."""
        return str(self.force)


class MusculotendonDeGroote2016(MusculotendonBase):
    r"""Musculotendon model using the curves of De Groote et al., 2016 [1]_.

    Examples
    ========

    This class models the musculotendon actuator parametrized by the
    characteristic curves described in De Groote et al., 2016 [1]_. Like all
    musculotendon models in SymPy's biomechanics module, it requires a pathway
    to define its line of action. We'll begin by creating a simple
    ``LinearPathway`` between two points that our musculotendon will follow.
    We'll create a point ``O`` to represent the musculotendon's origin and
    another ``I`` to represent its insertion.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, Point,
    ...     ReferenceFrame, dynamicsymbols)

    >>> N = ReferenceFrame('N')
    >>> O, I = O, P = symbols('O, I', cls=Point)
    >>> q, u = dynamicsymbols('q, u', real=True)
    >>> I.set_pos(O, q*N.x)
    >>> O.set_vel(N, 0)
    >>> I.set_vel(N, u*N.x)
    >>> pathway = LinearPathway(O, I)
    >>> pathway.attachments
    (O, I)
    >>> pathway.length
    Abs(q(t))
    >>> pathway.extension_velocity
    sign(q(t))*Derivative(q(t), t)

    A musculotendon also takes an instance of an activation dynamics model as
    this will be used to provide symbols for the activation in the formulation
    of the musculotendon dynamics. We'll use an instance of
    ``FirstOrderActivationDeGroote2016`` to represent first-order activation
    dynamics. Note that a single name argument needs to be provided as SymPy
    will use this as a suffix.

    >>> from sympy.physics.biomechanics import FirstOrderActivationDeGroote2016

    >>> activation = FirstOrderActivationDeGroote2016('muscle')
    >>> activation.x
    Matrix([[a_muscle(t)]])
    >>> activation.r
    Matrix([[e_muscle(t)]])
    >>> activation.p
    Matrix([
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])
    >>> activation.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class requires symbols or values to be passed to represent
    the constants in the musculotendon dynamics. We'll use SymPy's ``symbols``
    function to create symbols for the maximum isometric force ``F_M_max``,
    optimal fiber length ``l_M_opt``, tendon slack length ``l_T_slack``, maximum
    fiber velocity ``v_M_max``, optimal pennation angle ``alpha_opt, and fiber
    damping coefficient ``beta``.

    >>> F_M_max = symbols('F_M_max', real=True)
    >>> l_M_opt = symbols('l_M_opt', real=True)
    >>> l_T_slack = symbols('l_T_slack', real=True)
    >>> v_M_max = symbols('v_M_max', real=True)
    >>> alpha_opt = symbols('alpha_opt', real=True)
    >>> beta = symbols('beta', real=True)

    We can then import the class ``MusculotendonDeGroote2016`` from the
    biomechanics module and create an instance by passing in the various objects
    we have previously instantiated. By default, a musculotendon model with
    rigid tendon musculotendon dynamics will be created.

    >>> from sympy.physics.biomechanics import MusculotendonDeGroote2016

    >>> rigid_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    We can inspect the various properties of the musculotendon, including
    getting the symbolic expression describing the force it produces using its
    ``force`` attribute.

    >>> rigid_tendon_muscle.force
    -F_M_max*(beta*(-l_T_slack + Abs(q(t)))*sign(q(t))*Derivative(q(t), t)...

    When we created the musculotendon object, we passed in an instance of an
    activation dynamics object that governs the activation within the
    musculotendon. SymPy makes a design choice here that the activation dynamics
    instance will be treated as a child object of the musculotendon dynamics.
    Therefore, if we want to inspect the state and input variables associated
    with the musculotendon model, we will also be returned the state and input
    variables associated with the child object, or the activation dynamics in
    this case. As the musculotendon model that we created here uses rigid tendon
    dynamics, no additional states or inputs relating to the musculotendon are
    introduces. Consequently, the model has a single state associated with it,
    the activation, and a single input associated with it, the excitation. The
    states and inputs can be inspected using the ``x`` and ``r`` attributes
    respectively. Note that both ``x`` and ``r`` have the alias attributes of
    ``state_vars`` and ``input_vars``.

    >>> rigid_tendon_muscle.x
    Matrix([[a_muscle(t)]])
    >>> rigid_tendon_muscle.r
    Matrix([[e_muscle(t)]])

    To see which constants are symbolic in the musculotendon model, we can use
    the ``p`` or ``constants`` attribute. This returns a ``Matrix`` populated
    by the constants that are represented by a ``Symbol`` rather than a numeric
    value.

    >>> rigid_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])

    Finally, we can call the ``rhs`` method to return a ``Matrix`` that
    contains as its elements the righthand side of the ordinary differential
    equations corresponding to each of the musculotendon's states. Like the
    method with the same name on the ``Method`` classes in SymPy's mechanics
    module, this returns a column vector where the number of rows corresponds to
    the number of states. For our example here, we have a single state, the
    dynamic symbol ``a_muscle(t)``, so the returned value is a 1-by-1
    ``Matrix``.

    >>> rigid_tendon_muscle.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class supports elastic tendon musculotendon models in
    addition to rigid tendon ones. You can choose to either use the fiber length
    or tendon force as an additional state. You can also specify whether an
    explicit or implicit formulation should be used. To select a formulation,
    pass a member of the ``MusculotendonFormulation`` enumeration to the
    ``musculotendon_dynamics`` parameter when calling the constructor. This
    enumeration is an ``IntEnum``, so you can also pass an integer, however it
    is recommended to use the enumeration as it is clearer which formulation you
    are actually selecting. Below, we'll use the ``FIBER_LENGTH_EXPLICIT``
    member to create a musculotendon with an elastic tendon that will use the
    (normalized) muscle fiber length as an additional state and will produce
    the governing ordinary differential equation in explicit form.

    >>> from sympy.physics.biomechanics import MusculotendonFormulation

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    >>> elastic_tendon_muscle.force
    -F_M_max*TendonForceLengthDeGroote2016((-sqrt(l_M_opt**2*...
    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])
    >>> elastic_tendon_muscle.rhs()
    Matrix([
    [v_M_max*FiberForceVelocityInverseDeGroote2016((l_M_opt*...],
    [ ((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    It is strongly recommended to use the alternate ``with_defaults``
    constructor when creating an instance because this will ensure that the
    published constants are used in the musculotendon characteristic curves.

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016.with_defaults(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ... )

    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [   l_T_slack],
    [     F_M_max],
    [     l_M_opt],
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    curves = CharacteristicCurveCollection(
        tendon_force_length=TendonForceLengthDeGroote2016,
        tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
        fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
        fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
        fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
        fiber_force_velocity=FiberForceVelocityDeGroote2016,
        fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
    )
<|endoftext|>"""Implementations of characteristic curves for musculotendon models."""

from dataclasses import dataclass

from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Float, Integer
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.precedence import PRECEDENCE


__all__ = [
    'CharacteristicCurveCollection',
    'CharacteristicCurveFunction',
    'FiberForceLengthActiveDeGroote2016',
    'FiberForceLengthPassiveDeGroote2016',
    'FiberForceLengthPassiveInverseDeGroote2016',
    'FiberForceVelocityDeGroote2016',
    'FiberForceVelocityInverseDeGroote2016',
    'TendonForceLengthDeGroote2016',
    'TendonForceLengthInverseDeGroote2016',
]


class CharacteristicCurveFunction(Function):
    """Base class for all musculotendon characteristic curve functions."""

    @classmethod
    def eval(cls):
        msg = (
            f'Cannot directly instantiate {cls.__name__!r}, instances of '
            f'characteristic curves must be of a concrete subclass.'

        )
        raise TypeError(msg)

    def _print_code(self, printer):
        """Print code for the function defining the curve using a printer.

        Explanation
        ===========

        The order of operations may need to be controlled as constant folding
        the numeric terms within the equations of a musculotendon
        characteristic curve can sometimes results in a numerically-unstable
        expression.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print a string representation of the
            characteristic curve as valid code in the target language.

        """
        return printer._print(printer.parenthesize(
            self.doit(deep=False, evaluate=False), PRECEDENCE['Atom'],
        ))

    _ccode = _print_code
    _cupycode = _print_code
    _cxxcode = _print_code
    _fcode = _print_code
    _jaxcode = _print_code
    _lambdacode = _print_code
    _mpmathcode = _print_code
    _octave = _print_code
    _pythoncode = _print_code
    _numpycode = _print_code
    _scipycode = _print_code


class TendonForceLengthDeGroote2016(CharacteristicCurveFunction):
    r"""Tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon force produced as a function of normalized
    tendon length.

    The function is defined by the equation:

    $fl^T = c_0 \exp{c_3 \left( \tilde{l}^T - c_1 \right)} - c_2$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon length. We'll create a
    :class:`~.Symbol` called ``l_T_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    >>> l_T_tilde = Symbol('l_T_tilde')
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fl_T = TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_T`` and
    ``l_T_slack``, representing tendon length and tendon slack length
    respectively. We can then represent ``l_T_tilde`` as an expression, the
    ratio of these.

    >>> l_T, l_T_slack = symbols('l_T l_T_slack')
    >>> l_T_tilde = l_T/l_T_slack
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T/l_T_slack, 0.2, 0.995, 0.25,
    33.93669377311689)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_T.doit(evaluate=False)
    -0.25 + 0.2*exp(33.93669377311689*(l_T/l_T_slack - 0.995))

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> fl_T.diff(l_T)
    6.787338754623378*exp(33.93669377311689*(l_T/l_T_slack - 0.995))/l_T_slack

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_T_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the tendon force-length function using the
        four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
