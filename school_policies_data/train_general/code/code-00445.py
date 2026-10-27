    >>> list(Cycle(1, 2).items())
    [(1, 2), (2, 1)]

    See Also
    ========

    Permutation
    """
    def __missing__(self, arg):
        """Enter arg into dictionary and return arg."""
        return as_int(arg)

    def __iter__(self):
        yield from self.list()

    def __call__(self, *other):
        """Return product of cycles processed from R to L.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)(2, 3)
        (1 3 2)

        An instance of a Cycle will automatically parse list-like
        objects and Permutations that are on the right. It is more
        flexible than the Permutation in that all elements need not
        be present:

        >>> a = Cycle(1, 2)
        >>> a(2, 3)
        (1 3 2)
        >>> a(2, 3)(4, 5)
        (1 3 2)(4 5)

        """
        rv = Cycle(*other)
        for k, v in zip(list(self.keys()), [rv[self[k]] for k in self.keys()]):
            rv[k] = v
        return rv

    def list(self, size=None):
        """Return the cycles as an explicit list starting from 0 up
        to the greater of the largest value in the cycles and size.

        Truncation of trailing unmoved items will occur when size
        is less than the maximum element in the cycle; if this is
        desired, setting ``size=-1`` will guarantee such trimming.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> p = Cycle(2, 3)(4, 5)
        >>> p.list()
        [0, 1, 3, 2, 5, 4]
        >>> p.list(10)
        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]

        Passing a length too small will trim trailing, unchanged elements
        in the permutation:

        >>> Cycle(2, 4)(1, 2, 4).list(-1)
        [0, 2, 1]
        """
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        if size is not None:
            big = max([i for i in self.keys() if self[i] != i] + [0])
            size = max(size, big + 1)
        else:
            size = self.size
        return [self[i] for i in range(size)]

    def __repr__(self):
        """We want it to print as a Cycle, not as a dict.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> print(_)
        (1 2)
        >>> list(Cycle(1, 2).items())
        [(1, 2), (2, 1)]
        """
        if not self:
            return 'Cycle()'
        cycles = Permutation(self).cyclic_form
        s = ''.join(str(tuple(c)) for c in cycles)
        big = self.size - 1
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        return 'Cycle%s' % s

    def __str__(self):
        """We want it to be printed in a Cycle notation with no
        comma in-between.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2)
        (1 2)
        >>> Cycle(1, 2, 4)(5, 6)
        (1 2 4)(5 6)
        """
        if not self:
            return '()'
        cycles = Permutation(self).cyclic_form
        s = ''.join(str(tuple(c)) for c in cycles)
        big = self.size - 1
        if not any(i == big for c in cycles for i in c):
            s += '(%s)' % big
        s = s.replace(',', '')
        return s

    def __init__(self, *args):
        """Load up a Cycle instance with the values for the cycle.

        Examples
        ========

        >>> from sympy.combinatorics import Cycle
        >>> Cycle(1, 2, 6)
        (1 2 6)
        """

        if not args:
            return
        if len(args) == 1:
            if isinstance(args[0], Permutation):
                for c in args[0].cyclic_form:
                    self.update(self(*c))
                return
            elif isinstance(args[0], Cycle):
                for k, v in args[0].items():
                    self[k] = v
                return
        args = [as_int(a) for a in args]
        if any(i < 0 for i in args):
            raise ValueError('negative integers are not allowed in a cycle.')
        if has_dups(args):
            raise ValueError('All elements must be unique in a cycle.')
        for i in range(-len(args), 0):
            self[args[i]] = args[i + 1]

    @property
    def size(self):
        if not self:
            return 0
        return max(self.keys()) + 1

    def copy(self):
        return Cycle(self)


class Permutation(Atom):
    r"""
    A permutation, alternatively known as an 'arrangement number' or 'ordering'
    is an arrangement of the elements of an ordered list into a one-to-one
    mapping with itself. The permutation of a given arrangement is given by
    indicating the positions of the elements after re-arrangement [2]_. For
    example, if one started with elements ``[x, y, a, b]`` (in that order) and
    they were reordered as ``[x, y, b, a]`` then the permutation would be
    ``[0, 1, 3, 2]``. Notice that (in SymPy) the first element is always referred
    to as 0 and the permutation uses the indices of the elements in the
    original ordering, not the elements ``(a, b, ...)`` themselves.

    >>> from sympy.combinatorics import Permutation
    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    Permutations Notation
    =====================

    Permutations are commonly represented in disjoint cycle or array forms.

    Array Notation and 2-line Form
    ------------------------------------

    In the 2-line form, the elements and their final positions are shown
    as a matrix with 2 rows:

    [0    1    2     ... n-1]
    [p(0) p(1) p(2)  ... p(n-1)]

    Since the first line is always ``range(n)``, where n is the size of p,
    it is sufficient to represent the permutation by the second line,
    referred to as the "array form" of the permutation. This is entered
    in brackets as the argument to the Permutation class:

    >>> p = Permutation([0, 2, 1]); p
    Permutation([0, 2, 1])

    Given i in range(p.size), the permutation maps i to i^p

    >>> [i^p for i in range(p.size)]
    [0, 2, 1]

    The composite of two permutations p*q means first apply p, then q, so
    i^(p*q) = (i^p)^q which is i^p^q according to Python precedence rules:

    >>> q = Permutation([2, 1, 0])
    >>> [i^p^q for i in range(3)]
    [2, 0, 1]
    >>> [i^(p*q) for i in range(3)]
    [2, 0, 1]

    One can use also the notation p(i) = i^p, but then the composition
    rule is (p*q)(i) = q(p(i)), not p(q(i)):

    >>> [(p*q)(i) for i in range(p.size)]
    [2, 0, 1]
    >>> [q(p(i)) for i in range(p.size)]
    [2, 0, 1]
    >>> [p(q(i)) for i in range(p.size)]
    [1, 2, 0]

    Disjoint Cycle Notation
    -----------------------

    In disjoint cycle notation, only the elements that have shifted are
    indicated.

    For example, [1, 3, 2, 0] can be represented as (0, 1, 3)(2).
    This can be understood from the 2 line format of the given permutation.
    In the 2-line form,
    [0    1    2   3]
    [1    3    2   0]

    The element in the 0th position is 1, so 0 -> 1. The element in the 1st
    position is three, so 1 -> 3. And the element in the third position is again
    0, so 3 -> 0. Thus, 0 -> 1 -> 3 -> 0, and 2 -> 2. Thus, this can be represented
    as 2 cycles: (0, 1, 3)(2).
    In common notation, singular cycles are not explicitly written as they can be
    inferred implicitly.

    Only the relative ordering of elements in a cycle matter:

    >>> Permutation(1,2,3) == Permutation(2,3,1) == Permutation(3,1,2)
    True

    The disjoint cycle notation is convenient when representing
    permutations that have several cycles in them:

    >>> Permutation(1, 2)(3, 5) == Permutation([[1, 2], [3, 5]])
    True

    It also provides some economy in entry when computing products of
    permutations that are written in disjoint cycle notation:

    >>> Permutation(1, 2)(1, 3)(2, 3)
    Permutation([0, 3, 2, 1])
    >>> _ == Permutation([[1, 2]])*Permutation([[1, 3]])*Permutation([[2, 3]])
    True

        Caution: when the cycles have common elements between them then the order
        in which the permutations are applied matters. This module applies
        the permutations from *left to right*.

        >>> Permutation(1, 2)(2, 3) == Permutation([(1, 2), (2, 3)])
        True
        >>> Permutation(1, 2)(2, 3).list()
        [0, 3, 1, 2]

        In the above case, (1,2) is computed before (2,3).
        As 0 -> 0, 0 -> 0, element in position 0 is 0.
        As 1 -> 2, 2 -> 3, element in position 1 is 3.
        As 2 -> 1, 1 -> 1, element in position 2 is 1.
        As 3 -> 3, 3 -> 2, element in position 3 is 2.

        If the first and second elements had been
        swapped first, followed by the swapping of the second
        and third, the result would have been [0, 2, 3, 1].
        If, you want to apply the cycles in the conventional
        right to left order, call the function with arguments in reverse order
        as demonstrated below:

        >>> Permutation([(1, 2), (2, 3)][::-1]).list()
        [0, 2, 3, 1]

    Entering a singleton in a permutation is a way to indicate the size of the
    permutation. The ``size`` keyword can also be used.

    Array-form entry:

    >>> Permutation([[1, 2], [9]])
    Permutation([0, 2, 1], size=10)
    >>> Permutation([[1, 2]], size=10)
    Permutation([0, 2, 1], size=10)

    Cyclic-form entry:

    >>> Permutation(1, 2, size=10)
    Permutation([0, 2, 1], size=10)
    >>> Permutation(9)(1, 2)
    Permutation([0, 2, 1], size=10)

    Caution: no singleton containing an element larger than the largest
    in any previous cycle can be entered. This is an important difference
    in how Permutation and Cycle handle the ``__call__`` syntax. A singleton
    argument at the start of a Permutation performs instantiation of the
    Permutation and is permitted:

    >>> Permutation(5)
    Permutation([], size=6)

    A singleton entered after instantiation is a call to the permutation
    -- a function call -- and if the argument is out of range it will
    trigger an error. For this reason, it is better to start the cycle
    with the singleton:

    The following fails because there is no element 3:

    >>> Permutation(1, 2)(3)
    Traceback (most recent call last):
    ...
    IndexError: list index out of range

    This is ok: only the call to an out of range singleton is prohibited;
    otherwise the permutation autosizes:

    >>> Permutation(3)(1, 2)
    Permutation([0, 2, 1, 3])
    >>> Permutation(1, 2)(3, 4) == Permutation(3, 4)(1, 2)
    True


    Equality testing
    ----------------

    The array forms must be the same in order for permutations to be equal:

    >>> Permutation([1, 0, 2, 3]) == Permutation([1, 0])
    False


    Identity Permutation
    --------------------

    The identity permutation is a permutation in which no element is out of
    place. It can be entered in a variety of ways. All the following create
    an identity permutation of size 4:

    >>> I = Permutation([0, 1, 2, 3])
    >>> all(p == I for p in [
    ... Permutation(3),
    ... Permutation(range(4)),
    ... Permutation([], size=4),
    ... Permutation(size=4)])
    True

    Watch out for entering the range *inside* a set of brackets (which is
    cycle notation):

    >>> I == Permutation([range(4)])
    False


    Permutation Printing
    ====================

    There are a few things to note about how Permutations are printed.

    .. deprecated:: 1.6

       Configuring Permutation printing by setting
       ``Permutation.print_cyclic`` is deprecated. Users should use the
       ``perm_cyclic`` flag to the printers, as described below.

    1) If you prefer one form (array or cycle) over another, you can set
    ``init_printing`` with the ``perm_cyclic`` flag.

    >>> from sympy import init_printing
    >>> p = Permutation(1, 2)(4, 5)(3, 4)
    >>> p
    Permutation([0, 2, 1, 4, 5, 3])

    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (1 2)(3 4 5)

    2) Regardless of the setting, a list of elements in the array for cyclic
    form can be obtained and either of those can be copied and supplied as
    the argument to Permutation:

    >>> p.array_form
    [0, 2, 1, 4, 5, 3]
    >>> p.cyclic_form
    [[1, 2], [3, 4, 5]]
    >>> Permutation(_) == p
    True

    3) Printing is economical in that as little as possible is printed while
    retaining all information about the size of the permutation:

    >>> init_printing(perm_cyclic=False, pretty_print=False)
    >>> Permutation([1, 0, 2, 3])
    Permutation([1, 0, 2, 3])
    >>> Permutation([1, 0, 2, 3], size=20)
    Permutation([1, 0], size=20)
    >>> Permutation([1, 0, 2, 4, 3, 5, 6], size=20)
    Permutation([1, 0, 2, 4, 3], size=20)

    >>> p = Permutation([1, 0, 2, 3])
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> p
    (3)(0 1)
    >>> init_printing(perm_cyclic=False, pretty_print=False)

    The 2 was not printed but it is still there as can be seen with the
    array_form and size methods:

    >>> p.array_form
    [1, 0, 2, 3]
    >>> p.size
    4

    Short introduction to other methods
    ===================================

    The permutation can act as a bijective function, telling what element is
    located at a given position

    >>> q = Permutation([5, 2, 3, 4, 1, 0])
    >>> q.array_form[1] # the hard way
    2
    >>> q(1) # the easy way
    2
    >>> {i: q(i) for i in range(q.size)} # showing the bijection
    {0: 5, 1: 2, 2: 3, 3: 4, 4: 1, 5: 0}

    The full cyclic form (including singletons) can be obtained:

    >>> p.full_cyclic_form
    [[0, 1], [2], [3]]

    Any permutation can be factored into transpositions of pairs of elements:

    >>> Permutation([[1, 2], [3, 4, 5]]).transpositions()
    [(1, 2), (3, 5), (3, 4)]
    >>> Permutation.rmul(*[Permutation([ti], size=6) for ti in _]).cyclic_form
    [[1, 2], [3, 4, 5]]

    The number of permutations on a set of n elements is given by n! and is
    called the cardinality.

    >>> p.size
    4
    >>> p.cardinality
    24

    A given permutation has a rank among all the possible permutations of the
    same elements, but what that rank is depends on how the permutations are
    enumerated. (There are a number of different methods of doing so.) The
    lexicographic rank is given by the rank method and this rank is used to
    increment a permutation with addition/subtraction:

    >>> p.rank()
    6
    >>> p + 1
    Permutation([1, 0, 3, 2])
    >>> p.next_lex()
    Permutation([1, 0, 3, 2])
    >>> _.rank()
    7
    >>> p.unrank_lex(p.size, rank=7)
    Permutation([1, 0, 3, 2])

    The product of two permutations p and q is defined as their composition as
    functions, (p*q)(i) = q(p(i)) [6]_.

    >>> p = Permutation([1, 0, 2, 3])
    >>> q = Permutation([2, 3, 1, 0])
    >>> list(q*p)
    [2, 3, 0, 1]
    >>> list(p*q)
    [3, 2, 1, 0]
    >>> [q(p(i)) for i in range(p.size)]
    [3, 2, 1, 0]

    The permutation can be 'applied' to any list-like object, not only
    Permutations:

    >>> p(['zero', 'one', 'four', 'two'])
    ['one', 'zero', 'four', 'two']
    >>> p('zo42')
    ['o', 'z', '4', '2']

    If you have a list of arbitrary elements, the corresponding permutation
    can be found with the from_sequence method:

    >>> Permutation.from_sequence('SymPy')
    Permutation([1, 3, 2, 0, 4])

    Checking if a Permutation is contained in a Group
    =================================================

    Generally if you have a group of permutations G on n symbols, and
    you're checking if a permutation on less than n symbols is part
    of that group, the check will fail.

    Here is an example for n=5 and we check if the cycle
    (1,2,3) is in G:

    >>> from sympy import init_printing
    >>> init_printing(perm_cyclic=True, pretty_print=False)
    >>> from sympy.combinatorics import Cycle, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> G = PermutationGroup(Cycle(2, 3)(4, 5), Cycle(1, 2, 3, 4, 5))
    >>> p1 = Permutation(Cycle(2, 5, 3))
    >>> p2 = Permutation(Cycle(1, 2, 3))
    >>> a1 = Permutation(Cycle(1, 2, 3).list(6))
    >>> a2 = Permutation(Cycle(1, 2, 3)(5))
    >>> a3 = Permutation(Cycle(1, 2, 3),size=6)
    >>> for p in [p1,p2,a1,a2,a3]: p, G.contains(p)
    ((2 5 3), True)
    ((1 2 3), False)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)
    ((5)(1 2 3), True)

    The check for p2 above will fail.

    Checking if p1 is in G works because SymPy knows
    G is a group on 5 symbols, and p1 is also on 5 symbols
    (its largest element is 5).

    For ``a1``, the ``.list(6)`` call will extend the permutation to 5
    symbols, so the test will work as well. In the case of ``a2`` the
    permutation is being extended to 5 symbols by using a singleton,
    and in the case of ``a3`` it's extended through the constructor
    argument ``size=6``.

    There is another way to do this, which is to tell the ``contains``
    method that the number of symbols the group is on does not need to
    match perfectly the number of symbols for the permutation:

    >>> G.contains(p2,strict=False)
    True

    This can be via the ``strict`` argument to the ``contains`` method,
    and SymPy will try to extend the permutation on its own and then
    perform the containment check.

    See Also
    ========

    Cycle

    References
    ==========

    .. [1] Skiena, S. 'Permutations.' 1.1 in Implementing Discrete Mathematics
           Combinatorics and Graph Theory with Mathematica.  Reading, MA:
           Addison-Wesley, pp. 3-16, 1990.

    .. [2] Knuth, D. E. The Art of Computer Programming, Vol. 4: Combinatorial
           Algorithms, 1st ed. Reading, MA: Addison-Wesley, 2011.

    .. [3] Wendy Myrvold and Frank Ruskey. 2001. Ranking and unranking
           permutations in linear time. Inf. Process. Lett. 79, 6 (September 2001),
           281-284. DOI=10.1016/S0020-0190(01)00141-7

    .. [4] D. L. Kreher, D. R. Stinson 'Combinatorial Algorithms'
           CRC Press, 1999

    .. [5] Graham, R. L.; Knuth, D. E.; and Patashnik, O.
           Concrete Mathematics: A Foundation for Computer Science, 2nd ed.
           Reading, MA: Addison-Wesley, 1994.

    .. [6] https://en.wikipedia.org/w/index.php?oldid=499948155#Product_and_inverse

    .. [7] https://en.wikipedia.org/wiki/Lehmer_code

    """

    is_Permutation = True

    _array_form = None
    _cyclic_form = None
    _cycle_structure = None
    _size = None
    _rank = None

    def __new__(cls, *args, size=None, **kwargs):
        """
        Constructor for the Permutation object from a list or a
        list of lists in which all elements of the permutation may
        appear only once.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)

        Permutations entered in array-form are left unaltered:

        >>> Permutation([0, 2, 1])
        Permutation([0, 2, 1])

        Permutations entered in cyclic form are converted to array form;
        singletons need not be entered, but can be entered to indicate the
        largest element:

        >>> Permutation([[4, 5, 6], [0, 1]])
        Permutation([1, 0, 2, 3, 5, 6, 4])
        >>> Permutation([[4, 5, 6], [0, 1], [19]])
        Permutation([1, 0, 2, 3, 5, 6, 4], size=20)

        All manipulation of permutations assumes that the smallest element
        is 0 (in keeping with 0-based indexing in Python) so if the 0 is
        missing when entering a permutation in array form, an error will be
        raised:

        >>> Permutation([2, 1])
        Traceback (most recent call last):
        ...
        ValueError: Integers 0 through 2 must be present.

        If a permutation is entered in cyclic form, it can be entered without
        singletons and the ``size`` specified so those values can be filled
        in, otherwise the array form will only extend to the maximum value
        in the cycles:

        >>> Permutation([[1, 4], [3, 5, 2]], size=10)
        Permutation([0, 4, 3, 5, 1, 2], size=10)
        >>> _.array_form
        [0, 4, 3, 5, 1, 2, 6, 7, 8, 9]
        """
        if size is not None:
            size = int(size)

        #a) ()
        #b) (1) = identity
        #c) (1, 2) = cycle
        #d) ([1, 2, 3]) = array form
        #e) ([[1, 2]]) = cyclic form
        #f) (Cycle) = conversion to permutation
        #g) (Permutation) = adjust size or return copy
        ok = True
        if not args:  # a
            return cls._af_new(list(range(size or 0)))
        elif len(args) > 1:  # c
            return cls._af_new(Cycle(*args).list(size))
        if len(args) == 1:
            a = args[0]
            if isinstance(a, cls):  # g
                if size is None or size == a.size:
                    return a
                return cls(a.array_form, size=size)
            if isinstance(a, Cycle):  # f
                return cls._af_new(a.list(size))
            if not is_sequence(a):  # b
                if size is not None and a + 1 > size:
                    raise ValueError('size is too small when max is %s' % a)
                return cls._af_new(list(range(a + 1)))
            if has_variety(is_sequence(ai) for ai in a):
                ok = False
        else:
            ok = False
        if not ok:
            raise ValueError("Permutation argument must be a list of ints, "
                             "a list of lists, Permutation or Cycle.")

        # safe to assume args are valid; this also makes a copy
        # of the args
        args = list(args[0])

        is_cycle = args and is_sequence(args[0])
        if is_cycle:  # e
            args = [[int(i) for i in c] for c in args]
        else:  # d
            args = [int(i) for i in args]

        # if there are n elements present, 0, 1, ..., n-1 should be present
        # unless a cycle notation has been provided. A 0 will be added
        # for convenience in case one wants to enter permutations where
        # counting starts from 1.

        temp = flatten(args)
        if has_dups(temp) and not is_cycle:
            raise ValueError('there were repeated elements.')
        temp = set(temp)

        if not is_cycle:
            if temp != set(range(len(temp))):
                raise ValueError('Integers 0 through %s must be present.' %
                max(temp))
            if size is not None and temp and max(temp) + 1 > size:
                raise ValueError('max element should not exceed %s' % (size - 1))

        if is_cycle:
            # it's not necessarily canonical so we won't store
            # it -- use the array form instead
            c = Cycle()
            for ci in args:
                c = c(*ci)
            aform = c.list()
        else:
            aform = list(args)
        if size and size > len(aform):
            # don't allow for truncation of permutation which
            # might split a cycle and lead to an invalid aform
            # but do allow the permutation size to be increased
            aform.extend(list(range(len(aform), size)))

        return cls._af_new(aform)

    @classmethod
    def _af_new(cls, perm):
        """A method to produce a Permutation object from a list;
        the list is bound to the _array_form attribute, so it must
        not be modified; this method is meant for internal use only;
        the list ``a`` is supposed to be generated as a temporary value
        in a method, so p = Perm._af_new(a) is the only object
        to hold a reference to ``a``::

        Examples
        ========

        >>> from sympy.combinatorics.permutations import Perm
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = [2, 1, 3, 0]
        >>> p = Perm._af_new(a)
        >>> p
        Permutation([2, 1, 3, 0])

        """
        p = super().__new__(cls)
        p._array_form = perm
        p._size = len(perm)
        return p

    def copy(self):
        return self.__class__(self.array_form)

    def __getnewargs__(self):
        return (self.array_form,)

    def _hashable_content(self):
        # the array_form (a list) is the Permutation arg, so we need to
        # return a tuple, instead
        return tuple(self.array_form)

    @property
    def array_form(self):
        """
        Return a copy of the attribute _array_form
        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> p.array_form
        [2, 3, 0, 1]
        >>> Permutation([[2, 0, 3, 1]]).array_form
        [3, 2, 0, 1]
        >>> Permutation([2, 0, 3, 1]).array_form
        [2, 0, 3, 1]
        >>> Permutation([[1, 2], [4, 5]]).array_form
        [0, 2, 1, 3, 5, 4]
        """
        return self._array_form[:]

    def list(self, size=None):
        """Return the permutation as an explicit list, possibly
        trimming unmoved elements if size is less than the maximum
        element in the permutation; if this is desired, setting
        ``size=-1`` will guarantee such trimming.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation(2, 3)(4, 5)
        >>> p.list()
        [0, 1, 3, 2, 5, 4]
        >>> p.list(10)
        [0, 1, 3, 2, 5, 4, 6, 7, 8, 9]

        Passing a length too small will trim trailing, unchanged elements
        in the permutation:

        >>> Permutation(2, 4)(1, 2, 4).list(-1)
        [0, 2, 1]
        >>> Permutation(3).list(-1)
        []
        """
        if not self and size is None:
            raise ValueError('must give size for empty Cycle')
        rv = self.array_form
        if size is not None:
            if size > self.size:
                rv.extend(list(range(self.size, size)))
            else:
                # find first value from rhs where rv[i] != i
                i = self.size - 1
                while rv:
                    if rv[-1] != i:
                        break
                    rv.pop()
                    i -= 1
        return rv

    @property
    def cyclic_form(self):
        """
        This is used to convert to the cyclic notation
        from the canonical notation. Singletons are omitted.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 3, 1, 2])
        >>> p.cyclic_form
        [[1, 3, 2]]
        >>> Permutation([1, 0, 2, 4, 3, 5]).cyclic_form
        [[0, 1], [3, 4]]

        See Also
        ========

        array_form, full_cyclic_form
        """
        if self._cyclic_form is not None:
            return list(self._cyclic_form)
        array_form = self.array_form
        unchecked = [True] * len(array_form)
        cyclic_form = []
        for i in range(len(array_form)):
            if unchecked[i]:
                cycle = []
                cycle.append(i)
                unchecked[i] = False
                j = i
                while unchecked[array_form[j]]:
                    j = array_form[j]
                    cycle.append(j)
                    unchecked[j] = False
                if len(cycle) > 1:
                    cyclic_form.append(cycle)
                    assert cycle == list(minlex(cycle))
        cyclic_form.sort()
        self._cyclic_form = cyclic_form.copy()
        return cyclic_form

    @property
    def full_cyclic_form(self):
        """Return permutation in cyclic form including singletons.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 2, 1]).full_cyclic_form
        [[0], [1, 2]]
        """
        need = set(range(self.size)) - set(flatten(self.cyclic_form))
        rv = self.cyclic_form + [[i] for i in need]
        rv.sort()
        return rv

    @property
    def size(self):
        """
        Returns the number of elements in the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([[3, 2], [0, 1]]).size
        4

        See Also
        ========

        cardinality, length, order, rank
        """
        return self._size

    def support(self):
        """Return the elements in permutation, P, for which P[i] != i.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[3, 2], [0, 1], [4]])
        >>> p.array_form
        [1, 0, 3, 2, 4]
        >>> p.support()
        [0, 1, 2, 3]
        """
        a = self.array_form
        return [i for i, e in enumerate(a) if e != i]

    def __add__(self, other):
        """Return permutation that is other higher in rank than self.

        The rank is the lexicographical rank, with the identity permutation
        having rank of 0.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> I = Permutation([0, 1, 2, 3])
        >>> a = Permutation([2, 1, 3, 0])
        >>> I + a.rank() == a
        True

        See Also
        ========

        __sub__, inversion_vector

        """
        rank = (self.rank() + other) % self.cardinality
        rv = self.unrank_lex(self.size, rank)
        rv._rank = rank
        return rv

    def __sub__(self, other):
        """Return the permutation that is other lower in rank than self.

        See Also
        ========

        __add__
        """
        return self.__add__(-other)

    @staticmethod
    def rmul(*args):
        """
        Return product of Permutations [a, b, c, ...] as the Permutation whose
        ith value is a(b(c(i))).

        a, b, c, ... can be Permutation objects or tuples.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        >>> a, b = [1, 0, 2], [0, 2, 1]
        >>> a = Permutation(a); b = Permutation(b)
        >>> list(Permutation.rmul(a, b))
        [1, 2, 0]
        >>> [a(b(i)) for i in range(3)]
        [1, 2, 0]

        This handles the operands in reverse order compared to the ``*`` operator:

        >>> a = Permutation(a); b = Permutation(b)
        >>> list(a*b)
        [2, 0, 1]
        >>> [b(a(i)) for i in range(3)]
        [2, 0, 1]

        Notes
        =====

        All items in the sequence will be parsed by Permutation as
        necessary as long as the first item is a Permutation:

        >>> Permutation.rmul(a, [0, 2, 1]) == Permutation.rmul(a, b)
        True

        The reverse order of arguments will raise a TypeError.

        """
        rv = args[0]
        for i in range(1, len(args)):
            rv = args[i]*rv
        return rv

    @classmethod
    def rmul_with_af(cls, *args):
        """
        same as rmul, but the elements of args are Permutation objects
        which have _array_form
        """
        a = [x._array_form for x in args]
        rv = cls._af_new(_af_rmuln(*a))
        return rv

    def mul_inv(self, other):
        """
        other*~self, self and other have _array_form
        """
        a = _af_invert(self._array_form)
        b = other._array_form
        return self._af_new(_af_rmul(a, b))

    def __rmul__(self, other):
        """This is needed to coerce other to Permutation in rmul."""
        cls = type(self)
        return cls(other)*self

    def __mul__(self, other):
        """
        Return the product a*b as a Permutation; the ith value is b(a(i)).

        Examples
        ========

        >>> from sympy.combinatorics.permutations import _af_rmul, Permutation

        >>> a, b = [1, 0, 2], [0, 2, 1]
        >>> a = Permutation(a); b = Permutation(b)
        >>> list(a*b)
        [2, 0, 1]
        >>> [b(a(i)) for i in range(3)]
        [2, 0, 1]

        This handles operands in reverse order compared to _af_rmul and rmul:

        >>> al = list(a); bl = list(b)
        >>> _af_rmul(al, bl)
        [1, 2, 0]
        >>> [al[bl[i]] for i in range(3)]
        [1, 2, 0]

        It is acceptable for the arrays to have different lengths; the shorter
        one will be padded to match the longer one:

        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> b*Permutation([1, 0])
        Permutation([1, 2, 0])
        >>> Permutation([1, 0])*b
        Permutation([2, 0, 1])

        It is also acceptable to allow coercion to handle conversion of a
        single list to the left of a Permutation:

        >>> [0, 1]*a # no change: 2-element identity
        Permutation([1, 0, 2])
        >>> [[0, 1]]*a # exchange first two elements
        Permutation([0, 1, 2])

        You cannot use more than 1 cycle notation in a product of cycles
        since coercion can only handle one argument to the left. To handle
        multiple cycles it is convenient to use Cycle instead of Permutation:

        >>> [[1, 2]]*[[2, 3]]*Permutation([]) # doctest: +SKIP
        >>> from sympy.combinatorics.permutations import Cycle
        >>> Cycle(1, 2)(2, 3)
        (1 3 2)

        """
        from sympy.combinatorics.perm_groups import PermutationGroup, Coset
        if isinstance(other, PermutationGroup):
            return Coset(self, other, dir='-')
        a = self.array_form
        # __rmul__ makes sure the other is a Permutation
        b = other.array_form
        if not b:
            perm = a
        else:
            b.extend(list(range(len(b), len(a))))
            perm = [b[i] for i in a] + b[len(a):]
        return self._af_new(perm)

    def commutes_with(self, other):
        """
        Checks if the elements are commuting.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> a = Permutation([1, 4, 3, 0, 2, 5])
        >>> b = Permutation([0, 1, 2, 3, 4, 5])
        >>> a.commutes_with(b)
        True
        >>> b = Permutation([2, 3, 5, 4, 1, 0])
        >>> a.commutes_with(b)
        False
        """
        a = self.array_form
        b = other.array_form
        return _af_commutes_with(a, b)

    def __pow__(self, n):
        """
        Routine for finding powers of a permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([2, 0, 3, 1])
        >>> p.order()
        4
        >>> p**4
        Permutation([0, 1, 2, 3])
        """
        if isinstance(n, Permutation):
            raise NotImplementedError(
                'p**p is not defined; do you mean p^p (conjugate)?')
        n = int(n)
        return self._af_new(_af_pow(self.array_form, n))

    def __rxor__(self, i):
        """Return self(i) when ``i`` is an int.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation(1, 2, 9)
        >>> 2^p == p(2) == 9
        True
        """
        if int_valued(i):
            return self(i)
        else:
            raise NotImplementedError(
                "i^p = p(i) when i is an integer, not %s." % i)

    def __xor__(self, h):
        """Return the conjugate permutation ``~h*self*h` `.

        Explanation
        ===========

        If ``a`` and ``b`` are conjugates, ``a = h*b*~h`` and
        ``b = ~h*a*h`` and both have the same cycle structure.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation(1, 2, 9)
        >>> q = Permutation(6, 9, 8)
        >>> p*q != q*p
        True

        Calculate and check properties of the conjugate:

        >>> c = p^q
        >>> c == ~q*p*q and p == q*c*~q
        True

        The expression q^p^r is equivalent to q^(p*r):

        >>> r = Permutation(9)(4, 6, 8)
        >>> q^p^r == q^(p*r)
        True

        If the term to the left of the conjugate operator, i, is an integer
        then this is interpreted as selecting the ith element from the
        permutation to the right:

        >>> all(i^p == p(i) for i in range(p.size))
        True

        Note that the * operator as higher precedence than the ^ operator:

        >>> q^r*p^r == q^(r*p)^r == Permutation(9)(1, 6, 4)
        True

        Notes
        =====

        In Python the precedence rule is p^q^r = (p^q)^r which differs
        in general from p^(q^r)

        >>> q^p^r
        (9)(1 4 8)
        >>> q^(p^r)
        (9)(1 8 6)

        For a given r and p, both of the following are conjugates of p:
        ~r*p*r and r*p*~r. But these are not necessarily the same:

        >>> ~r*p*r == r*p*~r
        True

        >>> p = Permutation(1, 2, 9)(5, 6)
        >>> ~r*p*r == r*p*~r
        False

        The conjugate ~r*p*r was chosen so that ``p^q^r`` would be equivalent
        to ``p^(q*r)`` rather than ``p^(r*q)``. To obtain r*p*~r, pass ~r to
        this method:

        >>> p^~r == r*p*~r
        True
        """

        if self.size != h.size:
            raise ValueError("The permutations must be of equal size.")
        a = [None]*self.size
        h = h._array_form
        p = self._array_form
        for i in range(self.size):
            a[h[i]] = h[p[i]]
        return self._af_new(a)

    def transpositions(self):
        """
        Return the permutation decomposed into a list of transpositions.

        Explanation
        ===========

        It is always possible to express a permutation as the product of
        transpositions, see [1]

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[1, 2, 3], [0, 4, 5, 6, 7]])
        >>> t = p.transpositions()
        >>> t
        [(0, 7), (0, 6), (0, 5), (0, 4), (1, 3), (1, 2)]
        >>> print(''.join(str(c) for c in t))
        (0, 7)(0, 6)(0, 5)(0, 4)(1, 3)(1, 2)
        >>> Permutation.rmul(*[Permutation([ti], size=p.size) for ti in t]) == p
        True

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transposition_%28mathematics%29#Properties

        """
        a = self.cyclic_form
        res = []
        for x in a:
            nx = len(x)
            if nx == 2:
                res.append(tuple(x))
            elif nx > 2:
                first = x[0]
                res.extend((first, y) for y in x[nx - 1:0:-1])
        return res

    @classmethod
    def from_sequence(self, i, key=None):
        """Return the permutation needed to obtain ``i`` from the sorted
        elements of ``i``. If custom sorting is desired, a key can be given.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        >>> Permutation.from_sequence('SymPy')
        (4)(0 1 3)
        >>> _(sorted("SymPy"))
        ['S', 'y', 'm', 'P', 'y']
        >>> Permutation.from_sequence('SymPy', key=lambda x: x.lower())
        (4)(0 2)(1 3)
        """
        ic = list(zip(i, list(range(len(i)))))
        if key:
            ic.sort(key=lambda x: key(x[0]))
        else:
            ic.sort()
        return ~Permutation([i[1] for i in ic])

    def __invert__(self):
        """
        Return the inverse of the permutation.

        A permutation multiplied by its inverse is the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> ~p
        Permutation([2, 3, 0, 1])
        >>> _ == p**-1
        True
        >>> p*~p == ~p*p == Permutation([0, 1, 2, 3])
        True
        """
        return self._af_new(_af_invert(self._array_form))

    def __iter__(self):
        """Yield elements from array form.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> list(Permutation(range(3)))
        [0, 1, 2]
        """
        yield from self.array_form

    def __repr__(self):
        return srepr(self)

    def __call__(self, *i):
        """
        Allows applying a permutation instance as a bijective function.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([[2, 0], [3, 1]])
        >>> p.array_form
        [2, 3, 0, 1]
        >>> [p(i) for i in range(4)]
        [2, 3, 0, 1]

        If an array is given then the permutation selects the items
        from the array (i.e. the permutation is applied to the array):

        >>> from sympy.abc import x
        >>> p([x, 1, 0, x**2])
        [0, x**2, x, 1]
        """
        # list indices can be Integer or int; leave this
        # as it is (don't test or convert it) because this
        # gets called a lot and should be fast
        if len(i) == 1:
            i = i[0]
            if not isinstance(i, Iterable):
                i = as_int(i)
                if i < 0 or i > self.size:
                    raise TypeError(
                        "{} should be an integer between 0 and {}"
                        .format(i, self.size-1))
                return self._array_form[i]
            # P([a, b, c])
            if len(i) != self.size:
                raise TypeError(
                    "{} should have the length {}.".format(i, self.size))
            return [i[j] for j in self._array_form]
        # P(1, 2, 3)
        return self*Permutation(Cycle(*i), size=self.size)

    def atoms(self):
        """
        Returns all the elements of a permutation

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 1, 2, 3, 4, 5]).atoms()
        {0, 1, 2, 3, 4, 5}
        >>> Permutation([[0, 1], [2, 3], [4, 5]]).atoms()
        {0, 1, 2, 3, 4, 5}
        """
        return set(self.array_form)

    def apply(self, i):
        r"""Apply the permutation to an expression.

        Parameters
        ==========

        i : Expr
            It should be an integer between $0$ and $n-1$ where $n$
            is the size of the permutation.

            If it is a symbol or a symbolic expression that can
            have integer values, an ``AppliedPermutation`` object
            will be returned which can represent an unevaluated
            function.

        Notes
        =====

        Any permutation can be defined as a bijective function
        $\sigma : \{ 0, 1, \dots, n-1 \} \rightarrow \{ 0, 1, \dots, n-1 \}$
        where $n$ denotes the size of the permutation.

        The definition may even be extended for any set with distinctive
        elements, such that the permutation can even be applied for
        real numbers or such, however, it is not implemented for now for
        computational reasons and the integrity with the group theory
        module.

        This function is similar to the ``__call__`` magic, however,
        ``__call__`` magic already has some other applications like
        permuting an array or attaching new cycles, which would
        not always be mathematically consistent.

        This also guarantees that the return type is a SymPy integer,
        which guarantees the safety to use assumptions.
        """
        i = _sympify(i)
        if i.is_integer is False:
            raise NotImplementedError("{} should be an integer.".format(i))

        n = self.size
        if (i < 0) == True or (i >= n) == True:
            raise NotImplementedError(
                "{} should be an integer between 0 and {}".format(i, n-1))

        if i.is_Integer:
            return Integer(self._array_form[i])
        return AppliedPermutation(self, i)

    def next_lex(self):
        """
        Returns the next permutation in lexicographical order.
        If self is the last permutation in lexicographical order
        it returns None.
        See [4] section 2.4.


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 3, 1, 0])
        >>> p = Permutation([2, 3, 1, 0]); p.rank()
        17
        >>> p = p.next_lex(); p.rank()
        18

        See Also
        ========

        rank, unrank_lex
        """
        perm = self.array_form[:]
        n = len(perm)
        i = n - 2
        while perm[i + 1] < perm[i]:
            i -= 1
        if i == -1:
            return None
        else:
            j = n - 1
            while perm[j] < perm[i]:
                j -= 1
            perm[j], perm[i] = perm[i], perm[j]
            i += 1
            j = n - 1
            while i < j:
                perm[j], perm[i] = perm[i], perm[j]
                i += 1
                j -= 1
        return self._af_new(perm)

    @classmethod
    def unrank_nonlex(self, n, r):
        """
        This is a linear time unranking algorithm that does not
        respect lexicographic order [3].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_nonlex(4, 5)
        Permutation([2, 0, 3, 1])
        >>> Permutation.unrank_nonlex(4, -1)
        Permutation([0, 1, 2, 3])

        See Also
        ========

        next_nonlex, rank_nonlex
        """
        def _unrank1(n, r, a):
            if n > 0:
                a[n - 1], a[r % n] = a[r % n], a[n - 1]
                _unrank1(n - 1, r//n, a)

        id_perm = list(range(n))
        n = int(n)
        r = r % ifac(n)
        _unrank1(n, r, id_perm)
        return self._af_new(id_perm)

    def rank_nonlex(self, inv_perm=None):
        """
        This is a linear time ranking algorithm that does not
        enforce lexicographic order [3].


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_nonlex()
        23

        See Also
        ========

        next_nonlex, unrank_nonlex
        """
        def _rank1(n, perm, inv_perm):
            if n == 1:
                return 0
            s = perm[n - 1]
            t = inv_perm[n - 1]
            perm[n - 1], perm[t] = perm[t], s
            inv_perm[n - 1], inv_perm[s] = inv_perm[s], t
            return s + n*_rank1(n - 1, perm, inv_perm)

        if inv_perm is None:
            inv_perm = (~self).array_form
        if not inv_perm:
            return 0
        perm = self.array_form[:]
        r = _rank1(len(perm), perm, inv_perm)
        return r

    def next_nonlex(self):
        """
        Returns the next permutation in nonlex order [3].
        If self is the last permutation in this order it returns None.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([2, 0, 3, 1]); p.rank_nonlex()
        5
        >>> p = p.next_nonlex(); p
        Permutation([3, 0, 1, 2])
        >>> p.rank_nonlex()
        6

        See Also
        ========

        rank_nonlex, unrank_nonlex
        """
        r = self.rank_nonlex()
        if r == ifac(self.size) - 1:
            return None
        return self.unrank_nonlex(self.size, r + 1)

    def rank(self):
        """
        Returns the lexicographic rank of the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank()
        0
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.rank()
        23

        See Also
        ========

        next_lex, unrank_lex, cardinality, length, order, size
        """
        if self._rank is not None:
            return self._rank
        rank = 0
        rho = self.array_form[:]
        n = self.size - 1
        size = n + 1
        psize = int(ifac(n))
        for j in range(size - 1):
            rank += rho[j]*psize
            for i in range(j + 1, size):
                if rho[i] > rho[j]:
                    rho[i] -= 1
            psize //= n
            n -= 1
        self._rank = rank
        return rank

    @property
    def cardinality(self):
        """
        Returns the number of all possible permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.cardinality
        24

        See Also
        ========

        length, order, rank, size
        """
        return int(ifac(self.size))

    def parity(self):
        """
        Computes the parity of a permutation.

        Explanation
        ===========

        The parity of a permutation reflects the parity of the
        number of inversions in the permutation, i.e., the
        number of pairs of x and y such that ``x > y`` but ``p[x] < p[y]``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.parity()
        0
        >>> p = Permutation([3, 2, 0, 1])
        >>> p.parity()
        1

        See Also
        ========

        _af_parity
        """
        if self._cyclic_form is not None:
            return (self.size - self.cycles) % 2

        return _af_parity(self.array_form)

    @property
    def is_even(self):
        """
        Checks if a permutation is even.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.is_even
        True
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.is_even
        True

        See Also
        ========

        is_odd
        """
        return not self.is_odd

    @property
    def is_odd(self):
        """
        Checks if a permutation is odd.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.is_odd
        False
        >>> p = Permutation([3, 2, 0, 1])
        >>> p.is_odd
        True

        See Also
        ========

        is_even
        """
        return bool(self.parity() % 2)

    @property
    def is_Singleton(self):
        """
        Checks to see if the permutation contains only one number and is
        thus the only possible permutation of this set of numbers

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0]).is_Singleton
        True
        >>> Permutation([0, 1]).is_Singleton
        False

        See Also
        ========

        is_Empty
        """
        return self.size == 1

    @property
    def is_Empty(self):
        """
        Checks to see if the permutation is a set with zero elements

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([]).is_Empty
        True
        >>> Permutation([0]).is_Empty
        False

        See Also
        ========

        is_Singleton
        """
        return self.size == 0

    @property
    def is_identity(self):
        return self.is_Identity

    @property
    def is_Identity(self):
        """
        Returns True if the Permutation is an identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([])
        >>> p.is_Identity
        True
        >>> p = Permutation([[0], [1], [2]])
        >>> p.is_Identity
        True
        >>> p = Permutation([0, 1, 2])
        >>> p.is_Identity
        True
        >>> p = Permutation([0, 2, 1])
        >>> p.is_Identity
        False

        See Also
        ========

        order
        """
        af = self.array_form
        return not af or all(i == af[i] for i in range(self.size))

    def ascents(self):
        """
        Returns the positions of ascents in a permutation, ie, the location
        where p[i] < p[i+1]

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 0, 1, 3, 2])
        >>> p.ascents()
        [1, 2]

        See Also
        ========

        descents, inversions, min, max
        """
        a = self.array_form
        pos = [i for i in range(len(a) - 1) if a[i] < a[i + 1]]
        return pos

    def descents(self):
        """
        Returns the positions of descents in a permutation, ie, the location
        where p[i] > p[i+1]

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 0, 1, 3, 2])
        >>> p.descents()
        [0, 3]

        See Also
        ========

        ascents, inversions, min, max
        """
        a = self.array_form
        pos = [i for i in range(len(a) - 1) if a[i] > a[i + 1]]
        return pos

    def max(self) -> int:
        """
        The maximum element moved by the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([1, 0, 2, 3, 4])
        >>> p.max()
        1

        See Also
        ========

        min, descents, ascents, inversions
        """
        a = self.array_form
        if not a:
            return 0
        return max(_a for i, _a in enumerate(a) if _a != i)

    def min(self) -> int:
        """
        The minimum element moved by the permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 4, 3, 2])
        >>> p.min()
        2

        See Also
        ========

        max, descents, ascents, inversions
        """
        a = self.array_form
        if not a:
            return 0
        return min(_a for i, _a in enumerate(a) if _a != i)

    def inversions(self):
        """
        Computes the number of inversions of a permutation.

        Explanation
        ===========

        An inversion is where i > j but p[i] < p[j].

        For small length of p, it iterates over all i and j
        values and calculates the number of inversions.
        For large length of p, it uses a variation of merge
        sort to calculate the number of inversions.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3, 4, 5])
        >>> p.inversions()
        0
        >>> Permutation([3, 2, 1, 0]).inversions()
        6

        See Also
        ========

        descents, ascents, min, max

        References
        ==========

        .. [1] https://www.cp.eng.chula.ac.th/~prabhas//teaching/algo/algo2008/count-inv.htm

        """
        inversions = 0
        a = self.array_form
        n = len(a)
        if n < 130:
            for i in range(n - 1):
                b = a[i]
                for c in a[i + 1:]:
                    if b > c:
                        inversions += 1
        else:
            k = 1
            right = 0
            arr = a[:]
            temp = a[:]
            while k < n:
                i = 0
                while i + k < n:
                    right = i + k * 2 - 1
                    if right >= n:
                        right = n - 1
                    inversions += _merge(arr, temp, i, i + k, right)
                    i = i + k * 2
                k = k * 2
        return inversions

    def commutator(self, x):
        """Return the commutator of ``self`` and ``x``: ``~x*~self*x*self``

        If f and g are part of a group, G, then the commutator of f and g
        is the group identity iff f and g commute, i.e. fg == gf.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([0, 2, 3, 1])
        >>> x = Permutation([2, 0, 3, 1])
        >>> c = p.commutator(x); c
        Permutation([2, 1, 3, 0])
        >>> c == ~x*~p*x*p
        True

        >>> I = Permutation(3)
        >>> p = [I + i for i in range(6)]
        >>> for i in range(len(p)):
        ...     for j in range(len(p)):
        ...         c = p[i].commutator(p[j])
        ...         if p[i]*p[j] == p[j]*p[i]:
        ...             assert c == I
        ...         else:
        ...             assert c != I
        ...

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Commutator
        """

        a = self.array_form
        b = x.array_form
        n = len(a)
        if len(b) != n:
            raise ValueError("The permutations must be of equal size.")
        inva = [None]*n
        for i in range(n):
            inva[a[i]] = i
        invb = [None]*n
        for i in range(n):
            invb[b[i]] = i
        return self._af_new([a[b[inva[i]]] for i in invb])

    def signature(self):
        """
        Gives the signature of the permutation needed to place the
        elements of the permutation in canonical order.

        The signature is calculated as (-1)^<number of inversions>

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2])
        >>> p.inversions()
        0
        >>> p.signature()
        1
        >>> q = Permutation([0,2,1])
        >>> q.inversions()
        1
        >>> q.signature()
        -1

        See Also
        ========

        inversions
        """
        if self.is_even:
            return 1
        return -1

    def order(self):
        """
        Computes the order of a permutation.

        When the permutation is raised to the power of its
        order it equals the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([3, 1, 5, 2, 4, 0])
        >>> p.order()
        4
        >>> (p**(p.order()))
        Permutation([], size=6)

        See Also
        ========

        identity, cardinality, length, rank, size
        """

        return reduce(lcm, [len(cycle) for cycle in self.cyclic_form], 1)

    def length(self):
        """
        Returns the number of integers moved by a permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 3, 2, 1]).length()
        2
        >>> Permutation([[0, 1], [2, 3]]).length()
        4

        See Also
        ========

        min, max, support, cardinality, order, rank, size
        """

        return len(self.support())

    @property
    def cycle_structure(self):
        """Return the cycle structure of the permutation as a dictionary
        indicating the multiplicity of each cycle length.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation(3).cycle_structure
        {1: 4}
        >>> Permutation(0, 4, 3)(1, 2)(5, 6).cycle_structure
        {2: 2, 3: 1}
        """
        if self._cycle_structure:
            rv = self._cycle_structure
        else:
            rv = defaultdict(int)
            singletons = self.size
            for c in self.cyclic_form:
                rv[len(c)] += 1
                singletons -= len(c)
            if singletons:
                rv[1] = singletons
            self._cycle_structure = rv
        return dict(rv)  # make a copy

    @property
    def cycles(self):
        """
        Returns the number of cycles contained in the permutation
        (including singletons).

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([0, 1, 2]).cycles
        3
        >>> Permutation([0, 1, 2]).full_cyclic_form
        [[0], [1], [2]]
        >>> Permutation(0, 1)(2, 3).cycles
        2

        See Also
        ========
        sympy.functions.combinatorial.numbers.stirling
        """
        return len(self.full_cyclic_form)

    def index(self):
        """
        Returns the index of a permutation.

        The index of a permutation is the sum of all subscripts j such
        that p[j] is greater than p[j+1].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([3, 0, 2, 1, 4])
        >>> p.index()
        2
        """
        a = self.array_form

        return sum(j for j in range(len(a) - 1) if a[j] > a[j + 1])

    def runs(self):
        """
        Returns the runs of a permutation.

        An ascending sequence in a permutation is called a run [5].


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 5, 7, 3, 6, 0, 1, 4, 8])
        >>> p.runs()
        [[2, 5, 7], [3, 6], [0, 1, 4, 8]]
        >>> q = Permutation([1,3,2,0])
        >>> q.runs()
        [[1, 3], [2], [0]]
        """
        return runs(self.array_form)

    def inversion_vector(self):
        """Return the inversion vector of the permutation.

        The inversion vector consists of elements whose value
        indicates the number of elements in the permutation
        that are lesser than it and lie on its right hand side.

        The inversion vector is the same as the Lehmer encoding of a
        permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 8, 0, 7, 1, 5, 3, 6, 2])
        >>> p.inversion_vector()
        [4, 7, 0, 5, 0, 2, 1, 1]
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.inversion_vector()
        [3, 2, 1]

        The inversion vector increases lexicographically with the rank
        of the permutation, the -ith element cycling through 0..i.

        >>> p = Permutation(2)
        >>> while p:
        ...     print('%s %s %s' % (p, p.inversion_vector(), p.rank()))
        ...     p = p.next_lex()
        (2) [0, 0] 0
        (1 2) [0, 1] 1
        (2)(0 1) [1, 0] 2
        (0 1 2) [1, 1] 3
        (0 2 1) [2, 0] 4
        (0 2) [2, 1] 5

        See Also
        ========

        from_inversion_vector
        """
        self_array_form = self.array_form
        n = len(self_array_form)
        inversion_vector = [0] * (n - 1)

        for i in range(n - 1):
            val = 0
            for j in range(i + 1, n):
                if self_array_form[j] < self_array_form[i]:
                    val += 1
            inversion_vector[i] = val
        return inversion_vector

    def rank_trotterjohnson(self):
        """
        Returns the Trotter Johnson rank, which we get from the minimal
        change algorithm. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_trotterjohnson()
        0
        >>> p = Permutation([0, 2, 1, 3])
        >>> p.rank_trotterjohnson()
        7

        See Also
        ========

        unrank_trotterjohnson, next_trotterjohnson
        """
        if self.array_form == [] or self.is_Identity:
            return 0
        if self.array_form == [1, 0]:
            return 1
        perm = self.array_form
        n = self.size
        rank = 0
        for j in range(1, n):
            k = 1
            i = 0
            while perm[i] != j:
                if perm[i] < j:
                    k += 1
                i += 1
            j1 = j + 1
            if rank % 2 == 0:
                rank = j1*rank + j1 - k
            else:
                rank = j1*rank + k - 1
        return rank

    @classmethod
    def unrank_trotterjohnson(cls, size, rank):
        """
        Trotter Johnson permutation unranking. See [4] section 2.4.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.unrank_trotterjohnson(5, 10)
        Permutation([0, 3, 1, 2, 4])

        See Also
        ========

        rank_trotterjohnson, next_trotterjohnson
        """
        perm = [0]*size
        r2 = 0
        n = ifac(size)
        pj = 1
        for j in range(2, size + 1):
            pj *= j
            r1 = (rank * pj) // n
            k = r1 - j*r2
            if r2 % 2 == 0:
                for i in range(j - 1, j - k - 1, -1):
                    perm[i] = perm[i - 1]
                perm[j - k - 1] = j - 1
            else:
                for i in range(j - 1, k, -1):
                    perm[i] = perm[i - 1]
                perm[k] = j - 1
            r2 = r1
        return cls._af_new(perm)

    def next_trotterjohnson(self):
        """
        Returns the next permutation in Trotter-Johnson order.
        If self is the last permutation it returns None.
        See [4] section 2.4. If it is desired to generate all such
        permutations, they can be generated in order more quickly
        with the ``generate_bell`` function.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([3, 0, 2, 1])
        >>> p.rank_trotterjohnson()
        4
        >>> p = p.next_trotterjohnson(); p
        Permutation([0, 3, 2, 1])
        >>> p.rank_trotterjohnson()
        5

        See Also
        ========

        rank_trotterjohnson, unrank_trotterjohnson, sympy.utilities.iterables.generate_bell
        """
        pi = self.array_form[:]
        n = len(pi)
        st = 0
        rho = pi[:]
        done = False
        m = n-1
        while m > 0 and not done:
            d = rho.index(m)
            for i in range(d, m):
                rho[i] = rho[i + 1]
            par = _af_parity(rho[:m])
            if par == 1:
                if d == m:
                    m -= 1
                else:
                    pi[st + d], pi[st + d + 1] = pi[st + d + 1], pi[st + d]
                    done = True
            else:
                if d == 0:
                    m -= 1
                    st += 1
                else:
                    pi[st + d], pi[st + d - 1] = pi[st + d - 1], pi[st + d]
                    done = True
        if m == 0:
            return None
        return self._af_new(pi)

    def get_precedence_matrix(self):
        """
        Gets the precedence matrix. This is used for computing the
        distance between two permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation.josephus(3, 6, 1)
        >>> p
        Permutation([2, 5, 3, 1, 4, 0])
        >>> p.get_precedence_matrix()
        Matrix([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0]])

        See Also
        ========

        get_precedence_distance, get_adjacency_matrix, get_adjacency_distance
        """
        m = zeros(self.size)
        perm = self.array_form
        for i in range(m.rows):
            for j in range(i + 1, m.cols):
                m[perm[i], perm[j]] = 1
        return m

    def get_precedence_distance(self, other):
        """
        Computes the precedence distance between two permutations.

        Explanation
        ===========

        Suppose p and p' represent n jobs. The precedence metric
        counts the number of times a job j is preceded by job i
        in both p and p'. This metric is commutative.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([2, 0, 4, 3, 1])
        >>> q = Permutation([3, 1, 2, 4, 0])
        >>> p.get_precedence_distance(q)
        7
        >>> q.get_precedence_distance(p)
        7

        See Also
        ========

        get_precedence_matrix, get_adjacency_matrix, get_adjacency_distance
        """
        if self.size != other.size:
            raise ValueError("The permutations must be of equal size.")
        self_prec_mat = self.get_precedence_matrix()
        other_prec_mat = other.get_precedence_matrix()
        n_prec = 0
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                if self_prec_mat[i, j] * other_prec_mat[i, j] == 1:
                    n_prec += 1
        d = self.size * (self.size - 1)//2 - n_prec
        return d

    def get_adjacency_matrix(self):
        """
        Computes the adjacency matrix of a permutation.

        Explanation
        ===========

        If job i is adjacent to job j in a permutation p
        then we set m[i, j] = 1 where m is the adjacency
        matrix of p.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation.josephus(3, 6, 1)
        >>> p.get_adjacency_matrix()
        Matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]])
        >>> q = Permutation([0, 1, 2, 3])
        >>> q.get_adjacency_matrix()
        Matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]])

        See Also
        ========

        get_precedence_matrix, get_precedence_distance, get_adjacency_distance
        """
        m = zeros(self.size)
        perm = self.array_form
        for i in range(self.size - 1):
            m[perm[i], perm[i + 1]] = 1
        return m

    def get_adjacency_distance(self, other):
        """
        Computes the adjacency distance between two permutations.

        Explanation
        ===========

        This metric counts the number of times a pair i,j of jobs is
        adjacent in both p and p'. If n_adj is this quantity then
        the adjacency distance is n - n_adj - 1 [1]

        [1] Reeves, Colin R. Landscapes, Operators and Heuristic search, Annals
        of Operational Research, 86, pp 473-490. (1999)


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 3, 1, 2, 4])
        >>> q = Permutation.josephus(4, 5, 2)
        >>> p.get_adjacency_distance(q)
        3
        >>> r = Permutation([0, 2, 1, 4, 3])
        >>> p.get_adjacency_distance(r)
        4

        See Also
        ========

        get_precedence_matrix, get_precedence_distance, get_adjacency_matrix
        """
        if self.size != other.size:
            raise ValueError("The permutations must be of the same size.")
        self_adj_mat = self.get_adjacency_matrix()
        other_adj_mat = other.get_adjacency_matrix()
        n_adj = 0
        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue
                if self_adj_mat[i, j] * other_adj_mat[i, j] == 1:
                    n_adj += 1
        d = self.size - n_adj - 1
        return d

    def get_positional_distance(self, other):
        """
        Computes the positional distance between two permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 3, 1, 2, 4])
        >>> q = Permutation.josephus(4, 5, 2)
        >>> r = Permutation([3, 1, 4, 0, 2])
        >>> p.get_positional_distance(q)
        12
        >>> p.get_positional_distance(r)
        12

        See Also
        ========

        get_precedence_distance, get_adjacency_distance
        """
        a = self.array_form
        b = other.array_form
        if len(a) != len(b):
            raise ValueError("The permutations must be of the same size.")
        return sum(abs(a[i] - b[i]) for i in range(len(a)))

    @classmethod
    def josephus(cls, m, n, s=1):
        """Return as a permutation the shuffling of range(n) using the Josephus
        scheme in which every m-th item is selected until all have been chosen.
        The returned permutation has elements listed by the order in which they
        were selected.

        The parameter ``s`` stops the selection process when there are ``s``
        items remaining and these are selected by continuing the selection,
        counting by 1 rather than by ``m``.

        Consider selecting every 3rd item from 6 until only 2 remain::

            choices    chosen
            ========   ======
              012345
              01 345   2
              01 34    25
              01  4    253
              0   4    2531
              0        25314
                       253140

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation.josephus(3, 6, 2).array_form
        [2, 5, 3, 1, 4, 0]

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Flavius_Josephus
        .. [2] https://en.wikipedia.org/wiki/Josephus_problem
        .. [3] https://web.archive.org/web/20171008094331/http://www.wou.edu/~burtonl/josephus.html

        """
        from collections import deque
        m -= 1
        Q = deque(list(range(n)))
        perm = []
        while len(Q) > max(s, 1):
            for dp in range(m):
                Q.append(Q.popleft())
            perm.append(Q.popleft())
        perm.extend(list(Q))
        return cls(perm)

    @classmethod
    def from_inversion_vector(cls, inversion):
        """
        Calculates the permutation from the inversion vector.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.from_inversion_vector([3, 2, 1, 0, 0])
        Permutation([3, 2, 1, 0, 4, 5])

        """
        size = len(inversion)
        N = list(range(size + 1))
        perm = []
        try:
            for k in range(size):
                val = N[inversion[k]]
                perm.append(val)
                N.remove(val)
        except IndexError:
            raise ValueError("The inversion vector is not valid.")
        perm.extend(N)
        return cls._af_new(perm)

    @classmethod
    def random(cls, n):
        """
        Generates a random permutation of length ``n``.

        Uses the underlying Python pseudo-random number generator.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation.random(2) in (Permutation([1, 0]), Permutation([0, 1]))
        True

        """
        perm_array = list(range(n))
        random.shuffle(perm_array)
        return cls._af_new(perm_array)

    @classmethod
    def unrank_lex(cls, size, rank):
        """
        Lexicographic permutation unranking.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> a = Permutation.unrank_lex(5, 10)
        >>> a.rank()
        10
        >>> a
        Permutation([0, 2, 4, 1, 3])

        See Also
        ========

        rank, next_lex
        """
        perm_array = [0] * size
        psize = 1
        for i in range(size):
            new_psize = psize*(i + 1)
            d = (rank % new_psize) // psize
            rank -= d*psize
            perm_array[size - i - 1] = d
            for j in range(size - i, size):
                if perm_array[j] > d - 1:
                    perm_array[j] += 1
            psize = new_psize
        return cls._af_new(perm_array)

    def resize(self, n):
        """Resize the permutation to the new size ``n``.

        Parameters
        ==========

        n : int
            The new size of the permutation.

        Raises
        ======

        ValueError
            If the permutation cannot be resized to the given size.
            This may only happen when resized to a smaller size than
            the original.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation

        Increasing the size of a permutation:

        >>> p = Permutation(0, 1, 2)
        >>> p = p.resize(5)
        >>> p
        (4)(0 1 2)

        Decreasing the size of the permutation:

        >>> p = p.resize(4)
        >>> p
        (3)(0 1 2)

        If resizing to the specific size breaks the cycles:

        >>> p.resize(2)
        Traceback (most recent call last):
        ...
        ValueError: The permutation cannot be resized to 2 because the
        cycle (0, 1, 2) may break.
        """
        aform = self.array_form
        l = len(aform)
        if n > l:
            aform += list(range(l, n))
            return Permutation._af_new(aform)

        elif n < l:
            cyclic_form = self.full_cyclic_form
            new_cyclic_form = []
            for cycle in cyclic_form:
                cycle_min = min(cycle)
                cycle_max = max(cycle)
                if cycle_min <= n-1:
                    if cycle_max > n-1:
                        raise ValueError(
                            "The permutation cannot be resized to {} "
                            "because the cycle {} may break."
                            .format(n, tuple(cycle)))

                    new_cyclic_form.append(cycle)
            return Permutation(new_cyclic_form)

        return self

    # XXX Deprecated flag
    print_cyclic = None


def _merge(arr, temp, left, mid, right):
    """
    Merges two sorted arrays and calculates the inversion count.

    Helper function for calculating inversions. This method is
    for internal use only.
    """
    i = k = left
    j = mid
    inv_count = 0
    while i < mid and j <= right:
        if arr[i] < arr[j]:
            temp[k] = arr[i]
            k += 1
            i += 1
        else:
            temp[k] = arr[j]
            k += 1
            j += 1
            inv_count += (mid -i)
    while i < mid:
        temp[k] = arr[i]
        k += 1
        i += 1
    if j <= right:
        k += right - j + 1
        j += right - j + 1
        arr[left:k + 1] = temp[left:k + 1]
    else:
        arr[left:right + 1] = temp[left:right + 1]
    return inv_count

Perm = Permutation
_af_new = Perm._af_new


class AppliedPermutation(Expr):
    """A permutation applied to a symbolic variable.

    Parameters
    ==========

    perm : Permutation
    x : Expr

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.combinatorics import Permutation

    Creating a symbolic permutation function application:

    >>> x = Symbol('x')
    >>> p = Permutation(0, 1, 2)
    >>> p.apply(x)
    AppliedPermutation((0 1 2), x)
    >>> _.subs(x, 1)
    2
    """
    def __new__(cls, perm, x, evaluate=None):
        if evaluate is None:
            evaluate = global_parameters.evaluate

        perm = _sympify(perm)
        x = _sympify(x)

        if not isinstance(perm, Permutation):
            raise ValueError("{} must be a Permutation instance."
                .format(perm))

        if evaluate:
            if x.is_Integer:
                return perm.apply(x)

        obj = super().__new__(cls, perm, x)
        return obj


@dispatch(Permutation, Permutation)
def _eval_is_eq(lhs, rhs):
    if lhs._size != rhs._size:
        return None
    return lhs._array_form == rhs._array_form
<|endoftext|>class State:
    '''
    A representation of a state managed by a ``StateMachine``.

    Attributes:
        name (instance of FreeGroupElement or string) -- State name which is also assigned to the Machine.
        transisitons (OrderedDict) -- Represents all the transitions of the state object.
        state_type (string) -- Denotes the type (accept/start/dead) of the state.
        rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.
        state_machine (instance of StateMachine object) -- The finite state machine that the state belongs to.
    '''

    def __init__(self, name, state_machine, state_type=None, rh_rule=None):
        self.name = name
        self.transitions = {}
        self.state_machine = state_machine
        self.state_type = state_type[0]
        self.rh_rule = rh_rule

    def add_transition(self, letter, state):
        '''
        Add a transition from the current state to a new state.

        Keyword Arguments:
            letter -- The alphabet element the current state reads to make the state transition.
            state -- This will be an instance of the State object which represents a new state after in the transition after the alphabet is read.

        '''
        self.transitions[letter] = state

class StateMachine:
    '''
    Representation of a finite state machine the manages the states and the transitions of the automaton.

    Attributes:
        states (dictionary) -- Collection of all registered `State` objects.
        name (str) -- Name of the state machine.
    '''

    def __init__(self, name, automaton_alphabet):
        self.name = name
        self.automaton_alphabet = automaton_alphabet
        self.states = {} # Contains all the states in the machine.
        self.add_state('start', state_type='s')

    def add_state(self, state_name, state_type=None, rh_rule=None):
        '''
        Instantiate a state object and stores it in the 'states' dictionary.

        Arguments:
            state_name (instance of FreeGroupElement or string) -- name of the new states.
            state_type (string) -- Denotes the type (accept/start/dead) of the state added.
            rh_rule (instance of FreeGroupElement) -- right hand rule for dead state.

        '''
        new_state = State(state_name, self, state_type, rh_rule)
        self.states[state_name] = new_state

    def __repr__(self):
        return "%s" % (self.name)
<|endoftext|>from math import factorial as _factorial, log, prod
from itertools import chain, product


from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
    _af_rmul, _af_rmuln, _af_pow, Cycle)
from sympy.combinatorics.util import (_check_cycles_alt_sym,
    _distribute_gens_by_base, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs, _base_ordering, _strong_gens_from_distr,
    _strip, _strip_af)
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq

rmul = Permutation.rmul_with_af
_af_new = Permutation._af_new


class PermutationGroup(Basic):
    r"""The class defining a Permutation group.

    Explanation
    ===========

    ``PermutationGroup([p1, p2, ..., pn])`` returns the permutation group
    generated by the list of permutations. This group can be supplied
    to Polyhedron if one desires to decorate the elements to which the
    indices of the permutation refer.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> from sympy.combinatorics import Polyhedron

    The permutations corresponding to motion of the front, right and
    bottom face of a $2 \times 2$ Rubik's cube are defined:

    >>> F = Permutation(2, 19, 21, 8)(3, 17, 20, 10)(4, 6, 7, 5)
    >>> R = Permutation(1, 5, 21, 14)(3, 7, 23, 12)(8, 10, 11, 9)
    >>> D = Permutation(6, 18, 14, 10)(7, 19, 15, 11)(20, 22, 23, 21)

    These are passed as permutations to PermutationGroup:

    >>> G = PermutationGroup(F, R, D)
    >>> G.order()
    3674160

    The group can be supplied to a Polyhedron in order to track the
    objects being moved. An example involving the $2 \times 2$ Rubik's cube is
    given there, but here is a simple demonstration:

    >>> a = Permutation(2, 1)
    >>> b = Permutation(1, 0)
    >>> G = PermutationGroup(a, b)
    >>> P = Polyhedron(list('ABC'), pgroup=G)
    >>> P.corners
    (A, B, C)
    >>> P.rotate(0) # apply permutation 0
    >>> P.corners
    (A, C, B)
    >>> P.reset()
    >>> P.corners
    (A, B, C)

    Or one can make a permutation as a product of selected permutations
    and apply them to an iterable directly:

    >>> P10 = G.make_perm([0, 1])
    >>> P10('ABC')
    ['C', 'A', 'B']

    See Also
    ========

    sympy.combinatorics.polyhedron.Polyhedron,
    sympy.combinatorics.permutations.Permutation

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"

    .. [2] Seress, A.
           "Permutation Group Algorithms"

    .. [3] https://en.wikipedia.org/wiki/Schreier_vector

    .. [4] https://en.wikipedia.org/wiki/Nielsen_transformation#Product_replacement_algorithm

    .. [5] Frank Celler, Charles R.Leedham-Green, Scott H.Murray,
           Alice C.Niemeyer, and E.A.O'Brien. "Generating Random
           Elements of a Finite Group"

    .. [6] https://en.wikipedia.org/wiki/Block_%28permutation_group_theory%29

    .. [7] https://algorithmist.com/wiki/Union_find

    .. [8] https://en.wikipedia.org/wiki/Multiply_transitive_group#Multiply_transitive_groups

    .. [9] https://en.wikipedia.org/wiki/Center_%28group_theory%29

    .. [10] https://en.wikipedia.org/wiki/Centralizer_and_normalizer

    .. [11] https://groupprops.subwiki.org/wiki/Derived_subgroup

    .. [12] https://en.wikipedia.org/wiki/Nilpotent_group

    .. [13] https://www.math.colostate.edu/~hulpke/CGT/cgtnotes.pdf

    .. [14] https://docs.gap-system.org/doc/ref/manual.pdf

    """
    is_group = True

    def __new__(cls, *args, dups=True, **kwargs):
        """The default constructor. Accepts Cycle and Permutation forms.
        Removes duplicates unless ``dups`` keyword is ``False``.
        """
        if not args:
            args = [Permutation()]
        else:
            args = list(args[0] if is_sequence(args[0]) else args)
            if not args:
                args = [Permutation()]
        if any(isinstance(a, Cycle) for a in args):
            args = [Permutation(a) for a in args]
        if has_variety(a.size for a in args):
            degree = kwargs.pop('degree', None)
            if degree is None:
                degree = max(a.size for a in args)
            for i in range(len(args)):
                if args[i].size != degree:
                    args[i] = Permutation(args[i], size=degree)
        if dups:
            args = list(uniq([_af_new(list(a)) for a in args]))
        if len(args) > 1:
            args = [g for g in args if not g.is_identity]
        return Basic.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self._generators = list(self.args)
        self._order = None
        self._elements = []
        self._center = None
        self._is_abelian = None
        self._is_transitive = None
        self._is_sym = None
        self._is_alt = None
        self._is_primitive = None
        self._is_nilpotent = None
        self._is_solvable = None
        self._is_trivial = None
        self._transitivity_degree = None
        self._max_div = None
        self._is_perfect = None
        self._is_cyclic = None
        self._is_dihedral = None
        self._r = len(self._generators)
        self._degree = self._generators[0].size

        # these attributes are assigned after running schreier_sims
        self._base = []
        self._strong_gens = []
        self._strong_gens_slp = []
        self._basic_orbits = []
        self._transversals = []
        self._transversal_slp = []

        # these attributes are assigned after running _random_pr_init
        self._random_gens = []

        # finite presentation of the group as an instance of `FpGroup`
        self._fp_presentation = None

    def __getitem__(self, i):
        return self._generators[i]

    def __contains__(self, i):
        """Return ``True`` if *i* is contained in PermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = Permutation(1, 2, 3)
        >>> Permutation(3) in PermutationGroup(p)
        True

        """
        if not isinstance(i, Permutation):
            raise TypeError("A PermutationGroup contains only Permutations as "
                            "elements, not elements of type %s" % type(i))
        return self.contains(i)

    def __len__(self):
        return len(self._generators)

    def equals(self, other):
        """Return ``True`` if PermutationGroup generated by elements in the
        group are same i.e they represent the same PermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = Permutation(0, 1, 2, 3, 4, 5)
        >>> G = PermutationGroup([p, p**2])
        >>> H = PermutationGroup([p**2, p])
        >>> G.generators == H.generators
        False
        >>> G.equals(H)
        True

        """
        if not isinstance(other, PermutationGroup):
            return False

        set_self_gens = set(self.generators)
        set_other_gens = set(other.generators)

        # before reaching the general case there are also certain
        # optimisation and obvious cases requiring less or no actual
        # computation.
        if set_self_gens == set_other_gens:
            return True

        # in the most general case it will check that each generator of
        # one group belongs to the other PermutationGroup and vice-versa
        for gen1 in set_self_gens:
            if not other.contains(gen1):
                return False
        for gen2 in set_other_gens:
            if not self.contains(gen2):
                return False
        return True

    def __mul__(self, other):
        """
        Return the direct product of two permutation groups as a permutation
        group.

        Explanation
        ===========

        This implementation realizes the direct product by shifting the index
        set for the generators of the second group: so if we have ``G`` acting
        on ``n1`` points and ``H`` acting on ``n2`` points, ``G*H`` acts on
        ``n1 + n2`` points.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import CyclicGroup
        >>> G = CyclicGroup(5)
        >>> H = G*G
        >>> H
        PermutationGroup([
            (9)(0 1 2 3 4),
            (5 6 7 8 9)])
        >>> H.order()
        25

        """
        if isinstance(other, Permutation):
            return Coset(other, self, dir='+')
        gens1 = [perm._array_form for perm in self.generators]
        gens2 = [perm._array_form for perm in other.generators]
        n1 = self._degree
        n2 = other._degree
        start = list(range(n1))
        end = list(range(n1, n1 + n2))
        for i in range(len(gens2)):
            gens2[i] = [x + n1 for x in gens2[i]]
        gens2 = [start + gen for gen in gens2]
        gens1 = [gen + end for gen in gens1]
        together = gens1 + gens2
        gens = [_af_new(x) for x in together]
        return PermutationGroup(gens)

    def _random_pr_init(self, r, n, _random_prec_n=None):
        r"""Initialize random generators for the product replacement algorithm.

        Explanation
        ===========

        The implementation uses a modification of the original product
        replacement algorithm due to Leedham-Green, as described in [1],
        pp. 69-71; also, see [2], pp. 27-29 for a detailed theoretical
        analysis of the original product replacement algorithm, and [4].

        The product replacement algorithm is used for producing random,
        uniformly distributed elements of a group `G` with a set of generators
        `S`. For the initialization ``_random_pr_init``, a list ``R`` of
        `\max\{r, |S|\}` group generators is created as the attribute
        ``G._random_gens``, repeating elements of `S` if necessary, and the
        identity element of `G` is appended to ``R`` - we shall refer to this
        last element as the accumulator. Then the function ``random_pr()``
        is called ``n`` times, randomizing the list ``R`` while preserving
        the generation of `G` by ``R``. The function ``random_pr()`` itself
        takes two random elements ``g, h`` among all elements of ``R`` but
        the accumulator and replaces ``g`` with a randomly chosen element
        from `\{gh, g(~h), hg, (~h)g\}`. Then the accumulator is multiplied
        by whatever ``g`` was replaced by. The new value of the accumulator is
        then returned by ``random_pr()``.

        The elements returned will eventually (for ``n`` large enough) become
        uniformly distributed across `G` ([5]). For practical purposes however,
        the values ``n = 50, r = 11`` are suggested in [1].

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: it changes the attribute
        self._random_gens

        See Also
        ========

        random_pr

        """
        deg = self.degree
        random_gens = [x._array_form for x in self.generators]
        k = len(random_gens)
        if k < r:
            for i in range(k, r):
                random_gens.append(random_gens[i - k])
        acc = list(range(deg))
        random_gens.append(acc)
        self._random_gens = random_gens

        # handle randomized input for testing purposes
        if _random_prec_n is None:
            for i in range(n):
                self.random_pr()
        else:
            for i in range(n):
                self.random_pr(_random_prec=_random_prec_n[i])

    def _union_find_merge(self, first, second, ranks, parents, not_rep):
        """Merges two classes in a union-find data structure.

        Explanation
        ===========

        Used in the implementation of Atkinson's algorithm as suggested in [1],
        pp. 83-87. The class merging process uses union by rank as an
        optimization. ([7])

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: the list of class representatives,
        ``parents``, the list of class sizes, ``ranks``, and the list of
        elements that are not representatives, ``not_rep``, are changed due to
        class merging.

        See Also
        ========

        minimal_block, _union_find_rep

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of computational group theory"

        .. [7] https://algorithmist.com/wiki/Union_find

        """
        rep_first = self._union_find_rep(first, parents)
        rep_second = self._union_find_rep(second, parents)
        if rep_first != rep_second:
            # union by rank
            if ranks[rep_first] >= ranks[rep_second]:
                new_1, new_2 = rep_first, rep_second
            else:
                new_1, new_2 = rep_second, rep_first
            total_rank = ranks[new_1] + ranks[new_2]
            if total_rank > self.max_div:
                return -1
            parents[new_2] = new_1
            ranks[new_1] = total_rank
            not_rep.append(new_2)
            return 1
        return 0

    def _union_find_rep(self, num, parents):
        """Find representative of a class in a union-find data structure.

        Explanation
        ===========

        Used in the implementation of Atkinson's algorithm as suggested in [1],
        pp. 83-87. After the representative of the class to which ``num``
        belongs is found, path compression is performed as an optimization
        ([7]).

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: the list of class representatives,
        ``parents``, is altered due to path compression.

        See Also
        ========

        minimal_block, _union_find_merge

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of computational group theory"

        .. [7] https://algorithmist.com/wiki/Union_find

        """
        rep, parent = num, parents[num]
        while parent != rep:
            rep = parent
            parent = parents[rep]
        # path compression
        temp, parent = num, parents[num]
        while parent != rep:
            parents[temp] = rep
            temp = parent
            parent = parents[temp]
        return rep

    @property
    def base(self):
        r"""Return a base from the Schreier-Sims algorithm.

        Explanation
        ===========

        For a permutation group `G`, a base is a sequence of points
        `B = (b_1, b_2, \dots, b_k)` such that no element of `G` apart
        from the identity fixes all the points in `B`. The concepts of
        a base and strong generating set and their applications are
        discussed in depth in [1], pp. 87-89 and [2], pp. 55-57.

        An alternative way to think of `B` is that it gives the
        indices of the stabilizer cosets that contain more than the
        identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation(0, 1, 3)(2, 4)])
        >>> G.base
        [0, 2]

        See Also
        ========

        strong_gens, basic_transversals, basic_orbits, basic_stabilizers

        """
        if self._base == []:
            self.schreier_sims()
        return self._base

    def baseswap(self, base, strong_gens, pos, randomized=False,
                 transversals=None, basic_orbits=None, strong_gens_distr=None):
        r"""Swap two consecutive base points in base and strong generating set.

        Explanation
        ===========

        If a base for a group `G` is given by `(b_1, b_2, \dots, b_k)`, this
        function returns a base `(b_1, b_2, \dots, b_{i+1}, b_i, \dots, b_k)`,
        where `i` is given by ``pos``, and a strong generating set relative
        to that base. The original base and strong generating set are not
        modified.

        The randomized version (default) is of Las Vegas type.

        Parameters
        ==========

        base, strong_gens
            The base and strong generating set.
        pos
            The position at which swapping is performed.
        randomized
            A switch between randomized and deterministic version.
        transversals
            The transversals for the basic orbits, if known.
        basic_orbits
            The basic orbits, if known.
        strong_gens_distr
            The strong generators distributed by basic stabilizers, if known.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the new base, and ``strong_gens`` is a generating set
            relative to it.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> S = SymmetricGroup(4)
        >>> S.schreier_sims()
        >>> S.base
        [0, 1, 2]
        >>> base, gens = S.baseswap(S.base, S.strong_gens, 1, randomized=False)
        >>> base, gens
        ([0, 2, 1],
        [(0 1 2 3), (3)(0 1), (1 3 2),
         (2 3), (1 3)])

        check that base, gens is a BSGS

        >>> S1 = PermutationGroup(gens)
        >>> _verify_bsgs(S1, base, gens)
        True

        See Also
        ========

        schreier_sims

        Notes
        =====

        The deterministic version of the algorithm is discussed in
        [1], pp. 102-103; the randomized version is discussed in [1], p.103, and
        [2], p.98. It is of Las Vegas type.
        Notice that [1] contains a mistake in the pseudocode and
        discussion of BASESWAP: on line 3 of the pseudocode,
        `|\beta_{i+1}^{\left\langle T\right\rangle}|` should be replaced by
        `|\beta_{i}^{\left\langle T\right\rangle}|`, and the same for the
        discussion of the algorithm.

        """
        # construct the basic orbits, generators for the stabilizer chain
        # and transversal elements from whatever was provided
        transversals, basic_orbits, strong_gens_distr = \
            _handle_precomputed_bsgs(base, strong_gens, transversals,
                                 basic_orbits, strong_gens_distr)
        base_len = len(base)
        degree = self.degree
        # size of orbit of base[pos] under the stabilizer we seek to insert
        # in the stabilizer chain at position pos + 1
        size = len(basic_orbits[pos])*len(basic_orbits[pos + 1]) \
            //len(_orbit(degree, strong_gens_distr[pos], base[pos + 1]))
        # initialize the wanted stabilizer by a subgroup
        if pos + 2 > base_len - 1:
            T = []
        else:
            T = strong_gens_distr[pos + 2][:]
        # randomized version
        if randomized is True:
            stab_pos = PermutationGroup(strong_gens_distr[pos])
            schreier_vector = stab_pos.schreier_vector(base[pos + 1])
            # add random elements of the stabilizer until they generate it
            while len(_orbit(degree, T, base[pos])) != size:
                new = stab_pos.random_stab(base[pos + 1],
                                           schreier_vector=schreier_vector)
                T.append(new)
        # deterministic version
        else:
            Gamma = set(basic_orbits[pos])
            Gamma.remove(base[pos])
            if base[pos + 1] in Gamma:
                Gamma.remove(base[pos + 1])
            # add elements of the stabilizer until they generate it by
            # ruling out member of the basic orbit of base[pos] along the way
            while len(_orbit(degree, T, base[pos])) != size:
                gamma = next(iter(Gamma))
                x = transversals[pos][gamma]
                temp = x._array_form.index(base[pos + 1]) # (~x)(base[pos + 1])
                if temp not in basic_orbits[pos + 1]:
                    Gamma = Gamma - _orbit(degree, T, gamma)
                else:
                    y = transversals[pos + 1][temp]
                    el = rmul(x, y)
                    if el(base[pos]) not in _orbit(degree, T, base[pos]):
                        T.append(el)
                        Gamma = Gamma - _orbit(degree, T, base[pos])
        # build the new base and strong generating set
        strong_gens_new_distr = strong_gens_distr[:]
        strong_gens_new_distr[pos + 1] = T
        base_new = base[:]
        base_new[pos], base_new[pos + 1] = base_new[pos + 1], base_new[pos]
        strong_gens_new = _strong_gens_from_distr(strong_gens_new_distr)
        for gen in T:
            if gen not in strong_gens_new:
                strong_gens_new.append(gen)
        return base_new, strong_gens_new

    @property
    def basic_orbits(self):
        r"""
        Return the basic orbits relative to a base and strong generating set.

        Explanation
        ===========

        If `(b_1, b_2, \dots, b_k)` is a base for a group `G`, and
        `G^{(i)} = G_{b_1, b_2, \dots, b_{i-1}}` is the ``i``-th basic stabilizer
        (so that `G^{(1)} = G`), the ``i``-th basic orbit relative to this base
        is the orbit of `b_i` under `G^{(i)}`. See [1], pp. 87-89 for more
        information.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(4)
        >>> S.basic_orbits
        [[0, 1, 2, 3], [1, 2, 3], [2, 3]]

        See Also
        ========

        base, strong_gens, basic_transversals, basic_stabilizers

        """
        if self._basic_orbits == []:
            self.schreier_sims()
        return self._basic_orbits

    @property
    def basic_stabilizers(self):
        r"""
        Return a chain of stabilizers relative to a base and strong generating
        set.

        Explanation
        ===========

        The ``i``-th basic stabilizer `G^{(i)}` relative to a base
        `(b_1, b_2, \dots, b_k)` is `G_{b_1, b_2, \dots, b_{i-1}}`. For more
        information, see [1], pp. 87-89.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> A = AlternatingGroup(4)
        >>> A.schreier_sims()
        >>> A.base
        [0, 1]
        >>> for g in A.basic_stabilizers:
        ...     print(g)
        ...
        PermutationGroup([
            (3)(0 1 2),
            (1 2 3)])
        PermutationGroup([
            (1 2 3)])

        See Also
        ========

        base, strong_gens, basic_orbits, basic_transversals

        """

        if self._transversals == []:
            self.schreier_sims()
        strong_gens = self._strong_gens
        base = self._base
        if not base: # e.g. if self is trivial
            return []
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_stabilizers = []
        for gens in strong_gens_distr:
            basic_stabilizers.append(PermutationGroup(gens))
        return basic_stabilizers

    @property
    def basic_transversals(self):
        """
        Return basic transversals relative to a base and strong generating set.

        Explanation
        ===========

        The basic transversals are transversals of the basic orbits. They
        are provided as a list of dictionaries, each dictionary having
        keys - the elements of one of the basic orbits, and values - the
        corresponding transversal elements. See [1], pp. 87-89 for more
        information.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> A = AlternatingGroup(4)
        >>> A.basic_transversals
        [{0: (3), 1: (3)(0 1 2), 2: (3)(0 2 1), 3: (0 3 1)}, {1: (3), 2: (1 2 3), 3: (1 3 2)}]

        See Also
        ========

        strong_gens, base, basic_orbits, basic_stabilizers

        """

        if self._transversals == []:
            self.schreier_sims()
        return self._transversals

    def composition_series(self):
        r"""
        Return the composition series for a group as a list
        of permutation groups.

        Explanation
        ===========

        The composition series for a group `G` is defined as a
        subnormal series `G = H_0 > H_1 > H_2 \ldots` A composition
        series is a subnormal series such that each factor group
        `H(i+1) / H(i)` is simple.
        A subnormal series is a composition series only if it is of
        maximum length.

        The algorithm works as follows:
        Starting with the derived series the idea is to fill
        the gap between `G = der[i]` and `H = der[i+1]` for each
        `i` independently. Since, all subgroups of the abelian group
        `G/H` are normal so, first step is to take the generators
        `g` of `G` and add them to generators of `H` one by one.

        The factor groups formed are not simple in general. Each
        group is obtained from the previous one by adding one
        generator `g`, if the previous group is denoted by `H`
        then the next group `K` is generated by `g` and `H`.
        The factor group `K/H` is cyclic and it's order is
        `K.order()//G.order()`. The series is then extended between
        `K` and `H` by groups generated by powers of `g` and `H`.
        The series formed is then prepended to the already existing
        series.

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import CyclicGroup
        >>> S = SymmetricGroup(12)
        >>> G = S.sylow_subgroup(2)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        >>> G = S.sylow_subgroup(3)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [243, 81, 27, 9, 3, 1]
        >>> G = CyclicGroup(12)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [12, 6, 3, 1]

        """
        der = self.derived_series()
        if not all(g.is_identity for g in der[-1].generators):
            raise NotImplementedError('Group should be solvable')
        series = []

        for i in range(len(der)-1):
            H = der[i+1]
            up_seg = []
            for g in der[i].generators:
                K = PermutationGroup([g] + H.generators)
                order = K.order() // H.order()
                down_seg = []
                for p, e in factorint(order).items():
                    for _ in range(e):
                        down_seg.append(PermutationGroup([g] + H.generators))
                        g = g**p
                up_seg = down_seg + up_seg
                H = K
            up_seg[0] = der[i]
            series.extend(up_seg)
        series.append(der[-1])
        return series

    def coset_transversal(self, H):
        """Return a transversal of the right cosets of self by its subgroup H
        using the second method described in [1], Subsection 4.6.7

        """

        if not H.is_subgroup(self):
            raise ValueError("The argument must be a subgroup")

        if H.order() == 1:
            return self.elements

        self._schreier_sims(base=H.base) # make G.base an extension of H.base

        base = self.base
        base_ordering = _base_ordering(base, self.degree)
        identity = Permutation(self.degree - 1)

        transversals = self.basic_transversals[:]
        # transversals is a list of dictionaries. Get rid of the keys
        # so that it is a list of lists and sort each list in
        # the increasing order of base[l]^x
        for l, t in enumerate(transversals):
            transversals[l] = sorted(t.values(),
                                key = lambda x: base_ordering[base[l]^x])

        orbits = H.basic_orbits
        h_stabs = H.basic_stabilizers
        g_stabs = self.basic_stabilizers

        indices = [x.order()//y.order() for x, y in zip(g_stabs, h_stabs)]

        # T^(l) should be a right transversal of H^(l) in G^(l) for
        # 1<=l<=len(base). While H^(l) is the trivial group, T^(l)
        # contains all the elements of G^(l) so we might just as well
        # start with l = len(h_stabs)-1
        if len(g_stabs) > len(h_stabs):
            T = g_stabs[len(h_stabs)].elements
        else:
            T = [identity]
        l = len(h_stabs)-1
        t_len = len(T)
        while l > -1:
            T_next = []
            for u in transversals[l]:
                if u == identity:
                    continue
                b = base_ordering[base[l]^u]
                for t in T:
                    p = t*u
                    if all(base_ordering[h^p] >= b for h in orbits[l]):
                        T_next.append(p)
                    if t_len + len(T_next) == indices[l]:
                        break
                if t_len + len(T_next) == indices[l]:
                    break
            T += T_next
            t_len += len(T_next)
            l -= 1
        T.remove(identity)
        T = [identity] + T
        return T

    def _coset_representative(self, g, H):
        """Return the representative of Hg from the transversal that
        would be computed by ``self.coset_transversal(H)``.

        """
        if H.order() == 1:
            return g
        # The base of self must be an extension of H.base.
        if not(self.base[:len(H.base)] == H.base):
            self._schreier_sims(base=H.base)
        orbits = H.basic_orbits[:]
        h_transversals = [list(_.values()) for _ in H.basic_transversals]
        transversals = [list(_.values()) for _ in self.basic_transversals]
        base = self.base
        base_ordering = _base_ordering(base, self.degree)
        def step(l, x):
            gamma = min(orbits[l], key = lambda y: base_ordering[y^x])
            i = [base[l]^h for h in h_transversals[l]].index(gamma)
            x = h_transversals[l][i]*x
            if l < len(orbits)-1:
                for u in transversals[l]:
                    if base[l]^u == base[l]^x:
                        break
                x = step(l+1, x*u**-1)*u
            return x
        return step(0, g)

    def coset_table(self, H):
        """Return the standardised (right) coset table of self in H as
        a list of lists.
        """
        # Maybe this should be made to return an instance of CosetTable
        # from fp_groups.py but the class would need to be changed first
        # to be compatible with PermutationGroups

        if not H.is_subgroup(self):
            raise ValueError("The argument must be a subgroup")
        T = self.coset_transversal(H)
        n = len(T)

        A = list(chain.from_iterable((gen, gen**-1)
                    for gen in self.generators))

        table = []
        for i in range(n):
            row = [self._coset_representative(T[i]*x, H) for x in A]
            row = [T.index(r) for r in row]
            table.append(row)

        # standardize (this is the same as the algorithm used in coset_table)
        # If CosetTable is made compatible with PermutationGroups, this
        # should be replaced by table.standardize()
        A = range(len(A))
        gamma = 1
        for alpha, a in product(range(n), A):
            beta = table[alpha][a]
            if beta >= gamma:
                if beta > gamma:
                    for x in A:
                        z = table[gamma][x]
                        table[gamma][x] = table[beta][x]
                        table[beta][x] = z
                        for i in range(n):
                            if table[i][x] == beta:
                                table[i][x] = gamma
                            elif table[i][x] == gamma:
                                table[i][x] = beta
                gamma += 1
            if gamma >= n-1:
                return table

    def center(self):
        r"""
        Return the center of a permutation group.

        Explanation
        ===========

        The center for a group `G` is defined as
        `Z(G) = \{z\in G | \forall g\in G, zg = gz \}`,
        the set of elements of `G` that commute with all elements of `G`.
        It is equal to the centralizer of `G` inside `G`, and is naturally a
        subgroup of `G` ([9]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(4)
        >>> G = D.center()
        >>> G.order()
        2

        See Also
        ========

        centralizer

        Notes
        =====

        This is a naive implementation that is a straightforward application
        of ``.centralizer()``

        """
        if not self._center:
            self._center = self.centralizer(self)
        return self._center

    def centralizer(self, other):
        r"""
        Return the centralizer of a group/set/element.

        Explanation
        ===========

        The centralizer of a set of permutations ``S`` inside
        a group ``G`` is the set of elements of ``G`` that commute with all
        elements of ``S``::

            `C_G(S) = \{ g \in G | gs = sg \forall s \in S\}` ([10])

        Usually, ``S`` is a subset of ``G``, but if ``G`` is a proper subgroup of
        the full symmetric group, we allow for ``S`` to have elements outside
        ``G``.

        It is naturally a subgroup of ``G``; the centralizer of a permutation
        group is equal to the centralizer of any set of generators for that
        group, since any element commuting with the generators commutes with
        any product of the  generators.

        Parameters
        ==========

        other
            a permutation group/list of permutations/single permutation

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> S = SymmetricGroup(6)
        >>> C = CyclicGroup(6)
        >>> H = S.centralizer(C)
        >>> H.is_subgroup(C)
        True

        See Also
        ========

        subgroup_search

        Notes
        =====

        The implementation is an application of ``.subgroup_search()`` with
        tests using a specific base for the group ``G``.

        """
        if hasattr(other, 'generators'):
            if other.is_trivial or self.is_trivial:
                return self
            degree = self.degree
            identity = _af_new(list(range(degree)))
            orbits = other.orbits()
            num_orbits = len(orbits)
            orbits.sort(key=lambda x: -len(x))
            long_base = []
            orbit_reps = [None]*num_orbits
            orbit_reps_indices = [None]*num_orbits
            orbit_descr = [None]*degree
            for i in range(num_orbits):
                orbit = list(orbits[i])
                orbit_reps[i] = orbit[0]
                orbit_reps_indices[i] = len(long_base)
                for point in orbit:
                    orbit_descr[point] = i
                long_base = long_base + orbit
            base, strong_gens = self.schreier_sims_incremental(base=long_base)
            strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
            i = 0
            for i in range(len(base)):
                if strong_gens_distr[i] == [identity]:
                    break
            base = base[:i]
            base_len = i
            for j in range(num_orbits):
                if base[base_len - 1] in orbits[j]:
                    break
            rel_orbits = orbits[: j + 1]
            num_rel_orbits = len(rel_orbits)
            transversals = [None]*num_rel_orbits
            for j in range(num_rel_orbits):
                rep = orbit_reps[j]
                transversals[j] = dict(
                    other.orbit_transversal(rep, pairs=True))
            trivial_test = lambda x: True
            tests = [None]*base_len
            for l in range(base_len):
                if base[l] in orbit_reps:
                    tests[l] = trivial_test
                else:
                    def test(computed_words, l=l):
                        g = computed_words[l]
                        rep_orb_index = orbit_descr[base[l]]
                        rep = orbit_reps[rep_orb_index]
                        im = g._array_form[base[l]]
                        im_rep = g._array_form[rep]
                        tr_el = transversals[rep_orb_index][base[l]]
                        # using the definition of transversal,
                        # base[l]^g = rep^(tr_el*g);
                        # if g belongs to the centralizer, then
                        # base[l]^g = (rep^g)^tr_el
                        return im == tr_el._array_form[im_rep]
                    tests[l] = test

            def prop(g):
                return [rmul(g, gen) for gen in other.generators] == \
                       [rmul(gen, g) for gen in other.generators]
            return self.subgroup_search(prop, base=base,
                                        strong_gens=strong_gens, tests=tests)
        elif hasattr(other, '__getitem__'):
            gens = list(other)
            return self.centralizer(PermutationGroup(gens))
        elif hasattr(other, 'array_form'):
            return self.centralizer(PermutationGroup([other]))

    def commutator(self, G, H):
        """
        Return the commutator of two subgroups.

        Explanation
        ===========

        For a permutation group ``K`` and subgroups ``G``, ``H``, the
        commutator of ``G`` and ``H`` is defined as the group generated
        by all the commutators `[g, h] = hgh^{-1}g^{-1}` for ``g`` in ``G`` and
        ``h`` in ``H``. It is naturally a subgroup of ``K`` ([1], p.27).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> A = AlternatingGroup(5)
        >>> G = S.commutator(S, A)
        >>> G.is_subgroup(A)
        True

        See Also
        ========

        derived_subgroup

        Notes
        =====

        The commutator of two subgroups `H, G` is equal to the normal closure
        of the commutators of all the generators, i.e. `hgh^{-1}g^{-1}` for `h`
        a generator of `H` and `g` a generator of `G` ([1], p.28)

        """
        ggens = G.generators
        hgens = H.generators
        commutators = []
        for ggen in ggens:
            for hgen in hgens:
                commutator = rmul(hgen, ggen, ~hgen, ~ggen)
                if commutator not in commutators:
                    commutators.append(commutator)
        res = self.normal_closure(commutators)
        return res

    def coset_factor(self, g, factor_index=False):
        """Return ``G``'s (self's) coset factorization of ``g``

        Explanation
        ===========

        If ``g`` is an element of ``G`` then it can be written as the product
        of permutations drawn from the Schreier-Sims coset decomposition,

        The permutations returned in ``f`` are those for which
        the product gives ``g``: ``g = f[n]*...f[1]*f[0]`` where ``n = len(B)``
        and ``B = G.base``. f[i] is one of the permutations in
        ``self._basic_orbits[i]``.

        If factor_index==True,
        returns a tuple ``[b[0],..,b[n]]``, where ``b[i]``
        belongs to ``self._basic_orbits[i]``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])

        Define g:

        >>> g = Permutation(7)(1, 2, 4)(3, 6, 5)

        Confirm that it is an element of G:

        >>> G.contains(g)
        True

        Thus, it can be written as a product of factors (up to
        3) drawn from u. See below that a factor from u1 and u2
        and the Identity permutation have been used:

        >>> f = G.coset_factor(g)
        >>> f[2]*f[1]*f[0] == g
        True
        >>> f1 = G.coset_factor(g, True); f1
        [0, 4, 4]
        >>> tr = G.basic_transversals
        >>> f[0] == tr[0][f1[0]]
        True

        If g is not an element of G then [] is returned:

        >>> c = Permutation(5, 6, 7)
        >>> G.coset_factor(c)
        []

        See Also
        ========

        sympy.combinatorics.util._strip

        """
        if isinstance(g, (Cycle, Permutation)):
            g = g.list()
        if len(g) != self._degree:
            # this could either adjust the size or return [] immediately
            # but we don't choose between the two and just signal a possible
            # error
            raise ValueError('g should be the same size as permutations of G')
        I = list(range(self._degree))
        basic_orbits = self.basic_orbits
        transversals = self._transversals
        factors = []
        base = self.base
        h = g
        for i in range(len(base)):
            beta = h[base[i]]
            if beta == base[i]:
                factors.append(beta)
                continue
            if beta not in basic_orbits[i]:
                return []
            u = transversals[i][beta]._array_form
            h = _af_rmul(_af_invert(u), h)
            factors.append(beta)
        if h != I:
            return []
        if factor_index:
            return factors
        tr = self.basic_transversals
        factors = [tr[i][factors[i]] for i in range(len(base))]
        return factors

    def generator_product(self, g, original=False):
        r'''
        Return a list of strong generators `[s1, \dots, sn]`
        s.t `g = sn \times \dots \times s1`. If ``original=True``, make the
        list contain only the original group generators

        '''
        product = []
        if g.is_identity:
            return []
        if g in self.strong_gens:
            if not original or g in self.generators:
                return [g]
            else:
                slp = self._strong_gens_slp[g]
                for s in slp:
                    product.extend(self.generator_product(s, original=True))
                return product
        elif g**-1 in self.strong_gens:
            g = g**-1
            if not original or g in self.generators:
                return [g**-1]
            else:
                slp = self._strong_gens_slp[g]
                for s in slp:
                    product.extend(self.generator_product(s, original=True))
                l = len(product)
                product = [product[l-i-1]**-1 for i in range(l)]
                return product

        f = self.coset_factor(g, True)
        for i, j in enumerate(f):
            slp = self._transversal_slp[i][j]
            for s in slp:
                if not original:
                    product.append(self.strong_gens[s])
                else:
                    s = self.strong_gens[s]
                    product.extend(self.generator_product(s, original=True))
        return product

    def coset_rank(self, g):
        """rank using Schreier-Sims representation.

        Explanation
        ===========

        The coset rank of ``g`` is the ordering number in which
        it appears in the lexicographic listing according to the
        coset decomposition

        The ordering is the same as in G.generate(method='coset').
        If ``g`` does not belong to the group it returns None.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])
        >>> c = Permutation(7)(2, 4)(3, 5)
        >>> G.coset_rank(c)
        16
        >>> G.coset_unrank(16)
        (7)(2 4)(3 5)

        See Also
        ========

        coset_factor

        """
        factors = self.coset_factor(g, True)
        if not factors:
            return None
        rank = 0
        b = 1
        transversals = self._transversals
        base = self._base
        basic_orbits = self._basic_orbits
        for i in range(len(base)):
            k = factors[i]
            j = basic_orbits[i].index(k)
            rank += b*j
            b = b*len(transversals[i])
        return rank

    def coset_unrank(self, rank, af=False):
        """unrank using Schreier-Sims representation

        coset_unrank is the inverse operation of coset_rank
        if 0 <= rank < order; otherwise it returns None.

        """
        if rank < 0 or rank >= self.order():
            return None
        base = self.base
        transversals = self.basic_transversals
        basic_orbits = self.basic_orbits
        m = len(base)
        v = [0]*m
        for i in range(m):
            rank, c = divmod(rank, len(transversals[i]))
            v[i] = basic_orbits[i][c]
        a = [transversals[i][v[i]]._array_form for i in range(m)]
        h = _af_rmuln(*a)
        if af:
            return h
        else:
            return _af_new(h)

    @property
    def degree(self):
        """Returns the size of the permutations in the group.

        Explanation
        ===========

        The number of permutations comprising the group is given by
        ``len(group)``; the number of permutations that can be generated
        by the group is given by ``group.order()``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a])
        >>> G.degree
        3
        >>> len(G)
        1
        >>> G.order()
        2
        >>> list(G.generate())
        [(2), (2)(0 1)]

        See Also
        ========

        order
        """
        return self._degree

    @property
    def identity(self):
        '''
        Return the identity element of the permutation group.

        '''
        return _af_new(list(range(self.degree)))

    @property
    def elements(self):
        """Returns all the elements of the permutation group as a list

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = PermutationGroup(Permutation(1, 3), Permutation(1, 2))
        >>> p.elements
        [(3), (3)(1 2), (1 3), (2 3), (1 2 3), (1 3 2)]

        """
        if not self._elements:
            self._elements = list(self.generate())

        return self._elements

    def derived_series(self):
        r"""Return the derived series for the group.

        Explanation
        ===========

        The derived series for a group `G` is defined as
        `G = G_0 > G_1 > G_2 > \ldots` where `G_i = [G_{i-1}, G_{i-1}]`,
        i.e. `G_i` is the derived subgroup of `G_{i-1}`, for
        `i\in\mathbb{N}`. When we have `G_k = G_{k-1}` for some
        `k\in\mathbb{N}`, the series terminates.

        Returns
        =======

        A list of permutation groups containing the members of the derived
        series in the order `G = G_0, G_1, G_2, \ldots`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup, DihedralGroup)
        >>> A = AlternatingGroup(5)
        >>> len(A.derived_series())
        1
        >>> S = SymmetricGroup(4)
        >>> len(S.derived_series())
        4
        >>> S.derived_series()[1].is_subgroup(AlternatingGroup(4))
        True
        >>> S.derived_series()[2].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        derived_subgroup

        """
        res = [self]
        current = self
        nxt = self.derived_subgroup()
        while not current.is_subgroup(nxt):
            res.append(nxt)
            current = nxt
            nxt = nxt.derived_subgroup()
        return res

    def derived_subgroup(self):
        r"""Compute the derived subgroup.

        Explanation
        ===========

        The derived subgroup, or commutator subgroup is the subgroup generated
        by all commutators `[g, h] = hgh^{-1}g^{-1}` for `g, h\in G` ; it is
        equal to the normal closure of the set of commutators of the generators
        ([1], p.28, [11]).

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 0, 2, 4, 3])
        >>> b = Permutation([0, 1, 3, 2, 4])
        >>> G = PermutationGroup([a, b])
        >>> C = G.derived_subgroup()
        >>> list(C.generate(af=True))
        [[0, 1, 2, 3, 4], [0, 1, 3, 4, 2], [0, 1, 4, 2, 3]]

        See Also
        ========

        derived_series

        """
        r = self._r
        gens = [p._array_form for p in self.generators]
        set_commutators = set()
        degree = self._degree
        rng = list(range(degree))
        for i in range(r):
            for j in range(r):
                p1 = gens[i]
                p2 = gens[j]
                c = list(range(degree))
                for k in rng:
                    c[p2[p1[k]]] = p1[p2[k]]
                ct = tuple(c)
                if ct not in set_commutators:
                    set_commutators.add(ct)
        cms = [_af_new(p) for p in set_commutators]
        G2 = self.normal_closure(cms)
        return G2

    def generate(self, method="coset", af=False):
        """Return iterator to generate the elements of the group.

        Explanation
        ===========

        Iteration is done with one of these methods::

          method='coset'  using the Schreier-Sims coset representation
          method='dimino' using the Dimino method

        If ``af = True`` it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import PermutationGroup
        >>> from sympy.combinatorics.polyhedron import tetrahedron

        The permutation group given in the tetrahedron object is also
        true groups:

        >>> G = tetrahedron.pgroup
        >>> G.is_group
        True

        Also the group generated by the permutations in the tetrahedron
        pgroup -- even the first two -- is a proper group:

        >>> H = PermutationGroup(G[0], G[1])
        >>> J = PermutationGroup(list(H.generate())); J
        PermutationGroup([
            (0 1)(2 3),
            (1 2 3),
            (1 3 2),
            (0 3 1),
            (0 2 3),
            (0 3)(1 2),
            (0 1 3),
            (3)(0 2 1),
            (0 3 2),
            (3)(0 1 2),
            (0 2)(1 3)])
        >>> _.is_group
        True
        """
        if method == "coset":
            return self.generate_schreier_sims(af)
        elif method == "dimino":
            return self.generate_dimino(af)
        else:
            raise NotImplementedError('No generation defined for %s' % method)

    def generate_dimino(self, af=False):
        """Yield group elements using Dimino's algorithm.

        If ``af == True`` it yields the array form of the permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_dimino(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1],
         [0, 1, 3, 2], [0, 3, 2, 1], [0, 3, 1, 2]]

        References
        ==========

        .. [1] The Implementation of Various Algorithms for Permutation Groups in
               the Computer Algebra System: AXIOM, N.J. Doye, M.Sc. Thesis

        """
        idn = list(range(self.degree))
        order = 0
        element_list = [idn]
        set_element_list = {tuple(idn)}
        if af:
            yield idn
        else:
            yield _af_new(idn)
        gens = [p._array_form for p in self.generators]

        for i in range(len(gens)):
            # D elements of the subgroup G_i generated by gens[:i]
            D = element_list.copy()
            N = [idn]
            while N:
                A = N
                N = []
                for a in A:
                    for g in gens[:i + 1]:
                        ag = _af_rmul(a, g)
                        if tuple(ag) not in set_element_list:
                            # produce G_i*g
                            for d in D:
                                order += 1
                                ap = _af_rmul(d, ag)
                                if af:
                                    yield ap
                                else:
                                    p = _af_new(ap)
                                    yield p
                                element_list.append(ap)
                                set_element_list.add(tuple(ap))
                                N.append(ap)
        self._order = len(element_list)

    def generate_schreier_sims(self, af=False):
        """Yield group elements using the Schreier-Sims representation
        in coset_rank order

        If ``af = True`` it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_schreier_sims(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1],
         [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2]]
        """

        n = self._degree
        u = self.basic_transversals
        basic_orbits = self._basic_orbits
        if len(u) == 0:
            for x in self.generators:
                if af:
                    yield x._array_form
                else:
                    yield x
            return
        if len(u) == 1:
            for i in basic_orbits[0]:
                if af:
                    yield u[0][i]._array_form
                else:
                    yield u[0][i]
            return

        u = list(reversed(u))
        basic_orbits = basic_orbits[::-1]
        # stg stack of group elements
        stg = [list(range(n))]
        posmax = [len(x) for x in u]
        n1 = len(posmax) - 1
        pos = [0]*n1
        h = 0
        while 1:
            # backtrack when finished iterating over coset
            if pos[h] >= posmax[h]:
                if h == 0:
                    return
                pos[h] = 0
                h -= 1
                stg.pop()
                continue
            p = _af_rmul(u[h][basic_orbits[h][pos[h]]]._array_form, stg[-1])
            pos[h] += 1
            stg.append(p)
            h += 1
            if h == n1:
                if af:
                    for i in basic_orbits[-1]:
                        p = _af_rmul(u[-1][i]._array_form, stg[-1])
                        yield p
                else:
                    for i in basic_orbits[-1]:
                        p = _af_rmul(u[-1][i]._array_form, stg[-1])
                        p1 = _af_new(p)
                        yield p1
                stg.pop()
                h -= 1

    @property
    def generators(self):
        """Returns the generators of the group.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.generators
        [(1 2), (2)(0 1)]

        """
        return self._generators

    def contains(self, g, strict=True):
        """Test if permutation ``g`` belong to self, ``G``.

        Explanation
        ===========

        If ``g`` is an element of ``G`` it can be written as a product
        of factors drawn from the cosets of ``G``'s stabilizers. To see
        if ``g`` is one of the actual generators defining the group use
        ``G.has(g)``.

        If ``strict`` is not ``True``, ``g`` will be resized, if necessary,
        to match the size of permutations in ``self``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup

        >>> a = Permutation(1, 2)
        >>> b = Permutation(2, 3, 1)
        >>> G = PermutationGroup(a, b, degree=5)
        >>> G.contains(G[0]) # trivial check
        True
        >>> elem = Permutation([[2, 3]], size=5)
        >>> G.contains(elem)
        True
        >>> G.contains(Permutation(4)(0, 1, 2, 3))
        False

        If strict is False, a permutation will be resized, if
        necessary:

        >>> H = PermutationGroup(Permutation(5))
        >>> H.contains(Permutation(3))
        False
        >>> H.contains(Permutation(3), strict=False)
        True

        To test if a given permutation is present in the group:

        >>> elem in G.generators
        False
        >>> G.has(elem)
        False

        See Also
        ========

        coset_factor, sympy.core.basic.Basic.has, __contains__

        """
        if not isinstance(g, Permutation):
            return False
        if g.size != self.degree:
            if strict:
                return False
            g = Permutation(g, size=self.degree)
        if g in self.generators:
            return True
        return bool(self.coset_factor(g.array_form, True))

    @property
    def is_perfect(self):
        """Return ``True`` if the group is perfect.
        A group is perfect if it equals to its derived subgroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1,2,3)(4,5)
        >>> b = Permutation(1,2,3,4,5)
        >>> G = PermutationGroup([a, b])
        >>> G.is_perfect
        False

        """
        if self._is_perfect is None:
            self._is_perfect = self.equals(self.derived_subgroup())
        return self._is_perfect

    @property
    def is_abelian(self):
        """Test if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.is_abelian
        False
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_abelian
        True

        """
        if self._is_abelian is not None:
            return self._is_abelian

        self._is_abelian = True
        gens = [p._array_form for p in self.generators]
        for x in gens:
            for y in gens:
                if y <= x:
                    continue
                if not _af_commutes_with(x, y):
                    self._is_abelian = False
                    return False
        return True

    def abelian_invariants(self):
        """
        Returns the abelian invariants for the given group.
        Let ``G`` be a nontrivial finite abelian group. Then G is isomorphic to
        the direct product of finitely many nontrivial cyclic groups of
        prime-power order.

        Explanation
        ===========

        The prime-powers that occur as the orders of the factors are uniquely
        determined by G. More precisely, the primes that occur in the orders of the
        factors in any such decomposition of ``G`` are exactly the primes that divide
        ``|G|`` and for any such prime ``p``, if the orders of the factors that are
        p-groups in one such decomposition of ``G`` are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``,
        then the orders of the factors that are p-groups in any such decomposition of ``G``
        are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``.

        The uniquely determined integers ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``, taken
        for all primes that divide ``|G|`` are called the invariants of the nontrivial
        group ``G`` as suggested in ([14], p. 542).

        Notes
        =====

        We adopt the convention that the invariants of a trivial group are [].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.abelian_invariants()
        [2]
        >>> from sympy.combinatorics import CyclicGroup
        >>> G = CyclicGroup(7)
        >>> G.abelian_invariants()
        [7]

        """
        if self.is_trivial:
            return []
        gns = self.generators
        inv = []
        G = self
        H = G.derived_subgroup()
        Hgens = H.generators
        for p in primefactors(G.order()):
            ranks = []
            while True:
                pows = []
                for g in gns:
                    elm = g**p
                    if not H.contains(elm):
                        pows.append(elm)
                K = PermutationGroup(Hgens + pows) if pows else H
                r = G.order()//K.order()
                G = K
                gns = pows
                if r == 1:
                    break
                ranks.append(multiplicity(p, r))

            if ranks:
                pows = [1]*ranks[0]
                for i in ranks:
                    for j in range(i):
                        pows[j] = pows[j]*p
                inv.extend(pows)
        inv.sort()
        return inv

    def is_elementary(self, p):
        """Return ``True`` if the group is elementary abelian. An elementary
        abelian group is a finite abelian group, where every nontrivial
        element has order `p`, where `p` is a prime.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_elementary(2)
        True
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([3, 1, 2, 0])
        >>> G = PermutationGroup([a, b])
        >>> G.is_elementary(2)
        True
        >>> G.is_elementary(3)
        False

        """
        return self.is_abelian and all(g.order() == p for g in self.generators)

    def _eval_is_alt_sym_naive(self, only_sym=False, only_alt=False):
        """A naive test using the group order."""
        if only_sym and only_alt:
            raise ValueError(
                "Both {} and {} cannot be set to True"
                .format(only_sym, only_alt))

        n = self.degree
        sym_order = _factorial(n)
        order = self.order()

        if order == sym_order:
            self._is_sym = True
            self._is_alt = False
            return not only_alt

        if 2*order == sym_order:
            self._is_sym = False
            self._is_alt = True
            return not only_sym

        return False

    def _eval_is_alt_sym_monte_carlo(self, eps=0.05, perms=None):
        """A test using monte-carlo algorithm.

        Parameters
        ==========

        eps : float, optional
            The criterion for the incorrect ``False`` return.

        perms : list[Permutation], optional
            If explicitly given, it tests over the given candidates
            for testing.

            If ``None``, it randomly computes ``N_eps`` and chooses
            ``N_eps`` sample of the permutation from the group.

        See Also
        ========

        _check_cycles_alt_sym
        """
        if perms is None:
            n = self.degree
            if n < 17:
                c_n = 0.34
            else:
                c_n = 0.57
            d_n = (c_n*log(2))/log(n)
            N_eps = int(-log(eps)/d_n)

            perms = (self.random_pr() for i in range(N_eps))
            return self._eval_is_alt_sym_monte_carlo(perms=perms)

        for perm in perms:
            if _check_cycles_alt_sym(perm):
                return True
        return False

    def is_alt_sym(self, eps=0.05, _random_prec=None):
        r"""Monte Carlo test for the symmetric/alternating group for degrees
        >= 8.

        Explanation
        ===========

        More specifically, it is one-sided Monte Carlo with the
        answer True (i.e., G is symmetric/alternating) guaranteed to be
        correct, and the answer False being incorrect with probability eps.

        For degree < 8, the order of the group is checked so the test
        is deterministic.

        Notes
        =====

        The algorithm itself uses some nontrivial results from group theory and
        number theory:
        1) If a transitive group ``G`` of degree ``n`` contains an element
        with a cycle of length ``n/2 < p < n-2`` for ``p`` a prime, ``G`` is the
        symmetric or alternating group ([1], pp. 81-82)
        2) The proportion of elements in the symmetric/alternating group having
        the property described in 1) is approximately `\log(2)/\log(n)`
        ([1], p.82; [2], pp. 226-227).
        The helper function ``_check_cycles_alt_sym`` is used to
        go over the cycles in a permutation and look for ones satisfying 1).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_alt_sym()
        False

        See Also
        ========

        _check_cycles_alt_sym

        """
        if _random_prec is not None:
            N_eps = _random_prec['N_eps']
            perms= (_random_prec[i] for i in range(N_eps))
            return self._eval_is_alt_sym_monte_carlo(perms=perms)

        if self._is_sym or self._is_alt:
            return True
        if self._is_sym is False and self._is_alt is False:
            return False

        n = self.degree
        if n < 8:
            return self._eval_is_alt_sym_naive()
        elif self.is_transitive():
            return self._eval_is_alt_sym_monte_carlo(eps=eps)

        self._is_sym, self._is_alt = False, False
        return False

    @property
    def is_nilpotent(self):
        """Test if the group is nilpotent.

        Explanation
        ===========

        A group `G` is nilpotent if it has a central series of finite length.
        Alternatively, `G` is nilpotent if its lower central series terminates
        with the trivial group. Every nilpotent group is also solvable
        ([1], p.29, [12]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> C = CyclicGroup(6)
        >>> C.is_nilpotent
        True
        >>> S = SymmetricGroup(5)
        >>> S.is_nilpotent
        False

        See Also
        ========

        lower_central_series, is_solvable

        """
        if self._is_nilpotent is None:
            lcs = self.lower_central_series()
            terminator = lcs[len(lcs) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                self._is_nilpotent = True
                return True
            else:
                self._is_nilpotent = False
                return False
        else:
            return self._is_nilpotent

    def is_normal(self, gr, strict=True):
        """Test if ``G=self`` is a normal subgroup of ``gr``.

        Explanation
        ===========

        G is normal in gr if
        for each g2 in G, g1 in gr, ``g = g1*g2*g1**-1`` belongs to G
        It is sufficient to check this for each g1 in gr.generators and
        g2 in G.generators.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G1 = PermutationGroup([a, Permutation([2, 0, 1])])
        >>> G1.is_normal(G)
        True

        """
        if not self.is_subgroup(gr, strict=strict):
            return False
        d_self = self.degree
        d_gr = gr.degree
        if self.is_trivial and (d_self == d_gr or not strict):
            return True
        if self._is_abelian:
            return True
        new_self = self.copy()
        if not strict and d_self != d_gr:
            if d_self < d_gr:
                new_self = PermGroup(new_self.generators + [Permutation(d_gr - 1)])
            else:
                gr = PermGroup(gr.generators + [Permutation(d_self - 1)])
        gens2 = [p._array_form for p in new_self.generators]
        gens1 = [p._array_form for p in gr.generators]
        for g1 in gens1:
            for g2 in gens2:
                p = _af_rmuln(g1, g2, _af_invert(g1))
                if not new_self.coset_factor(p, True):
                    return False
        return True

    def is_primitive(self, randomized=True):
        r"""Test if a group is primitive.

        Explanation
        ===========

        A permutation group ``G`` acting on a set ``S`` is called primitive if
        ``S`` contains no nontrivial block under the action of ``G``
        (a block is nontrivial if its cardinality is more than ``1``).

        Notes
        =====

        The algorithm is described in [1], p.83, and uses the function
        minimal_block to search for blocks of the form `\{0, k\}` for ``k``
        ranging over representatives for the orbits of `G_0`, the stabilizer of
        ``0``. This algorithm has complexity `O(n^2)` where ``n`` is the degree
        of the group, and will perform badly if `G_0` is small.

        There are two implementations offered: one finds `G_0`
        deterministically using the function ``stabilizer``, and the other
        (default) produces random elements of `G_0` using ``random_stab``,
        hoping that they generate a subgroup of `G_0` with not too many more
        orbits than `G_0` (this is suggested in [1], p.83). Behavior is changed
        by the ``randomized`` flag.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_primitive()
        False

        See Also
        ========

        minimal_block, random_stab

        """
        if self._is_primitive is not None:
            return self._is_primitive

        if self.is_transitive() is False:
            return False

        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for _ in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0 and any(e != 0 for e in self.minimal_block([0, x])):
                self._is_primitive = False
                return False
        self._is_primitive = True
        return True

    def minimal_blocks(self, randomized=True):
        '''
        For a transitive group, return the list of all minimal
        block systems. If a group is intransitive, return `False`.

        Examples
        ========
        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> DihedralGroup(6).minimal_blocks()
        [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]
        >>> G = PermutationGroup(Permutation(1,2,5))
        >>> G.minimal_blocks()
        False

        See Also
        ========

        minimal_block, is_transitive, is_primitive

        '''
        def _number_blocks(blocks):
            # number the blocks of a block system
            # in order and return the number of
            # blocks and the tuple with the
            # reordering
            n = len(blocks)
            appeared = {}
            m = 0
            b = [None]*n
            for i in range(n):
                if blocks[i] not in appeared:
                    appeared[blocks[i]] = m
                    b[i] = m
                    m += 1
                else:
                    b[i] = appeared[blocks[i]]
            return tuple(b), m

        if not self.is_transitive():
            return False
        blocks = []
        num_blocks = []
        rep_blocks = []
        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for i in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0:
                block = self.minimal_block([0, x])
                num_block, _ = _number_blocks(block)
                # a representative block (containing 0)
                rep = {j for j in range(self.degree) if num_block[j] == 0}
                # check if the system is minimal with
                # respect to the already discovere ones
                minimal = True
                blocks_remove_mask = [False] * len(blocks)
                for i, r in enumerate(rep_blocks):
                    if len(r) > len(rep) and rep.issubset(r):
                        # i-th block system is not minimal
                        blocks_remove_mask[i] = True
                    elif len(r) < len(rep) and r.issubset(rep):
                        # the system being checked is not minimal
                        minimal = False
                        break
                # remove non-minimal representative blocks
                blocks = [b for i, b in enumerate(blocks) if not blocks_remove_mask[i]]
                num_blocks = [n for i, n in enumerate(num_blocks) if not blocks_remove_mask[i]]
                rep_blocks = [r for i, r in enumerate(rep_blocks) if not blocks_remove_mask[i]]

                if minimal and num_block not in num_blocks:
                    blocks.append(block)
                    num_blocks.append(num_block)
                    rep_blocks.append(rep)
        return blocks

    @property
    def is_solvable(self):
        """Test if the group is solvable.

        ``G`` is solvable if its derived series terminates with the trivial
        group ([1], p.29).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(3)
        >>> S.is_solvable
        True

        See Also
        ========

        is_nilpotent, derived_series

        """
        if self._is_solvable is None:
            if self.order() % 2 != 0:
                return True
            ds = self.derived_series()
            terminator = ds[len(ds) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                return True
            else:
                self._is_solvable = False
                return False
        else:
            return self._is_solvable

    def is_subgroup(self, G, strict=True):
        """Return ``True`` if all elements of ``self`` belong to ``G``.

        If ``strict`` is ``False`` then if ``self``'s degree is smaller
        than ``G``'s, the elements will be resized to have the same degree.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.combinatorics import SymmetricGroup, CyclicGroup

        Testing is strict by default: the degree of each group must be the
        same:

        >>> p = Permutation(0, 1, 2, 3, 4, 5)
        >>> G1 = PermutationGroup([Permutation(0, 1, 2), Permutation(0, 1)])
        >>> G2 = PermutationGroup([Permutation(0, 2), Permutation(0, 1, 2)])
        >>> G3 = PermutationGroup([p, p**2])
        >>> assert G1.order() == G2.order() == G3.order() == 6
        >>> G1.is_subgroup(G2)
        True
        >>> G1.is_subgroup(G3)
        False
        >>> G3.is_subgroup(PermutationGroup(G3[1]))
        False
        >>> G3.is_subgroup(PermutationGroup(G3[0]))
        True

        To ignore the size, set ``strict`` to ``False``:

        >>> S3 = SymmetricGroup(3)
        >>> S5 = SymmetricGroup(5)
        >>> S3.is_subgroup(S5, strict=False)
        True
        >>> C7 = CyclicGroup(7)
        >>> G = S5*C7
        >>> S5.is_subgroup(G, False)
        True
        >>> C7.is_subgroup(G, 0)
        False

        """
        if isinstance(G, SymmetricPermutationGroup):
            if self.degree != G.degree:
                return False
            return True
        if not isinstance(G, PermutationGroup):
            return False
        if self == G or self.generators[0]==Permutation():
            return True
        if G.order() % self.order() != 0:
            return False
        if self.degree == G.degree or \
                (self.degree < G.degree and not strict):
            gens = self.generators
        else:
            return False
        return all(G.contains(g, strict=strict) for g in gens)

    @property
    def is_polycyclic(self):
        """Return ``True`` if a group is polycyclic. A group is polycyclic if
        it has a subnormal series with cyclic factors. For finite groups,
        this is the same as if the group is solvable.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G = PermutationGroup([a, b])
        >>> G.is_polycyclic
        True

        """
        return self.is_solvable

    def is_transitive(self, strict=True):
        """Test if the group is transitive.

        Explanation
        ===========

        A group is transitive if it has a single orbit.

        If ``strict`` is ``False`` the group is transitive if it has
        a single orbit of length different from 1.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G1 = PermutationGroup([a, b])
        >>> G1.is_transitive()
        False
        >>> G1.is_transitive(strict=False)
        True
        >>> c = Permutation([2, 3, 0, 1])
        >>> G2 = PermutationGroup([a, c])
        >>> G2.is_transitive()
        True
        >>> d = Permutation([1, 0, 2, 3])
        >>> e = Permutation([0, 1, 3, 2])
        >>> G3 = PermutationGroup([d, e])
        >>> G3.is_transitive() or G3.is_transitive(strict=False)
        False

        """
        if self._is_transitive:  # strict or not, if True then True
            return self._is_transitive
        if strict:
            if self._is_transitive is not None:  # we only store strict=True
                return self._is_transitive

            ans = len(self.orbit(0)) == self.degree
            self._is_transitive = ans
            return ans

        got_orb = False
        for x in self.orbits():
            if len(x) > 1:
                if got_orb:
                    return False
                got_orb = True
        return got_orb

    @property
    def is_trivial(self):
        """Test if the group is the trivial group.

        This is true if the group contains only the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 1, 2])])
        >>> G.is_trivial
        True

        """
        if self._is_trivial is None:
            self._is_trivial = len(self) == 1 and self[0].is_Identity
        return self._is_trivial

    def lower_central_series(self):
        r"""Return the lower central series for the group.

        The lower central series for a group `G` is the series
        `G = G_0 > G_1 > G_2 > \ldots` where
        `G_k = [G, G_{k-1}]`, i.e. every term after the first is equal to the
        commutator of `G` and the previous term in `G1` ([1], p.29).

        Returns
        =======

        A list of permutation groups in the order `G = G_0, G_1, G_2, \ldots`

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (AlternatingGroup,
        ... DihedralGroup)
        >>> A = AlternatingGroup(4)
        >>> len(A.lower_central_series())
        2
        >>> A.lower_central_series()[1].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        commutator, derived_series

        """
        res = [self]
        current = self
        nxt = self.commutator(self, current)
        while not current.is_subgroup(nxt):
            res.append(nxt)
            current = nxt
            nxt = self.commutator(self, current)
        return res

    @property
    def max_div(self):
        """Maximum proper divisor of the degree of a permutation group.

        Explanation
        ===========

        Obviously, this is the degree divided by its minimal proper divisor
        (larger than ``1``, if one exists). As it is guaranteed to be prime,
        the ``sieve`` from ``sympy.ntheory`` is used.
        This function is also used as an optimization tool for the functions
        ``minimal_block`` and ``_union_find_merge``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 2, 1, 3])])
        >>> G.max_div
        2

        See Also
        ========

        minimal_block, _union_find_merge

        """
        if self._max_div is not None:
            return self._max_div
        n = self.degree
        if n == 1:
            return 1
        for x in sieve:
            if n % x == 0:
                d = n//x
                self._max_div = d
                return d

    def minimal_block(self, points):
        r"""For a transitive group, finds the block system generated by
        ``points``.

        Explanation
        ===========

        If a group ``G`` acts on a set ``S``, a nonempty subset ``B`` of ``S``
        is called a block under the action of ``G`` if for all ``g`` in ``G``
        we have ``gB = B`` (``g`` fixes ``B``) or ``gB`` and ``B`` have no
        common points (``g`` moves ``B`` entirely). ([1], p.23; [6]).

        The distinct translates ``gB`` of a block ``B`` for ``g`` in ``G``
        partition the set ``S`` and this set of translates is known as a block
        system. Moreover, we obviously have that all blocks in the partition
        have the same size, hence the block size divides ``|S|`` ([1], p.23).
        A ``G``-congruence is an equivalence relation ``~`` on the set ``S``
        such that ``a ~ b`` implies ``g(a) ~ g(b)`` for all ``g`` in ``G``.
        For a transitive group, the equivalence classes of a ``G``-congruence
        and the blocks of a block system are the same thing ([1], p.23).

        The algorithm below checks the group for transitivity, and then finds
        the ``G``-congruence generated by the pairs ``(p_0, p_1), (p_0, p_2),
        ..., (p_0,p_{k-1})`` which is the same as finding the maximal block
        system (i.e., the one with minimum block size) such that
        ``p_0, ..., p_{k-1}`` are in the same block ([1], p.83).

        It is an implementation of Atkinson's algorithm, as suggested in [1],
        and manipulates an equivalence relation on the set ``S`` using a
        union-find data structure. The running time is just above
        `O(|points||S|)`. ([1], pp. 83-87; [7]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.minimal_block([0, 5])
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        >>> D.minimal_block([0, 1])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        See Also
        ========

        _union_find_rep, _union_find_merge, is_transitive, is_primitive

        """
        if not self.is_transitive():
            return False
        n = self.degree
        gens = self.generators
        # initialize the list of equivalence class representatives
        parents = list(range(n))
        ranks = [1]*n
        not_rep = []
        k = len(points)
        # the block size must divide the degree of the group
        if k > self.max_div:
            return [0]*n
        for i in range(k - 1):
            parents[points[i + 1]] = points[0]
            not_rep.append(points[i + 1])
        ranks[points[0]] = k
        i = 0
        len_not_rep = k - 1
        while i < len_not_rep:
            gamma = not_rep[i]
            i += 1
            for gen in gens:
                # find has side effects: performs path compression on the list
                # of representatives
                delta = self._union_find_rep(gamma, parents)
                # union has side effects: performs union by rank on the list
                # of representatives
                temp = self._union_find_merge(gen(gamma), gen(delta), ranks,
                                              parents, not_rep)
                if temp == -1:
                    return [0]*n
                len_not_rep += temp
        for i in range(n):
            # force path compression to get the final state of the equivalence
            # relation
            self._union_find_rep(i, parents)

        # rewrite result so that block representatives are minimal
        new_reps = {}
        return [new_reps.setdefault(r, i) for i, r in enumerate(parents)]

    def conjugacy_class(self, x):
        r"""Return the conjugacy class of an element in the group.

        Explanation
        ===========

        The conjugacy class of an element ``g`` in a group ``G`` is the set of
        elements ``x`` in ``G`` that are conjugate with ``g``, i.e. for which

            ``g = xax^{-1}``

        for some ``a`` in ``G``.

        Note that conjugacy is an equivalence relation, and therefore that
        conjugacy classes are partitions of ``G``. For a list of all the
        conjugacy classes of the group, use the conjugacy_classes() method.

        In a permutation group, each conjugacy class corresponds to a particular
        `cycle structure': for example, in ``S_3``, the conjugacy classes are:

            * the identity class, ``{()}``
            * all transpositions, ``{(1 2), (1 3), (2 3)}``
            * all 3-cycles, ``{(1 2 3), (1 3 2)}``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricGroup
        >>> S3 = SymmetricGroup(3)
        >>> S3.conjugacy_class(Permutation(0, 1, 2))
        {(0 1 2), (0 2 1)}

        Notes
        =====

        This procedure computes the conjugacy class directly by finding the
        orbit of the element under conjugation in G. This algorithm is only
        feasible for permutation groups of relatively small order, but is like
        the orbit() function itself in that respect.
        """
        # Ref: "Computing the conjugacy classes of finite groups"; Butler, G.
        # Groups '93 Galway/St Andrews; edited by Campbell, C. M.
        new_class = {x}
        last_iteration = new_class

        while len(last_iteration) > 0:
            this_iteration = set()

            for y in last_iteration:
                for s in self.generators:
                    conjugated = s * y * (~s)
                    if conjugated not in new_class:
                        this_iteration.add(conjugated)

            new_class.update(last_iteration)
            last_iteration = this_iteration

        return new_class


    def conjugacy_classes(self):
        r"""Return the conjugacy classes of the group.

        Explanation
        ===========

        As described in the documentation for the .conjugacy_class() function,
        conjugacy is an equivalence relation on a group G which partitions the
        set of elements. This method returns a list of all these conjugacy
        classes of G.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricGroup
        >>> SymmetricGroup(3).conjugacy_classes()
        [{(2)}, {(0 1 2), (0 2 1)}, {(0 2), (1 2), (2)(0 1)}]

        """
        identity = _af_new(list(range(self.degree)))
        known_elements = {identity}
        classes = [known_elements.copy()]

        for x in self.generate():
            if x not in known_elements:
                new_class = self.conjugacy_class(x)
                classes.append(new_class)
                known_elements.update(new_class)

        return classes

    def normal_closure(self, other, k=10):
        r"""Return the normal closure of a subgroup/set of permutations.

        Explanation
        ===========

        If ``S`` is a subset of a group ``G``, the normal closure of ``A`` in ``G``
        is defined as the intersection of all normal subgroups of ``G`` that
        contain ``A`` ([1], p.14). Alternatively, it is the group generated by
        the conjugates ``x^{-1}yx`` for ``x`` a generator of ``G`` and ``y`` a
        generator of the subgroup ``\left\langle S\right\rangle`` generated by
        ``S`` (for some chosen generating set for ``\left\langle S\right\rangle``)
        ([1], p.73).

        Parameters
        ==========

        other
            a subgroup/list of permutations/single permutation
        k
            an implementation-specific parameter that determines the number
            of conjugates that are adjoined to ``other`` at once

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup, AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> C = CyclicGroup(5)
        >>> G = S.normal_closure(C)
        >>> G.order()
        60
        >>> G.is_subgroup(AlternatingGroup(5))
        True

        See Also
        ========

        commutator, derived_subgroup, random_pr

        Notes
        =====

        The algorithm is described in [1], pp. 73-74; it makes use of the
        generation of random elements for permutation groups by the product
        replacement algorithm.

        """
        if hasattr(other, 'generators'):
            degree = self.degree
            identity = _af_new(list(range(degree)))

            if all(g == identity for g in other.generators):
                return other
            Z = PermutationGroup(other.generators[:])
            base, strong_gens = Z.schreier_sims_incremental()
            strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
            basic_orbits, basic_transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr)

            self._random_pr_init(r=10, n=20)

            _loop = True
            while _loop:
                Z._random_pr_init(r=10, n=10)
                for _ in range(k):
                    g = self.random_pr()
                    h = Z.random_pr()
                    conj = h^g
                    res = _strip(conj, base, basic_orbits, basic_transversals)
                    if res[0] != identity or res[1] != len(base) + 1:
                        gens = Z.generators
                        gens.append(conj)
                        Z = PermutationGroup(gens)
                        strong_gens.append(conj)
                        temp_base, temp_strong_gens = \
                            Z.schreier_sims_incremental(base, strong_gens)
                        base, strong_gens = temp_base, temp_strong_gens
                        strong_gens_distr = \
                            _distribute_gens_by_base(base, strong_gens)
                        basic_orbits, basic_transversals = \
                            _orbits_transversals_from_bsgs(base,
                                strong_gens_distr)
                _loop = False
                for g in self.generators:
                    for h in Z.generators:
                        conj = h^g
                        res = _strip(conj, base, basic_orbits,
                                     basic_transversals)
                        if res[0] != identity or res[1] != len(base) + 1:
                            _loop = True
                            break
                    if _loop:
                        break
            return Z
        elif hasattr(other, '__getitem__'):
            return self.normal_closure(PermutationGroup(other))
        elif hasattr(other, 'array_form'):
            return self.normal_closure(PermutationGroup([other]))

    def orbit(self, alpha, action='tuples'):
        r"""Compute the orbit of alpha `\{g(\alpha) | g \in G\}` as a set.

        Explanation
        ===========

        The time complexity of the algorithm used here is `O(|Orb|*r)` where
        `|Orb|` is the size of the orbit and ``r`` is the number of generators of
        the group. For a more detailed analysis, see [1], p.78, [2], pp. 19-21.
        Here alpha can be a single point, or a list of points.

        If alpha is a single point, the ordinary orbit is computed.
        if alpha is a list of points, there are three available options:

        'union' - computes the union of the orbits of the points in the list
        'tuples' - computes the orbit of the list interpreted as an ordered
        tuple under the group action ( i.e., g((1,2,3)) = (g(1), g(2), g(3)) )
        'sets' - computes the orbit of the list interpreted as a sets

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0, 4, 5, 6, 3])
        >>> G = PermutationGroup([a])
        >>> G.orbit(0)
        {0, 1, 2}
        >>> G.orbit([0, 4], 'union')
        {0, 1, 2, 3, 4, 5, 6}

        See Also
        ========

        orbit_transversal

        """
        return _orbit(self.degree, self.generators, alpha, action)

    def orbit_rep(self, alpha, beta, schreier_vector=None):
        """Return a group element which sends ``alpha`` to ``beta``.

        Explanation
        ===========

        If ``beta`` is not in the orbit of ``alpha``, the function returns
        ``False``. This implementation makes use of the schreier vector.
        For a proof of correctness, see [1], p.80

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> G = AlternatingGroup(5)
        >>> G.orbit_rep(0, 4)
        (0 4 1 2 3)

        See Also
        ========

        schreier_vector

        """
        if schreier_vector is None:
            schreier_vector = self.schreier_vector(alpha)
        if schreier_vector[beta] is None:
            return False
        k = schreier_vector[beta]
        gens = [x._array_form for x in self.generators]
        a = []
        while k != -1:
            a.append(gens[k])
            beta = gens[k].index(beta) # beta = (~gens[k])(beta)
            k = schreier_vector[beta]
        if a:
            return _af_new(_af_rmuln(*a))
        else:
            return _af_new(list(range(self._degree)))

    def orbit_transversal(self, alpha, pairs=False):
        r"""Computes a transversal for the orbit of ``alpha`` as a set.

        Explanation
        ===========

        For a permutation group `G`, a transversal for the orbit
        `Orb = \{g(\alpha) | g \in G\}` is a set
        `\{g_\beta | g_\beta(\alpha) = \beta\}` for `\beta \in Orb`.
        Note that there may be more than one possible transversal.
        If ``pairs`` is set to ``True``, it returns the list of pairs
        `(\beta, g_\beta)`. For a proof of correctness, see [1], p.79

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.orbit_transversal(0)
        [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]

        See Also
        ========

        orbit

        """
        return _orbit_transversal(self._degree, self.generators, alpha, pairs)

    def orbits(self, rep=False):
        """Return the orbits of ``self``, ordered according to lowest element
        in each orbit.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1, 5)(2, 3)(4, 0, 6)
        >>> b = Permutation(1, 5)(3, 4)(2, 6, 0)
        >>> G = PermutationGroup([a, b])
        >>> G.orbits()
        [{0, 2, 3, 4, 6}, {1, 5}]
        """
        return _orbits(self._degree, self._generators)

    def order(self):
        """Return the order of the group: the number of permutations that
        can be generated from elements of the group.

        The number of permutations comprising the group is given by
        ``len(group)``; the length of each permutation in the group is
        given by ``group.size``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup

        >>> a = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a])
        >>> G.degree
        3
        >>> len(G)
        1
        >>> G.order()
        2
        >>> list(G.generate())
        [(2), (2)(0 1)]

        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.order()
        6

        See Also
        ========

        degree

        """
        if self._order is not None:
            return self._order
        if self._is_sym:
            n = self._degree
            self._order = factorial(n)
            return self._order
        if self._is_alt:
            n = self._degree
            self._order = factorial(n)/2
            return self._order

        m = prod([len(x) for x in self.basic_transversals])
        self._order = m
        return m

    def index(self, H):
        """
        Returns the index of a permutation group.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1,2,3)
        >>> b =Permutation(3)
        >>> G = PermutationGroup([a])
        >>> H = PermutationGroup([b])
        >>> G.index(H)
        3

        """
        if H.is_subgroup(self):
            return self.order()//H.order()

    @property
    def is_symmetric(self):
        """Return ``True`` if the group is symmetric.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricGroup
        >>> g = SymmetricGroup(5)
        >>> g.is_symmetric
        True

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> g = PermutationGroup(
        ...     Permutation(0, 1, 2, 3, 4),
        ...     Permutation(2, 3))
        >>> g.is_symmetric
        True

        Notes
        =====

        This uses a naive test involving the computation of the full
        group order.
        If you need more quicker taxonomy for large groups, you can use
        :meth:`PermutationGroup.is_alt_sym`.
        However, :meth:`PermutationGroup.is_alt_sym` may not be accurate
        and is not able to distinguish between an alternating group and
        a symmetric group.

        See Also
        ========

        is_alt_sym
        """
        _is_sym = self._is_sym
        if _is_sym is not None:
            return _is_sym

        n = self.degree
        if n >= 8:
            if self.is_transitive():
                _is_alt_sym = self._eval_is_alt_sym_monte_carlo()
                if _is_alt_sym:
                    if any(g.is_odd for g in self.generators):
                        self._is_sym, self._is_alt = True, False
                        return True

                    self._is_sym, self._is_alt = False, True
                    return False

                return self._eval_is_alt_sym_naive(only_sym=True)

            self._is_sym, self._is_alt = False, False
            return False

        return self._eval_is_alt_sym_naive(only_sym=True)


    @property
    def is_alternating(self):
        """Return ``True`` if the group is alternating.

        Examples
        ========

        >>> from sympy.combinatorics import AlternatingGroup
        >>> g = AlternatingGroup(5)
        >>> g.is_alternating
        True

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> g = PermutationGroup(
        ...     Permutation(0, 1, 2, 3, 4),
        ...     Permutation(2, 3, 4))
        >>> g.is_alternating
        True

        Notes
        =====

        This uses a naive test involving the computation of the full
        group order.
        If you need more quicker taxonomy for large groups, you can use
        :meth:`PermutationGroup.is_alt_sym`.
        However, :meth:`PermutationGroup.is_alt_sym` may not be accurate
        and is not able to distinguish between an alternating group and
        a symmetric group.

        See Also
        ========

        is_alt_sym
        """
        _is_alt = self._is_alt
        if _is_alt is not None:
            return _is_alt

        n = self.degree
        if n >= 8:
            if self.is_transitive():
                _is_alt_sym = self._eval_is_alt_sym_monte_carlo()
                if _is_alt_sym:
                    if all(g.is_even for g in self.generators):
                        self._is_sym, self._is_alt = False, True
                        return True

                    self._is_sym, self._is_alt = True, False
                    return False

                return self._eval_is_alt_sym_naive(only_alt=True)

            self._is_sym, self._is_alt = False, False
            return False

        return self._eval_is_alt_sym_naive(only_alt=True)

    @classmethod
    def _distinct_primes_lemma(cls, primes):
        """Subroutine to test if there is only one cyclic group for the
        order."""
        primes = sorted(primes)
        l = len(primes)
        for i in range(l):
            for j in range(i+1, l):
                if primes[j] % primes[i] == 1:
                    return None
        return True

    @property
    def is_cyclic(self):
        r"""
        Return ``True`` if the group is Cyclic.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AbelianGroup
        >>> G = AbelianGroup(3, 4)
        >>> G.is_cyclic
        True
        >>> G = AbelianGroup(4, 4)
        >>> G.is_cyclic
        False

        Notes
        =====

        If the order of a group $n$ can be factored into the distinct
        primes $p_1, p_2, \dots , p_s$ and if

        .. math::
            \forall i, j \in \{1, 2, \dots, s \}:
            p_i \not \equiv 1 \pmod {p_j}

        holds true, there is only one group of the order $n$ which
        is a cyclic group [1]_. This is a generalization of the lemma
        that the group of order $15, 35, \dots$ are cyclic.

        And also, these additional lemmas can be used to test if a
        group is cyclic if the order of the group is already found.

        - If the group is abelian and the order of the group is
          square-free, the group is cyclic.
        - If the order of the group is less than $6$ and is not $4$, the
          group is cyclic.
        - If the order of the group is prime, the group is cyclic.

        References
        ==========

        .. [1] 1978: John S. Rose: A Course on Group Theory,
            Introduction to Finite Group Theory: 1.4
        """
        if self._is_cyclic is not None:
            return self._is_cyclic

        if len(self.generators) == 1:
            self._is_cyclic = True
            self._is_abelian = True
            return True

        if self._is_abelian is False:
            self._is_cyclic = False
            return False

        order = self.order()

        if order < 6:
            self._is_abelian = True
            if order != 4:
                self._is_cyclic = True
                return True

        factors = factorint(order)
        if all(v == 1 for v in factors.values()):
            if self._is_abelian:
                self._is_cyclic = True
                return True

            primes = list(factors.keys())
            if PermutationGroup._distinct_primes_lemma(primes) is True:
                self._is_cyclic = True
                self._is_abelian = True
                return True

        if not self.is_abelian:
            self._is_cyclic = False
            return False

        self._is_cyclic = all(
            any(g**(order//p) != self.identity for g in self.generators)
            for p, e in factors.items() if e > 1
        )
        return self._is_cyclic

    @property
    def is_dihedral(self):
        r"""
        Return ``True`` if the group is dihedral.

        Examples
        ========

        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup
        >>> G = PermutationGroup(Permutation(1, 6)(2, 5)(3, 4), Permutation(0, 1, 2, 3, 4, 5, 6))
        >>> G.is_dihedral
        True
        >>> G = SymmetricGroup(3)
        >>> G.is_dihedral
        True
        >>> G = CyclicGroup(6)
        >>> G.is_dihedral
        False

        References
        ==========

        .. [Di1] https://math.stackexchange.com/questions/827230/given-a-cayley-table-is-there-an-algorithm-to-determine-if-it-is-a-dihedral-gro/827273#827273
        .. [Di2] https://kconrad.math.uconn.edu/blurbs/grouptheory/dihedral.pdf
        .. [Di3] https://kconrad.math.uconn.edu/blurbs/grouptheory/dihedral2.pdf
        .. [Di4] https://en.wikipedia.org/wiki/Dihedral_group
        """
        if self._is_dihedral is not None:
            return self._is_dihedral

        order = self.order()

        if order % 2 == 1:
            self._is_dihedral = False
            return False
        if order == 2:
            self._is_dihedral = True
            return True
        if order == 4:
            # The dihedral group of order 4 is the Klein 4-group.
            self._is_dihedral = not self.is_cyclic
            return self._is_dihedral
        if self.is_abelian:
            # The only abelian dihedral groups are the ones of orders 2 and 4.
            self._is_dihedral = False
            return False

        # Now we know the group is of even order >= 6, and nonabelian.
        n = order // 2

        # Handle special cases where there are exactly two generators.
        gens = self.generators
        if len(gens) == 2:
            x, y = gens
            a, b = x.order(), y.order()
            # Make a >= b
            if a < b:
                x, y, a, b = y, x, b, a
            # Using Theorem 2.1 of [Di3]:
            if a == 2 == b:
                self._is_dihedral = True
                return True
            # Using Theorem 1.1 of [Di3]:
            if a == n and b == 2 and y*x*y == ~x:
                self._is_dihedral = True
                return True

        # Proceed with algorithm of [Di1]
        # Find elements of orders 2 and n
        order_2, order_n = [], []
        for p in self.elements:
            k = p.order()
            if k == 2:
                order_2.append(p)
            elif k == n:
                order_n.append(p)

        if len(order_2) != n + 1 - (n % 2):
            self._is_dihedral = False
            return False

        if not order_n:
            self._is_dihedral = False
            return False

        x = order_n[0]
        # Want an element y of order 2 that is not a power of x
        # (i.e. that is not the 180-deg rotation, when n is even).
        y = order_2[0]
        if n % 2 == 0 and y == x**(n//2):
            y = order_2[1]

        self._is_dihedral = (y*x*y == ~x)
        return self._is_dihedral

    def pointwise_stabilizer(self, points, incremental=True):
        r"""Return the pointwise stabilizer for a set of points.

        Explanation
        ===========

        For a permutation group `G` and a set of points
        `\{p_1, p_2,\ldots, p_k\}`, the pointwise stabilizer of
        `p_1, p_2, \ldots, p_k` is defined as
        `G_{p_1,\ldots, p_k} =
        \{g\in G | g(p_i) = p_i \forall i\in\{1, 2,\ldots,k\}\}` ([1],p20).
        It is a subgroup of `G`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(7)
        >>> Stab = S.pointwise_stabilizer([2, 3, 5])
        >>> Stab.is_subgroup(S.stabilizer(2).stabilizer(3).stabilizer(5))
        True

        See Also
        ========

        stabilizer, schreier_sims_incremental

        Notes
        =====

        When incremental == True,
        rather than the obvious implementation using successive calls to
        ``.stabilizer()``, this uses the incremental Schreier-Sims algorithm
        to obtain a base with starting segment - the given points.

        """
        if incremental:
            base, strong_gens = self.schreier_sims_incremental(base=points)
            stab_gens = []
            degree = self.degree
            for gen in strong_gens:
                if [gen(point) for point in points] == points:
                    stab_gens.append(gen)
            if not stab_gens:
                stab_gens = _af_new(list(range(degree)))
            return PermutationGroup(stab_gens)
        else:
            gens = self._generators
            degree = self.degree
            for x in points:
                gens = _stabilizer(degree, gens, x)
        return PermutationGroup(gens)

    def make_perm(self, n, seed=None):
        """
        Multiply ``n`` randomly selected permutations from
        pgroup together, starting with the identity
        permutation. If ``n`` is a list of integers, those
        integers will be used to select the permutations and they
        will be applied in L to R order: make_perm((A, B, C)) will
        give CBA(I) where I is the identity permutation.

        ``seed`` is used to set the seed for the random selection
        of permutations from pgroup. If this is a list of integers,
        the corresponding permutations from pgroup will be selected
        in the order give. This is mainly used for testing purposes.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a, b = [Permutation([1, 0, 3, 2]), Permutation([1, 3, 0, 2])]
        >>> G = PermutationGroup([a, b])
        >>> G.make_perm(1, [0])
        (0 1)(2 3)
        >>> G.make_perm(3, [0, 1, 0])
        (0 2 3 1)
        >>> G.make_perm([0, 1, 0])
        (0 2 3 1)

        See Also
        ========

        random
        """
        if is_sequence(n):
            if seed is not None:
                raise ValueError('If n is a sequence, seed should be None')
            n, seed = len(n), n
        else:
            try:
                n = int(n)
            except TypeError:
                raise ValueError('n must be an integer or a sequence.')
        randomrange = _randrange(seed)

        # start with the identity permutation
        result = Permutation(list(range(self.degree)))
        m = len(self)
        for _ in range(n):
            p = self[randomrange(m)]
            result = rmul(result, p)
        return result

    def random(self, af=False):
        """Return a random group element
        """
        rank = randrange(self.order())
        return self.coset_unrank(rank, af)

    def random_pr(self, gen_count=11, iterations=50, _random_prec=None):
        """Return a random group element using product replacement.

        Explanation
        ===========

        For the details of the product replacement algorithm, see
        ``_random_pr_init`` In ``random_pr`` the actual 'product replacement'
        is performed. Notice that if the attribute ``_random_gens``
        is empty, it needs to be initialized by ``_random_pr_init``.

        See Also
        ========

        _random_pr_init

        """
        if self._random_gens == []:
            self._random_pr_init(gen_count, iterations)
        random_gens = self._random_gens
        r = len(random_gens) - 1

        # handle randomized input for testing purposes
        if _random_prec is None:
            s = randrange(r)
            t = randrange(r - 1)
            if t == s:
                t = r - 1
            x = choice([1, 2])
            e = choice([-1, 1])
        else:
            s = _random_prec['s']
            t = _random_prec['t']
            if t == s:
                t = r - 1
            x = _random_prec['x']
            e = _random_prec['e']

        if x == 1:
            random_gens[s] = _af_rmul(random_gens[s], _af_pow(random_gens[t], e))
            random_gens[r] = _af_rmul(random_gens[r], random_gens[s])
        else:
            random_gens[s] = _af_rmul(_af_pow(random_gens[t], e), random_gens[s])
            random_gens[r] = _af_rmul(random_gens[s], random_gens[r])
        return _af_new(random_gens[r])

    def random_stab(self, alpha, schreier_vector=None, _random_prec=None):
        """Random element from the stabilizer of ``alpha``.

        The schreier vector for ``alpha`` is an optional argument used
        for speeding up repeated calls. The algorithm is described in [1], p.81

        See Also
        ========

        random_pr, orbit_rep

        """
        if schreier_vector is None:
            schreier_vector = self.schreier_vector(alpha)
        if _random_prec is None:
            rand = self.random_pr()
        else:
            rand = _random_prec['rand']
        beta = rand(alpha)
        h = self.orbit_rep(alpha, beta, schreier_vector)
        return rmul(~h, rand)

    def schreier_sims(self):
        """Schreier-Sims algorithm.

        Explanation
        ===========

        It computes the generators of the chain of stabilizers
        `G > G_{b_1} > .. > G_{b1,..,b_r} > 1`
        in which `G_{b_1,..,b_i}` stabilizes `b_1,..,b_i`,
        and the corresponding ``s`` cosets.
        An element of the group can be written as the product
        `h_1*..*h_s`.

        We use the incremental Schreier-Sims algorithm.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_sims()
        >>> G.basic_transversals
        [{0: (2)(0 1), 1: (2), 2: (1 2)},
         {0: (2), 2: (0 2)}]
        """
        if self._transversals:
            return
        self._schreier_sims()
        return

    def _schreier_sims(self, base=None):
        schreier = self.schreier_sims_incremental(base=base, slp_dict=True)
        base, strong_gens = schreier[:2]
        self._base = base
        self._strong_gens = strong_gens
        self._strong_gens_slp = schreier[2]
        if not base:
            self._transversals = []
            self._basic_orbits = []
            return

        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_orbits, transversals, slps = _orbits_transversals_from_bsgs(base,\
                strong_gens_distr, slp=True)

        # rewrite the indices stored in slps in terms of strong_gens
        for i, slp in enumerate(slps):
            gens = strong_gens_distr[i]
            for k in slp:
                slp[k] = [strong_gens.index(gens[s]) for s in slp[k]]

        self._transversals = transversals
        self._basic_orbits = [sorted(x) for x in basic_orbits]
        self._transversal_slp = slps

    def schreier_sims_incremental(self, base=None, gens=None, slp_dict=False):
        """Extend a sequence of points and generating set to a base and strong
        generating set.

        Parameters
        ==========

        base
            The sequence of points to be extended to a base. Optional
            parameter with default value ``[]``.
        gens
            The generating set to be extended to a strong generating set
            relative to the base obtained. Optional parameter with default
            value ``self.generators``.

        slp_dict
            If `True`, return a dictionary `{g: gens}` for each strong
            generator `g` where `gens` is a list of strong generators
            coming before `g` in `strong_gens`, such that the product
            of the elements of `gens` is equal to `g`.

        Returns
        =======

        (base, strong_gens)
