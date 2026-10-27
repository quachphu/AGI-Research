            return C.table

    def order(self, strategy="relator_based"):
        """
        Returns the order of the finitely presented group ``self``. It uses
        the coset enumeration with identity group as subgroup, i.e ``H=[]``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x, y**2])
        >>> f.order(strategy="coset_table_based")
        2

        """
        if self._order is not None:
            return self._order
        if self._coset_table is not None:
            self._order = len(self._coset_table.table)
        elif len(self.relators) == 0:
            self._order = self.free_group.order()
        elif len(self.generators) == 1:
            self._order = abs(gcd([r.array_form[0][1] for r in self.relators]))
        elif self._is_infinite():
            self._order = S.Infinity
        else:
            gens, C = self._finite_index_subgroup()
            if C:
                ind = len(C.table)
                self._order = ind*self.subgroup(gens, C=C).order()
            else:
                self._order = self.index([])
        return self._order

    def _is_infinite(self):
        '''
        Test if the group is infinite. Return `True` if the test succeeds
        and `None` otherwise

        '''
        used_gens = set()
        for r in self.relators:
            used_gens.update(r.contains_generators())
        if not set(self.generators) <= used_gens:
            return True
        # Abelianisation test: check is the abelianisation is infinite
        abelian_rels = []
        for rel in self.relators:
            abelian_rels.append([rel.exponent_sum(g) for g in self.generators])
        m = Matrix(Matrix(abelian_rels))
        if 0 in invariant_factors(m):
            return True
        else:
            return None


    def _finite_index_subgroup(self, s=None):
        '''
        Find the elements of `self` that generate a finite index subgroup
        and, if found, return the list of elements and the coset table of `self` by
        the subgroup, otherwise return `(None, None)`

        '''
        gen = self.most_frequent_generator()
        rels = list(self.generators)
        rels.extend(self.relators)
        if not s:
            if len(self.generators) == 2:
                s = [gen] + [g for g in self.generators if g != gen]
            else:
                rand = self.free_group.identity
                i = 0
                while ((rand in rels or rand**-1 in rels or rand.is_identity)
                        and i<10):
                    rand = self.random()
                    i += 1
                s = [gen, rand] + [g for g in self.generators if g != gen]
        mid = (len(s)+1)//2
        half1 = s[:mid]
        half2 = s[mid:]
        draft1 = None
        draft2 = None
        m = 200
        C = None
        while not C and (m/2 < CosetTable.coset_table_max_limit):
            m = min(m, CosetTable.coset_table_max_limit)
            draft1 = self.coset_enumeration(half1, max_cosets=m,
                                 draft=draft1, incomplete=True)
            if draft1.is_complete():
                C = draft1
                half = half1
            else:
                draft2 = self.coset_enumeration(half2, max_cosets=m,
                                 draft=draft2, incomplete=True)
                if draft2.is_complete():
                    C = draft2
                    half = half2
            if not C:
                m *= 2
        if not C:
            return None, None
        C.compress()
        return half, C

    def most_frequent_generator(self):
        gens = self.generators
        rels = self.relators
        freqs = [sum(r.generator_count(g) for r in rels) for g in gens]
        return gens[freqs.index(max(freqs))]

    def random(self):
        import random
        r = self.free_group.identity
        for i in range(random.randint(2,3)):
            r = r*random.choice(self.generators)**random.choice([1,-1])
        return r

    def index(self, H, strategy="relator_based"):
        """
        Return the index of subgroup ``H`` in group ``self``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**5, y**4, y*x*y**3*x**3])
        >>> f.index([x])
        4

        """
        # TODO: use |G:H| = |G|/|H| (currently H can't be made into a group)
        # when we know |G| and |H|

        if H == []:
            return self.order()
        else:
            C = self.coset_enumeration(H, strategy)
            return len(C.table)

    def __str__(self):
        if self.free_group.rank > 30:
            str_form = "<fp group with %s generators>" % self.free_group.rank
        else:
            str_form = "<fp group on the generators %s>" % str(self.generators)
        return str_form

    __repr__ = __str__

#==============================================================================
#                       PERMUTATION GROUP METHODS
#==============================================================================

    def _to_perm_group(self):
        '''
        Return an isomorphic permutation group and the isomorphism.
        The implementation is dependent on coset enumeration so
        will only terminate for finite groups.

        '''
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.homomorphisms import homomorphism
        if self.order() is S.Infinity:
            raise NotImplementedError("Permutation presentation of infinite "
                                                  "groups is not implemented")
        if self._perm_isomorphism:
            T = self._perm_isomorphism
            P = T.image()
        else:
            C = self.coset_table([])
            gens = self.generators
            images = [[C[i][2*gens.index(g)] for i in range(len(C))] for g in gens]
            images = [Permutation(i) for i in images]
            P = PermutationGroup(images)
            T = homomorphism(self, P, gens, images, check=False)
            self._perm_isomorphism = T
        return P, T

    def _perm_group_list(self, method_name, *args):
        '''
        Given the name of a `PermutationGroup` method (returning a subgroup
        or a list of subgroups) and (optionally) additional arguments it takes,
        return a list or a list of lists containing the generators of this (or
        these) subgroups in terms of the generators of `self`.

        '''
        P, T = self._to_perm_group()
        perm_result = getattr(P, method_name)(*args)
        single = False
        if isinstance(perm_result, PermutationGroup):
            perm_result, single = [perm_result], True
        result = []
        for group in perm_result:
            gens = group.generators
            result.append(T.invert(gens))
        return result[0] if single else result

    def derived_series(self):
        '''
        Return the list of lists containing the generators
        of the subgroups in the derived series of `self`.

        '''
        return self._perm_group_list('derived_series')

    def lower_central_series(self):
        '''
        Return the list of lists containing the generators
        of the subgroups in the lower central series of `self`.

        '''
        return self._perm_group_list('lower_central_series')

    def center(self):
        '''
        Return the list of generators of the center of `self`.

        '''
        return self._perm_group_list('center')


    def derived_subgroup(self):
        '''
        Return the list of generators of the derived subgroup of `self`.

        '''
        return self._perm_group_list('derived_subgroup')


    def centralizer(self, other):
        '''
        Return the list of generators of the centralizer of `other`
        (a list of elements of `self`) in `self`.

        '''
        T = self._to_perm_group()[1]
        other = T(other)
        return self._perm_group_list('centralizer', other)

    def normal_closure(self, other):
        '''
        Return the list of generators of the normal closure of `other`
        (a list of elements of `self`) in `self`.

        '''
        T = self._to_perm_group()[1]
        other = T(other)
        return self._perm_group_list('normal_closure', other)

    def _perm_property(self, attr):
        '''
        Given an attribute of a `PermutationGroup`, return
        its value for a permutation group isomorphic to `self`.

        '''
        P = self._to_perm_group()[0]
        return getattr(P, attr)

    @property
    def is_abelian(self):
        '''
        Check if `self` is abelian.

        '''
        return self._perm_property("is_abelian")

    @property
    def is_nilpotent(self):
        '''
        Check if `self` is nilpotent.

        '''
        return self._perm_property("is_nilpotent")

    @property
    def is_solvable(self):
        '''
        Check if `self` is solvable.

        '''
        return self._perm_property("is_solvable")

    @property
    def elements(self):
        '''
        List the elements of `self`.

        '''
        P, T = self._to_perm_group()
        return T.invert(P.elements)

    @property
    def is_cyclic(self):
        """
        Return ``True`` if group is Cyclic.

        """
        if len(self.generators) <= 1:
            return True
        try:
            P, T = self._to_perm_group()
        except NotImplementedError:
            raise NotImplementedError("Check for infinite Cyclic group "
                                      "is not implemented")
        return P.is_cyclic

    def abelian_invariants(self):
        """
        Return Abelian Invariants of a group.
        """
        try:
            P, T = self._to_perm_group()
        except NotImplementedError:
            raise NotImplementedError("abelian invariants is not implemented"
                                      "for infinite group")
        return P.abelian_invariants()

    def composition_series(self):
        """
        Return subnormal series of maximum length for a group.
        """
        try:
            P, T = self._to_perm_group()
        except NotImplementedError:
            raise NotImplementedError("composition series is not implemented"
                                      "for infinite group")
        return P.composition_series()


class FpSubgroup(DefaultPrinting):
    '''
    The class implementing a subgroup of an FpGroup or a FreeGroup
    (only finite index subgroups are supported at this point). This
    is to be used if one wishes to check if an element of the original
    group belongs to the subgroup

    '''
    def __init__(self, G, gens, normal=False):
        super().__init__()
        self.parent = G
        self.generators = list({g for g in gens if g != G.identity})
        self._min_words = None #for use in __contains__
        self.C = None
        self.normal = normal

    def __contains__(self, g):

        if isinstance(self.parent, FreeGroup):
            if self._min_words is None:
                # make _min_words - a list of subwords such that
                # g is in the subgroup if and only if it can be
                # partitioned into these subwords. Infinite families of
                # subwords are presented by tuples, e.g. (r, w)
                # stands for the family of subwords r*w**n*r**-1

                def _process(w):
                    # this is to be used before adding new words
                    # into _min_words; if the word w is not cyclically
                    # reduced, it will generate an infinite family of
                    # subwords so should be written as a tuple;
                    # if it is, w**-1 should be added to the list
                    # as well
                    p, r = w.cyclic_reduction(removed=True)
                    if not r.is_identity:
                        return [(r, p)]
                    else:
                        return [w, w**-1]

                # make the initial list
                gens = []
                for w in self.generators:
                    if self.normal:
                        w = w.cyclic_reduction()
                    gens.extend(_process(w))

                for w1 in gens:
                    for w2 in gens:
                        # if w1 and w2 are equal or are inverses, continue
                        if w1 == w2 or (not isinstance(w1, tuple)
                                                        and w1**-1 == w2):
                            continue

                        # if the start of one word is the inverse of the
                        # end of the other, their multiple should be added
                        # to _min_words because of cancellation
                        if isinstance(w1, tuple):
                            # start, end
                            s1, s2 = w1[0][0], w1[0][0]**-1
                        else:
                            s1, s2 = w1[0], w1[len(w1)-1]

                        if isinstance(w2, tuple):
                            # start, end
                            r1, r2 = w2[0][0], w2[0][0]**-1
                        else:
                            r1, r2 = w2[0], w2[len(w1)-1]

                        # p1 and p2 are w1 and w2 or, in case when
                        # w1 or w2 is an infinite family, a representative
                        p1, p2 = w1, w2
                        if isinstance(w1, tuple):
                            p1 = w1[0]*w1[1]*w1[0]**-1
                        if isinstance(w2, tuple):
                            p2 = w2[0]*w2[1]*w2[0]**-1

                        # add the product of the words to the list is necessary
                        if r1**-1 == s2 and not (p1*p2).is_identity:
                            new = _process(p1*p2)
                            if new not in gens:
                                gens.extend(new)

                        if r2**-1 == s1 and not (p2*p1).is_identity:
                            new = _process(p2*p1)
                            if new not in gens:
                                gens.extend(new)

                self._min_words = gens

            min_words = self._min_words

            def _is_subword(w):
                # check if w is a word in _min_words or one of
                # the infinite families in it
                w, r = w.cyclic_reduction(removed=True)
                if r.is_identity or self.normal:
                    return w in min_words
                else:
                    t = [s[1] for s in min_words if isinstance(s, tuple)
                                                                and s[0] == r]
                    return [s for s in t if w.power_of(s)] != []

            # store the solution of words for which the result of
            # _word_break (below) is known
            known = {}

            def _word_break(w):
                # check if w can be written as a product of words
                # in min_words
                if len(w) == 0:
                    return True
                i = 0
                while i < len(w):
                    i += 1
                    prefix = w.subword(0, i)
                    if not _is_subword(prefix):
                        continue
                    rest = w.subword(i, len(w))
                    if rest not in known:
                        known[rest] = _word_break(rest)
                    if known[rest]:
                        return True
                return False

            if self.normal:
                g = g.cyclic_reduction()
            return _word_break(g)
        else:
            if self.C is None:
                C = self.parent.coset_enumeration(self.generators)
                self.C = C
            i = 0
            C = self.C
            for j in range(len(g)):
                i = C.table[i][C.A_dict[g[j]]]
            return i == 0

    def order(self):
        if not self.generators:
            return S.One
        if isinstance(self.parent, FreeGroup):
            return S.Infinity
        if self.C is None:
            C = self.parent.coset_enumeration(self.generators)
            self.C = C
        # This is valid because `len(self.C.table)` (the index of the subgroup)
        # will always be finite - otherwise coset enumeration doesn't terminate
        return self.parent.order()/len(self.C.table)

    def to_FpGroup(self):
        if isinstance(self.parent, FreeGroup):
            gen_syms = [('x_%d'%i) for i in range(len(self.generators))]
            return free_group(', '.join(gen_syms))[0]
        return self.parent.subgroup(C=self.C)

    def __str__(self):
        if len(self.generators) > 30:
            str_form = "<fp subgroup with %s generators>" % len(self.generators)
        else:
            str_form = "<fp subgroup on the generators %s>" % str(self.generators)
        return str_form

    __repr__ = __str__


###############################################################################
#                           LOW INDEX SUBGROUPS                               #
###############################################################################

def low_index_subgroups(G, N, Y=()):
    """
    Implements the Low Index Subgroups algorithm, i.e find all subgroups of
    ``G`` upto a given index ``N``. This implements the method described in
    [Sim94]. This procedure involves a backtrack search over incomplete Coset
    Tables, rather than over forced coincidences.

    Parameters
    ==========

    G: An FpGroup < X|R >
    N: positive integer, representing the maximum index value for subgroups
    Y: (an optional argument) specifying a list of subgroup generators, such
    that each of the resulting subgroup contains the subgroup generated by Y.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, low_index_subgroups
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**4])
    >>> L = low_index_subgroups(f, 4)
    >>> for coset_table in L:
    ...     print(coset_table.table)
    [[0, 0, 0, 0]]
    [[0, 0, 1, 2], [1, 1, 2, 0], [3, 3, 0, 1], [2, 2, 3, 3]]
    [[0, 0, 1, 2], [2, 2, 2, 0], [1, 1, 0, 1]]
    [[1, 1, 0, 0], [0, 0, 1, 1]]

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           Section 5.4

    .. [2] Marston Conder and Peter Dobcsanyi
           "Applications and Adaptions of the Low Index Subgroups Procedure"

    """
    C = CosetTable(G, [])
    R = G.relators
    # length chosen for the length of the short relators
    len_short_rel = 5
    # elements of R2 only checked at the last step for complete
    # coset tables
    R2 = {rel for rel in R if len(rel) > len_short_rel}
    # elements of R1 are used in inner parts of the process to prune
    # branches of the search tree,
    R1 = {rel.identity_cyclic_reduction() for rel in set(R) - R2}
    R1_c_list = C.conjugates(R1)
    S = []
    descendant_subgroups(S, C, R1_c_list, C.A[0], R2, N, Y)
    return S


def descendant_subgroups(S, C, R1_c_list, x, R2, N, Y):
    A_dict = C.A_dict
    A_dict_inv = C.A_dict_inv
    if C.is_complete():
        # if C is complete then it only needs to test
        # whether the relators in R2 are satisfied
        for w, alpha in product(R2, C.omega):
            if not C.scan_check(alpha, w):
                return
        # relators in R2 are satisfied, append the table to list
        S.append(C)
    else:
        # find the first undefined entry in Coset Table
        for alpha, x in product(range(len(C.table)), C.A):
            if C.table[alpha][A_dict[x]] is None:
                # this is "x" in pseudo-code (using "y" makes it clear)
                undefined_coset, undefined_gen = alpha, x
                break
        # for filling up the undefine entry we try all possible values
        # of beta in Omega or beta = n where beta^(undefined_gen^-1) is undefined
        reach = C.omega + [C.n]
        for beta in reach:
            if beta < N:
                if beta == C.n or C.table[beta][A_dict_inv[undefined_gen]] is None:
                    try_descendant(S, C, R1_c_list, R2, N, undefined_coset, \
                            undefined_gen, beta, Y)


def try_descendant(S, C, R1_c_list, R2, N, alpha, x, beta, Y):
    r"""
    Solves the problem of trying out each individual possibility
    for `\alpha^x.

    """
    D = C.copy()
    if beta == D.n and beta < N:
        D.table.append([None]*len(D.A))
        D.p.append(beta)
    D.table[alpha][D.A_dict[x]] = beta
    D.table[beta][D.A_dict_inv[x]] = alpha
    D.deduction_stack.append((alpha, x))
    if not D.process_deductions_check(R1_c_list[D.A_dict[x]], \
            R1_c_list[D.A_dict_inv[x]]):
        return
    for w in Y:
        if not D.scan_check(0, w):
            return
    if first_in_class(D, Y):
        descendant_subgroups(S, D, R1_c_list, x, R2, N, Y)


def first_in_class(C, Y=()):
    """
    Checks whether the subgroup ``H=G1`` corresponding to the Coset Table
    could possibly be the canonical representative of its conjugacy class.

    Parameters
    ==========

    C: CosetTable

    Returns
    =======

    bool: True/False

    If this returns False, then no descendant of C can have that property, and
    so we can abandon C. If it returns True, then we need to process further
    the node of the search tree corresponding to C, and so we call
    ``descendant_subgroups`` recursively on C.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, CosetTable, first_in_class
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**4])
    >>> C = CosetTable(f, [])
    >>> C.table = [[0, 0, None, None]]
    >>> first_in_class(C)
    True
    >>> C.table = [[1, 1, 1, None], [0, 0, None, 1]]; C.p = [0, 1]
    >>> first_in_class(C)
    True
    >>> C.table = [[1, 1, 2, 1], [0, 0, 0, None], [None, None, None, 0]]
    >>> C.p = [0, 1, 2]
    >>> first_in_class(C)
    False
    >>> C.table = [[1, 1, 1, 2], [0, 0, 2, 0], [2, None, 0, 1]]
    >>> first_in_class(C)
    False

    # TODO:: Sims points out in [Sim94] that performance can be improved by
    # remembering some of the information computed by ``first_in_class``. If
    # the ``continue alpha`` statement is executed at line 14, then the same thing
    # will happen for that value of alpha in any descendant of the table C, and so
    # the values the values of alpha for which this occurs could profitably be
    # stored and passed through to the descendants of C. Of course this would
    # make the code more complicated.

    # The code below is taken directly from the function on page 208 of [Sim94]
    # nu[alpha]

    """
    n = C.n
    # lamda is the largest numbered point in Omega_c_alpha which is currently defined
    lamda = -1
    # for alpha in Omega_c, nu[alpha] is the point in Omega_c_alpha corresponding to alpha
    nu = [None]*n
    # for alpha in Omega_c_alpha, mu[alpha] is the point in Omega_c corresponding to alpha
    mu = [None]*n
    # mutually nu and mu are the mutually-inverse equivalence maps between
    # Omega_c_alpha and Omega_c
    next_alpha = False
    # For each 0!=alpha in [0 .. nc-1], we start by constructing the equivalent
    # standardized coset table C_alpha corresponding to H_alpha
    for alpha in range(1, n):
        # reset nu to "None" after previous value of alpha
        for beta in range(lamda+1):
            nu[mu[beta]] = None
        # we only want to reject our current table in favour of a preceding
        # table in the ordering in which 1 is replaced by alpha, if the subgroup
        # G_alpha corresponding to this preceding table definitely contains the
        # given subgroup
        for w in Y:
            # TODO: this should support input of a list of general words
            # not just the words which are in "A" (i.e gen and gen^-1)
            if C.table[alpha][C.A_dict[w]] != alpha:
                # continue with alpha
                next_alpha = True
                break
        if next_alpha:
            next_alpha = False
            continue
        # try alpha as the new point 0 in Omega_C_alpha
        mu[0] = alpha
        nu[alpha] = 0
        # compare corresponding entries in C and C_alpha
        lamda = 0
        for beta in range(n):
            for x in C.A:
                gamma = C.table[beta][C.A_dict[x]]
                delta = C.table[mu[beta]][C.A_dict[x]]
                # if either of the entries is undefined,
                # we move with next alpha
                if gamma is None or delta is None:
                    # continue with alpha
                    next_alpha = True
                    break
                if nu[delta] is None:
                    # delta becomes the next point in Omega_C_alpha
                    lamda += 1
                    nu[delta] = lamda
                    mu[lamda] = delta
                if nu[delta] < gamma:
                    return False
                if nu[delta] > gamma:
                    # continue with alpha
                    next_alpha = True
                    break
            if next_alpha:
                next_alpha = False
                break
    return True

#========================================================================
#                    Simplifying Presentation
#========================================================================

def simplify_presentation(*args, change_gens=False):
    '''
    For an instance of `FpGroup`, return a simplified isomorphic copy of
    the group (e.g. remove redundant generators or relators). Alternatively,
    a list of generators and relators can be passed in which case the
    simplified lists will be returned.

    By default, the generators of the group are unchanged. If you would
    like to remove redundant generators, set the keyword argument
    `change_gens = True`.

    '''
    if len(args) == 1:
        if not isinstance(args[0], FpGroup):
            raise TypeError("The argument must be an instance of FpGroup")
        G = args[0]
        gens, rels = simplify_presentation(G.generators, G.relators,
                                              change_gens=change_gens)
        if gens:
            return FpGroup(gens[0].group, rels)
        return FpGroup(FreeGroup([]), [])
    elif len(args) == 2:
        gens, rels = args[0][:], args[1][:]
        if not gens:
            return gens, rels
        identity = gens[0].group.identity
    else:
        if len(args) == 0:
            m = "Not enough arguments"
        else:
            m = "Too many arguments"
        raise RuntimeError(m)

    prev_gens = []
    prev_rels = []
    while not set(prev_rels) == set(rels):
        prev_rels = rels
        while change_gens and not set(prev_gens) == set(gens):
            prev_gens = gens
            gens, rels = elimination_technique_1(gens, rels, identity)
        rels = _simplify_relators(rels)

    if change_gens:
        syms = [g.array_form[0][0] for g in gens]
        F = free_group(syms)[0]
        identity = F.identity
        gens = F.generators
        subs = dict(zip(syms, gens))
        for j, r in enumerate(rels):
            a = r.array_form
            rel = identity
            for sym, p in a:
                rel = rel*subs[sym]**p
            rels[j] = rel
    return gens, rels

def _simplify_relators(rels):
    """
    Simplifies a set of relators. All relators are checked to see if they are
    of the form `gen^n`. If any such relators are found then all other relators
    are processed for strings in the `gen` known order.

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import _simplify_relators
    >>> F, x, y = free_group("x, y")
    >>> w1 = [x**2*y**4, x**3]
    >>> _simplify_relators(w1)
    [x**3, x**-1*y**4]

    >>> w2 = [x**2*y**-4*x**5, x**3, x**2*y**8, y**5]
    >>> _simplify_relators(w2)
    [x**-1*y**-2, x**-1*y*x**-1, x**3, y**5]

    >>> w3 = [x**6*y**4, x**4]
    >>> _simplify_relators(w3)
    [x**4, x**2*y**4]

    >>> w4 = [x**2, x**5, y**3]
    >>> _simplify_relators(w4)
    [x, y**3]

    """
    rels = rels[:]

    if not rels:
        return []

    identity = rels[0].group.identity

    # build dictionary with "gen: n" where gen^n is one of the relators
    exps = {}
    for i in range(len(rels)):
        rel = rels[i]
        if rel.number_syllables() == 1:
            g = rel[0]
            exp = abs(rel.array_form[0][1])
            if rel.array_form[0][1] < 0:
                rels[i] = rels[i]**-1
                g = g**-1
            if g in exps:
                exp = gcd(exp, exps[g].array_form[0][1])
            exps[g] = g**exp

    one_syllables_words = list(exps.values())
    # decrease some of the exponents in relators, making use of the single
    # syllable relators
    for i, rel in enumerate(rels):
        if rel in one_syllables_words:
            continue
        rel = rel.eliminate_words(one_syllables_words, _all = True)
        # if rels[i] contains g**n where abs(n) is greater than half of the power p
        # of g in exps, g**n can be replaced by g**(n-p) (or g**(p-n) if n<0)
        for g in rel.contains_generators():
            if g in exps:
                exp = exps[g].array_form[0][1]
                max_exp = (exp + 1)//2
                rel = rel.eliminate_word(g**(max_exp), g**(max_exp-exp), _all = True)
                rel = rel.eliminate_word(g**(-max_exp), g**(-(max_exp-exp)), _all = True)
        rels[i] = rel

    rels = [r.identity_cyclic_reduction() for r in rels]

    rels += one_syllables_words # include one_syllable_words in the list of relators
    rels = list(set(rels)) # get unique values in rels
    rels.sort()

    # remove <identity> entries in rels
    try:
        rels.remove(identity)
    except ValueError:
        pass
    return rels

# Pg 350, section 2.5.1 from [2]
def elimination_technique_1(gens, rels, identity):
    rels = rels[:]
    # the shorter relators are examined first so that generators selected for
    # elimination will have shorter strings as equivalent
    rels.sort()
    gens = gens[:]
    redundant_gens = {}
    redundant_rels = []
    used_gens = set()
    # examine each relator in relator list for any generator occurring exactly
    # once
    for rel in rels:
        # don't look for a redundant generator in a relator which
        # depends on previously found ones
        contained_gens = rel.contains_generators()
        if any(g in contained_gens for g in redundant_gens):
            continue
        contained_gens = list(contained_gens)
        contained_gens.sort(reverse = True)
        for gen in contained_gens:
            if rel.generator_count(gen) == 1 and gen not in used_gens:
                k = rel.exponent_sum(gen)
                gen_index = rel.index(gen**k)
                bk = rel.subword(gen_index + 1, len(rel))
                fw = rel.subword(0, gen_index)
                chi = bk*fw
                redundant_gens[gen] = chi**(-1*k)
                used_gens.update(chi.contains_generators())
                redundant_rels.append(rel)
                break
    rels = [r for r in rels if r not in redundant_rels]
    # eliminate the redundant generators from remaining relators
    rels = [r.eliminate_words(redundant_gens, _all = True).identity_cyclic_reduction() for r in rels]
    rels = list(set(rels))
    try:
        rels.remove(identity)
    except ValueError:
        pass
    gens = [g for g in gens if g not in redundant_gens]
    return gens, rels

###############################################################################
#                           SUBGROUP PRESENTATIONS                            #
###############################################################################

# Pg 175 [1]
def define_schreier_generators(C, homomorphism=False):
    '''
    Parameters
    ==========

    C -- Coset table.
    homomorphism -- When set to True, return a dictionary containing the images
                     of the presentation generators in the original group.
    '''
    y = []
    gamma = 1
    f = C.fp_group
    X = f.generators
    if homomorphism:
        # `_gens` stores the elements of the parent group to
        # to which the schreier generators correspond to.
        _gens = {}
        # compute the schreier Traversal
        tau = {}
        tau[0] = f.identity
    C.P = [[None]*len(C.A) for i in range(C.n)]
    for alpha, x in product(C.omega, C.A):
        beta = C.table[alpha][C.A_dict[x]]
        if beta == gamma:
            C.P[alpha][C.A_dict[x]] = "<identity>"
            C.P[beta][C.A_dict_inv[x]] = "<identity>"
            gamma += 1
            if homomorphism:
                tau[beta] = tau[alpha]*x
        elif x in X and C.P[alpha][C.A_dict[x]] is None:
            y_alpha_x = '%s_%s' % (x, alpha)
            y.append(y_alpha_x)
            C.P[alpha][C.A_dict[x]] = y_alpha_x
            if homomorphism:
                _gens[y_alpha_x] = tau[alpha]*x*tau[beta]**-1
    grp_gens = list(free_group(', '.join(y)))
    C._schreier_free_group = grp_gens.pop(0)
    C._schreier_generators = grp_gens
    if homomorphism:
        C._schreier_gen_elem = _gens
    # replace all elements of P by, free group elements
    for i, j in product(range(len(C.P)), range(len(C.A))):
        # if equals "<identity>", replace by identity element
        if C.P[i][j] == "<identity>":
            C.P[i][j] = C._schreier_free_group.identity
        elif isinstance(C.P[i][j], str):
            r = C._schreier_generators[y.index(C.P[i][j])]
            C.P[i][j] = r
            beta = C.table[i][j]
            C.P[beta][j + 1] = r**-1

def reidemeister_relators(C):
    R = C.fp_group.relators
    rels = [rewrite(C, coset, word) for word in R for coset in range(C.n)]
    order_1_gens = {i for i in rels if len(i) == 1}

    # remove all the order 1 generators from relators
    rels = list(filter(lambda rel: rel not in order_1_gens, rels))

    # replace order 1 generators by identity element in reidemeister relators
    for i in range(len(rels)):
        w = rels[i]
        w = w.eliminate_words(order_1_gens, _all=True)
        rels[i] = w

    C._schreier_generators = [i for i in C._schreier_generators
                    if not (i in order_1_gens or i**-1 in order_1_gens)]

    # Tietze transformation 1 i.e TT_1
    # remove cyclic conjugate elements from relators
    i = 0
    while i < len(rels):
        w = rels[i]
        j = i + 1
        while j < len(rels):
            if w.is_cyclic_conjugate(rels[j]):
                del rels[j]
            else:
                j += 1
        i += 1

    C._reidemeister_relators = rels


def rewrite(C, alpha, w):
    """
    Parameters
    ==========

    C: CosetTable
    alpha: A live coset
    w: A word in `A*`

    Returns
    =======

    rho(tau(alpha), w)

    Examples
    ========

    >>> from sympy.combinatorics.fp_groups import FpGroup, CosetTable, define_schreier_generators, rewrite
    >>> from sympy.combinatorics import free_group
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**6])
    >>> C = CosetTable(f, [])
    >>> C.table = [[1, 1, 2, 3], [0, 0, 4, 5], [4, 4, 3, 0], [5, 5, 0, 2], [2, 2, 5, 1], [3, 3, 1, 4]]
    >>> C.p = [0, 1, 2, 3, 4, 5]
    >>> define_schreier_generators(C)
    >>> rewrite(C, 0, (x*y)**6)
    x_4*y_2*x_3*x_1*x_2*y_4*x_5

    """
    v = C._schreier_free_group.identity
    for i in range(len(w)):
        x_i = w[i]
        v = v*C.P[alpha][C.A_dict[x_i]]
        alpha = C.table[alpha][C.A_dict[x_i]]
    return v

# Pg 350, section 2.5.2 from [2]
def elimination_technique_2(C):
    """
    This technique eliminates one generator at a time. Heuristically this
    seems superior in that we may select for elimination the generator with
    shortest equivalent string at each stage.

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r, \
            reidemeister_relators, define_schreier_generators, elimination_technique_2
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**5, (x*y)**2]); H = [x*y, x**-1*y**-1*x*y*x]
    >>> C = coset_enumeration_r(f, H)
    >>> C.compress(); C.standardize()
    >>> define_schreier_generators(C)
    >>> reidemeister_relators(C)
    >>> elimination_technique_2(C)
    ([y_1, y_2], [y_2**-3, y_2*y_1*y_2*y_1*y_2*y_1, y_1**2])

    """
    rels = C._reidemeister_relators
    rels.sort(reverse=True)
    gens = C._schreier_generators
    for i in range(len(gens) - 1, -1, -1):
        rel = rels[i]
        for j in range(len(gens) - 1, -1, -1):
            gen = gens[j]
            if rel.generator_count(gen) == 1:
                k = rel.exponent_sum(gen)
                gen_index = rel.index(gen**k)
                bk = rel.subword(gen_index + 1, len(rel))
                fw = rel.subword(0, gen_index)
                rep_by = (bk*fw)**(-1*k)
                del rels[i]
                del gens[j]
                rels = [rel.eliminate_word(gen, rep_by) for rel in rels]
                break
    C._reidemeister_relators = rels
    C._schreier_generators = gens
    return C._schreier_generators, C._reidemeister_relators

def reidemeister_presentation(fp_grp, H, C=None, homomorphism=False):
    """
    Parameters
    ==========

    fp_group: A finitely presented group, an instance of FpGroup
    H: A subgroup whose presentation is to be found, given as a list
    of words in generators of `fp_grp`
    homomorphism: When set to True, return a homomorphism from the subgroup
                    to the parent group

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, reidemeister_presentation
    >>> F, x, y = free_group("x, y")

    Example 5.6 Pg. 177 from [1]
    >>> f = FpGroup(F, [x**3, y**5, (x*y)**2])
    >>> H = [x*y, x**-1*y**-1*x*y*x]
    >>> reidemeister_presentation(f, H)
    ((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))

    Example 5.8 Pg. 183 from [1]
    >>> f = FpGroup(F, [x**3, y**3, (x*y)**3])
    >>> H = [x*y, x*y**-1]
    >>> reidemeister_presentation(f, H)
    ((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))

    Exercises Q2. Pg 187 from [1]
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> H = [x]
    >>> reidemeister_presentation(f, H)
    ((x_0,), (x_0**4,))

    Example 5.9 Pg. 183 from [1]
    >>> f = FpGroup(F, [x**3*y**-3, (x*y)**3, (x*y**-1)**2])
    >>> H = [x]
    >>> reidemeister_presentation(f, H)
    ((x_0,), (x_0**6,))

    """
    if not C:
        C = coset_enumeration_r(fp_grp, H)
    C.compress(); C.standardize()
    define_schreier_generators(C, homomorphism=homomorphism)
    reidemeister_relators(C)
    gens, rels = C._schreier_generators, C._reidemeister_relators
    gens, rels = simplify_presentation(gens, rels, change_gens=True)

    C.schreier_generators = tuple(gens)
    C.reidemeister_relators = tuple(rels)

    if homomorphism:
        _gens = [C._schreier_gen_elem[str(gen)] for gen in gens]
        return C.schreier_generators, C.reidemeister_relators, _gens

    return C.schreier_generators, C.reidemeister_relators


FpGroupElement = FreeGroupElement
<|endoftext|>from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime

rmul = Permutation.rmul
_af_new = Permutation._af_new

############################################
#
# Utilities for computational group theory
#
############################################


def _base_ordering(base, degree):
    r"""
    Order `\{0, 1, \dots, n-1\}` so that base points come first and in order.

    Parameters
    ==========

    base : the base
    degree : the degree of the associated permutation group

    Returns
    =======

    A list ``base_ordering`` such that ``base_ordering[point]`` is the
    number of ``point`` in the ordering.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _base_ordering
    >>> S = SymmetricGroup(4)
    >>> S.schreier_sims()
    >>> _base_ordering(S.base, S.degree)
    [0, 1, 2, 3]

    Notes
    =====

    This is used in backtrack searches, when we define a relation `\ll` on
    the underlying set for a permutation group of degree `n`,
    `\{0, 1, \dots, n-1\}`, so that if `(b_1, b_2, \dots, b_k)` is a base we
    have `b_i \ll b_j` whenever `i<j` and `b_i \ll a` for all
    `i\in\{1,2, \dots, k\}` and `a` is not in the base. The idea is developed
    and applied to backtracking algorithms in [1], pp.108-132. The points
    that are not in the base are taken in increasing order.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    base_len = len(base)
    ordering = [0]*degree
    for i in range(base_len):
        ordering[base[i]] = i
    current = base_len
    for i in range(degree):
        if i not in base:
            ordering[i] = current
            current += 1
    return ordering


def _check_cycles_alt_sym(perm):
    """
    Checks for cycles of prime length p with n/2 < p < n-2.

    Explanation
    ===========

    Here `n` is the degree of the permutation. This is a helper function for
    the function is_alt_sym from sympy.combinatorics.perm_groups.

    Examples
    ========

    >>> from sympy.combinatorics.util import _check_cycles_alt_sym
    >>> from sympy.combinatorics import Permutation
    >>> a = Permutation([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    >>> _check_cycles_alt_sym(a)
    False
    >>> b = Permutation([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
    >>> _check_cycles_alt_sym(b)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.is_alt_sym

    """
    n = perm.size
    af = perm.array_form
    current_len = 0
    total_len = 0
    used = set()
    for i in range(n//2):
        if i not in used and i < n//2 - total_len:
            current_len = 1
            used.add(i)
            j = i
            while af[j] != i:
                current_len += 1
                j = af[j]
                used.add(j)
            total_len += current_len
            if current_len > n//2 and current_len < n - 2 and isprime(current_len):
                return True
    return False


def _distribute_gens_by_base(base, gens):
    r"""
    Distribute the group elements ``gens`` by membership in basic stabilizers.

    Explanation
    ===========

    Notice that for a base `(b_1, b_2, \dots, b_k)`, the basic stabilizers
    are defined as `G^{(i)} = G_{b_1, \dots, b_{i-1}}` for
    `i \in\{1, 2, \dots, k\}`.

    Parameters
    ==========

    base : a sequence of points in `\{0, 1, \dots, n-1\}`
    gens : a list of elements of a permutation group of degree `n`.

    Returns
    =======
    list
        List of length `k`, where `k` is the length of *base*. The `i`-th entry
        contains those elements in *gens* which fix the first `i` elements of
        *base* (so that the `0`-th entry is equal to *gens* itself). If no
        element fixes the first `i` elements of *base*, the `i`-th element is
        set to a list containing the identity element.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> D.strong_gens
    [(0 1 2), (0 2), (1 2)]
    >>> D.base
    [0, 1]
    >>> _distribute_gens_by_base(D.base, D.strong_gens)
    [[(0 1 2), (0 2), (1 2)],
     [(1 2)]]

    See Also
    ========

    _strong_gens_from_distr, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs

    """
    base_len = len(base)
    degree = gens[0].size
    stabs = [[] for _ in range(base_len)]
    max_stab_index = 0
    for gen in gens:
        j = 0
        while j < base_len - 1 and gen._array_form[base[j]] == base[j]:
            j += 1
        if j > max_stab_index:
            max_stab_index = j
        for k in range(j + 1):
            stabs[k].append(gen)
    for i in range(max_stab_index + 1, base_len):
        stabs[i].append(_af_new(list(range(degree))))
    return stabs


def _handle_precomputed_bsgs(base, strong_gens, transversals=None,
                             basic_orbits=None, strong_gens_distr=None):
    """
    Calculate BSGS-related structures from those present.

    Explanation
    ===========

    The base and strong generating set must be provided; if any of the
    transversals, basic orbits or distributed strong generators are not
    provided, they will be calculated from the base and strong generating set.

    Parameters
    ==========

    base : the base
    strong_gens : the strong generators
    transversals : basic transversals
    basic_orbits : basic orbits
    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Returns
    =======

    (transversals, basic_orbits, strong_gens_distr)
        where *transversals* are the basic transversals, *basic_orbits* are the
        basic orbits, and *strong_gens_distr* are the strong generators distributed
        by membership in basic stabilizers.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _handle_precomputed_bsgs
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> _handle_precomputed_bsgs(D.base, D.strong_gens,
    ... basic_orbits=D.basic_orbits)
    ([{0: (2), 1: (0 1 2), 2: (0 2)}, {1: (2), 2: (1 2)}], [[0, 1, 2], [1, 2]], [[(0 1 2), (0 2), (1 2)], [(1 2)]])

    See Also
    ========

    _orbits_transversals_from_bsgs, _distribute_gens_by_base

    """
    if strong_gens_distr is None:
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    if transversals is None:
        if basic_orbits is None:
            basic_orbits, transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr)
        else:
            transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr,
                                           transversals_only=True)
    else:
        if basic_orbits is None:
            base_len = len(base)
            basic_orbits = [None]*base_len
            for i in range(base_len):
                basic_orbits[i] = list(transversals[i].keys())
    return transversals, basic_orbits, strong_gens_distr


def _orbits_transversals_from_bsgs(base, strong_gens_distr,
                                   transversals_only=False, slp=False):
    """
    Compute basic orbits and transversals from a base and strong generating set.

    Explanation
    ===========

    The generators are provided as distributed across the basic stabilizers.
    If the optional argument ``transversals_only`` is set to True, only the
    transversals are returned.

    Parameters
    ==========

    base : The base.
    strong_gens_distr : Strong generators distributed by membership in basic stabilizers.
    transversals_only : bool, default: False
        A flag switching between returning only the
        transversals and both orbits and transversals.
    slp : bool, default: False
        If ``True``, return a list of dictionaries containing the
        generator presentations of the elements of the transversals,
        i.e. the list of indices of generators from ``strong_gens_distr[i]``
        such that their product is the relevant transversal element.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> (S.base, strong_gens_distr)
    ([0, 1], [[(0 1 2), (2)(0 1), (1 2)], [(1 2)]])

    See Also
    ========

    _distribute_gens_by_base, _handle_precomputed_bsgs

    """
    from sympy.combinatorics.perm_groups import _orbit_transversal
    base_len = len(base)
    degree = strong_gens_distr[0][0].size
    transversals = [None]*base_len
    slps = [None]*base_len
    if transversals_only is False:
        basic_orbits = [None]*base_len
    for i in range(base_len):
        transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i],
                                 base[i], pairs=True, slp=True)
        transversals[i] = dict(transversals[i])
        if transversals_only is False:
            basic_orbits[i] = list(transversals[i].keys())
    if transversals_only:
        return transversals
    else:
        if not slp:
            return basic_orbits, transversals
        return basic_orbits, transversals, slps


def _remove_gens(base, strong_gens, basic_orbits=None, strong_gens_distr=None):
    """
    Remove redundant generators from a strong generating set.

    Parameters
    ==========

    base : a base
    strong_gens : a strong generating set relative to *base*
    basic_orbits : basic orbits
    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Returns
    =======

    A strong generating set with respect to ``base`` which is a subset of
    ``strong_gens``.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _remove_gens
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> S = SymmetricGroup(15)
    >>> base, strong_gens = S.schreier_sims_incremental()
    >>> new_gens = _remove_gens(base, strong_gens)
    >>> len(new_gens)
    14
    >>> _verify_bsgs(S, base, new_gens)
    True

    Notes
    =====

    This procedure is outlined in [1],p.95.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    from sympy.combinatorics.perm_groups import _orbit
    base_len = len(base)
    degree = strong_gens[0].size
    if strong_gens_distr is None:
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    if basic_orbits is None:
        basic_orbits = []
        for i in range(base_len):
            basic_orbit = _orbit(degree, strong_gens_distr[i], base[i])
            basic_orbits.append(basic_orbit)
    strong_gens_distr.append([])
    res = strong_gens[:]
    for i in range(base_len - 1, -1, -1):
        gens_copy = strong_gens_distr[i][:]
        for gen in strong_gens_distr[i]:
            if gen not in strong_gens_distr[i + 1]:
                temp_gens = gens_copy[:]
                temp_gens.remove(gen)
                if temp_gens == []:
                    continue
                temp_orbit = _orbit(degree, temp_gens, base[i])
                if temp_orbit == basic_orbits[i]:
                    gens_copy.remove(gen)
                    res.remove(gen)
    return res


def _strip(g, base, orbits, transversals):
    """
    Attempt to decompose a permutation using a (possibly partial) BSGS
    structure.

    Explanation
    ===========

    This is done by treating the sequence ``base`` as an actual base, and
    the orbits ``orbits`` and transversals ``transversals`` as basic orbits and
    transversals relative to it.

    This process is called "sifting". A sift is unsuccessful when a certain
    orbit element is not found or when after the sift the decomposition
    does not end with the identity element.

    The argument ``transversals`` is a list of dictionaries that provides
    transversal elements for the orbits ``orbits``.

    Parameters
    ==========

    g : permutation to be decomposed
    base : sequence of points
    orbits : list
        A list in which the ``i``-th entry is an orbit of ``base[i]``
        under some subgroup of the pointwise stabilizer of `
        `base[0], base[1], ..., base[i - 1]``. The groups themselves are implicit
        in this function since the only information we need is encoded in the orbits
        and transversals
    transversals : list
        A list of orbit transversals associated with the orbits *orbits*.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, SymmetricGroup
    >>> from sympy.combinatorics.util import _strip
    >>> S = SymmetricGroup(5)
    >>> S.schreier_sims()
    >>> g = Permutation([0, 2, 3, 1, 4])
    >>> _strip(g, S.base, S.basic_orbits, S.basic_transversals)
    ((4), 5)

    Notes
    =====

    The algorithm is described in [1],pp.89-90. The reason for returning
    both the current state of the element being decomposed and the level
    at which the sifting ends is that they provide important information for
    the randomized version of the Schreier-Sims algorithm.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E."Handbook of computational group theory"

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims
    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims_random

    """
    h = g._array_form
    base_len = len(base)
    for i in range(base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in orbits[i]:
            return _af_new(h), i + 1
        u = transversals[i][beta]._array_form
        h = _af_rmul(_af_invert(u), h)
    return _af_new(h), base_len + 1


def _strip_af(h, base, orbits, transversals, j, slp=[], slps={}):
    """
    optimized _strip, with h, transversals and result in array form
    if the stripped elements is the identity, it returns False, base_len + 1

    j    h[base[i]] == base[i] for i <= j

    """
    base_len = len(base)
    for i in range(j+1, base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in orbits[i]:
            if not slp:
                return h, i + 1
            return h, i + 1, slp
        u = transversals[i][beta]
        if h == u:
            if not slp:
                return False, base_len + 1
            return False, base_len + 1, slp
        h = _af_rmul(_af_invert(u), h)
        if slp:
            u_slp = slps[i][beta][:]
            u_slp.reverse()
            u_slp = [(i, (g,)) for g in u_slp]
            slp = u_slp + slp
    if not slp:
        return h, base_len + 1
    return h, base_len + 1, slp


def _strong_gens_from_distr(strong_gens_distr):
    """
    Retrieve strong generating set from generators of basic stabilizers.

    This is just the union of the generators of the first and second basic
    stabilizers.

    Parameters
    ==========

    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
    ... _distribute_gens_by_base)
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> S.strong_gens
    [(0 1 2), (2)(0 1), (1 2)]
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> _strong_gens_from_distr(strong_gens_distr)
    [(0 1 2), (2)(0 1), (1 2)]

    See Also
    ========

    _distribute_gens_by_base

    """
    if len(strong_gens_distr) == 1:
        return strong_gens_distr[0][:]
    else:
        result = strong_gens_distr[0]
        for gen in strong_gens_distr[1]:
            if gen not in result:
                result.append(gen)
        return result
<|endoftext|>from sympy.combinatorics.permutations import Permutation, _af_rmul, \
    _af_invert, _af_new
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
    _orbit_transversal
from sympy.combinatorics.util import _distribute_gens_by_base, \
    _orbits_transversals_from_bsgs

"""
    References for tensor canonicalization:

    [1] R. Portugal "Algorithmic simplification of tensor expressions",
        J. Phys. A 32 (1999) 7779-7789

    [2] R. Portugal, B.F. Svaiter "Group-theoretic Approach for Symbolic
        Tensor Manipulation: I. Free Indices"
        arXiv:math-ph/0107031v1

    [3] L.R.U. Manssur, R. Portugal "Group-theoretic Approach for Symbolic
        Tensor Manipulation: II. Dummy Indices"
        arXiv:math-ph/0107032v1

    [4] xperm.c part of XPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
"""


def dummy_sgs(dummies, sym, n):
    """
    Return the strong generators for dummy indices.

    Parameters
    ==========

    dummies : List of dummy indices.
        `dummies[2k], dummies[2k+1]` are paired indices.
        In base form, the dummy indices are always in
        consecutive positions.
    sym : symmetry under interchange of contracted dummies::
        * None  no symmetry
        * 0     commuting
        * 1     anticommuting

    n : number of indices

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import dummy_sgs
    >>> dummy_sgs(list(range(2, 8)), 0, 8)
    [[0, 1, 3, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 7, 6, 8, 9], [0, 1, 4, 5, 2, 3, 6, 7, 8, 9],
     [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]]
    """
    if len(dummies) > n:
        raise ValueError("List too large")
    res = []
    # exchange of contravariant and covariant indices
    if sym is not None:
        for j in dummies[::2]:
            a = list(range(n + 2))
            if sym == 1:
                a[n] = n + 1
                a[n + 1] = n
            a[j], a[j + 1] = a[j + 1], a[j]
            res.append(a)
    # rename dummy indices
    for j in dummies[:-3:2]:
        a = list(range(n + 2))
        a[j:j + 4] = a[j + 2], a[j + 3], a[j], a[j + 1]
        res.append(a)
    return res


def _min_dummies(dummies, sym, indices):
    """
    Return list of minima of the orbits of indices in group of dummies.
    See ``double_coset_can_rep`` for the description of ``dummies`` and ``sym``.
    ``indices`` is the initial list of dummy indices.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import _min_dummies
    >>> _min_dummies([list(range(2, 8))], [0], list(range(10)))
    [0, 1, 2, 2, 2, 2, 2, 2, 8, 9]
    """
    num_types = len(sym)
    m = [min(dx) if dx else None for dx in dummies]
    res = indices[:]
    for i in range(num_types):
        for c, i in enumerate(indices):
            for j in range(num_types):
                if i in dummies[j]:
                    res[c] = m[j]
                    break
    return res


def _trace_S(s, j, b, S_cosets):
    """
    Return the representative h satisfying s[h[b]] == j

    If there is not such a representative return None
    """
    for h in S_cosets[b]:
        if s[h[b]] == j:
            return h
    return None


def _trace_D(gj, p_i, Dxtrav):
    """
    Return the representative h satisfying h[gj] == p_i

    If there is not such a representative return None
    """
    for h in Dxtrav:
        if h[gj] == p_i:
            return h
    return None


def _dumx_remove(dumx, dumx_flat, p0):
    """
    remove p0 from dumx
    """
    res = []
    for dx in dumx:
        if p0 not in dx:
            res.append(dx)
            continue
        k = dx.index(p0)
        if k % 2 == 0:
            p0_paired = dx[k + 1]
        else:
            p0_paired = dx[k - 1]
        dx.remove(p0)
        dx.remove(p0_paired)
        dumx_flat.remove(p0)
        dumx_flat.remove(p0_paired)
        res.append(dx)


def transversal2coset(size, base, transversal):
    a = []
    j = 0
    for i in range(size):
        if i in base:
            a.append(sorted(transversal[j].values()))
            j += 1
        else:
            a.append([list(range(size))])
    j = len(a) - 1
    while a[j] == [list(range(size))]:
        j -= 1
    return a[:j + 1]


def double_coset_can_rep(dummies, sym, b_S, sgens, S_transversals, g):
    r"""
    Butler-Portugal algorithm for tensor canonicalization with dummy indices.

    Parameters
    ==========

      dummies
        list of lists of dummy indices,
        one list for each type of index;
        the dummy indices are put in order contravariant, covariant
        [d0, -d0, d1, -d1, ...].

      sym
        list of the symmetries of the index metric for each type.

      possible symmetries of the metrics
              * 0     symmetric
              * 1     antisymmetric
              * None  no symmetry

      b_S
        base of a minimal slot symmetry BSGS.

      sgens
        generators of the slot symmetry BSGS.

      S_transversals
        transversals for the slot BSGS.

      g
        permutation representing the tensor.

    Returns
    =======

    Return 0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Notes
    =====

    A tensor with dummy indices can be represented in a number
    of equivalent ways which typically grows exponentially with
    the number of indices. To be able to establish if two tensors
    with many indices are equal becomes computationally very slow
    in absence of an efficient algorithm.

    The Butler-Portugal algorithm [3] is an efficient algorithm to
    put tensors in canonical form, solving the above problem.

    Portugal observed that a tensor can be represented by a permutation,
    and that the class of tensors equivalent to it under slot and dummy
    symmetries is equivalent to the double coset `D*g*S`
    (Note: in this documentation we use the conventions for multiplication
    of permutations p, q with (p*q)(i) = p[q[i]] which is opposite
    to the one used in the Permutation class)

    Using the algorithm by Butler to find a representative of the
    double coset one can find a canonical form for the tensor.

    To see this correspondence,
    let `g` be a permutation in array form; a tensor with indices `ind`
    (the indices including both the contravariant and the covariant ones)
    can be written as

    `t = T(ind[g[0]], \dots, ind[g[n-1]])`,

    where `n = len(ind)`;
    `g` has size `n + 2`, the last two indices for the sign of the tensor
    (trick introduced in [4]).

    A slot symmetry transformation `s` is a permutation acting on the slots
    `t \rightarrow T(ind[(g*s)[0]], \dots, ind[(g*s)[n-1]])`

    A dummy symmetry transformation acts on `ind`
    `t \rightarrow T(ind[(d*g)[0]], \dots, ind[(d*g)[n-1]])`

    Being interested only in the transformations of the tensor under
    these symmetries, one can represent the tensor by `g`, which transforms
    as

    `g -> d*g*s`, so it belongs to the coset `D*g*S`, or in other words
    to the set of all permutations allowed by the slot and dummy symmetries.

    Let us explain the conventions by an example.

    Given a tensor `T^{d3 d2 d1}{}_{d1 d2 d3}` with the slot symmetries
          `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

          `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    and symmetric metric, find the tensor equivalent to it which
    is the lowest under the ordering of indices:
    lexicographic ordering `d1, d2, d3` and then contravariant
    before covariant index; that is the canonical form of the tensor.

    The canonical form is `-T^{d1 d2 d3}{}_{d1 d2 d3}`
    obtained using `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`.

    To convert this problem in the input for this function,
    use the following ordering of the index names
    (- for covariant for short) `d1, -d1, d2, -d2, d3, -d3`

    `T^{d3 d2 d1}{}_{d1 d2 d3}` corresponds to `g = [4, 2, 0, 1, 3, 5, 6, 7]`
    where the last two indices are for the sign

    `sgens = [Permutation(0, 2)(6, 7), Permutation(0, 4)(6, 7)]`

    sgens[0] is the slot symmetry `-(0, 2)`
    `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

    sgens[1] is the slot symmetry `-(0, 4)`
    `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    The dummy symmetry group D is generated by the strong base generators
    `[(0, 1), (2, 3), (4, 5), (0, 2)(1, 3), (0, 4)(1, 5)]`
    where the first three interchange covariant and contravariant
    positions of the same index (d1 <-> -d1) and the last two interchange
    the dummy indices themselves (d1 <-> d2).

    The dummy symmetry acts from the left
    `d = [1, 0, 2, 3, 4, 5, 6, 7]`  exchange `d1 \leftrightarrow -d1`
    `T^{d3 d2 d1}{}_{d1 d2 d3} == T^{d3 d2}{}_{d1}{}^{d1}{}_{d2 d3}`

    `g=[4, 2, 0, 1, 3, 5, 6, 7]  -> [4, 2, 1, 0, 3, 5, 6, 7] = _af_rmul(d, g)`
    which differs from `_af_rmul(g, d)`.

    The slot symmetry acts from the right
    `s = [2, 1, 0, 3, 4, 5, 7, 6]`  exchanges slots 0 and 2 and changes sign
    `T^{d3 d2 d1}{}_{d1 d2 d3} == -T^{d1 d2 d3}{}_{d1 d2 d3}`

    `g=[4,2,0,1,3,5,6,7]  -> [0, 2, 4, 1, 3, 5, 7, 6] = _af_rmul(g, s)`

    Example in which the tensor is zero, same slot symmetries as above:
    `T^{d2}{}_{d1 d3}{}^{d1 d3}{}_{d2}`

    `= -T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`   under slot symmetry `-(0,4)`;

    `= T_{d3 d1}{}^{d3}{}^{d1 d2}{}_{d2}`    under slot symmetry `-(0,2)`;

    `= T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`    symmetric metric;

    `= 0`  since two of these lines have tensors differ only for the sign.

    The double coset D*g*S consists of permutations `h = d*g*s` corresponding
    to equivalent tensors; if there are two `h` which are the same apart
    from the sign, return zero; otherwise
    choose as representative the tensor with indices
    ordered lexicographically according to `[d1, -d1, d2, -d2, d3, -d3]`
    that is ``rep = min(D*g*S) = min([d*g*s for d in D for s in S])``

    The indices are fixed one by one; first choose the lowest index
    for slot 0, then the lowest remaining index for slot 1, etc.
    Doing this one obtains a chain of stabilizers

    `S \rightarrow S_{b0} \rightarrow S_{b0,b1} \rightarrow \dots` and
    `D \rightarrow D_{p0} \rightarrow D_{p0,p1} \rightarrow \dots`

    where ``[b0, b1, ...] = range(b)`` is a base of the symmetric group;
    the strong base `b_S` of S is an ordered sublist of it;
    therefore it is sufficient to compute once the
    strong base generators of S using the Schreier-Sims algorithm;
    the stabilizers of the strong base generators are the
    strong base generators of the stabilizer subgroup.

    ``dbase = [p0, p1, ...]`` is not in general in lexicographic order,
    so that one must recompute the strong base generators each time;
    however this is trivial, there is no need to use the Schreier-Sims
    algorithm for D.

    The algorithm keeps a TAB of elements `(s_i, d_i, h_i)`
    where `h_i = d_i \times g \times s_i` satisfying `h_i[j] = p_j` for `0 \le j < i`
    starting from `s_0 = id, d_0 = id, h_0 = g`.

    The equations `h_0[0] = p_0, h_1[1] = p_1, \dots` are solved in this order,
    choosing each time the lowest possible value of p_i

    For `j < i`
    `d_i*g*s_i*S_{b_0, \dots, b_{i-1}}*b_j = D_{p_0, \dots, p_{i-1}}*p_j`
    so that for dx in `D_{p_0,\dots,p_{i-1}}` and sx in
    `S_{base[0], \dots, base[i-1]}` one has `dx*d_i*g*s_i*sx*b_j = p_j`

    Search for dx, sx such that this equation holds for `j = i`;
    it can be written as `s_i*sx*b_j = J, dx*d_i*g*J = p_j`
    `sx*b_j = s_i**-1*J; sx = trace(s_i**-1, S_{b_0,...,b_{i-1}})`
    `dx**-1*p_j = d_i*g*J; dx = trace(d_i*g*J, D_{p_0,...,p_{i-1}})`

    `s_{i+1} = s_i*trace(s_i**-1*J, S_{b_0,...,b_{i-1}})`
    `d_{i+1} = trace(d_i*g*J, D_{p_0,...,p_{i-1}})**-1*d_i`
    `h_{i+1}*b_i = d_{i+1}*g*s_{i+1}*b_i = p_i`

    `h_n*b_j = p_j` for all j, so that `h_n` is the solution.

    Add the found `(s, d, h)` to TAB1.

    At the end of the iteration sort TAB1 with respect to the `h`;
    if there are two consecutive `h` in TAB1 which differ only for the
    sign, the tensor is zero, so return 0;
    if there are two consecutive `h` which are equal, keep only one.

    Then stabilize the slot generators under `i` and the dummy generators
    under `p_i`.

    Assign `TAB = TAB1` at the end of the iteration step.

    At the end `TAB` contains a unique `(s, d, h)`, since all the slots
    of the tensor `h` have been fixed to have the minimum value according
    to the symmetries. The algorithm returns `h`.

    It is important that the slot BSGS has lexicographic minimal base,
    otherwise there is an `i` which does not belong to the slot base
    for which `p_i` is fixed by the dummy symmetry only, while `i`
    is not invariant from the slot stabilizer, so `p_i` is not in
    general the minimal value.

    This algorithm differs slightly from the original algorithm [3]:
      the canonical form is minimal lexicographically, and
      the BSGS has minimal base under lexicographic order.
      Equal tensors `h` are eliminated from TAB.


    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.tensor_can import double_coset_can_rep, get_transversals
    >>> gens = [Permutation(x) for x in [[2, 1, 0, 3, 4, 5, 7, 6], [4, 1, 2, 3, 0, 5, 7, 6]]]
    >>> base = [0, 2]
    >>> g = Permutation([4, 2, 0, 1, 3, 5, 6, 7])
    >>> transversals = get_transversals(base, gens)
    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)
    [0, 1, 2, 3, 4, 5, 7, 6]

    >>> g = Permutation([4, 1, 3, 0, 5, 2, 6, 7])
    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)
    0
    """
    size = g.size
    g = g.array_form
    num_dummies = size - 2
    indices = list(range(num_dummies))
    all_metrics_with_sym = not any(_ is None for _ in sym)
    num_types = len(sym)
    dumx = dummies[:]
    dumx_flat = []
    for dx in dumx:
        dumx_flat.extend(dx)
    b_S = b_S[:]
    sgensx = [h._array_form for h in sgens]
    if b_S:
        S_transversals = transversal2coset(size, b_S, S_transversals)
    # strong generating set for D
    dsgsx = []
    for i in range(num_types):
        dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    idn = list(range(size))
    # TAB = list of entries (s, d, h) where h = _af_rmuln(d,g,s)
    # for short, in the following d*g*s means _af_rmuln(d,g,s)
    TAB = [(idn, idn, g)]
    for i in range(size - 2):
        b = i
        testb = b in b_S and sgensx
        if testb:
            sgensx1 = [_af_new(_) for _ in sgensx]
            deltab = _orbit(size, sgensx1, b)
        else:
            deltab = {b}
        # p1 = min(IMAGES) = min(Union D_p*h*deltab for h in TAB)
        if all_metrics_with_sym:
            md = _min_dummies(dumx, sym, indices)
        else:
            md = [min(_orbit(size, [_af_new(
                ddx) for ddx in dsgsx], ii)) for ii in range(size - 2)]

        p_i = min(min(md[h[x]] for x in deltab) for s, d, h in TAB)
        dsgsx1 = [_af_new(_) for _ in dsgsx]
        Dxtrav = _orbit_transversal(size, dsgsx1, p_i, False, af=True) \
            if dsgsx else None
        if Dxtrav:
            Dxtrav = [_af_invert(x) for x in Dxtrav]
        # compute the orbit of p_i
        for ii in range(num_types):
            if p_i in dumx[ii]:
                # the orbit is made by all the indices in dum[ii]
                if sym[ii] is not None:
                    deltap = dumx[ii]
                else:
                    # the orbit is made by all the even indices if p_i
                    # is even, by all the odd indices if p_i is odd
                    p_i_index = dumx[ii].index(p_i) % 2
                    deltap = dumx[ii][p_i_index::2]
                break
        else:
            deltap = [p_i]
        TAB1 = []
        while TAB:
            s, d, h = TAB.pop()
            if min(md[h[x]] for x in deltab) != p_i:
                continue
            deltab1 = [x for x in deltab if md[h[x]] == p_i]
            # NEXT = s*deltab1 intersection (d*g)**-1*deltap
            dg = _af_rmul(d, g)
            dginv = _af_invert(dg)
            sdeltab = [s[x] for x in deltab1]
            gdeltap = [dginv[x] for x in deltap]
            NEXT = [x for x in sdeltab if x in gdeltap]
            # d, s satisfy
            # d*g*s*base[i-1] = p_{i-1}; using the stabilizers
            # d*g*s*S_{base[0],...,base[i-1]}*base[i-1] =
            # D_{p_0,...,p_{i-1}}*p_{i-1}
            # so that to find d1, s1 satisfying d1*g*s1*b = p_i
            # one can look for dx in D_{p_0,...,p_{i-1}} and
            # sx in S_{base[0],...,base[i-1]}
            # d1 = dx*d; s1 = s*sx
            # d1*g*s1*b = dx*d*g*s*sx*b = p_i
            for j in NEXT:
                if testb:
                    # solve s1*b = j with s1 = s*sx for some element sx
                    # of the stabilizer of ..., base[i-1]
                    # sx*b = s**-1*j; sx = _trace_S(s, j,...)
                    # s1 = s*trace_S(s**-1*j,...)
                    s1 = _trace_S(s, j, b, S_transversals)
                    if not s1:
                        continue
                    else:
                        s1 = [s[ix] for ix in s1]
                else:
                    s1 = s
                # assert s1[b] == j  # invariant
                # solve d1*g*j = p_i with d1 = dx*d for some element dg
                # of the stabilizer of ..., p_{i-1}
                # dx**-1*p_i = d*g*j; dx**-1 = trace_D(d*g*j,...)
                # d1 = trace_D(d*g*j,...)**-1*d
                # to save an inversion in the inner loop; notice we did
                # Dxtrav = [perm_af_invert(x) for x in Dxtrav] out of the loop
                if Dxtrav:
                    d1 = _trace_D(dg[j], p_i, Dxtrav)
                    if not d1:
                        continue
                else:
                    if p_i != dg[j]:
                        continue
                    d1 = idn
                assert d1[dg[j]] == p_i  # invariant
                d1 = [d1[ix] for ix in d]
                h1 = [d1[g[ix]] for ix in s1]
                # assert h1[b] == p_i  # invariant
                TAB1.append((s1, d1, h1))

        # if TAB contains equal permutations, keep only one of them;
        # if TAB contains equal permutations up to the sign, return 0
        TAB1.sort(key=lambda x: x[-1])
        prev = [0] * size
        while TAB1:
            s, d, h = TAB1.pop()
            if h[:-2] == prev[:-2]:
                if h[-1] != prev[-1]:
                    return 0
            else:
                TAB.append((s, d, h))
            prev = h

        # stabilize the SGS
        sgensx = [h for h in sgensx if h[b] == b]
        if b in b_S:
            b_S.remove(b)
        _dumx_remove(dumx, dumx_flat, p_i)
        dsgsx = []
        for i in range(num_types):
            dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    return TAB[0][-1]


def canonical_free(base, gens, g, num_free):
    """
    Canonicalization of a tensor with respect to free indices
    choosing the minimum with respect to lexicographical ordering
    in the free indices.

    Explanation
    ===========

    ``base``, ``gens``  BSGS for slot permutation group
    ``g``               permutation representing the tensor
    ``num_free``        number of free indices
    The indices must be ordered with first the free indices

    See explanation in double_coset_can_rep
    The algorithm is a variation of the one given in [2].

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import canonical_free
    >>> gens = [[1, 0, 2, 3, 5, 4], [2, 3, 0, 1, 4, 5],[0, 1, 3, 2, 5, 4]]
    >>> gens = [Permutation(h) for h in gens]
    >>> base = [0, 2]
    >>> g = Permutation([2, 1, 0, 3, 4, 5])
    >>> canonical_free(base, gens, g, 4)
    [0, 3, 1, 2, 5, 4]

    Consider the product of Riemann tensors
    ``T = R^{a}_{d0}^{d1,d2}*R_{d2,d1}^{d0,b}``
    The order of the indices is ``[a, b, d0, -d0, d1, -d1, d2, -d2]``
    The permutation corresponding to the tensor is
    ``g = [0, 3, 4, 6, 7, 5, 2, 1, 8, 9]``

    In particular ``a`` is position ``0``, ``b`` is in position ``9``.
    Use the slot symmetries to get `T` is a form which is the minimal
    in lexicographic order in the free indices ``a`` and ``b``, e.g.
    ``-R^{a}_{d0}^{d1,d2}*R^{b,d0}_{d2,d1}`` corresponding to
    ``[0, 3, 4, 6, 1, 2, 7, 5, 9, 8]``

    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, tensor_gens
    >>> base, gens = riemann_bsgs
    >>> size, sbase, sgens = tensor_gens(base, gens, [[], []], 0)
    >>> g = Permutation([0, 3, 4, 6, 7, 5, 2, 1, 8, 9])
    >>> canonical_free(sbase, [Permutation(h) for h in sgens], g, 2)
    [0, 3, 4, 6, 1, 2, 7, 5, 9, 8]
    """
    g = g.array_form
    size = len(g)
    if not base:
        return g[:]

    transversals = get_transversals(base, gens)
    for x in sorted(g[:-2]):
        if x not in base:
            base.append(x)
    h = g
    for transv in transversals:
        h_i = [size]*num_free
        # find the element s in transversals[i] such that
        # _af_rmul(h, s) has its free elements with the lowest position in h
        s = None
        for sk in transv.values():
            h1 = _af_rmul(h, sk)
            hi = [h1.index(ix) for ix in range(num_free)]
            if hi < h_i:
                h_i = hi
                s = sk
        if s:
            h = _af_rmul(h, s)
    return h


def _get_map_slots(size, fixed_slots):
    res = list(range(size))
    pos = 0
    for i in range(size):
        if i in fixed_slots:
            continue
        res[i] = pos
        pos += 1
    return res


def _lift_sgens(size, fixed_slots, free, s):
    a = []
    j = k = 0
    fd = [y for _, y in sorted(zip(fixed_slots, free))]
    num_free = len(free)
    for i in range(size):
        if i in fixed_slots:
            a.append(fd[k])
            k += 1
        else:
            a.append(s[j] + num_free)
            j += 1
    return a


def canonicalize(g, dummies, msym, *v):
    """
    canonicalize tensor formed by tensors

    Parameters
    ==========

    g : permutation representing the tensor

    dummies : list representing the dummy indices
      it can be a list of dummy indices of the same type
      or a list of lists of dummy indices, one list for each
      type of index;
      the dummy indices must come after the free indices,
      and put in order contravariant, covariant
      [d0, -d0, d1,-d1,...]

    msym :  symmetry of the metric(s)
        it can be an integer or a list;
        in the first case it is the symmetry of the dummy index metric;
        in the second case it is the list of the symmetries of the
        index metric for each type

    v : list, (base_i, gens_i, n_i, sym_i) for tensors of type `i`

    base_i, gens_i : BSGS for tensors of this type.
        The BSGS should have minimal base under lexicographic ordering;
        if not, an attempt is made do get the minimal BSGS;
        in case of failure,
        canonicalize_naive is used, which is much slower.

    n_i :    number of tensors of type `i`.

    sym_i :  symmetry under exchange of component tensors of type `i`.

        Both for msym and sym_i the cases are
            * None  no symmetry
            * 0     commuting
            * 1     anticommuting

    Returns
    =======

    0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Algorithm
    =========

    First one uses canonical_free to get the minimum tensor under
    lexicographic order, using only the slot symmetries.
    If the component tensors have not minimal BSGS, it is attempted
    to find it; if the attempt fails canonicalize_naive
    is used instead.

    Compute the residual slot symmetry keeping fixed the free indices
    using tensor_gens(base, gens, list_free_indices, sym).

    Reduce the problem eliminating the free indices.

    Then use double_coset_can_rep and lift back the result reintroducing
    the free indices.

    Examples
    ========

    one type of index with commuting metric;

    `A_{a b}` and `B_{a b}` antisymmetric and commuting

    `T = A_{d0 d1} * B^{d0}{}_{d2} * B^{d2 d1}`

    `ord = [d0,-d0,d1,-d1,d2,-d2]` order of the indices

    g = [1, 3, 0, 5, 4, 2, 6, 7]

    `T_c = 0`

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize, bsgs_direct_product
    >>> from sympy.combinatorics import Permutation
    >>> base2a, gens2a = get_symmetric_group_sgs(2, 1)
    >>> t0 = (base2a, gens2a, 1, 0)
    >>> t1 = (base2a, gens2a, 2, 0)
    >>> g = Permutation([1, 3, 0, 5, 4, 2, 6, 7])
    >>> canonicalize(g, range(6), 0, t0, t1)
    0

    same as above, but with `B_{a b}` anticommuting

    `T_c = -A^{d0 d1} * B_{d0}{}^{d2} * B_{d1 d2}`

    can = [0,2,1,4,3,5,7,6]

    >>> t1 = (base2a, gens2a, 2, 1)
    >>> canonicalize(g, range(6), 0, t0, t1)
    [0, 2, 1, 4, 3, 5, 7, 6]

    two types of indices `[a,b,c,d,e,f]` and `[m,n]`, in this order,
    both with commuting metric

    `f^{a b c}` antisymmetric, commuting

    `A_{m a}` no symmetry, commuting

    `T = f^c{}_{d a} * f^f{}_{e b} * A_m{}^d * A^{m b} * A_n{}^a * A^{n e}`

    ord = [c,f,a,-a,b,-b,d,-d,e,-e,m,-m,n,-n]

    g = [0,7,3, 1,9,5, 11,6, 10,4, 13,2, 12,8, 14,15]

    The canonical tensor is
    `T_c = -f^{c a b} * f^{f d e} * A^m{}_a * A_{m d} * A^n{}_b * A_{n e}`

    can = [0,2,4, 1,6,8, 10,3, 11,7, 12,5, 13,9, 15,14]

    >>> base_f, gens_f = get_symmetric_group_sgs(3, 1)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    >>> t0 = (base_f, gens_f, 2, 0)
    >>> t1 = (base_A, gens_A, 4, 0)
    >>> dummies = [range(2, 10), range(10, 14)]
    >>> g = Permutation([0, 7, 3, 1, 9, 5, 11, 6, 10, 4, 13, 2, 12, 8, 14, 15])
    >>> canonicalize(g, dummies, [0, 0], t0, t1)
    [0, 2, 4, 1, 6, 8, 10, 3, 11, 7, 12, 5, 13, 9, 15, 14]
    """
    from sympy.combinatorics.testutil import canonicalize_naive
    if not isinstance(msym, list):
        if msym not in (0, 1, None):
            raise ValueError('msym must be 0, 1 or None')
        num_types = 1
    else:
        num_types = len(msym)
        if not all(msymx in (0, 1, None) for msymx in msym):
            raise ValueError('msym entries must be 0, 1 or None')
        if len(dummies) != num_types:
            raise ValueError(
                'dummies and msym must have the same number of elements')
    size = g.size
    num_tensors = 0
    v1 = []
    for base_i, gens_i, n_i, sym_i in v:
        # check that the BSGS is minimal;
        # this property is used in double_coset_can_rep;
        # if it is not minimal use canonicalize_naive
        if not _is_minimal_bsgs(base_i, gens_i):
            mbsgs = get_minimal_bsgs(base_i, gens_i)
            if not mbsgs:
                can = canonicalize_naive(g, dummies, msym, *v)
                return can
            base_i, gens_i = mbsgs
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
        num_tensors += n_i

    if num_types == 1 and not isinstance(msym, list):
        dummies = [dummies]
        msym = [msym]
    flat_dummies = []
    for dumx in dummies:
        flat_dummies.extend(dumx)

    if flat_dummies and flat_dummies != list(range(flat_dummies[0], flat_dummies[-1] + 1)):
        raise ValueError('dummies is not valid')

    # slot symmetry of the tensor
    size1, sbase, sgens = gens_products(*v1)
    if size != size1:
        raise ValueError(
            'g has size %d, generators have size %d' % (size, size1))
    free = [i for i in range(size - 2) if i not in flat_dummies]
    num_free = len(free)

    # g1 minimal tensor under slot symmetry
    g1 = canonical_free(sbase, sgens, g, num_free)
    if not flat_dummies:
        return g1
    # save the sign of g1
    sign = 0 if g1[-1] == size - 1 else 1

    # the free indices are kept fixed.
    # Determine free_i, the list of slots of tensors which are fixed
    # since they are occupied by free indices, which are fixed.
    start = 0
    for i, (base_i, gens_i, n_i, sym_i) in enumerate(v):
        free_i = []
        len_tens = gens_i[0].size - 2
        # for each component tensor get a list od fixed islots
        for j in range(n_i):
            # get the elements corresponding to the component tensor
            h = g1[start:(start + len_tens)]
            fr = []
            # get the positions of the fixed elements in h
            for k in free:
                if k in h:
                    fr.append(h.index(k))
            free_i.append(fr)
            start += len_tens
        v1[i] = (base_i, gens_i, free_i, sym_i)
    # BSGS of the tensor with fixed free indices
    # if tensor_gens fails in gens_product, use canonicalize_naive
    size, sbase, sgens = gens_products(*v1)

    # reduce the permutations getting rid of the free indices
    pos_free = [g1.index(x) for x in range(num_free)]
    size_red = size - num_free
    g1_red = [x - num_free for x in g1 if x in flat_dummies]
    if sign:
        g1_red.extend([size_red - 1, size_red - 2])
    else:
        g1_red.extend([size_red - 2, size_red - 1])
    map_slots = _get_map_slots(size, pos_free)
    sbase_red = [map_slots[i] for i in sbase if i not in pos_free]
    sgens_red = [_af_new([map_slots[i] for i in y._array_form if i not in pos_free]) for y in sgens]
    dummies_red = [[x - num_free for x in y] for y in dummies]
    transv_red = get_transversals(sbase_red, sgens_red)
    g1_red = _af_new(g1_red)
    g2 = double_coset_can_rep(
        dummies_red, msym, sbase_red, sgens_red, transv_red, g1_red)
    if g2 == 0:
        return 0
    # lift to the case with the free indices
    g3 = _lift_sgens(size, pos_free, free, g2)
    return g3


def perm_af_direct_product(gens1, gens2, signed=True):
    """
    Direct products of the generators gens1 and gens2.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import perm_af_direct_product
    >>> gens1 = [[1, 0, 2, 3], [0, 1, 3, 2]]
    >>> gens2 = [[1, 0]]
    >>> perm_af_direct_product(gens1, gens2, False)
    [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]
    >>> gens1 = [[1, 0, 2, 3, 5, 4], [0, 1, 3, 2, 4, 5]]
    >>> gens2 = [[1, 0, 2, 3]]
    >>> perm_af_direct_product(gens1, gens2, True)
    [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]
    """
    gens1 = [list(x) for x in gens1]
    gens2 = [list(x) for x in gens2]
    s = 2 if signed else 0
    n1 = len(gens1[0]) - s
    n2 = len(gens2[0]) - s
    start = list(range(n1))
    end = list(range(n1, n1 + n2))
    if signed:
        gens1 = [gen[:-2] + end + [gen[-2] + n2, gen[-1] + n2]
                 for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    else:
        gens1 = [gen + end for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]

    res = gens1 + gens2

    return res


def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
    """
    Direct product of two BSGS.

    Parameters
    ==========

    base1 : base of the first BSGS.

    gens1 : strong generating sequence of the first BSGS.

    base2, gens2 : similarly for the second BSGS.

    signed : flag for signed permutations.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> bsgs_direct_product(base1, gens1, base2, gens2)
    ([1], [(4)(1 2)])
    """
    s = 2 if signed else 0
    n1 = gens1[0].size - s
    base = list(base1)
    base += [x + n1 for x in base2]
    gens1 = [h._array_form for h in gens1]
    gens2 = [h._array_form for h in gens2]
    gens = perm_af_direct_product(gens1, gens2, signed)
    size = len(gens[0])
    id_af = list(range(size))
    gens = [h for h in gens if h != id_af]
    if not gens:
        gens = [id_af]
    return base, [_af_new(h) for h in gens]


def get_symmetric_group_sgs(n, antisym=False):
    """
    Return base, gens of the minimal BSGS for (anti)symmetric tensor

    Parameters
    ==========

    n : rank of the tensor
    antisym : bool
        ``antisym = False`` symmetric tensor
        ``antisym = True``  antisymmetric tensor

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
    >>> get_symmetric_group_sgs(3)
    ([0, 1], [(4)(0 1), (4)(1 2)])
    """
    if n == 1:
        return [], [_af_new(list(range(3)))]
    gens = [Permutation(n - 1)(i, i + 1)._array_form for i in range(n - 1)]
    if antisym == 0:
        gens = [x + [n, n + 1] for x in gens]
    else:
        gens = [x + [n + 1, n] for x in gens]
    base = list(range(n - 1))
    return base, [_af_new(h) for h in gens]

riemann_bsgs = [0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5),
                        Permutation(5)(0, 2)(1, 3)]


def get_transversals(base, gens):
    """
    Return transversals for the group with BSGS base, gens
    """
    if not base:
        return []
    stabs = _distribute_gens_by_base(base, gens)
    orbits, transversals = _orbits_transversals_from_bsgs(base, stabs)
    transversals = [{x: h._array_form for x, h in y.items()} for y in
                    transversals]
    return transversals


def _is_minimal_bsgs(base, gens):
    """
    Check if the BSGS has minimal base under lexigographic order.

    base, gens BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, _is_minimal_bsgs
    >>> _is_minimal_bsgs(*riemann_bsgs)
    True
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> _is_minimal_bsgs(*riemann_bsgs1)
    False
    """
    base1 = []
    sgs1 = gens[:]
    size = gens[0].size
    for i in range(size):
        if not all(h._array_form[i] == i for h in sgs1):
            base1.append(i)
            sgs1 = [h for h in sgs1 if h._array_form[i] == i]
    return base1 == base


def get_minimal_bsgs(base, gens):
    """
    Compute a minimal GSGS

    base, gens BSGS

    If base, gens is a minimal BSGS return it; else return a minimal BSGS
    if it fails in finding one, it returns None

    TODO: use baseswap in the case in which if it fails in finding a
    minimal BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> get_minimal_bsgs(*riemann_bsgs1)
    ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
    """
    G = PermutationGroup(gens)
    base, gens = G.schreier_sims_incremental()
    if not _is_minimal_bsgs(base, gens):
        return None
    return base, gens


def tensor_gens(base, gens, list_free_indices, sym=0):
    """
    Returns size, res_base, res_gens BSGS for n tensors of the
    same type.

    Explanation
    ===========

    base, gens BSGS for tensors of this type
    list_free_indices  list of the slots occupied by fixed indices
                       for each of the tensors

    sym symmetry under commutation of two tensors
    sym   None  no symmetry
    sym   0     commuting
    sym   1     anticommuting

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs

    two symmetric tensors with 3 indices without free indices

    >>> base, gens = get_symmetric_group_sgs(3)
    >>> tensor_gens(base, gens, [[], []])
    (8, [0, 1, 3, 4], [(7)(0 1), (7)(1 2), (7)(3 4), (7)(4 5), (7)(0 3)(1 4)(2 5)])

    two symmetric tensors with 3 indices with free indices in slot 1 and 0

    >>> tensor_gens(base, gens, [[1], [0]])
    (8, [0, 4], [(7)(0 2), (7)(4 5)])

    four symmetric tensors with 3 indices, two of which with free indices

    """
    def _get_bsgs(G, base, gens, free_indices):
        """
        return the BSGS for G.pointwise_stabilizer(free_indices)
        """
        if not free_indices:
            return base[:], gens[:]
        else:
            H = G.pointwise_stabilizer(free_indices)
            base, sgs = H.schreier_sims_incremental()
            return base, sgs

    # if not base there is no slot symmetry for the component tensors
    # if list_free_indices.count([]) < 2 there is no commutation symmetry
    # so there is no resulting slot symmetry
    if not base and list_free_indices.count([]) < 2:
        n = len(list_free_indices)
        size = gens[0].size
        size = n * (size - 2) + 2
        return size, [], [_af_new(list(range(size)))]

    # if any(list_free_indices) one needs to compute the pointwise
    # stabilizer, so G is needed
    if any(list_free_indices):
        G = PermutationGroup(gens)
    else:
        G = None

    # no_free list of lists of indices for component tensors without fixed
    # indices
    no_free = []
    size = gens[0].size
    id_af = list(range(size))
    num_indices = size - 2
    if not list_free_indices[0]:
        no_free.append(list(range(num_indices)))
    res_base, res_gens = _get_bsgs(G, base, gens, list_free_indices[0])
    for i in range(1, len(list_free_indices)):
        base1, gens1 = _get_bsgs(G, base, gens, list_free_indices[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens,
                                                 base1, gens1, 1)
        if not list_free_indices[i]:
            no_free.append(list(range(size - 2, size - 2 + num_indices)))
        size += num_indices
    nr = size - 2
    res_gens = [h for h in res_gens if h._array_form != id_af]
    # if sym there are no commuting tensors stop here
    if sym is None or not no_free:
        if not res_gens:
            res_gens = [_af_new(id_af)]
        return size, res_base, res_gens

    # if the component tensors have moinimal BSGS, so is their direct
    # product P; the slot symmetry group is S = P*C, where C is the group
    # to (anti)commute the component tensors with no free indices
    # a stabilizer has the property S_i = P_i*C_i;
    # the BSGS of P*C has SGS_P + SGS_C and the base is
    # the ordered union of the bases of P and C.
    # If P has minimal BSGS, so has S with this base.
    base_comm = []
    for i in range(len(no_free) - 1):
        ind1 = no_free[i]
        ind2 = no_free[i + 1]
        a = list(range(ind1[0]))
        a.extend(ind2)
        a.extend(ind1)
        base_comm.append(ind1[0])
        a.extend(list(range(ind2[-1] + 1, nr)))
        if sym == 0:
            a.extend([nr, nr + 1])
        else:
            a.extend([nr + 1, nr])
        res_gens.append(_af_new(a))
    res_base = list(res_base)
    # each base is ordered; order the union of the two bases
    for i in base_comm:
        if i not in res_base:
            res_base.append(i)
    res_base.sort()
    if not res_gens:
        res_gens = [_af_new(id_af)]

    return size, res_base, res_gens


def gens_products(*v):
    """
    Returns size, res_base, res_gens BSGS for n tensors of different types.

    Explanation
    ===========

    v is a sequence of (base_i, gens_i, free_i, sym_i)
    where
    base_i, gens_i  BSGS of tensor of type `i`
    free_i          list of the fixed slots for each of the tensors
                    of type `i`; if there are `n_i` tensors of type `i`
                    and none of them have fixed slots, `free = [[]]*n_i`
    sym   0 (1) if the tensors of type `i` (anti)commute among themselves

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
    >>> base, gens = get_symmetric_group_sgs(2)
    >>> gens_products((base, gens, [[], []], 0))
    (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
    >>> gens_products((base, gens, [[1], []], 0))
    (6, [2], [(5)(2 3)])
    """
    res_size, res_base, res_gens = tensor_gens(*v[0])
    for i in range(1, len(v)):
        size, base, gens = tensor_gens(*v[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens, base,
                                                 gens, 1)
    res_size = res_gens[0].size
    id_af = list(range(res_size))
    res_gens = [h for h in res_gens if h != id_af]
    if not res_gens:
        res_gens = [id_af]
    return res_size, res_base, res_gens
<|endoftext|>from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group


class PolycyclicGroup(DefaultPrinting):

    is_group = True
    is_solvable = True

    def __init__(self, pc_sequence, pc_series, relative_order, collector=None):
        """

        Parameters
        ==========

        pc_sequence : list
            A sequence of elements whose classes generate the cyclic factor
            groups of pc_series.
        pc_series : list
            A subnormal sequence of subgroups where each factor group is cyclic.
        relative_order : list
            The orders of factor groups of pc_series.
        collector : Collector
            By default, it is None. Collector class provides the
            polycyclic presentation with various other functionalities.

        """
        self.pcgs = pc_sequence
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.collector = Collector(self.pcgs, pc_series, relative_order) if not collector else collector

    def is_prime_order(self):
        return all(isprime(order) for order in self.relative_order)

    def length(self):
        return len(self.pcgs)


class Collector(DefaultPrinting):

    """
    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           Section 8.1.3
    """

    def __init__(self, pcgs, pc_series, relative_order, free_group_=None, pc_presentation=None):
        """

        Most of the parameters for the Collector class are the same as for PolycyclicGroup.
        Others are described below.

        Parameters
        ==========

        free_group_ : tuple
            free_group_ provides the mapping of polycyclic generating
            sequence with the free group elements.
        pc_presentation : dict
            Provides the presentation of polycyclic groups with the
            help of power and conjugate relators.

        See Also
        ========

        PolycyclicGroup

        """
        self.pcgs = pcgs
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.free_group = free_group('x:{}'.format(len(pcgs)))[0] if not free_group_ else free_group_
        self.index = {s: i for i, s in enumerate(self.free_group.symbols)}
        self.pc_presentation = self.pc_relators()

    def minimal_uncollected_subword(self, word):
        r"""
        Returns the minimal uncollected subwords.

        Explanation
        ===========

        A word ``v`` defined on generators in ``X`` is a minimal
        uncollected subword of the word ``w`` if ``v`` is a subword
        of ``w`` and it has one of the following form

        * `v = {x_{i+1}}^{a_j}x_i`

        * `v = {x_{i+1}}^{a_j}{x_i}^{-1}`

        * `v = {x_i}^{a_j}`

        for `a_j` not in `\{1, \ldots, s-1\}`. Where, ``s`` is the power
        exponent of the corresponding generator.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> collector.minimal_uncollected_subword(word)
        ((x2, 2),)

        """
        # To handle the case word = <identity>
        if not word:
            return None

        array = word.array_form
        re = self.relative_order
        index = self.index

        for i in range(len(array)):
            s1, e1 = array[i]

            if re[index[s1]] and (e1 < 0 or e1 > re[index[s1]]-1):
                return ((s1, e1), )

        for i in range(len(array)-1):
            s1, e1 = array[i]
            s2, e2 = array[i+1]

            if index[s1] > index[s2]:
                e = 1 if e2 > 0 else -1
                return ((s1, e1), (s2, e))

        return None

    def relations(self):
        """
        Separates the given relators of pc presentation in power and
        conjugate relations.

        Returns
        =======

        (power_rel, conj_rel)
            Separates pc presentation into power and conjugate relations.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> power_rel, conj_rel = collector.relations()
        >>> power_rel
        {x0**2: (), x1**3: ()}
        >>> conj_rel
        {x0**-1*x1*x0: x1**2}

        See Also
        ========

        pc_relators

        """
        power_relators = {}
        conjugate_relators = {}
        for key, value in self.pc_presentation.items():
            if len(key.array_form) == 1:
                power_relators[key] = value
            else:
                conjugate_relators[key] = value
        return power_relators, conjugate_relators

    def subword_index(self, word, w):
        """
        Returns the start and ending index of a given
        subword in a word.

        Parameters
        ==========

        word : FreeGroupElement
            word defined on free group elements for a
            polycyclic group.
        w : FreeGroupElement
            subword of a given word, whose starting and
            ending index to be computed.

        Returns
        =======

        (i, j)
            A tuple containing starting and ending index of ``w``
            in the given word. If not exists, (-1,-1) is returned.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> w = x2**2*x1
        >>> collector.subword_index(word, w)
        (0, 3)
        >>> w = x1**7
        >>> collector.subword_index(word, w)
        (2, 9)
        >>> w = x1**8
        >>> collector.subword_index(word, w)
        (-1, -1)

        """
        low = -1
        high = -1
        for i in range(len(word)-len(w)+1):
            if word.subword(i, i+len(w)) == w:
                low = i
                high = i+len(w)
                break
        return low, high

    def map_relation(self, w):
        """
        Return a conjugate relation.

        Explanation
        ===========

        Given a word formed by two free group elements, the
        corresponding conjugate relation with those free
        group elements is formed and mapped with the collected
        word in the polycyclic presentation.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1 = free_group("x0, x1")
        >>> w = x1*x0
        >>> collector.map_relation(w)
        x1**2

        See Also
        ========

        pc_presentation

        """
        array = w.array_form
        s1 = array[0][0]
        s2 = array[1][0]
        key = ((s2, -1), (s1, 1), (s2, 1))
        key = self.free_group.dtype(key)
        return self.pc_presentation[key]


    def collected_word(self, word):
        r"""
        Return the collected form of a word.

        Explanation
        ===========

        A word ``w`` is called collected, if `w = {x_{i_1}}^{a_1} * \ldots *
        {x_{i_r}}^{a_r}` with `i_1 < i_2< \ldots < i_r` and `a_j` is in
        `\{1, \ldots, {s_j}-1\}`.

        Otherwise w is uncollected.

        Parameters
        ==========

        word : FreeGroupElement
            An uncollected word.

        Returns
        =======

        word
            A collected word of form `w = {x_{i_1}}^{a_1}, \ldots,
            {x_{i_r}}^{a_r}` with `i_1, i_2, \ldots, i_r` and `a_j \in
            \{1, \ldots, {s_j}-1\}`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1, x2, x3 = free_group("x0, x1, x2, x3")
        >>> word = x3*x2*x1*x0
        >>> collected_word = collector.collected_word(word)
        >>> free_to_perm = {}
        >>> free_group = collector.free_group
        >>> for sym, gen in zip(free_group.symbols, collector.pcgs):
        ...     free_to_perm[sym] = gen
        >>> G1 = PermutationGroup()
        >>> for w in word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G1 = PermutationGroup([perm] + G1.generators)
        >>> G2 = PermutationGroup()
        >>> for w in collected_word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G2 = PermutationGroup([perm] + G2.generators)

        The two are not identical, but they are equivalent:

        >>> G1.equals(G2), G1 == G2
        (True, False)

        See Also
        ========

        minimal_uncollected_subword

        """
        free_group = self.free_group
        while True:
            w = self.minimal_uncollected_subword(word)
            if not w:
                break

            low, high = self.subword_index(word, free_group.dtype(w))
            if low == -1:
                continue

            s1, e1 = w[0]
            if len(w) == 1:
                re = self.relative_order[self.index[s1]]
                q = e1 // re
                r = e1-q*re

                key = ((w[0][0], re), )
                key = free_group.dtype(key)
                if self.pc_presentation[key]:
                    presentation = self.pc_presentation[key].array_form
                    sym, exp = presentation[0]
                    word_ = ((w[0][0], r), (sym, q*exp))
                    word_ = free_group.dtype(word_)
                else:
                    if r != 0:
                        word_ = ((w[0][0], r), )
                        word_ = free_group.dtype(word_)
                    else:
                        word_ = None
                word = word.eliminate_word(free_group.dtype(w), word_)

            if len(w) == 2 and w[1][1] > 0:
                s2, e2 = w[1]
                s2 = ((s2, 1), )
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2*word_**e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)

            elif len(w) == 2 and w[1][1] < 0:
                s2, e2 = w[1]
                s2 = ((s2, 1), )
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2**-1*word_**e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)

        return word


    def pc_relators(self):
        r"""
        Return the polycyclic presentation.

        Explanation
        ===========

        There are two types of relations used in polycyclic
        presentation.

        * Power relations : Power relators are of the form `x_i^{re_i}`,
          where `i \in \{0, \ldots, \mathrm{len(pcgs)}\}`, ``x`` represents polycyclic
          generator and ``re`` is the corresponding relative order.

        * Conjugate relations : Conjugate relators are of the form `x_j^-1x_ix_j`,
          where `j < i \in \{0, \ldots, \mathrm{len(pcgs)}\}`.

        Returns
        =======

        A dictionary with power and conjugate relations as key and
        their collected form as corresponding values.

        Notes
        =====

        Identity Permutation is mapped with empty ``()``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> S = SymmetricGroup(49).sylow_subgroup(7)
        >>> der = S.derived_series()
        >>> G = der[len(der)-2]
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> len(pcgs)
        6
        >>> free_group = collector.free_group
        >>> pc_resentation = collector.pc_presentation
        >>> free_to_perm = {}
        >>> for s, g in zip(free_group.symbols, pcgs):
        ...     free_to_perm[s] = g

        >>> for k, v in pc_resentation.items():
        ...     k_array = k.array_form
        ...     if v != ():
        ...        v_array = v.array_form
        ...     lhs = Permutation()
        ...     for gen in k_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         lhs = lhs*free_to_perm[s]**e
        ...     if v == ():
        ...         assert lhs.is_identity
        ...         continue
        ...     rhs = Permutation()
        ...     for gen in v_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         rhs = rhs*free_to_perm[s]**e
        ...     assert lhs == rhs

        """
        free_group = self.free_group
        rel_order = self.relative_order
        pc_relators = {}
        perm_to_free = {}
        pcgs = self.pcgs

        for gen, s in zip(pcgs, free_group.generators):
            perm_to_free[gen**-1] = s**-1
            perm_to_free[gen] = s

        pcgs = pcgs[::-1]
        series = self.pc_series[::-1]
        rel_order = rel_order[::-1]
        collected_gens = []

        for i, gen in enumerate(pcgs):
            re = rel_order[i]
            relation = perm_to_free[gen]**re
            G = series[i]

            l = G.generator_product(gen**re, original = True)
            l.reverse()

            word = free_group.identity
            for g in l:
                word = word*perm_to_free[g]

            word = self.collected_word(word)
            pc_relators[relation] = word if word else ()
            self.pc_presentation = pc_relators

            collected_gens.append(gen)
            if len(collected_gens) > 1:
                conj = collected_gens[len(collected_gens)-1]
                conjugator = perm_to_free[conj]

                for j in range(len(collected_gens)-1):
                    conjugated = perm_to_free[collected_gens[j]]

                    relation = conjugator**-1*conjugated*conjugator
                    gens = conj**-1*collected_gens[j]*conj

                    l = G.generator_product(gens, original = True)
                    l.reverse()
                    word = free_group.identity
                    for g in l:
                        word = word*perm_to_free[g]

                    word = self.collected_word(word)
                    pc_relators[relation] = word if word else ()
                    self.pc_presentation = pc_relators

        return pc_relators

    def exponent_vector(self, element):
        r"""
        Return the exponent vector of length equal to the
        length of polycyclic generating sequence.

        Explanation
        ===========

        For a given generator/element ``g`` of the polycyclic group,
        it can be represented as `g = {x_1}^{e_1}, \ldots, {x_n}^{e_n}`,
        where `x_i` represents polycyclic generators and ``n`` is
        the number of generators in the free_group equal to the length
        of pcgs.

        Parameters
        ==========

        element : Permutation
            Generator of a polycyclic group.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> collector.exponent_vector(G[0])
        [1, 0, 0, 0]
        >>> exp = collector.exponent_vector(G[1])
        >>> g = Permutation()
        >>> for i in range(len(exp)):
        ...     g = g*pcgs[i]**exp[i] if exp[i] else g
        >>> assert g == G[1]

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.4

        """
        free_group = self.free_group
        G = PermutationGroup()
        for g in self.pcgs:
            G = PermutationGroup([g] + G.generators)
        gens = G.generator_product(element, original = True)
        gens.reverse()

        perm_to_free = {}
        for sym, g in zip(free_group.generators, self.pcgs):
            perm_to_free[g**-1] = sym**-1
            perm_to_free[g] = sym
        w = free_group.identity
        for g in gens:
            w = w*perm_to_free[g]

        word = self.collected_word(w)

        index = self.index
        exp_vector = [0]*len(free_group)
        word = word.array_form
        for t in word:
            exp_vector[index[t[0]]] = t[1]
        return exp_vector

    def depth(self, element):
        r"""
        Return the depth of a given element.

        Explanation
        ===========

        The depth of a given element ``g`` is defined by
        `\mathrm{dep}[g] = i` if `e_1 = e_2 = \ldots = e_{i-1} = 0`
        and `e_i != 0`, where ``e`` represents the exponent-vector.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.depth(G[0])
        2
        >>> collector.depth(G[1])
        1

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.5

        """
        exp_vector = self.exponent_vector(element)
        return next((i+1 for i, x in enumerate(exp_vector) if x), len(self.pcgs)+1)

    def leading_exponent(self, element):
        r"""
        Return the leading non-zero exponent.

        Explanation
        ===========

        The leading exponent for a given element `g` is defined
        by `\mathrm{leading\_exponent}[g]` `= e_i`, if `\mathrm{depth}[g] = i`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.leading_exponent(G[1])
        1

        """
        exp_vector = self.exponent_vector(element)
        depth = self.depth(element)
        if depth != len(self.pcgs)+1:
            return exp_vector[depth-1]
        return None

    def _sift(self, z, g):
        h = g
        d = self.depth(h)
        while d < len(self.pcgs) and z[d-1] != 1:
            k = z[d-1]
            e = self.leading_exponent(h)*(self.leading_exponent(k))**-1
            e = e % self.relative_order[d-1]
            h = k**-e*h
            d = self.depth(h)
        return h

    def induced_pcgs(self, gens):
        """

        Parameters
        ==========

        gens : list
            A list of generators on which polycyclic subgroup
            is to be defined.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(8)
        >>> G = S.sylow_subgroup(2)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [2, 2, 2]
        >>> G = S.sylow_subgroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [3]

        """
        z = [1]*len(self.pcgs)
        G = gens
        while G:
            g = G.pop(0)
            h = self._sift(z, g)
            d = self.depth(h)
            if d < len(self.pcgs):
                for gen in z:
                    if gen != 1:
                        G.append(h**-1*gen**-1*h*gen)
                z[d-1] = h
        z = [gen for gen in z if gen != 1]
        return z

    def constructive_membership_test(self, ipcgs, g):
        """
        Return the exponent vector for induced pcgs.
        """
        e = [0]*len(ipcgs)
        h = g
        d = self.depth(h)
        for i, gen in enumerate(ipcgs):
            while self.depth(gen) == d:
                f = self.leading_exponent(h)*self.leading_exponent(gen)
                f = f % self.relative_order[d-1]
                h = gen**(-f)*h
                e[i] = f
                d = self.depth(h)
        if h == 1:
            return e
        return False
<|endoftext|>from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
from sympy.utilities.iterables import uniq

_af_new = Permutation._af_new


def DirectProduct(*groups):
    """
    Returns the direct product of several groups as a permutation group.

    Explanation
    ===========

    This is implemented much like the __mul__ procedure for taking the direct
    product of two permutation groups, but the idea of shifting the
    generators is realized in the case of an arbitrary number of groups.
    A call to DirectProduct(G1, G2, ..., Gn) is generally expected to be faster
    than a call to G1*G2*...*Gn (and thus the need for this algorithm).

    Examples
    ========

    >>> from sympy.combinatorics.group_constructs import DirectProduct
    >>> from sympy.combinatorics.named_groups import CyclicGroup
    >>> C = CyclicGroup(4)
    >>> G = DirectProduct(C, C, C)
    >>> G.order()
    64

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.__mul__

    """
    degrees = []
    gens_count = []
    total_degree = 0
    total_gens = 0
    for group in groups:
        current_deg = group.degree
        current_num_gens = len(group.generators)
        degrees.append(current_deg)
        total_degree += current_deg
        gens_count.append(current_num_gens)
        total_gens += current_num_gens
    array_gens = []
    for i in range(total_gens):
        array_gens.append(list(range(total_degree)))
    current_gen = 0
    current_deg = 0
    for i in range(len(gens_count)):
        for j in range(current_gen, current_gen + gens_count[i]):
            gen = ((groups[i].generators)[j - current_gen]).array_form
            array_gens[j][current_deg:current_deg + degrees[i]] = \
                [x + current_deg for x in gen]
        current_gen += gens_count[i]
        current_deg += degrees[i]
    perm_gens = list(uniq([_af_new(list(a)) for a in array_gens]))
    return PermutationGroup(perm_gens, dups=False)
<|endoftext|>from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int

from collections import defaultdict


class Prufer(Basic):
    """
    The Prufer correspondence is an algorithm that describes the
    bijection between labeled trees and the Prufer code. A Prufer
    code of a labeled tree is unique up to isomorphism and has
    a length of n - 2.

    Prufer sequences were first used by Heinz Prufer to give a
    proof of Cayley's formula.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/LabeledTree.html

    """
    _prufer_repr = None
    _tree_repr = None
    _nodes = None
    _rank = None

    @property
    def prufer_repr(self):
        """Returns Prufer sequence for the Prufer object.

        This sequence is found by removing the highest numbered vertex,
        recording the node it was attached to, and continuing until only
        two vertices remain. The Prufer sequence is the list of recorded nodes.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).prufer_repr
        [3, 3, 3, 4]
        >>> Prufer([1, 0, 0]).prufer_repr
        [1, 0, 0]

        See Also
        ========

        to_prufer

        """
        if self._prufer_repr is None:
            self._prufer_repr = self.to_prufer(self._tree_repr[:], self.nodes)
        return self._prufer_repr

    @property
    def tree_repr(self):
        """Returns the tree representation of the Prufer object.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).tree_repr
        [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]
        >>> Prufer([1, 0, 0]).tree_repr
        [[1, 2], [0, 1], [0, 3], [0, 4]]

        See Also
        ========

        to_tree

        """
        if self._tree_repr is None:
            self._tree_repr = self.to_tree(self._prufer_repr[:])
        return self._tree_repr

    @property
    def nodes(self):
        """Returns the number of nodes in the tree.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).nodes
        6
        >>> Prufer([1, 0, 0]).nodes
        5

        """
        return self._nodes

    @property
    def rank(self):
        """Returns the rank of the Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> p = Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]])
        >>> p.rank
        778
        >>> p.next(1).rank
        779
        >>> p.prev().rank
        777

        See Also
        ========

        prufer_rank, next, prev, size

        """
        if self._rank is None:
            self._rank = self.prufer_rank()
        return self._rank

    @property
    def size(self):
        """Return the number of possible trees of this Prufer object.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([0]*4).size == Prufer([6]*4).size == 1296
        True

        See Also
        ========

        prufer_rank, rank, next, prev

        """
        return self.prev(self.rank).prev().rank + 1

    @staticmethod
    def to_prufer(tree, n):
        """Return the Prufer sequence for a tree given as a list of edges where
        ``n`` is the number of nodes in the tree.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_repr
        [0, 0]
        >>> Prufer.to_prufer([[0, 1], [0, 2], [0, 3]], 4)
        [0, 0]

        See Also
        ========
        prufer_repr: returns Prufer sequence of a Prufer object.

        """
        d = defaultdict(int)
        L = []
        for edge in tree:
            # Increment the value of the corresponding
            # node in the degree list as we encounter an
            # edge involving it.
            d[edge[0]] += 1
            d[edge[1]] += 1
        for i in range(n - 2):
            # find the smallest leaf
            for x in range(n):
                if d[x] == 1:
                    break
            # find the node it was connected to
            y = None
            for edge in tree:
                if x == edge[0]:
                    y = edge[1]
                elif x == edge[1]:
                    y = edge[0]
                if y is not None:
                    break
            # record and update
            L.append(y)
            for j in (x, y):
                d[j] -= 1
                if not d[j]:
                    d.pop(j)
            tree.remove(edge)
        return L

    @staticmethod
    def to_tree(prufer):
        """Return the tree (as a list of edges) of the given Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([0, 2], 4)
        >>> a.tree_repr
        [[0, 1], [0, 2], [2, 3]]
        >>> Prufer.to_tree([0, 2])
        [[0, 1], [0, 2], [2, 3]]

        References
        ==========

        .. [1] https://hamberg.no/erlend/posts/2010-11-06-prufer-sequence-compact-tree-representation.html

        See Also
        ========
        tree_repr: returns tree representation of a Prufer object.

        """
        tree = []
        last = []
        n = len(prufer) + 2
        d = defaultdict(lambda: 1)
        for p in prufer:
            d[p] += 1
        for i in prufer:
            for j in range(n):
            # find the smallest leaf (degree = 1)
                if d[j] == 1:
                    break
            # (i, j) is the new edge that we append to the tree
            # and remove from the degree dictionary
            d[i] -= 1
            d[j] -= 1
            tree.append(sorted([i, j]))
        last = [i for i in range(n) if d[i] == 1] or [0, 1]
        tree.append(last)

        return tree

    @staticmethod
    def edges(*runs):
        """Return a list of edges and the number of nodes from the given runs
        that connect nodes in an integer-labelled tree.

        All node numbers will be shifted so that the minimum node is 0. It is
        not a problem if edges are repeated in the runs; only unique edges are
        returned. There is no assumption made about what the range of the node
        labels should be, but all nodes from the smallest through the largest
        must be present.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer.edges([1, 2, 3], [2, 4, 5]) # a T
        ([[0, 1], [1, 2], [1, 3], [3, 4]], 5)

        Duplicate edges are removed:

        >>> Prufer.edges([0, 1, 2, 3], [1, 4, 5], [1, 4, 6]) # a K
        ([[0, 1], [1, 2], [1, 4], [2, 3], [4, 5], [4, 6]], 7)

        """
        e = set()
        nmin = runs[0][0]
        for r in runs:
            for i in range(len(r) - 1):
                a, b = r[i: i + 2]
                if b < a:
                    a, b = b, a
                e.add((a, b))
        rv = []
        got = set()
        nmin = nmax = None
        for ei in e:
            got.update(ei)
            nmin = min(ei[0], nmin) if nmin is not None else ei[0]
            nmax = max(ei[1], nmax) if nmax is not None else ei[1]
            rv.append(list(ei))
        missing = set(range(nmin, nmax + 1)) - got
        if missing:
            missing = [i + nmin for i in missing]
            if len(missing) == 1:
                msg = 'Node %s is missing.' % missing.pop()
            else:
                msg = 'Nodes %s are missing.' % sorted(missing)
            raise ValueError(msg)
        if nmin != 0:
            for i, ei in enumerate(rv):
                rv[i] = [n - nmin for n in ei]
            nmax -= nmin
        return sorted(rv), nmax + 1

    def prufer_rank(self):
        """Computes the rank of a Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_rank()
        0

        See Also
        ========

        rank, next, prev, size

        """
        r = 0
        p = 1
        for i in range(self.nodes - 3, -1, -1):
            r += p*self.prufer_repr[i]
            p *= self.nodes
        return r

    @classmethod
    def unrank(self, rank, n):
        """Finds the unranked Prufer sequence.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer.unrank(0, 4)
        Prufer([0, 0])

        """
        n, rank = as_int(n), as_int(rank)
        L = defaultdict(int)
        for i in range(n - 3, -1, -1):
            L[i] = rank % n
            rank = (rank - L[i])//n
        return Prufer([L[i] for i in range(len(L))])

    def __new__(cls, *args, **kw_args):
        """The constructor for the Prufer object.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer

        A Prufer object can be constructed from a list of edges:

        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_repr
        [0, 0]

        If the number of nodes is given, no checking of the nodes will
        be performed; it will be assumed that nodes 0 through n - 1 are
        present:

        >>> Prufer([[0, 1], [0, 2], [0, 3]], 4)
        Prufer([[0, 1], [0, 2], [0, 3]], 4)

        A Prufer object can be constructed from a Prufer sequence:

        >>> b = Prufer([1, 3])
        >>> b.tree_repr
        [[0, 1], [1, 3], [2, 3]]

        """
        arg0 = Array(args[0]) if args[0] else Tuple()
        args = (arg0,) + tuple(_sympify(arg) for arg in args[1:])
        ret_obj = Basic.__new__(cls, *args, **kw_args)
        args = [list(args[0])]
        if args[0] and iterable(args[0][0]):
            if not args[0][0]:
                raise ValueError(
                    'Prufer expects at least one edge in the tree.')
            if len(args) > 1:
                nnodes = args[1]
            else:
                nodes = set(flatten(args[0]))
                nnodes = max(nodes) + 1
                if nnodes != len(nodes):
                    missing = set(range(nnodes)) - nodes
                    if len(missing) == 1:
                        msg = 'Node %s is missing.' % missing.pop()
                    else:
                        msg = 'Nodes %s are missing.' % sorted(missing)
                    raise ValueError(msg)
            ret_obj._tree_repr = [list(i) for i in args[0]]
            ret_obj._nodes = nnodes
        else:
            ret_obj._prufer_repr = args[0]
            ret_obj._nodes = len(ret_obj._prufer_repr) + 2
        return ret_obj

    def next(self, delta=1):
        """Generates the Prufer sequence that is delta beyond the current one.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> b = a.next(1) # == a.next()
        >>> b.tree_repr
        [[0, 2], [0, 1], [1, 3]]
        >>> b.rank
        1

        See Also
        ========

        prufer_rank, rank, prev, size

        """
        return Prufer.unrank(self.rank + delta, self.nodes)

    def prev(self, delta=1):
        """Generates the Prufer sequence that is -delta before the current one.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [1, 2], [2, 3], [1, 4]])
        >>> a.rank
        36
        >>> b = a.prev()
        >>> b
        Prufer([1, 2, 0])
        >>> b.rank
        35

        See Also
        ========

        prufer_rank, rank, next, size

        """
        return Prufer.unrank(self.rank -delta, self.nodes)
<|endoftext|>from sympy.combinatorics.permutations import Permutation, Cycle
from sympy.combinatorics.prufer import Prufer
from sympy.combinatorics.generators import cyclic, alternating, symmetric, dihedral
from sympy.combinatorics.subsets import Subset
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
    RGS_rank, RGS_unrank, RGS_enum)
from sympy.combinatorics.polyhedron import (Polyhedron, tetrahedron, cube,
    octahedron, dodecahedron, icosahedron)
from sympy.combinatorics.perm_groups import PermutationGroup, Coset, SymmetricPermutationGroup
from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.named_groups import (SymmetricGroup, DihedralGroup,
    CyclicGroup, AlternatingGroup, AbelianGroup, RubikGroup)
from sympy.combinatorics.pc_groups import PolycyclicGroup, Collector
from sympy.combinatorics.free_groups import free_group

__all__ = [
    'Permutation', 'Cycle',

    'Prufer',

    'cyclic', 'alternating', 'symmetric', 'dihedral',

    'Subset',

    'Partition', 'IntegerPartition', 'RGS_rank', 'RGS_unrank', 'RGS_enum',

    'Polyhedron', 'tetrahedron', 'cube', 'octahedron', 'dodecahedron',
    'icosahedron',

    'PermutationGroup', 'Coset', 'SymmetricPermutationGroup',

    'DirectProduct',

    'GrayCode',

    'SymmetricGroup', 'DihedralGroup', 'CyclicGroup', 'AlternatingGroup',
    'AbelianGroup', 'RubikGroup',

    'PolycyclicGroup', 'Collector',

    'free_group',
]
<|endoftext|>from sympy.core import Basic, Integer

import random


class GrayCode(Basic):
    """
    A Gray code is essentially a Hamiltonian walk on
    a n-dimensional cube with edge length of one.
    The vertices of the cube are represented by vectors
    whose values are binary. The Hamilton walk visits
    each vertex exactly once. The Gray code for a 3d
    cube is ['000','100','110','010','011','111','101',
    '001'].

    A Gray code solves the problem of sequentially
    generating all possible subsets of n objects in such
    a way that each subset is obtained from the previous
    one by either deleting or adding a single object.
    In the above example, 1 indicates that the object is
    present, and 0 indicates that its absent.

    Gray codes have applications in statistics as well when
    we want to compute various statistics related to subsets
    in an efficient manner.

    Examples
    ========

    >>> from sympy.combinatorics import GrayCode
    >>> a = GrayCode(3)
    >>> list(a.generate_gray())
    ['000', '001', '011', '010', '110', '111', '101', '100']
    >>> a = GrayCode(4)
    >>> list(a.generate_gray())
    ['0000', '0001', '0011', '0010', '0110', '0111', '0101', '0100', \
    '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']

    References
    ==========

    .. [1] Nijenhuis,A. and Wilf,H.S.(1978).
           Combinatorial Algorithms. Academic Press.
    .. [2] Knuth, D. (2011). The Art of Computer Programming, Vol 4
           Addison Wesley


    """

    _skip = False
    _current = 0
    _rank = None

    def __new__(cls, n, *args, **kw_args):
        """
        Default constructor.

        It takes a single argument ``n`` which gives the dimension of the Gray
        code. The starting Gray code string (``start``) or the starting ``rank``
        may also be given; the default is to start at rank = 0 ('0...0').

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> a
        GrayCode(3)
        >>> a.n
        3

        >>> a = GrayCode(3, start='100')
        >>> a.current
        '100'

        >>> a = GrayCode(4, rank=4)
        >>> a.current
        '0110'
        >>> a.rank
        4

        """
        if n < 1 or int(n) != n:
            raise ValueError(
                'Gray code dimension must be a positive integer, not %i' % n)
        n = Integer(n)
        args = (n,) + args
        obj = Basic.__new__(cls, *args)
        if 'start' in kw_args:
            obj._current = kw_args["start"]
            if len(obj._current) > n:
                raise ValueError('Gray code start has length %i but '
                'should not be greater than %i' % (len(obj._current), n))
        elif 'rank' in kw_args:
            if int(kw_args["rank"]) != kw_args["rank"]:
                raise ValueError('Gray code rank must be a positive integer, '
                'not %i' % kw_args["rank"])
            obj._rank = int(kw_args["rank"]) % obj.selections
            obj._current = obj.unrank(n, obj._rank)
        return obj

    def next(self, delta=1):
        """
        Returns the Gray code a distance ``delta`` (default = 1) from the
        current value in canonical order.


        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3, start='110')
        >>> a.next().current
        '111'
        >>> a.next(-1).current
        '010'
        """
        return GrayCode(self.n, rank=(self.rank + delta) % self.selections)

    @property
    def selections(self):
        """
        Returns the number of bit vectors in the Gray code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> a.selections
        8
        """
        return 2**self.n

    @property
    def n(self):
        """
        Returns the dimension of the Gray code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(5)
        >>> a.n
        5
        """
        return self.args[0]

    def generate_gray(self, **hints):
        """
        Generates the sequence of bit vectors of a Gray Code.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(start='011'))
        ['011', '010', '110', '111', '101', '100']
        >>> list(a.generate_gray(rank=4))
        ['110', '111', '101', '100']

        See Also
        ========

        skip

        References
        ==========

        .. [1] Knuth, D. (2011). The Art of Computer Programming,
               Vol 4, Addison Wesley

        """
        bits = self.n
        start = None
        if "start" in hints:
            start = hints["start"]
        elif "rank" in hints:
            start = GrayCode.unrank(self.n, hints["rank"])
        if start is not None:
            self._current = start
        current = self.current
        graycode_bin = gray_to_bin(current)
        if len(graycode_bin) > self.n:
            raise ValueError('Gray code start has length %i but should '
            'not be greater than %i' % (len(graycode_bin), bits))
        self._current = int(current, 2)
        graycode_int = int(''.join(graycode_bin), 2)
        for i in range(graycode_int, 1 << bits):
            if self._skip:
                self._skip = False
            else:
                yield self.current
            bbtc = (i ^ (i + 1))
            gbtc = (bbtc ^ (bbtc >> 1))
            self._current = (self._current ^ gbtc)
        self._current = 0

    def skip(self):
        """
        Skips the bit generation.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> for i in a.generate_gray():
        ...     if i == '010':
        ...         a.skip()
        ...     print(i)
        ...
        000
        001
        011
        010
        111
        101
        100

        See Also
        ========

        generate_gray
        """
        self._skip = True

    @property
    def rank(self):
        """
        Ranks the Gray code.

        A ranking algorithm determines the position (or rank)
        of a combinatorial object among all the objects w.r.t.
        a given order. For example, the 4 bit binary reflected
        Gray code (BRGC) '0101' has a rank of 6 as it appears in
        the 6th position in the canonical ordering of the family
        of 4 bit Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> a = GrayCode(3)
        >>> list(a.generate_gray())
        ['000', '001', '011', '010', '110', '111', '101', '100']
        >>> GrayCode(3, start='100').rank
        7
        >>> GrayCode(3, rank=7).current
        '100'

        See Also
        ========

        unrank

        References
        ==========

        .. [1] https://web.archive.org/web/20200224064753/http://statweb.stanford.edu/~susan/courses/s208/node12.html

        """
        if self._rank is None:
            self._rank = int(gray_to_bin(self.current), 2)
        return self._rank

    @property
    def current(self):
        """
        Returns the currently referenced Gray code as a bit string.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(3, start='100').current
        '100'
        """
        rv = self._current or '0'
        if not isinstance(rv, str):
            rv = bin(rv)[2:]
        return rv.rjust(self.n, '0')

    @classmethod
    def unrank(self, n, rank):
        """
        Unranks an n-bit sized Gray code of rank k. This method exists
        so that a derivative GrayCode class can define its own code of
        a given rank.

        The string here is generated in reverse order to allow for tail-call
        optimization.

        Examples
        ========

        >>> from sympy.combinatorics import GrayCode
        >>> GrayCode(5, rank=3).current
        '00010'
        >>> GrayCode.unrank(5, 3)
        '00010'

        See Also
        ========

        rank
        """
        def _unrank(k, n):
            if n == 1:
                return str(k % 2)
            m = 2**(n - 1)
            if k < m:
                return '0' + _unrank(k, n - 1)
            return '1' + _unrank(m - (k % m) - 1, n - 1)
        return _unrank(rank, n)


def random_bitstring(n):
    """
    Generates a random bitlist of length n.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import random_bitstring
    >>> random_bitstring(3) # doctest: +SKIP
    100
    """
    return ''.join([random.choice('01') for i in range(n)])


def gray_to_bin(bin_list):
    """
    Convert from Gray coding to binary coding.

    We assume big endian encoding.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import gray_to_bin
    >>> gray_to_bin('100')
    '111'

    See Also
    ========

    bin_to_gray
    """
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(b[i - 1] != bin_list[i]))
    return ''.join(b)


def bin_to_gray(bin_list):
    """
    Convert from binary coding to gray coding.

    We assume big endian encoding.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import bin_to_gray
    >>> bin_to_gray('111')
    '100'

    See Also
    ========

    gray_to_bin
    """
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(bin_list[i]) ^ int(bin_list[i - 1]))
    return ''.join(b)


def get_subset_from_bitstring(super_set, bitstring):
    """
    Gets the subset defined by the bitstring.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import get_subset_from_bitstring
    >>> get_subset_from_bitstring(['a', 'b', 'c', 'd'], '0011')
    ['c', 'd']
    >>> get_subset_from_bitstring(['c', 'a', 'c', 'c'], '1100')
    ['c', 'a']

    See Also
    ========

    graycode_subsets
    """
    if len(super_set) != len(bitstring):
        raise ValueError("The sizes of the lists are not equal")
    return [super_set[i] for i, j in enumerate(bitstring)
            if bitstring[i] == '1']


def graycode_subsets(gray_code_set):
    """
    Generates the subsets as enumerated by a Gray code.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import graycode_subsets
    >>> list(graycode_subsets(['a', 'b', 'c']))
    [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'], \
    ['a', 'c'], ['a']]
    >>> list(graycode_subsets(['a', 'b', 'c', 'c']))
    [[], ['c'], ['c', 'c'], ['c'], ['b', 'c'], ['b', 'c', 'c'], \
    ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'c'], \
    ['a', 'b', 'c'], ['a', 'c'], ['a', 'c', 'c'], ['a', 'c'], ['a']]

    See Also
    ========

    get_subset_from_bitstring
    """
    for bitstring in list(GrayCode(len(gray_code_set)).generate_gray()):
        yield get_subset_from_bitstring(gray_code_set, bitstring)
<|endoftext|>from sympy.core import Basic, Dict, sympify, Tuple
from sympy.core.numbers import Integer
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from sympy.utilities.misc import as_int


from collections import defaultdict


class Partition(FiniteSet):
    """
    This class represents an abstract partition.

    A partition is a set of disjoint sets whose union equals a given set.

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions
    """

    _rank = None
    _partition = None

    def __new__(cls, *partition):
        """
        Generates a new partition object.

        This method also verifies if the arguments passed are
        valid and raises a ValueError if they are not.

        Examples
        ========

        Creating Partition from Python lists:

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a
        Partition({3}, {1, 2})
        >>> a.partition
        [[1, 2], [3]]
        >>> len(a)
        2
        >>> a.members
        (1, 2, 3)

        Creating Partition from Python sets:

        >>> Partition({1, 2, 3}, {4, 5})
        Partition({4, 5}, {1, 2, 3})

        Creating Partition from SymPy finite sets:

        >>> from sympy import FiniteSet
        >>> a = FiniteSet(1, 2, 3)
        >>> b = FiniteSet(4, 5)
        >>> Partition(a, b)
        Partition({4, 5}, {1, 2, 3})
        """
        args = []
        dups = False
        for arg in partition:
            if isinstance(arg, list):
                as_set = set(arg)
                if len(as_set) < len(arg):
                    dups = True
                    break  # error below
                arg = as_set
            args.append(_sympify(arg))

        if not all(isinstance(part, FiniteSet) for part in args):
            raise ValueError(
                "Each argument to Partition should be " \
                "a list, set, or a FiniteSet")

        # sort so we have a canonical reference for RGS
        U = Union(*args)
        if dups or len(U) < sum(len(arg) for arg in args):
            raise ValueError("Partition contained duplicate elements.")

        obj = FiniteSet.__new__(cls, *args)
        obj.members = tuple(U)
        obj.size = len(U)
        return obj

    def sort_key(self, order=None):
        """Return a canonical key that can be used for sorting.

        Ordering is based on the size and sorted elements of the partition
        and ties are broken with the rank.

        Examples
        ========

        >>> from sympy import default_sort_key
        >>> from sympy.combinatorics import Partition
        >>> from sympy.abc import x
        >>> a = Partition([1, 2])
        >>> b = Partition([3, 4])
        >>> c = Partition([1, x])
        >>> d = Partition(list(range(4)))
        >>> l = [d, b, a + 1, a, c]
        >>> l.sort(key=default_sort_key); l
        [Partition({1, 2}), Partition({1}, {2}), Partition({1, x}), Partition({3, 4}), Partition({0, 1, 2, 3})]
        """
        if order is None:
            members = self.members
        else:
            members = tuple(sorted(self.members,
                             key=lambda w: default_sort_key(w, order)))
        return tuple(map(default_sort_key, (self.size, members, self.rank)))

    @property
    def partition(self):
        """Return partition as a sorted list of lists.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> Partition([1], [2, 3]).partition
        [[1], [2, 3]]
        """
        if self._partition is None:
            self._partition = sorted([sorted(p, key=default_sort_key)
                                      for p in self.args])
        return self._partition

    def __add__(self, other):
        """
        Return permutation whose rank is ``other`` greater than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a + 1).rank
        2
        >>> (a + 100).rank
        1
        """
        other = as_int(other)
        offset = self.rank + other
        result = RGS_unrank((offset) %
                            RGS_enum(self.size),
                            self.size)
        return Partition.from_rgs(result, self.members)

    def __sub__(self, other):
        """
        Return permutation whose rank is ``other`` less than current rank,
        (mod the maximum rank for the set).

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3])
        >>> a.rank
        1
        >>> (a - 1).rank
        0
        >>> (a - 100).rank
        1
        """
        return self.__add__(-other)

    def __le__(self, other):
        """
        Checks if a partition is less than or equal to
        the other based on rank.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3, 4, 5])
        >>> b = Partition([1], [2, 3], [4], [5])
        >>> a.rank, b.rank
        (9, 34)
        >>> a <= a
        True
        >>> a <= b
        True
        """
        return self.sort_key() <= sympify(other).sort_key()

    def __lt__(self, other):
        """
        Checks if a partition is less than the other.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3, 4, 5])
        >>> b = Partition([1], [2, 3], [4], [5])
        >>> a.rank, b.rank
        (9, 34)
        >>> a < b
        True
        """
        return self.sort_key() < sympify(other).sort_key()

    @property
    def rank(self):
        """
        Gets the rank of a partition.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.rank
        13
        """
        if self._rank is not None:
            return self._rank
        self._rank = RGS_rank(self.RGS)
        return self._rank

    @property
    def RGS(self):
        """
        Returns the "restricted growth string" of the partition.

        Explanation
        ===========

        The RGS is returned as a list of indices, L, where L[i] indicates
        the block in which element i appears. For example, in a partition
        of 3 elements (a, b, c) into 2 blocks ([c], [a, b]) the RGS is
        [1, 1, 0]: "a" is in block 1, "b" is in block 1 and "c" is in block 0.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> a = Partition([1, 2], [3], [4, 5])
        >>> a.members
        (1, 2, 3, 4, 5)
        >>> a.RGS
        (0, 0, 1, 2, 2)
        >>> a + 1
        Partition({3}, {4}, {5}, {1, 2})
        >>> _.RGS
        (0, 0, 1, 2, 3)
        """
        rgs = {}
        partition = self.partition
        for i, part in enumerate(partition):
            for j in part:
                rgs[j] = i
        return tuple([rgs[i] for i in sorted(
            [i for p in partition for i in p], key=default_sort_key)])

    @classmethod
    def from_rgs(self, rgs, elements):
        """
        Creates a set partition from a restricted growth string.

        Explanation
        ===========

        The indices given in rgs are assumed to be the index
        of the element as given in elements *as provided* (the
        elements are not sorted by this routine). Block numbering
        starts from 0. If any block was not referenced in ``rgs``
        an error will be raised.

        Examples
        ========

        >>> from sympy.combinatorics import Partition
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('abcde'))
        Partition({c}, {a, d}, {b, e})
        >>> Partition.from_rgs([0, 1, 2, 0, 1], list('cbead'))
        Partition({e}, {a, c}, {b, d})
        >>> a = Partition([1, 4], [2], [3, 5])
        >>> Partition.from_rgs(a.RGS, a.members)
        Partition({2}, {1, 4}, {3, 5})
        """
        if len(rgs) != len(elements):
            raise ValueError('mismatch in rgs and element lengths')
        max_elem = max(rgs) + 1
        partition = [[] for i in range(max_elem)]
        j = 0
        for i in rgs:
            partition[i].append(elements[j])
            j += 1
        if not all(p for p in partition):
            raise ValueError('some blocks of the partition were empty.')
        return Partition(*partition)


class IntegerPartition(Basic):
    """
    This class represents an integer partition.

    Explanation
    ===========

    In number theory and combinatorics, a partition of a positive integer,
    ``n``, also called an integer partition, is a way of writing ``n`` as a
    list of positive integers that sum to n. Two partitions that differ only
    in the order of summands are considered to be the same partition; if order
    matters then the partitions are referred to as compositions. For example,
    4 has five partitions: [4], [3, 1], [2, 2], [2, 1, 1], and [1, 1, 1, 1];
    the compositions [1, 2, 1] and [1, 1, 2] are the same as partition
    [2, 1, 1].

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    """

    _dict = None
    _keys = None

    def __new__(cls, partition, integer=None):
        """
        Generates a new IntegerPartition object from a list or dictionary.

        Explanation
        ===========

        The partition can be given as a list of positive integers or a
        dictionary of (integer, multiplicity) items. If the partition is
        preceded by an integer an error will be raised if the partition
        does not sum to that given integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([5, 4, 3, 1, 1])
        >>> a
        IntegerPartition(14, (5, 4, 3, 1, 1))
        >>> print(a)
        [5, 4, 3, 1, 1]
        >>> IntegerPartition({1:3, 2:1})
        IntegerPartition(5, (2, 1, 1, 1))

        If the value that the partition should sum to is given first, a check
        will be made to see n error will be raised if there is a discrepancy:

        >>> IntegerPartition(10, [5, 4, 3, 1])
        Traceback (most recent call last):
        ...
        ValueError: The partition is not valid

        """
        if integer is not None:
            integer, partition = partition, integer
        if isinstance(partition, (dict, Dict)):
            _ = []
            for k, v in sorted(partition.items(), reverse=True):
                if not v:
                    continue
                k, v = as_int(k), as_int(v)
                _.extend([k]*v)
            partition = tuple(_)
        else:
            partition = tuple(sorted(map(as_int, partition), reverse=True))
        sum_ok = False
        if integer is None:
            integer = sum(partition)
            sum_ok = True
        else:
            integer = as_int(integer)

        if not sum_ok and sum(partition) != integer:
            raise ValueError("Partition did not add to %s" % integer)
        if any(i < 1 for i in partition):
            raise ValueError("All integer summands must be greater than one")

        obj = Basic.__new__(cls, Integer(integer), Tuple(*partition))
        obj.partition = list(partition)
        obj.integer = integer
        return obj

    def prev_lex(self):
        """Return the previous partition of the integer, n, in lexical order,
        wrapping around to [1, ..., 1] if the partition is [n].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([4])
        >>> print(p.prev_lex())
        [3, 1]
        >>> p.partition > p.prev_lex().partition
        True
        """
        d = defaultdict(int)
        d.update(self.as_dict())
        keys = self._keys
        if keys == [1]:
            return IntegerPartition({self.integer: 1})
        if keys[-1] != 1:
            d[keys[-1]] -= 1
            if keys[-1] == 2:
                d[1] = 2
            else:
                d[keys[-1] - 1] = d[1] = 1
        else:
            d[keys[-2]] -= 1
            left = d[1] + keys[-2]
            new = keys[-2]
            d[1] = 0
            while left:
                new -= 1
                if left - new >= 0:
                    d[new] += left//new
                    left -= d[new]*new
        return IntegerPartition(self.integer, d)

    def next_lex(self):
        """Return the next partition of the integer, n, in lexical order,
        wrapping around to [n] if the partition is [1, ..., 1].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([3, 1])
        >>> print(p.next_lex())
        [4]
        >>> p.partition < p.next_lex().partition
        True
        """
        d = defaultdict(int)
        d.update(self.as_dict())
        key = self._keys
        a = key[-1]
        if a == self.integer:
            d.clear()
            d[1] = self.integer
        elif a == 1:
            if d[a] > 1:
                d[a + 1] += 1
                d[a] -= 2
            else:
                b = key[-2]
                d[b + 1] += 1
                d[1] = (d[b] - 1)*b
                d[b] = 0
        else:
            if d[a] > 1:
                if len(key) == 1:
                    d.clear()
                    d[a + 1] = 1
                    d[1] = self.integer - a - 1
                else:
                    a1 = a + 1
                    d[a1] += 1
                    d[1] = d[a]*a - a1
                    d[a] = 0
            else:
                b = key[-2]
                b1 = b + 1
                d[b1] += 1
                need = d[b]*b + d[a]*a - b1
                d[a] = d[b] = 0
                d[1] = need
        return IntegerPartition(self.integer, d)

    def as_dict(self):
        """Return the partition as a dictionary whose keys are the
        partition integers and the values are the multiplicity of that
        integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> IntegerPartition([1]*3 + [2] + [3]*4).as_dict()
        {1: 3, 2: 1, 3: 4}
        """
        if self._dict is None:
            groups = group(self.partition, multiple=False)
            self._keys = [g[0] for g in groups]
            self._dict = dict(groups)
        return self._dict

    @property
    def conjugate(self):
        """
        Computes the conjugate partition of itself.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([6, 3, 3, 2, 1])
        >>> a.conjugate
        [5, 4, 3, 1, 1, 1]
        """
        j = 1
        temp_arr = list(self.partition) + [0]
        k = temp_arr[0]
        b = [0]*k
        while k > 0:
            while k > temp_arr[j]:
                b[k - 1] = j
                k -= 1
            j += 1
        return b

    def __lt__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([3, 1])
        >>> a < a
        False
        >>> b = a.next_lex()
        >>> a < b
        True
        >>> a == b
        False
        """
        return list(reversed(self.partition)) < list(reversed(other.partition))

    def __le__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([4])
        >>> a <= a
        True
        """
        return list(reversed(self.partition)) <= list(reversed(other.partition))

    def as_ferrers(self, char='#'):
        """
        Prints the ferrer diagram of a partition.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> print(IntegerPartition([1, 1, 5]).as_ferrers())
        #####
        #
        #
        """
        return "\n".join([char*i for i in self.partition])

    def __str__(self):
        return str(list(self.partition))


def random_integer_partition(n, seed=None):
    """
    Generates a random integer partition summing to ``n`` as a list
    of reverse-sorted integers.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import random_integer_partition

    For the following, a seed is given so a known value can be shown; in
    practice, the seed would not be given.

    >>> random_integer_partition(100, seed=[1, 1, 12, 1, 2, 1, 85, 1])
    [85, 12, 2, 1]
    >>> random_integer_partition(10, seed=[1, 2, 3, 1, 5, 1])
    [5, 3, 1, 1]
    >>> random_integer_partition(1)
    [1]
    """
    from sympy.core.random import _randint

    n = as_int(n)
    if n < 1:
        raise ValueError('n must be a positive integer')

    randint = _randint(seed)

    partition = []
    while (n > 0):
        k = randint(1, n)
        mult = randint(1, n//k)
        partition.append((k, mult))
        n -= k*mult
    partition.sort(reverse=True)
    partition = flatten([[k]*m for k, m in partition])
    return partition


def RGS_generalized(m):
    """
    Computes the m + 1 generalized unrestricted growth strings
    and returns them as rows in matrix.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_generalized
    >>> RGS_generalized(6)
    Matrix([
    [  1,   1,   1,  1,  1, 1, 1],
    [  1,   2,   3,  4,  5, 6, 0],
    [  2,   5,  10, 17, 26, 0, 0],
    [  5,  15,  37, 77,  0, 0, 0],
    [ 15,  52, 151,  0,  0, 0, 0],
    [ 52, 203,   0,  0,  0, 0, 0],
    [203,   0,   0,  0,  0, 0, 0]])
    """
    d = zeros(m + 1)
    for i in range(m + 1):
        d[0, i] = 1

    for i in range(1, m + 1):
        for j in range(m):
            if j <= m - i:
                d[i, j] = j * d[i - 1, j] + d[i - 1, j + 1]
            else:
                d[i, j] = 0
    return d


def RGS_enum(m):
    """
    RGS_enum computes the total number of restricted growth strings
    possible for a superset of size m.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_enum
    >>> from sympy.combinatorics import Partition
    >>> RGS_enum(4)
    15
    >>> RGS_enum(5)
    52
    >>> RGS_enum(6)
    203

    We can check that the enumeration is correct by actually generating
    the partitions. Here, the 15 partitions of 4 items are generated:

    >>> a = Partition(list(range(4)))
    >>> s = set()
    >>> for i in range(20):
    ...     s.add(a)
    ...     a += 1
    ...
    >>> assert len(s) == 15

    """
    if (m < 1):
        return 0
    elif (m == 1):
        return 1
    else:
        return bell(m)


def RGS_unrank(rank, m):
    """
    Gives the unranked restricted growth string for a given
    superset size.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_unrank
    >>> RGS_unrank(14, 4)
    [0, 1, 2, 3]
    >>> RGS_unrank(0, 4)
    [0, 0, 0, 0]
    """
    if m < 1:
        raise ValueError("The superset size must be >= 1")
    if rank < 0 or RGS_enum(m) <= rank:
        raise ValueError("Invalid arguments")

    L = [1] * (m + 1)
    j = 1
    D = RGS_generalized(m)
    for i in range(2, m + 1):
        v = D[m - i, j]
        cr = j*v
        if cr <= rank:
            L[i] = j + 1
            rank -= cr
            j += 1
        else:
            L[i] = int(rank / v + 1)
            rank %= v
    return [x - 1 for x in L[1:]]


def RGS_rank(rgs):
    """
    Computes the rank of a restricted growth string.

    Examples
    ========

    >>> from sympy.combinatorics.partitions import RGS_rank, RGS_unrank
    >>> RGS_rank([0, 1, 2, 1, 3])
    42
    >>> RGS_rank(RGS_unrank(4, 7))
    4
    """
    rgs_size = len(rgs)
    rank = 0
    D = RGS_generalized(rgs_size)
    for i in range(1, rgs_size):
        n = len(rgs[(i + 1):])
        m = max(rgs[0:i])
        rank += D[n, m + 1] * rgs[i]
    return rank
<|endoftext|>import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.intfunc import igcd
from sympy.functions.combinatorial.numbers import totient
from sympy.core.singleton import S

class GroupHomomorphism:
    '''
    A class representing group homomorphisms. Instantiate using `homomorphism()`.

    References
    ==========

    .. [1] Holt, D., Eick, B. and O'Brien, E. (2005). Handbook of computational group theory.

    '''

    def __init__(self, domain, codomain, images):
        self.domain = domain
        self.codomain = codomain
        self.images = images
        self._inverses = None
        self._kernel = None
        self._image = None

    def _invs(self):
        '''
        Return a dictionary with `{gen: inverse}` where `gen` is a rewriting
        generator of `codomain` (e.g. strong generator for permutation groups)
        and `inverse` is an element of its preimage

        '''
        image = self.image()
        inverses = {}
        for k in list(self.images.keys()):
            v = self.images[k]
            if not (v in inverses
                    or v.is_identity):
                inverses[v] = k
        if isinstance(self.codomain, PermutationGroup):
            gens = image.strong_gens
        else:
            gens = image.generators
        for g in gens:
            if g in inverses or g.is_identity:
                continue
            w = self.domain.identity
            if isinstance(self.codomain, PermutationGroup):
                parts = image._strong_gens_slp[g][::-1]
            else:
                parts = g
            for s in parts:
                if s in inverses:
                    w = w*inverses[s]
                else:
                    w = w*inverses[s**-1]**-1
            inverses[g] = w

        return inverses

    def invert(self, g):
        '''
        Return an element of the preimage of ``g`` or of each element
        of ``g`` if ``g`` is a list.

        Explanation
        ===========

        If the codomain is an FpGroup, the inverse for equal
        elements might not always be the same unless the FpGroup's
        rewriting system is confluent. However, making a system
        confluent can be time-consuming. If it's important, try
        `self.codomain.make_confluent()` first.

        '''
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.free_groups import FreeGroupElement
        if isinstance(g, (Permutation, FreeGroupElement)):
            if isinstance(self.codomain, FpGroup):
                g = self.codomain.reduce(g)
            if self._inverses is None:
                self._inverses = self._invs()
            image = self.image()
            w = self.domain.identity
            if isinstance(self.codomain, PermutationGroup):
                gens = image.generator_product(g)[::-1]
            else:
                gens = g
            # the following can't be "for s in gens:"
            # because that would be equivalent to
            # "for s in gens.array_form:" when g is
            # a FreeGroupElement. On the other hand,
            # when you call gens by index, the generator
            # (or inverse) at position i is returned.
            for i in range(len(gens)):
                s = gens[i]
                if s.is_identity:
                    continue
                if s in self._inverses:
                    w = w*self._inverses[s]
                else:
                    w = w*self._inverses[s**-1]**-1
            return w
        elif isinstance(g, list):
            return [self.invert(e) for e in g]

    def kernel(self):
        '''
        Compute the kernel of `self`.

        '''
        if self._kernel is None:
            self._kernel = self._compute_kernel()
        return self._kernel

    def _compute_kernel(self):
        G = self.domain
        G_order = G.order()
        if G_order is S.Infinity:
            raise NotImplementedError(
                "Kernel computation is not implemented for infinite groups")
        gens = []
        if isinstance(G, PermutationGroup):
            K = PermutationGroup(G.identity)
        else:
            K = FpSubgroup(G, gens, normal=True)
        i = self.image().order()
        while K.order()*i != G_order:
            r = G.random()
            k = r*self.invert(self(r))**-1
            if k not in K:
                gens.append(k)
                if isinstance(G, PermutationGroup):
                    K = PermutationGroup(gens)
                else:
                    K = FpSubgroup(G, gens, normal=True)
        return K

    def image(self):
        '''
        Compute the image of `self`.

        '''
        if self._image is None:
            values = list(set(self.images.values()))
            if isinstance(self.codomain, PermutationGroup):
                self._image = self.codomain.subgroup(values)
            else:
                self._image = FpSubgroup(self.codomain, values)
        return self._image

    def _apply(self, elem):
        '''
        Apply `self` to `elem`.

        '''
        if elem not in self.domain:
            if isinstance(elem, (list, tuple)):
                return [self._apply(e) for e in elem]
            raise ValueError("The supplied element does not belong to the domain")
        if elem.is_identity:
            return self.codomain.identity
        else:
            images = self.images
            value = self.codomain.identity
            if isinstance(self.domain, PermutationGroup):
                gens = self.domain.generator_product(elem, original=True)
                for g in gens:
                    if g in self.images:
                        value = images[g]*value
                    else:
                        value = images[g**-1]**-1*value
            else:
                i = 0
                for _, p in elem.array_form:
                    if p < 0:
                        g = elem[i]**-1
                    else:
                        g = elem[i]
                    value = value*images[g]**p
                    i += abs(p)
        return value

    def __call__(self, elem):
        return self._apply(elem)

    def is_injective(self):
        '''
        Check if the homomorphism is injective

        '''
        return self.kernel().order() == 1

    def is_surjective(self):
        '''
        Check if the homomorphism is surjective

        '''
        im = self.image().order()
        oth = self.codomain.order()
        if im is S.Infinity and oth is S.Infinity:
            return None
        else:
            return im == oth

    def is_isomorphism(self):
        '''
        Check if `self` is an isomorphism.

        '''
        return self.is_injective() and self.is_surjective()

    def is_trivial(self):
        '''
        Check is `self` is a trivial homomorphism, i.e. all elements
        are mapped to the identity.

        '''
        return self.image().order() == 1

    def compose(self, other):
        '''
        Return the composition of `self` and `other`, i.e.
        the homomorphism phi such that for all g in the domain
        of `other`, phi(g) = self(other(g))

        '''
        if not other.image().is_subgroup(self.domain):
            raise ValueError("The image of `other` must be a subgroup of "
                    "the domain of `self`")
        images = {g: self(other(g)) for g in other.images}
        return GroupHomomorphism(other.domain, self.codomain, images)

    def restrict_to(self, H):
        '''
        Return the restriction of the homomorphism to the subgroup `H`
        of the domain.

        '''
        if not isinstance(H, PermutationGroup) or not H.is_subgroup(self.domain):
            raise ValueError("Given H is not a subgroup of the domain")
        domain = H
        images = {g: self(g) for g in H.generators}
        return GroupHomomorphism(domain, self.codomain, images)

    def invert_subgroup(self, H):
        '''
        Return the subgroup of the domain that is the inverse image
        of the subgroup ``H`` of the homomorphism image

        '''
        if not H.is_subgroup(self.image()):
            raise ValueError("Given H is not a subgroup of the image")
        gens = []
        P = PermutationGroup(self.image().identity)
        for h in H.generators:
            h_i = self.invert(h)
            if h_i not in P:
                gens.append(h_i)
                P = PermutationGroup(gens)
            for k in self.kernel().generators:
                if k*h_i not in P:
                    gens.append(k*h_i)
                    P = PermutationGroup(gens)
        return P

def homomorphism(domain, codomain, gens, images=(), check=True):
    '''
    Create (if possible) a group homomorphism from the group ``domain``
    to the group ``codomain`` defined by the images of the domain's
    generators ``gens``. ``gens`` and ``images`` can be either lists or tuples
    of equal sizes. If ``gens`` is a proper subset of the group's generators,
    the unspecified generators will be mapped to the identity. If the
    images are not specified, a trivial homomorphism will be created.

    If the given images of the generators do not define a homomorphism,
    an exception is raised.

    If ``check`` is ``False``, do not check whether the given images actually
    define a homomorphism.

    '''
    if not isinstance(domain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The domain must be a group")
    if not isinstance(codomain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The codomain must be a group")

    generators = domain.generators
    if not all(g in generators for g in gens):
        raise ValueError("The supplied generators must be a subset of the domain's generators")
    if not all(g in codomain for g in images):
        raise ValueError("The images must be elements of the codomain")

    if images and len(images) != len(gens):
        raise ValueError("The number of images must be equal to the number of generators")

    gens = list(gens)
    images = list(images)

    images.extend([codomain.identity]*(len(generators)-len(images)))
    gens.extend([g for g in generators if g not in gens])
    images = dict(zip(gens,images))

    if check and not _check_homomorphism(domain, codomain, images):
        raise ValueError("The given images do not define a homomorphism")
    return GroupHomomorphism(domain, codomain, images)

def _check_homomorphism(domain, codomain, images):
    """
    Check that a given mapping of generators to images defines a homomorphism.

    Parameters
    ==========
    domain : PermutationGroup, FpGroup, FreeGroup
    codomain : PermutationGroup, FpGroup, FreeGroup
    images : dict
        The set of keys must be equal to domain.generators.
        The values must be elements of the codomain.

    """
    pres = domain if hasattr(domain, 'relators') else domain.presentation()
    rels = pres.relators
    gens = pres.generators
    symbols = [g.ext_rep[0] for g in gens]
    symbols_to_domain_generators = dict(zip(symbols, domain.generators))
    identity = codomain.identity

    def _image(r):
        w = identity
        for symbol, power in r.array_form:
            g = symbols_to_domain_generators[symbol]
            w *= images[g]**power
        return w

    for r in rels:
        if isinstance(codomain, FpGroup):
            s = codomain.equals(_image(r), identity)
            if s is None:
                # only try to make the rewriting system
                # confluent when it can't determine the
                # truth of equality otherwise
                success = codomain.make_confluent()
                s = codomain.equals(_image(r), identity)
                if s is None and not success:
                    raise RuntimeError("Can't determine if the images "
                        "define a homomorphism. Try increasing "
                        "the maximum number of rewriting rules "
                        "(group._rewriting_system.set_max(new_value); "
                        "the current value is stored in group._rewriting"
                        "_system.maxeqns)")
        else:
            s = _image(r).is_identity
        if not s:
            return False
    return True

def orbit_homomorphism(group, omega):
    '''
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the set ``omega`` that is closed under the action.

    '''
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup
    codomain = SymmetricGroup(len(omega))
    identity = codomain.identity
    omega = list(omega)
    images = {g: identity*Permutation([omega.index(o^g) for o in omega]) for g in group.generators}
    group._schreier_sims(base=omega)
    H = GroupHomomorphism(group, codomain, images)
    if len(group.basic_stabilizers) > len(omega):
        H._kernel = group.basic_stabilizers[len(omega)]
    else:
        H._kernel = PermutationGroup([group.identity])
    return H

def block_homomorphism(group, blocks):
    '''
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the block system ``blocks``. The latter should be
    of the same form as returned by the ``minimal_block`` method for
    permutation groups, namely a list of length ``group.degree`` where
    the i-th entry is a representative of the block i belongs to.

    '''
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup

    n = len(blocks)

    # number the blocks; m is the total number,
    # b is such that b[i] is the number of the block i belongs to,
    # p is the list of length m such that p[i] is the representative
    # of the i-th block
    m = 0
    p = []
    b = [None]*n
    for i in range(n):
        if blocks[i] == i:
            p.append(i)
            b[i] = m
            m += 1
    for i in range(n):
        b[i] = b[blocks[i]]

    codomain = SymmetricGroup(m)
    # the list corresponding to the identity permutation in codomain
    identity = range(m)
    images = {g: Permutation([b[p[i]^g] for i in identity]) for g in group.generators}
    H = GroupHomomorphism(group, codomain, images)
    return H

def group_isomorphism(G, H, isomorphism=True):
    '''
    Compute an isomorphism between 2 given groups.

    Parameters
    ==========

    G : A finite ``FpGroup`` or a ``PermutationGroup``.
        First group.

    H : A finite ``FpGroup`` or a ``PermutationGroup``
        Second group.

    isomorphism : bool
        This is used to avoid the computation of homomorphism
        when the user only wants to check if there exists
        an isomorphism between the groups.

    Returns
    =======

    If isomorphism = False -- Returns a boolean.
    If isomorphism = True  -- Returns a boolean and an isomorphism between `G` and `H`.

    Examples
    ========

    >>> from sympy.combinatorics import free_group, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.homomorphisms import group_isomorphism
    >>> from sympy.combinatorics.named_groups import DihedralGroup, AlternatingGroup

    >>> D = DihedralGroup(8)
    >>> p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    >>> P = PermutationGroup(p)
    >>> group_isomorphism(D, P)
    (False, None)

    >>> F, a, b = free_group("a, b")
    >>> G = FpGroup(F, [a**3, b**3, (a*b)**2])
    >>> H = AlternatingGroup(4)
    >>> (check, T) = group_isomorphism(G, H)
    >>> check
    True
    >>> T(b*a*b**-1*a**-1*b**-1)
    (0 2 3)

    Notes
    =====

    Uses the approach suggested by Robert Tarjan to compute the isomorphism between two groups.
    First, the generators of ``G`` are mapped to the elements of ``H`` and
    we check if the mapping induces an isomorphism.

    '''
    if not isinstance(G, (PermutationGroup, FpGroup)):
        raise TypeError("The group must be a PermutationGroup or an FpGroup")
    if not isinstance(H, (PermutationGroup, FpGroup)):
        raise TypeError("The group must be a PermutationGroup or an FpGroup")

    if isinstance(G, FpGroup) and isinstance(H, FpGroup):
        G = simplify_presentation(G)
        H = simplify_presentation(H)
        # Two infinite FpGroups with the same generators are isomorphic
        # when the relators are same but are ordered differently.
        if G.generators == H.generators and (G.relators).sort() == (H.relators).sort():
            if not isomorphism:
                return True
            return (True, homomorphism(G, H, G.generators, H.generators))

    #  `_H` is the permutation group isomorphic to `H`.
    _H = H
    g_order = G.order()
    h_order = H.order()

    if g_order is S.Infinity:
        raise NotImplementedError("Isomorphism methods are not implemented for infinite groups.")

    if isinstance(H, FpGroup):
        if h_order is S.Infinity:
            raise NotImplementedError("Isomorphism methods are not implemented for infinite groups.")
        _H, h_isomorphism = H._to_perm_group()

    if (g_order != h_order) or (G.is_abelian != H.is_abelian):
        if not isomorphism:
            return False
        return (False, None)

    if not isomorphism:
        # Two groups of the same cyclic numbered order
        # are isomorphic to each other.
        n = g_order
        if (igcd(n, totient(n))) == 1:
            return True

    # Match the generators of `G` with subsets of `_H`
    gens = list(G.generators)
    for subset in itertools.permutations(_H, len(gens)):
        images = list(subset)
        images.extend([_H.identity]*(len(G.generators)-len(images)))
        _images = dict(zip(gens,images))
        if _check_homomorphism(G, _H, _images):
            if isinstance(H, FpGroup):
                images = h_isomorphism.invert(images)
            T =  homomorphism(G, H, G.generators, images, check=False)
            if T.is_isomorphism():
                # It is a valid isomorphism
                if not isomorphism:
                    return True
                return (True, T)

    if not isomorphism:
        return False
    return (False, None)

def is_isomorphic(G, H):
    '''
    Check if the groups are isomorphic to each other

    Parameters
    ==========

    G : A finite ``FpGroup`` or a ``PermutationGroup``
        First group.

    H : A finite ``FpGroup`` or a ``PermutationGroup``
        Second group.

    Returns
    =======

    boolean
    '''
    return group_isomorphism(G, H, isomorphism=False)
<|endoftext|>from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left


def symmetric(n):
    """
    Generates the symmetric group of order n, Sn.

    Examples
    ========

    >>> from sympy.combinatorics.generators import symmetric
    >>> list(symmetric(3))
    [(2), (1 2), (2)(0 1), (0 1 2), (0 2 1), (0 2)]
    """
    yield from (Permutation(perm) for perm in variations(range(n), n))


def cyclic(n):
    """
    Generates the cyclic group of order n, Cn.

    Examples
    ========

    >>> from sympy.combinatorics.generators import cyclic
    >>> list(cyclic(5))
    [(4), (0 1 2 3 4), (0 2 4 1 3),
     (0 3 1 4 2), (0 4 3 2 1)]

    See Also
    ========

    dihedral
    """
    gen = list(range(n))
    for i in range(n):
        yield Permutation(gen)
        gen = rotate_left(gen, 1)


def alternating(n):
    """
    Generates the alternating group of order n, An.

    Examples
    ========

    >>> from sympy.combinatorics.generators import alternating
    >>> list(alternating(3))
    [(2), (0 1 2), (0 2 1)]
    """
    for perm in variations(range(n), n):
        p = Permutation(perm)
        if p.is_even:
            yield p


def dihedral(n):
    """
    Generates the dihedral group of order 2n, Dn.

    The result is given as a subgroup of Sn, except for the special cases n=1
    (the group S2) and n=2 (the Klein 4-group) where that's not possible
    and embeddings in S2 and S4 respectively are given.

    Examples
    ========

    >>> from sympy.combinatorics.generators import dihedral
    >>> list(dihedral(3))
    [(2), (0 2), (0 1 2), (1 2), (0 2 1), (2)(0 1)]

    See Also
    ========

    cyclic
    """
    if n == 1:
        yield Permutation([0, 1])
        yield Permutation([1, 0])
    elif n == 2:
        yield Permutation([0, 1, 2, 3])
        yield Permutation([1, 0, 3, 2])
        yield Permutation([2, 3, 0, 1])
        yield Permutation([3, 2, 1, 0])
    else:
        gen = list(range(n))
        for i in range(n):
            yield Permutation(gen)
            yield Permutation(gen[::-1])
            gen = rotate_left(gen, 1)


def rubik_cube_generators():
    """Return the permutations of the 3x3 Rubik's cube, see
    https://www.gap-system.org/Doc/Examples/rubik.html
    """
    a = [
        [(1, 3, 8, 6), (2, 5, 7, 4), (9, 33, 25, 17), (10, 34, 26, 18),
         (11, 35, 27, 19)],
        [(9, 11, 16, 14), (10, 13, 15, 12), (1, 17, 41, 40), (4, 20, 44, 37),
         (6, 22, 46, 35)],
        [(17, 19, 24, 22), (18, 21, 23, 20), (6, 25, 43, 16), (7, 28, 42, 13),
         (8, 30, 41, 11)],
        [(25, 27, 32, 30), (26, 29, 31, 28), (3, 38, 43, 19), (5, 36, 45, 21),
         (8, 33, 48, 24)],
        [(33, 35, 40, 38), (34, 37, 39, 36), (3, 9, 46, 32), (2, 12, 47, 29),
         (1, 14, 48, 27)],
        [(41, 43, 48, 46), (42, 45, 47, 44), (14, 22, 30, 38),
         (15, 23, 31, 39), (16, 24, 32, 40)]
    ]
    return [Permutation([[i - 1 for i in xi] for xi in x], size=48) for x in a]


def rubik(n):
    """Return permutations for an nxn Rubik's cube.

    Permutations returned are for rotation of each of the slice
    from the face up to the last face for each of the 3 sides (in this order):
    front, right and bottom. Hence, the first n - 1 permutations are for the
    slices from the front.
    """

    if n < 2:
        raise ValueError('dimension of cube must be > 1')

    # 1-based reference to rows and columns in Matrix
    def getr(f, i):
        return faces[f].col(n - i)

    def getl(f, i):
        return faces[f].col(i - 1)

    def getu(f, i):
        return faces[f].row(i - 1)

    def getd(f, i):
        return faces[f].row(n - i)

    def setr(f, i, s):
        faces[f][:, n - i] = Matrix(n, 1, s)

    def setl(f, i, s):
        faces[f][:, i - 1] = Matrix(n, 1, s)

    def setu(f, i, s):
        faces[f][i - 1, :] = Matrix(1, n, s)

    def setd(f, i, s):
        faces[f][n - i, :] = Matrix(1, n, s)

    # motion of a single face
    def cw(F, r=1):
        for _ in range(r):
            face = faces[F]
            rv = []
            for c in range(n):
                for r in range(n - 1, -1, -1):
                    rv.append(face[r, c])
            faces[F] = Matrix(n, n, rv)

    def ccw(F):
        cw(F, 3)

    # motion of plane i from the F side;
    # fcw(0) moves the F face, fcw(1) moves the plane
    # just behind the front face, etc...
    def fcw(i, r=1):
        for _ in range(r):
            if i == 0:
                cw(F)
            i += 1
            temp = getr(L, i)
            setr(L, i, list(getu(D, i)))
            setu(D, i, list(reversed(getl(R, i))))
            setl(R, i, list(getd(U, i)))
            setd(U, i, list(reversed(temp)))
            i -= 1

    def fccw(i):
        fcw(i, 3)

    # motion of the entire cube from the F side
    def FCW(r=1):
        for _ in range(r):
            cw(F)
            ccw(B)
            cw(U)
            t = faces[U]
            cw(L)
            faces[U] = faces[L]
            cw(D)
            faces[L] = faces[D]
            cw(R)
            faces[D] = faces[R]
            faces[R] = t

    def FCCW():
        FCW(3)

    # motion of the entire cube from the U side
    def UCW(r=1):
        for _ in range(r):
            cw(U)
            ccw(D)
            t = faces[F]
            faces[F] = faces[R]
            faces[R] = faces[B]
            faces[B] = faces[L]
            faces[L] = t

    def UCCW():
        UCW(3)

    # defining the permutations for the cube

    U, F, R, B, L, D = names = symbols('U, F, R, B, L, D')

    # the faces are represented by nxn matrices
    faces = {}
    count = 0
    for fi in range(6):
        f = []
        for a in range(n**2):
            f.append(count)
            count += 1
        faces[names[fi]] = Matrix(n, n, f)

    # this will either return the value of the current permutation
    # (show != 1) or else append the permutation to the group, g
    def perm(show=0):
        # add perm to the list of perms
        p = []
        for f in names:
            p.extend(faces[f])
        if show:
            return p
        g.append(Permutation(p))

    g = []  # container for the group's permutations
    I = list(range(6*n**2))  # the identity permutation used for checking

    # define permutations corresponding to cw rotations of the planes
    # up TO the last plane from that direction; by not including the
    # last plane, the orientation of the cube is maintained.

    # F slices
    for i in range(n - 1):
        fcw(i)
        perm()
        fccw(i)  # restore
    assert perm(1) == I

    # R slices
    # bring R to front
    UCW()
    for i in range(n - 1):
        fcw(i)
        # put it back in place
        UCCW()
        # record
        perm()
        # restore
        # bring face to front
        UCW()
        fccw(i)
    # restore
    UCCW()
    assert perm(1) == I

    # D slices
    # bring up bottom
    FCW()
    UCCW()
    FCCW()
    for i in range(n - 1):
        # turn strip
        fcw(i)
        # put bottom back on the bottom
        FCW()
        UCW()
        FCCW()
        # record
        perm()
        # restore
        # bring up bottom
        FCW()
        UCCW()
        FCCW()
        # turn strip
        fccw(i)
    # put bottom back on the bottom
    FCW()
    UCW()
    FCCW()
    assert perm(1) == I

    return g
<|endoftext|>import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce

from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import int_valued
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
    has_dups, runs, is_sequence)
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch

def _af_rmul(a, b):
    """
    Return the product b*a; input and output are array forms. The ith value
    is a[b[i]].

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_rmul, Permutation

    >>> a, b = [1, 0, 2], [0, 2, 1]
    >>> _af_rmul(a, b)
    [1, 2, 0]
    >>> [a[b[i]] for i in range(3)]
    [1, 2, 0]

    This handles the operands in reverse order compared to the ``*`` operator:

    >>> a = Permutation(a)
    >>> b = Permutation(b)
    >>> list(a*b)
    [2, 0, 1]
    >>> [b(a(i)) for i in range(3)]
    [2, 0, 1]

    See Also
    ========

    rmul, _af_rmuln
    """
    return [a[i] for i in b]


def _af_rmuln(*abc):
    """
    Given [a, b, c, ...] return the product of ...*c*b*a using array forms.
    The ith value is a[b[c[i]]].

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_rmul, Permutation

    >>> a, b = [1, 0, 2], [0, 2, 1]
    >>> _af_rmul(a, b)
    [1, 2, 0]
    >>> [a[b[i]] for i in range(3)]
    [1, 2, 0]

    This handles the operands in reverse order compared to the ``*`` operator:

    >>> a = Permutation(a); b = Permutation(b)
    >>> list(a*b)
    [2, 0, 1]
    >>> [b(a(i)) for i in range(3)]
    [2, 0, 1]

    See Also
    ========

    rmul, _af_rmul
    """
    a = abc
    m = len(a)
    if m == 3:
        p0, p1, p2 = a
        return [p0[p1[i]] for i in p2]
    if m == 4:
        p0, p1, p2, p3 = a
        return [p0[p1[p2[i]]] for i in p3]
    if m == 5:
        p0, p1, p2, p3, p4 = a
        return [p0[p1[p2[p3[i]]]] for i in p4]
    if m == 6:
        p0, p1, p2, p3, p4, p5 = a
        return [p0[p1[p2[p3[p4[i]]]]] for i in p5]
    if m == 7:
        p0, p1, p2, p3, p4, p5, p6 = a
        return [p0[p1[p2[p3[p4[p5[i]]]]]] for i in p6]
    if m == 8:
        p0, p1, p2, p3, p4, p5, p6, p7 = a
        return [p0[p1[p2[p3[p4[p5[p6[i]]]]]]] for i in p7]
    if m == 1:
        return a[0][:]
    if m == 2:
        a, b = a
        return [a[i] for i in b]
    if m == 0:
        raise ValueError("String must not be empty")
    p0 = _af_rmuln(*a[:m//2])
    p1 = _af_rmuln(*a[m//2:])
    return [p0[i] for i in p1]


def _af_parity(pi):
    """
    Computes the parity of a permutation in array form.

    Explanation
    ===========

    The parity of a permutation reflects the parity of the
    number of inversions in the permutation, i.e., the
    number of pairs of x and y such that x > y but p[x] < p[y].

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_parity
    >>> _af_parity([0, 1, 2, 3])
    0
    >>> _af_parity([3, 2, 0, 1])
    1

    See Also
    ========

    Permutation
    """
    n = len(pi)
    a = [0] * n
    c = 0
    for j in range(n):
        if a[j] == 0:
            c += 1
            a[j] = 1
            i = j
            while pi[i] != j:
                i = pi[i]
                a[i] = 1
    return (n - c) % 2


def _af_invert(a):
    """
    Finds the inverse, ~A, of a permutation, A, given in array form.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_invert, _af_rmul
    >>> A = [1, 2, 0, 3]
    >>> _af_invert(A)
    [2, 0, 1, 3]
    >>> _af_rmul(_, A)
    [0, 1, 2, 3]

    See Also
    ========

    Permutation, __invert__
    """
    inv_form = [0] * len(a)
    for i, ai in enumerate(a):
        inv_form[ai] = i
    return inv_form


def _af_pow(a, n):
    """
    Routine for finding powers of a permutation.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.permutations import _af_pow
    >>> p = Permutation([2, 0, 3, 1])
    >>> p.order()
    4
    >>> _af_pow(p._array_form, 4)
    [0, 1, 2, 3]
    """
    if n == 0:
        return list(range(len(a)))
    if n < 0:
        return _af_pow(_af_invert(a), -n)
    if n == 1:
        return a[:]
    elif n == 2:
        b = [a[i] for i in a]
    elif n == 3:
        b = [a[a[i]] for i in a]
    elif n == 4:
        b = [a[a[a[i]]] for i in a]
    else:
        # use binary multiplication
        b = list(range(len(a)))
        while 1:
            if n & 1:
                b = [b[i] for i in a]
                n -= 1
                if not n:
                    break
            if n % 4 == 0:
                a = [a[a[a[i]]] for i in a]
                n = n // 4
            elif n % 2 == 0:
                a = [a[i] for i in a]
                n = n // 2
    return b


def _af_commutes_with(a, b):
    """
    Checks if the two permutations with array forms
    given by ``a`` and ``b`` commute.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import _af_commutes_with
    >>> _af_commutes_with([1, 2, 0], [0, 2, 1])
    False

    See Also
    ========

    Permutation, commutes_with
    """
    return not any(a[b[i]] != b[a[i]] for i in range(len(a) - 1))


class Cycle(dict):
    """
    Wrapper around dict which provides the functionality of a disjoint cycle.

    Explanation
    ===========

    A cycle shows the rule to use to move subsets of elements to obtain
    a permutation. The Cycle class is more flexible than Permutation in
    that 1) all elements need not be present in order to investigate how
    multiple cycles act in sequence and 2) it can contain singletons:

    >>> from sympy.combinatorics.permutations import Perm, Cycle

    A Cycle will automatically parse a cycle given as a tuple on the rhs:

    >>> Cycle(1, 2)(2, 3)
    (1 3 2)

    The identity cycle, Cycle(), can be used to start a product:

    >>> Cycle()(1, 2)(2, 3)
    (1 3 2)

    The array form of a Cycle can be obtained by calling the list
    method (or passing it to the list function) and all elements from
    0 will be shown:

    >>> a = Cycle(1, 2)
    >>> a.list()
    [0, 2, 1]
    >>> list(a)
    [0, 2, 1]

    If a larger (or smaller) range is desired use the list method and
    provide the desired size -- but the Cycle cannot be truncated to
    a size smaller than the largest element that is out of place:

    >>> b = Cycle(2, 4)(1, 2)(3, 1, 4)(1, 3)
    >>> b.list()
    [0, 2, 1, 3, 4]
    >>> b.list(b.size + 1)
    [0, 2, 1, 3, 4, 5]
    >>> b.list(-1)
    [0, 2, 1]

    Singletons are not shown when printing with one exception: the largest
    element is always shown -- as a singleton if necessary:

    >>> Cycle(1, 4, 10)(4, 5)
    (1 5 4 10)
    >>> Cycle(1, 2)(4)(5)(10)
    (1 2)(10)

    The array form can be used to instantiate a Permutation so other
    properties of the permutation can be investigated:

    >>> Perm(Cycle(1, 2)(3, 4).list()).transpositions()
    [(1, 2), (3, 4)]

    Notes
    =====

    The underlying structure of the Cycle is a dictionary and although
    the __iter__ method has been redefined to give the array form of the
    cycle, the underlying dictionary items are still available with the
    such methods as items():

