            ``base`` is the base obtained, and ``strong_gens`` is the strong
            generating set relative to it. The original parameters ``base``,
            ``gens`` remain unchanged.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> A = AlternatingGroup(7)
        >>> base = [2, 3]
        >>> seq = [2, 3]
        >>> base, strong_gens = A.schreier_sims_incremental(base=seq)
        >>> _verify_bsgs(A, base, strong_gens)
        True
        >>> base[:2]
        [2, 3]

        Notes
        =====

        This version of the Schreier-Sims algorithm runs in polynomial time.
        There are certain assumptions in the implementation - if the trivial
        group is provided, ``base`` and ``gens`` are returned immediately,
        as any sequence of points is a base for the trivial group. If the
        identity is present in the generators ``gens``, it is removed as
        it is a redundant generator.
        The implementation is described in [1], pp. 90-93.

        See Also
        ========

        schreier_sims, schreier_sims_random

        """
        if base is None:
            base = []
        if gens is None:
            gens = self.generators[:]
        degree = self.degree
        id_af = list(range(degree))
        # handle the trivial group
        if len(gens) == 1 and gens[0].is_Identity:
            if slp_dict:
                return base, gens, {gens[0]: [gens[0]]}
            return base, gens
        # prevent side effects
        _base, _gens = base[:], gens[:]
        # remove the identity as a generator
        _gens = [x for x in _gens if not x.is_Identity]
        # make sure no generator fixes all base points
        for gen in _gens:
            if all(x == gen._array_form[x] for x in _base):
                for new in id_af:
                    if gen._array_form[new] != new:
                        break
                else:
                    assert None  # can this ever happen?
                _base.append(new)
        # distribute generators according to basic stabilizers
        strong_gens_distr = _distribute_gens_by_base(_base, _gens)
        strong_gens_slp = []
        # initialize the basic stabilizers, basic orbits and basic transversals
        orbs = {}
        transversals = {}
        slps = {}
        base_len = len(_base)
        for i in range(base_len):
            transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i],
                _base[i], pairs=True, af=True, slp=True)
            transversals[i] = dict(transversals[i])
            orbs[i] = list(transversals[i].keys())
        # main loop: amend the stabilizer chain until we have generators
        # for all stabilizers
        i = base_len - 1
        while i >= 0:
            # this flag is used to continue with the main loop from inside
            # a nested loop
            continue_i = False
            # test the generators for being a strong generating set
            db = {}
            for beta, u_beta in list(transversals[i].items()):
                for j, gen in enumerate(strong_gens_distr[i]):
                    gb = gen._array_form[beta]
                    u1 = transversals[i][gb]
                    g1 = _af_rmul(gen._array_form, u_beta)
                    slp = [(i, g) for g in slps[i][beta]]
                    slp = [(i, j)] + slp
                    if g1 != u1:
                        # test if the schreier generator is in the i+1-th
                        # would-be basic stabilizer
                        y = True
                        try:
                            u1_inv = db[gb]
                        except KeyError:
                            u1_inv = db[gb] = _af_invert(u1)
                        schreier_gen = _af_rmul(u1_inv, g1)
                        u1_inv_slp = slps[i][gb][:]
                        u1_inv_slp.reverse()
                        u1_inv_slp = [(i, (g,)) for g in u1_inv_slp]
                        slp = u1_inv_slp + slp
                        h, j, slp = _strip_af(schreier_gen, _base, orbs, transversals, i, slp=slp, slps=slps)
                        if j <= base_len:
                            # new strong generator h at level j
                            y = False
                        elif h:
                            # h fixes all base points
                            y = False
                            moved = 0
                            while h[moved] == moved:
                                moved += 1
                            _base.append(moved)
                            base_len += 1
                            strong_gens_distr.append([])
                        if y is False:
                            # if a new strong generator is found, update the
                            # data structures and start over
                            h = _af_new(h)
                            strong_gens_slp.append((h, slp))
                            for l in range(i + 1, j):
                                strong_gens_distr[l].append(h)
                                transversals[l], slps[l] =\
                                _orbit_transversal(degree, strong_gens_distr[l],
                                    _base[l], pairs=True, af=True, slp=True)
                                transversals[l] = dict(transversals[l])
                                orbs[l] = list(transversals[l].keys())
                            i = j - 1
                            # continue main loop using the flag
                            continue_i = True
                    if continue_i is True:
                        break
                if continue_i is True:
                    break
            if continue_i is True:
                continue
            i -= 1

        strong_gens = _gens[:]

        if slp_dict:
            # create the list of the strong generators strong_gens and
            # rewrite the indices of strong_gens_slp in terms of the
            # elements of strong_gens
            for k, slp in strong_gens_slp:
                strong_gens.append(k)
                for i in range(len(slp)):
                    s = slp[i]
                    if isinstance(s[1], tuple):
                        slp[i] = strong_gens_distr[s[0]][s[1][0]]**-1
                    else:
                        slp[i] = strong_gens_distr[s[0]][s[1]]
            strong_gens_slp = dict(strong_gens_slp)
            # add the original generators
            for g in _gens:
                strong_gens_slp[g] = [g]
            return (_base, strong_gens, strong_gens_slp)

        strong_gens.extend([k for k, _ in strong_gens_slp])
        return _base, strong_gens

    def schreier_sims_random(self, base=None, gens=None, consec_succ=10,
                             _random_prec=None):
        r"""Randomized Schreier-Sims algorithm.

        Explanation
        ===========

        The randomized Schreier-Sims algorithm takes the sequence ``base``
        and the generating set ``gens``, and extends ``base`` to a base, and
        ``gens`` to a strong generating set relative to that base with
        probability of a wrong answer at most `2^{-consec\_succ}`,
        provided the random generators are sufficiently random.

        Parameters
        ==========

        base
            The sequence to be extended to a base.
        gens
            The generating set to be extended to a strong generating set.
        consec_succ
            The parameter defining the probability of a wrong answer.
        _random_prec
            An internal parameter used for testing purposes.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the base and ``strong_gens`` is the strong generating
            set relative to it.

        Examples
        ========

        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(5)
        >>> base, strong_gens = S.schreier_sims_random(consec_succ=5)
        >>> _verify_bsgs(S, base, strong_gens) #doctest: +SKIP
        True

        Notes
        =====

        The algorithm is described in detail in [1], pp. 97-98. It extends
        the orbits ``orbs`` and the permutation groups ``stabs`` to
        basic orbits and basic stabilizers for the base and strong generating
        set produced in the end.
        The idea of the extension process
        is to "sift" random group elements through the stabilizer chain
        and amend the stabilizers/orbits along the way when a sift
        is not successful.
        The helper function ``_strip`` is used to attempt
        to decompose a random group element according to the current
        state of the stabilizer chain and report whether the element was
        fully decomposed (successful sift) or not (unsuccessful sift). In
        the latter case, the level at which the sift failed is reported and
        used to amend ``stabs``, ``base``, ``gens`` and ``orbs`` accordingly.
        The halting condition is for ``consec_succ`` consecutive successful
        sifts to pass. This makes sure that the current ``base`` and ``gens``
        form a BSGS with probability at least `1 - 1/\text{consec\_succ}`.

        See Also
        ========

        schreier_sims

        """
        if base is None:
            base = []
        if gens is None:
            gens = self.generators
        base_len = len(base)
        n = self.degree
        # make sure no generator fixes all base points
        for gen in gens:
            if all(gen(x) == x for x in base):
                new = 0
                while gen._array_form[new] == new:
                    new += 1
                base.append(new)
                base_len += 1
        # distribute generators according to basic stabilizers
        strong_gens_distr = _distribute_gens_by_base(base, gens)
        # initialize the basic stabilizers, basic transversals and basic orbits
        transversals = {}
        orbs = {}
        for i in range(base_len):
            transversals[i] = dict(_orbit_transversal(n, strong_gens_distr[i],
                base[i], pairs=True))
            orbs[i] = list(transversals[i].keys())
        # initialize the number of consecutive elements sifted
        c = 0
        # start sifting random elements while the number of consecutive sifts
        # is less than consec_succ
        while c < consec_succ:
            if _random_prec is None:
                g = self.random_pr()
            else:
                g = _random_prec['g'].pop()
            h, j = _strip(g, base, orbs, transversals)
            y = True
            # determine whether a new base point is needed
            if j <= base_len:
                y = False
            elif not h.is_Identity:
                y = False
                moved = 0
                while h(moved) == moved:
                    moved += 1
                base.append(moved)
                base_len += 1
                strong_gens_distr.append([])
            # if the element doesn't sift, amend the strong generators and
            # associated stabilizers and orbits
            if y is False:
                for l in range(1, j):
                    strong_gens_distr[l].append(h)
                    transversals[l] = dict(_orbit_transversal(n,
                        strong_gens_distr[l], base[l], pairs=True))
                    orbs[l] = list(transversals[l].keys())
                c = 0
            else:
                c += 1
        # build the strong generating set
        strong_gens = strong_gens_distr[0][:]
        for gen in strong_gens_distr[1]:
            if gen not in strong_gens:
                strong_gens.append(gen)
        return base, strong_gens

    def schreier_vector(self, alpha):
        """Computes the schreier vector for ``alpha``.

        Explanation
        ===========

        The Schreier vector efficiently stores information
        about the orbit of ``alpha``. It can later be used to quickly obtain
        elements of the group that send ``alpha`` to a particular element
        in the orbit. Notice that the Schreier vector depends on the order
        in which the group generators are listed. For a definition, see [3].
        Since list indices start from zero, we adopt the convention to use
        "None" instead of 0 to signify that an element does not belong
        to the orbit.
        For the algorithm and its correctness, see [2], pp.78-80.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([2, 4, 6, 3, 1, 5, 0])
        >>> b = Permutation([0, 1, 3, 5, 4, 6, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_vector(0)
        [-1, None, 0, 1, None, 1, 0]

        See Also
        ========

        orbit

        """
        n = self.degree
        v = [None]*n
        v[alpha] = -1
        orb = [alpha]
        used = [False]*n
        used[alpha] = True
        gens = self.generators
        r = len(gens)
        for b in orb:
            for i in range(r):
                temp = gens[i]._array_form[b]
                if used[temp] is False:
                    orb.append(temp)
                    used[temp] = True
                    v[temp] = i
        return v

    def stabilizer(self, alpha):
        r"""Return the stabilizer subgroup of ``alpha``.

        Explanation
        ===========

        The stabilizer of `\alpha` is the group `G_\alpha =
        \{g \in G | g(\alpha) = \alpha\}`.
        For a proof of correctness, see [1], p.79.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.stabilizer(5)
        PermutationGroup([
            (5)(0 4)(1 3)])

        See Also
        ========

        orbit

        """
        return PermGroup(_stabilizer(self._degree, self._generators, alpha))

    @property
    def strong_gens(self):
        r"""Return a strong generating set from the Schreier-Sims algorithm.

        Explanation
        ===========

        A generating set `S = \{g_1, g_2, \dots, g_t\}` for a permutation group
        `G` is a strong generating set relative to the sequence of points
        (referred to as a "base") `(b_1, b_2, \dots, b_k)` if, for
        `1 \leq i \leq k` we have that the intersection of the pointwise
        stabilizer `G^{(i+1)} := G_{b_1, b_2, \dots, b_i}` with `S` generates
        the pointwise stabilizer `G^{(i+1)}`. The concepts of a base and
        strong generating set and their applications are discussed in depth
        in [1], pp. 87-89 and [2], pp. 55-57.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(4)
        >>> D.strong_gens
        [(0 1 2 3), (0 3)(1 2), (1 3)]
        >>> D.base
        [0, 1]

        See Also
        ========

        base, basic_transversals, basic_orbits, basic_stabilizers

        """
        if self._strong_gens == []:
            self.schreier_sims()
        return self._strong_gens

    def subgroup(self, gens):
        """
           Return the subgroup generated by `gens` which is a list of
           elements of the group
        """

        if not all(g in self for g in gens):
            raise ValueError("The group does not contain the supplied generators")

        G = PermutationGroup(gens)
        return G

    def subgroup_search(self, prop, base=None, strong_gens=None, tests=None,
                        init_subgroup=None):
        """Find the subgroup of all elements satisfying the property ``prop``.

        Explanation
        ===========

        This is done by a depth-first search with respect to base images that
        uses several tests to prune the search tree.

        Parameters
        ==========

        prop
            The property to be used. Has to be callable on group elements
            and always return ``True`` or ``False``. It is assumed that
            all group elements satisfying ``prop`` indeed form a subgroup.
        base
            A base for the supergroup.
        strong_gens
            A strong generating set for the supergroup.
        tests
            A list of callables of length equal to the length of ``base``.
            These are used to rule out group elements by partial base images,
            so that ``tests[l](g)`` returns False if the element ``g`` is known
            not to satisfy prop base on where g sends the first ``l + 1`` base
            points.
        init_subgroup
            if a subgroup of the sought group is
            known in advance, it can be passed to the function as this
            parameter.

        Returns
        =======

        res
            The subgroup of all elements satisfying ``prop``. The generating
            set for this group is guaranteed to be a strong generating set
            relative to the base ``base``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> S = SymmetricGroup(7)
        >>> prop_even = lambda x: x.is_even
        >>> base, strong_gens = S.schreier_sims_incremental()
        >>> G = S.subgroup_search(prop_even, base=base, strong_gens=strong_gens)
        >>> G.is_subgroup(AlternatingGroup(7))
        True
        >>> _verify_bsgs(G, base, G.generators)
        True

        Notes
        =====

        This function is extremely lengthy and complicated and will require
        some careful attention. The implementation is described in
        [1], pp. 114-117, and the comments for the code here follow the lines
        of the pseudocode in the book for clarity.

        The complexity is exponential in general, since the search process by
        itself visits all members of the supergroup. However, there are a lot
        of tests which are used to prune the search tree, and users can define
        their own tests via the ``tests`` parameter, so in practice, and for
        some computations, it's not terrible.

        A crucial part in the procedure is the frequent base change performed
        (this is line 11 in the pseudocode) in order to obtain a new basic
        stabilizer. The book mentiones that this can be done by using
        ``.baseswap(...)``, however the current implementation uses a more
        straightforward way to find the next basic stabilizer - calling the
        function ``.stabilizer(...)`` on the previous basic stabilizer.

        """
        # initialize BSGS and basic group properties
        def get_reps(orbits):
            # get the minimal element in the base ordering
            return [min(orbit, key = lambda x: base_ordering[x]) \
              for orbit in orbits]

        def update_nu(l):
            temp_index = len(basic_orbits[l]) + 1 -\
                         len(res_basic_orbits_init_base[l])
            # this corresponds to the element larger than all points
            if temp_index >= len(sorted_orbits[l]):
                nu[l] = base_ordering[degree]
            else:
                nu[l] = sorted_orbits[l][temp_index]

        if base is None:
            base, strong_gens = self.schreier_sims_incremental()
        base_len = len(base)
        degree = self.degree
        identity = _af_new(list(range(degree)))
        base_ordering = _base_ordering(base, degree)
        # add an element larger than all points
        base_ordering.append(degree)
        # add an element smaller than all points
        base_ordering.append(-1)
        # compute BSGS-related structures
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_orbits, transversals = _orbits_transversals_from_bsgs(base,
                                     strong_gens_distr)
        # handle subgroup initialization and tests
        if init_subgroup is None:
            init_subgroup = PermutationGroup([identity])
        if tests is None:
            trivial_test = lambda x: True
            tests = []
            for i in range(base_len):
                tests.append(trivial_test)
        # line 1: more initializations.
        res = init_subgroup
        f = base_len - 1
        l = base_len - 1
        # line 2: set the base for K to the base for G
        res_base = base[:]
        # line 3: compute BSGS and related structures for K
        res_base, res_strong_gens = res.schreier_sims_incremental(
            base=res_base)
        res_strong_gens_distr = _distribute_gens_by_base(res_base,
                                res_strong_gens)
        res_generators = res.generators
        res_basic_orbits_init_base = \
        [_orbit(degree, res_strong_gens_distr[i], res_base[i])\
         for i in range(base_len)]
        # initialize orbit representatives
        orbit_reps = [None]*base_len
        # line 4: orbit representatives for f-th basic stabilizer of K
        orbits = _orbits(degree, res_strong_gens_distr[f])
        orbit_reps[f] = get_reps(orbits)
        # line 5: remove the base point from the representatives to avoid
        # getting the identity element as a generator for K
        orbit_reps[f].remove(base[f])
        # line 6: more initializations
        c = [0]*base_len
        u = [identity]*base_len
        sorted_orbits = [None]*base_len
        for i in range(base_len):
            sorted_orbits[i] = basic_orbits[i][:]
            sorted_orbits[i].sort(key=lambda point: base_ordering[point])
        # line 7: initializations
        mu = [None]*base_len
        nu = [None]*base_len
        # this corresponds to the element smaller than all points
        mu[l] = degree + 1
        update_nu(l)
        # initialize computed words
        computed_words = [identity]*base_len
        # line 8: main loop
        while True:
            # apply all the tests
            while l < base_len - 1 and \
                computed_words[l](base[l]) in orbit_reps[l] and \
                base_ordering[mu[l]] < \
                base_ordering[computed_words[l](base[l])] < \
                base_ordering[nu[l]] and \
                    tests[l](computed_words):
                # line 11: change the (partial) base of K
                new_point = computed_words[l](base[l])
                res_base[l] = new_point
                new_stab_gens = _stabilizer(degree, res_strong_gens_distr[l],
                        new_point)
                res_strong_gens_distr[l + 1] = new_stab_gens
                # line 12: calculate minimal orbit representatives for the
                # l+1-th basic stabilizer
                orbits = _orbits(degree, new_stab_gens)
                orbit_reps[l + 1] = get_reps(orbits)
                # line 13: amend sorted orbits
                l += 1
                temp_orbit = [computed_words[l - 1](point) for point
                             in basic_orbits[l]]
                temp_orbit.sort(key=lambda point: base_ordering[point])
                sorted_orbits[l] = temp_orbit
                # lines 14 and 15: update variables used minimality tests
                new_mu = degree + 1
                for i in range(l):
                    if base[l] in res_basic_orbits_init_base[i]:
                        candidate = computed_words[i](base[i])
                        if base_ordering[candidate] > base_ordering[new_mu]:
                            new_mu = candidate
                mu[l] = new_mu
                update_nu(l)
                # line 16: determine the new transversal element
                c[l] = 0
                temp_point = sorted_orbits[l][c[l]]
                gamma = computed_words[l - 1]._array_form.index(temp_point)
                u[l] = transversals[l][gamma]
                # update computed words
                computed_words[l] = rmul(computed_words[l - 1], u[l])
            # lines 17 & 18: apply the tests to the group element found
            g = computed_words[l]
            temp_point = g(base[l])
            if l == base_len - 1 and \
                base_ordering[mu[l]] < \
                base_ordering[temp_point] < base_ordering[nu[l]] and \
                temp_point in orbit_reps[l] and \
                tests[l](computed_words) and \
                    prop(g):
                # line 19: reset the base of K
                res_generators.append(g)
                res_base = base[:]
                # line 20: recalculate basic orbits (and transversals)
                res_strong_gens.append(g)
                res_strong_gens_distr = _distribute_gens_by_base(res_base,
                                                          res_strong_gens)
                res_basic_orbits_init_base = \
                [_orbit(degree, res_strong_gens_distr[i], res_base[i]) \
                 for i in range(base_len)]
                # line 21: recalculate orbit representatives
                # line 22: reset the search depth
                orbit_reps[f] = get_reps(orbits)
                l = f
            # line 23: go up the tree until in the first branch not fully
            # searched
            while l >= 0 and c[l] == len(basic_orbits[l]) - 1:
                l = l - 1
            # line 24: if the entire tree is traversed, return K
            if l == -1:
                return PermutationGroup(res_generators)
            # lines 25-27: update orbit representatives
            if l < f:
                # line 26
                f = l
                c[l] = 0
                # line 27
                temp_orbits = _orbits(degree, res_strong_gens_distr[f])
                orbit_reps[f] = get_reps(temp_orbits)
                # line 28: update variables used for minimality testing
                mu[l] = degree + 1
                temp_index = len(basic_orbits[l]) + 1 - \
                    len(res_basic_orbits_init_base[l])
                if temp_index >= len(sorted_orbits[l]):
                    nu[l] = base_ordering[degree]
                else:
                    nu[l] = sorted_orbits[l][temp_index]
            # line 29: set the next element from the current branch and update
            # accordingly
            c[l] += 1
            if l == 0:
                gamma  = sorted_orbits[l][c[l]]
            else:
                gamma = computed_words[l - 1]._array_form.index(sorted_orbits[l][c[l]])

            u[l] = transversals[l][gamma]
            if l == 0:
                computed_words[l] = u[l]
            else:
                computed_words[l] = rmul(computed_words[l - 1], u[l])

    @property
    def transitivity_degree(self):
        r"""Compute the degree of transitivity of the group.

        Explanation
        ===========

        A permutation group `G` acting on `\Omega = \{0, 1, \dots, n-1\}` is
        ``k``-fold transitive, if, for any `k` points
        `(a_1, a_2, \dots, a_k) \in \Omega` and any `k` points
        `(b_1, b_2, \dots, b_k) \in \Omega` there exists `g \in  G` such that
        `g(a_1) = b_1, g(a_2) = b_2, \dots, g(a_k) = b_k`
        The degree of transitivity of `G` is the maximum ``k`` such that
        `G` is ``k``-fold transitive. ([8])

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.transitivity_degree
        3

        See Also
        ========

        is_transitive, orbit

        """
        if self._transitivity_degree is None:
            n = self.degree
            G = self
            # if G is k-transitive, a tuple (a_0,..,a_k)
            # can be brought to (b_0,...,b_(k-1), b_k)
            # where b_0,...,b_(k-1) are fixed points;
            # consider the group G_k which stabilizes b_0,...,b_(k-1)
            # if G_k is transitive on the subset excluding b_0,...,b_(k-1)
            # then G is (k+1)-transitive
            for i in range(n):
                orb = G.orbit(i)
                if len(orb) != n - i:
                    self._transitivity_degree = i
                    return i
                G = G.stabilizer(i)
            self._transitivity_degree = n
            return n
        else:
            return self._transitivity_degree

    def _p_elements_group(self, p):
        '''
        For an abelian p-group, return the subgroup consisting of
        all elements of order p (and the identity)

        '''
        gens = self.generators[:]
        gens = sorted(gens, key=lambda x: x.order(), reverse=True)
        gens_p = [g**(g.order()/p) for g in gens]
        gens_r = []
        for i in range(len(gens)):
            x = gens[i]
            x_order = x.order()
            # x_p has order p
            x_p = x**(x_order/p)
            if i > 0:
                P = PermutationGroup(gens_p[:i])
            else:
                P = PermutationGroup(self.identity)
            if x**(x_order/p) not in P:
                gens_r.append(x**(x_order/p))
            else:
                # replace x by an element of order (x.order()/p)
                # so that gens still generates G
                g = P.generator_product(x_p, original=True)
                for s in g:
                    x = x*s**-1
                x_order = x_order/p
                # insert x to gens so that the sorting is preserved
                del gens[i]
                del gens_p[i]
                j = i - 1
                while j < len(gens) and gens[j].order() >= x_order:
                    j += 1
                gens = gens[:j] + [x] + gens[j:]
                gens_p = gens_p[:j] + [x] + gens_p[j:]
        return PermutationGroup(gens_r)

    def _sylow_alt_sym(self, p):
        '''
        Return a p-Sylow subgroup of a symmetric or an
        alternating group.

        Explanation
        ===========

        The algorithm for this is hinted at in [1], Chapter 4,
        Exercise 4.

        For Sym(n) with n = p^i, the idea is as follows. Partition
        the interval [0..n-1] into p equal parts, each of length p^(i-1):
        [0..p^(i-1)-1], [p^(i-1)..2*p^(i-1)-1]...[(p-1)*p^(i-1)..p^i-1].
        Find a p-Sylow subgroup of Sym(p^(i-1)) (treated as a subgroup
        of ``self``) acting on each of the parts. Call the subgroups
        P_1, P_2...P_p. The generators for the subgroups P_2...P_p
        can be obtained from those of P_1 by applying a "shifting"
        permutation to them, that is, a permutation mapping [0..p^(i-1)-1]
        to the second part (the other parts are obtained by using the shift
        multiple times). The union of this permutation and the generators
        of P_1 is a p-Sylow subgroup of ``self``.

        For n not equal to a power of p, partition
        [0..n-1] in accordance with how n would be written in base p.
        E.g. for p=2 and n=11, 11 = 2^3 + 2^2 + 1 so the partition
        is [[0..7], [8..9], {10}]. To generate a p-Sylow subgroup,
        take the union of the generators for each of the parts.
        For the above example, {(0 1), (0 2)(1 3), (0 4), (1 5)(2 7)}
        from the first part, {(8 9)} from the second part and
        nothing from the third. This gives 4 generators in total, and
        the subgroup they generate is p-Sylow.

        Alternating groups are treated the same except when p=2. In this
        case, (0 1)(s s+1) should be added for an appropriate s (the start
        of a part) for each part in the partitions.

        See Also
        ========

        sylow_subgroup, is_alt_sym

        '''
        n = self.degree
        gens = []
        identity = Permutation(n-1)
        # the case of 2-sylow subgroups of alternating groups
        # needs special treatment
        alt = p == 2 and all(g.is_even for g in self.generators)

        # find the presentation of n in base p
        coeffs = []
        m = n
        while m > 0:
            coeffs.append(m % p)
            m = m // p

        power = len(coeffs)-1
        # for a symmetric group, gens[:i] is the generating
        # set for a p-Sylow subgroup on [0..p**(i-1)-1]. For
        # alternating groups, the same is given by gens[:2*(i-1)]
        for i in range(1, power+1):
            if i == 1 and alt:
                # (0 1) shouldn't be added for alternating groups
                continue
            gen = Permutation([(j + p**(i-1)) % p**i for j in range(p**i)])
            gens.append(identity*gen)
            if alt:
                gen = Permutation(0, 1)*gen*Permutation(0, 1)*gen
                gens.append(gen)

        # the first point in the current part (see the algorithm
        # description in the docstring)
        start = 0

        while power > 0:
            a = coeffs[power]

            # make the permutation shifting the start of the first
            # part ([0..p^i-1] for some i) to the current one
            for _ in range(a):
                shift = Permutation()
                if start > 0:
                    for i in range(p**power):
                        shift = shift(i, start + i)

                    if alt:
                        gen = Permutation(0, 1)*shift*Permutation(0, 1)*shift
                        gens.append(gen)
                        j = 2*(power - 1)
                    else:
                        j = power

                    for i, gen in enumerate(gens[:j]):
                        if alt and i % 2 == 1:
                            continue
                        # shift the generator to the start of the
                        # partition part
                        gen = shift*gen*shift
                        gens.append(gen)

                start += p**power
            power = power-1

        return gens

    def sylow_subgroup(self, p):
        '''
        Return a p-Sylow subgroup of the group.

        The algorithm is described in [1], Chapter 4, Section 7

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import AlternatingGroup

        >>> D = DihedralGroup(6)
        >>> S = D.sylow_subgroup(2)
        >>> S.order()
        4
        >>> G = SymmetricGroup(6)
        >>> S = G.sylow_subgroup(5)
        >>> S.order()
        5

        >>> G1 = AlternatingGroup(3)
        >>> G2 = AlternatingGroup(5)
        >>> G3 = AlternatingGroup(9)

        >>> S1 = G1.sylow_subgroup(3)
        >>> S2 = G2.sylow_subgroup(3)
        >>> S3 = G3.sylow_subgroup(3)

        >>> len1 = len(S1.lower_central_series())
        >>> len2 = len(S2.lower_central_series())
        >>> len3 = len(S3.lower_central_series())

        >>> len1 == len2
        True
        >>> len1 < len3
        True

        '''
        from sympy.combinatorics.homomorphisms import (
                orbit_homomorphism, block_homomorphism)

        if not isprime(p):
            raise ValueError("p must be a prime")

        def is_p_group(G):
            # check if the order of G is a power of p
            # and return the power
            m = G.order()
            n = 0
            while m % p == 0:
                m = m/p
                n += 1
                if m == 1:
                    return True, n
            return False, n

        def _sylow_reduce(mu, nu):
            # reduction based on two homomorphisms
            # mu and nu with trivially intersecting
            # kernels
            Q = mu.image().sylow_subgroup(p)
            Q = mu.invert_subgroup(Q)
            nu = nu.restrict_to(Q)
            R = nu.image().sylow_subgroup(p)
            return nu.invert_subgroup(R)

        order = self.order()
        if order % p != 0:
            return PermutationGroup([self.identity])
        p_group, n = is_p_group(self)
        if p_group:
            return self

        if self.is_alt_sym():
            return PermutationGroup(self._sylow_alt_sym(p))

        # if there is a non-trivial orbit with size not divisible
        # by p, the sylow subgroup is contained in its stabilizer
        # (by orbit-stabilizer theorem)
        orbits = self.orbits()
        non_p_orbits = [o for o in orbits if len(o) % p != 0 and len(o) != 1]
        if non_p_orbits:
            G = self.stabilizer(list(non_p_orbits[0]).pop())
            return G.sylow_subgroup(p)

        if not self.is_transitive():
            # apply _sylow_reduce to orbit actions
            orbits = sorted(orbits, key=len)
            omega1 = orbits.pop()
            omega2 = orbits[0].union(*orbits)
            mu = orbit_homomorphism(self, omega1)
            nu = orbit_homomorphism(self, omega2)
            return _sylow_reduce(mu, nu)

        blocks = self.minimal_blocks()
        if len(blocks) > 1:
            # apply _sylow_reduce to block system actions
            mu = block_homomorphism(self, blocks[0])
            nu = block_homomorphism(self, blocks[1])
            return _sylow_reduce(mu, nu)
        elif len(blocks) == 1:
            block = list(blocks)[0]
            if any(e != 0 for e in block):
                # self is imprimitive
                mu = block_homomorphism(self, block)
                if not is_p_group(mu.image())[0]:
                    S = mu.image().sylow_subgroup(p)
                    return mu.invert_subgroup(S).sylow_subgroup(p)

        # find an element of order p
        g = self.random()
        g_order = g.order()
        while g_order % p != 0 or g_order == 0:
            g = self.random()
            g_order = g.order()
        g = g**(g_order // p)
        if order % p**2 != 0:
            return PermutationGroup(g)

        C = self.centralizer(g)
        while C.order() % p**n != 0:
            S = C.sylow_subgroup(p)
            s_order = S.order()
            Z = S.center()
            P = Z._p_elements_group(p)
            h = P.random()
            C_h = self.centralizer(h)
            while C_h.order() % p*s_order != 0:
                h = P.random()
                C_h = self.centralizer(h)
            C = C_h

        return C.sylow_subgroup(p)

    def _block_verify(self, L, alpha):
        delta = sorted(self.orbit(alpha))
        # p[i] will be the number of the block
        # delta[i] belongs to
        p = [-1]*len(delta)
        blocks = [-1]*len(delta)

        B = [[]] # future list of blocks
        u = [0]*len(delta) # u[i] in L s.t. alpha^u[i] = B[0][i]

        t = L.orbit_transversal(alpha, pairs=True)
        for a, beta in t:
            B[0].append(a)
            i_a = delta.index(a)
            p[i_a] = 0
            blocks[i_a] = alpha
            u[i_a] = beta

        rho = 0
        m = 0 # number of blocks - 1

        while rho <= m:
            beta = B[rho][0]
            for g in self.generators:
                d = beta^g
                i_d = delta.index(d)
                sigma = p[i_d]
                if sigma < 0:
                    # define a new block
                    m += 1
                    sigma = m
                    u[i_d] = u[delta.index(beta)]*g
                    p[i_d] = sigma
                    rep = d
                    blocks[i_d] = rep
                    newb = [rep]
                    for gamma in B[rho][1:]:
                        i_gamma = delta.index(gamma)
                        d = gamma^g
                        i_d = delta.index(d)
                        if p[i_d] < 0:
                            u[i_d] = u[i_gamma]*g
                            p[i_d] = sigma
                            blocks[i_d] = rep
                            newb.append(d)
                        else:
                            # B[rho] is not a block
                            s = u[i_gamma]*g*u[i_d]**(-1)
                            return False, s

                    B.append(newb)
                else:
                    for h in B[rho][1:]:
                        if h^g not in B[sigma]:
                            # B[rho] is not a block
                            s = u[delta.index(beta)]*g*u[i_d]**(-1)
                            return False, s
            rho += 1

        return True, blocks

    def _verify(H, K, phi, z, alpha):
        '''
        Return a list of relators ``rels`` in generators ``gens`_h` that
        are mapped to ``H.generators`` by ``phi`` so that given a finite
        presentation <gens_k | rels_k> of ``K`` on a subset of ``gens_h``
        <gens_h | rels_k + rels> is a finite presentation of ``H``.

        Explanation
        ===========

        ``H`` should be generated by the union of ``K.generators`` and ``z``
        (a single generator), and ``H.stabilizer(alpha) == K``; ``phi`` is a
        canonical injection from a free group into a permutation group
        containing ``H``.

        The algorithm is described in [1], Chapter 6.

        Examples
        ========

        >>> from sympy.combinatorics import free_group, Permutation, PermutationGroup
        >>> from sympy.combinatorics.homomorphisms import homomorphism
        >>> from sympy.combinatorics.fp_groups import FpGroup

        >>> H = PermutationGroup(Permutation(0, 2), Permutation (1, 5))
        >>> K = PermutationGroup(Permutation(5)(0, 2))
        >>> F = free_group("x_0 x_1")[0]
        >>> gens = F.generators
        >>> phi = homomorphism(F, H, F.generators, H.generators)
        >>> rels_k = [gens[0]**2] # relators for presentation of K
        >>> z= Permutation(1, 5)
        >>> check, rels_h = H._verify(K, phi, z, 1)
        >>> check
        True
        >>> rels = rels_k + rels_h
        >>> G = FpGroup(F, rels) # presentation of H
        >>> G.order() == H.order()
        True

        See also
        ========

        strong_presentation, presentation, stabilizer

        '''

        orbit = H.orbit(alpha)
        beta = alpha^(z**-1)

        K_beta = K.stabilizer(beta)

        # orbit representatives of K_beta
        gammas = [alpha, beta]
        orbits = list({tuple(K_beta.orbit(o)) for o in orbit})
        orbit_reps = [orb[0] for orb in orbits]
        for rep in orbit_reps:
            if rep not in gammas:
                gammas.append(rep)

        # orbit transversal of K
        betas = [alpha, beta]
        transversal = {alpha: phi.invert(H.identity), beta: phi.invert(z**-1)}

        for s, g in K.orbit_transversal(beta, pairs=True):
            if s not in transversal:
                transversal[s] = transversal[beta]*phi.invert(g)


        union = K.orbit(alpha).union(K.orbit(beta))
        while (len(union) < len(orbit)):
            for gamma in gammas:
                if gamma in union:
                    r = gamma^z
                    if r not in union:
                        betas.append(r)
                        transversal[r] = transversal[gamma]*phi.invert(z)
                        for s, g in K.orbit_transversal(r, pairs=True):
                            if s not in transversal:
                                transversal[s] = transversal[r]*phi.invert(g)
                        union = union.union(K.orbit(r))
                        break

        # compute relators
        rels = []

        for b in betas:
            k_gens = K.stabilizer(b).generators
            for y in k_gens:
                new_rel = transversal[b]
                gens = K.generator_product(y, original=True)
                for g in gens[::-1]:
                    new_rel = new_rel*phi.invert(g)
                new_rel = new_rel*transversal[b]**-1

                perm = phi(new_rel)
                try:
                    gens = K.generator_product(perm, original=True)
                except ValueError:
                    return False, perm
                for g in gens:
                    new_rel = new_rel*phi.invert(g)**-1
                if new_rel not in rels:
                    rels.append(new_rel)

        for gamma in gammas:
            new_rel = transversal[gamma]*phi.invert(z)*transversal[gamma^z]**-1
            perm = phi(new_rel)
            try:
                gens = K.generator_product(perm, original=True)
            except ValueError:
                return False, perm
            for g in gens:
                new_rel = new_rel*phi.invert(g)**-1
            if new_rel not in rels:
                rels.append(new_rel)

        return True, rels

    def strong_presentation(self):
        '''
        Return a strong finite presentation of group. The generators
        of the returned group are in the same order as the strong
        generators of group.

        The algorithm is based on Sims' Verify algorithm described
        in [1], Chapter 6.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> P = DihedralGroup(4)
        >>> G = P.strong_presentation()
        >>> P.order() == G.order()
        True

        See Also
        ========

        presentation, _verify

        '''
        from sympy.combinatorics.fp_groups import (FpGroup,
                                            simplify_presentation)
        from sympy.combinatorics.free_groups import free_group
        from sympy.combinatorics.homomorphisms import (block_homomorphism,
                                           homomorphism, GroupHomomorphism)

        strong_gens = self.strong_gens[:]
        stabs = self.basic_stabilizers[:]
        base = self.base[:]

        # injection from a free group on len(strong_gens)
        # generators into G
        gen_syms = [('x_%d'%i) for i in range(len(strong_gens))]
        F = free_group(', '.join(gen_syms))[0]
        phi = homomorphism(F, self, F.generators, strong_gens)

        H = PermutationGroup(self.identity)
        while stabs:
            alpha = base.pop()
            K = H
            H = stabs.pop()
            new_gens = [g for g in H.generators if g not in K]

            if K.order() == 1:
                z = new_gens.pop()
                rels = [F.generators[-1]**z.order()]
                intermediate_gens = [z]
                K = PermutationGroup(intermediate_gens)

            # add generators one at a time building up from K to H
            while new_gens:
                z = new_gens.pop()
                intermediate_gens = [z] + intermediate_gens
                K_s = PermutationGroup(intermediate_gens)
                orbit = K_s.orbit(alpha)
                orbit_k = K.orbit(alpha)

                # split into cases based on the orbit of K_s
                if orbit_k == orbit:
                    if z in K:
                        rel = phi.invert(z)
                        perm = z
                    else:
                        t = K.orbit_rep(alpha, alpha^z)
                        rel = phi.invert(z)*phi.invert(t)**-1
                        perm = z*t**-1
                    for g in K.generator_product(perm, original=True):
                        rel = rel*phi.invert(g)**-1
                    new_rels = [rel]
                elif len(orbit_k) == 1:
                    # `success` is always true because `strong_gens`
                    # and `base` are already a verified BSGS. Later
                    # this could be changed to start with a randomly
                    # generated (potential) BSGS, and then new elements
                    # would have to be appended to it when `success`
                    # is false.
                    success, new_rels = K_s._verify(K, phi, z, alpha)
                else:
                    # K.orbit(alpha) should be a block
                    # under the action of K_s on K_s.orbit(alpha)
                    check, block = K_s._block_verify(K, alpha)
                    if check:
                        # apply _verify to the action of K_s
                        # on the block system; for convenience,
                        # add the blocks as additional points
                        # that K_s should act on
                        t = block_homomorphism(K_s, block)
                        m = t.codomain.degree # number of blocks
                        d = K_s.degree

                        # conjugating with p will shift
                        # permutations in t.image() to
                        # higher numbers, e.g.
                        # p*(0 1)*p = (m m+1)
                        p = Permutation()
                        for i in range(m):
                            p *= Permutation(i, i+d)

                        t_img = t.images
                        # combine generators of K_s with their
                        # action on the block system
                        images = {g: g*p*t_img[g]*p for g in t_img}
                        for g in self.strong_gens[:-len(K_s.generators)]:
                            images[g] = g
                        K_s_act = PermutationGroup(list(images.values()))
                        f = GroupHomomorphism(self, K_s_act, images)

                        K_act = PermutationGroup([f(g) for g in K.generators])
                        success, new_rels = K_s_act._verify(K_act, f.compose(phi), f(z), d)

                for n in new_rels:
                    if n not in rels:
                        rels.append(n)
                K = K_s

        group = FpGroup(F, rels)
        return simplify_presentation(group)

    def presentation(self, eliminate_gens=True):
        '''
        Return an `FpGroup` presentation of the group.

        The algorithm is described in [1], Chapter 6.1.

        '''
        from sympy.combinatorics.fp_groups import (FpGroup,
                                            simplify_presentation)
        from sympy.combinatorics.coset_table import CosetTable
        from sympy.combinatorics.free_groups import free_group
        from sympy.combinatorics.homomorphisms import homomorphism

        if self._fp_presentation:
            return self._fp_presentation

        def _factor_group_by_rels(G, rels):
            if isinstance(G, FpGroup):
                rels.extend(G.relators)
                return FpGroup(G.free_group, list(set(rels)))
            return FpGroup(G, rels)

        gens = self.generators
        len_g = len(gens)

        if len_g == 1:
            order = gens[0].order()
            # handle the trivial group
            if order == 1:
                return free_group([])[0]
            F, x = free_group('x')
            return FpGroup(F, [x**order])

        if self.order() > 20:
            half_gens = self.generators[0:(len_g+1)//2]
        else:
            half_gens = []
        H = PermutationGroup(half_gens)
        H_p = H.presentation()

        len_h = len(H_p.generators)

        C = self.coset_table(H)
        n = len(C) # subgroup index

        gen_syms = [('x_%d'%i) for i in range(len(gens))]
        F = free_group(', '.join(gen_syms))[0]

        # mapping generators of H_p to those of F
        images = [F.generators[i] for i in range(len_h)]
        R = homomorphism(H_p, F, H_p.generators, images, check=False)

        # rewrite relators
        rels = R(H_p.relators)
        G_p = FpGroup(F, rels)

        # injective homomorphism from G_p into self
        T = homomorphism(G_p, self, G_p.generators, gens)

        C_p = CosetTable(G_p, [])

        C_p.table = [[None]*(2*len_g) for i in range(n)]

        # initiate the coset transversal
        transversal = [None]*n
        transversal[0] = G_p.identity

        # fill in the coset table as much as possible
        for i in range(2*len_h):
            C_p.table[0][i] = 0

        gamma = 1
        for alpha, x in product(range(n), range(2*len_g)):
            beta = C[alpha][x]
            if beta == gamma:
                gen = G_p.generators[x//2]**((-1)**(x % 2))
                transversal[beta] = transversal[alpha]*gen
                C_p.table[alpha][x] = beta
                C_p.table[beta][x + (-1)**(x % 2)] = alpha
                gamma += 1
                if gamma == n:
                    break

        C_p.p = list(range(n))
        beta = x = 0

        while not C_p.is_complete():
            # find the first undefined entry
            while C_p.table[beta][x] == C[beta][x]:
                x = (x + 1) % (2*len_g)
                if x == 0:
                    beta = (beta + 1) % n

            # define a new relator
            gen = G_p.generators[x//2]**((-1)**(x % 2))
            new_rel = transversal[beta]*gen*transversal[C[beta][x]]**-1
            perm = T(new_rel)
            nxt = G_p.identity
            for s in H.generator_product(perm, original=True):
                nxt = nxt*T.invert(s)**-1
            new_rel = new_rel*nxt

            # continue coset enumeration
            G_p = _factor_group_by_rels(G_p, [new_rel])
            C_p.scan_and_fill(0, new_rel)
            C_p = G_p.coset_enumeration([], strategy="coset_table",
                                draft=C_p, max_cosets=n, incomplete=True)

        self._fp_presentation = simplify_presentation(G_p)
        return self._fp_presentation

    def polycyclic_group(self):
        """
        Return the PolycyclicGroup instance with below parameters:

        Explanation
        ===========

        * pc_sequence : Polycyclic sequence is formed by collecting all
          the missing generators between the adjacent groups in the
          derived series of given permutation group.

        * pc_series : Polycyclic series is formed by adding all the missing
          generators of ``der[i+1]`` in ``der[i]``, where ``der`` represents
          the derived series.

        * relative_order : A list, computed by the ratio of adjacent groups in
          pc_series.

        """
        from sympy.combinatorics.pc_groups import PolycyclicGroup
        if not self.is_polycyclic:
            raise ValueError("The group must be solvable")

        der = self.derived_series()
        pc_series = []
        pc_sequence = []
        relative_order = []
        pc_series.append(der[-1])
        der.reverse()

        for i in range(len(der)-1):
            H = der[i]
            for g in der[i+1].generators:
                if g not in H:
                    H = PermutationGroup([g] + H.generators)
                    pc_series.insert(0, H)
                    pc_sequence.insert(0, g)

                    G1 = pc_series[0].order()
                    G2 = pc_series[1].order()
                    relative_order.insert(0, G1 // G2)

        return PolycyclicGroup(pc_sequence, pc_series, relative_order, collector=None)


def _orbit(degree, generators, alpha, action='tuples'):
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
    tuple under the group action ( i.e., g((1, 2, 3)) = (g(1), g(2), g(3)) )
    'sets' - computes the orbit of the list interpreted as a sets

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> from sympy.combinatorics.perm_groups import _orbit
    >>> a = Permutation([1, 2, 0, 4, 5, 6, 3])
    >>> G = PermutationGroup([a])
    >>> _orbit(G.degree, G.generators, 0)
    {0, 1, 2}
    >>> _orbit(G.degree, G.generators, [0, 4], 'union')
    {0, 1, 2, 3, 4, 5, 6}

    See Also
    ========

    orbit, orbit_transversal

    """
    if not hasattr(alpha, '__getitem__'):
        alpha = [alpha]

    gens = [x._array_form for x in generators]
    if len(alpha) == 1 or action == 'union':
        orb = alpha
        used = [False]*degree
        for el in alpha:
            used[el] = True
        for b in orb:
            for gen in gens:
                temp = gen[b]
                if used[temp] == False:
                    orb.append(temp)
                    used[temp] = True
        return set(orb)
    elif action == 'tuples':
        alpha = tuple(alpha)
        orb = [alpha]
        used = {alpha}
        for b in orb:
            for gen in gens:
                temp = tuple([gen[x] for x in b])
                if temp not in used:
                    orb.append(temp)
                    used.add(temp)
        return set(orb)
    elif action == 'sets':
        alpha = frozenset(alpha)
        orb = [alpha]
        used = {alpha}
        for b in orb:
            for gen in gens:
                temp = frozenset([gen[x] for x in b])
                if temp not in used:
                    orb.append(temp)
                    used.add(temp)
        return {tuple(x) for x in orb}


def _orbits(degree, generators):
    """Compute the orbits of G.

    If ``rep=False`` it returns a list of sets else it returns a list of
    representatives of the orbits

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.perm_groups import _orbits
    >>> a = Permutation([0, 2, 1])
    >>> b = Permutation([1, 0, 2])
    >>> _orbits(a.size, [a, b])
    [{0, 1, 2}]
    """

    orbs = []
    sorted_I = list(range(degree))
    I = set(sorted_I)
    while I:
        i = sorted_I[0]
        orb = _orbit(degree, generators, i)
        orbs.append(orb)
        # remove all indices that are in this orbit
        I -= orb
        sorted_I = [i for i in sorted_I if i not in orb]
    return orbs


def _orbit_transversal(degree, generators, alpha, pairs, af=False, slp=False):
    r"""Computes a transversal for the orbit of ``alpha`` as a set.

    Explanation
    ===========

    generators   generators of the group ``G``

    For a permutation group ``G``, a transversal for the orbit
    `Orb = \{g(\alpha) | g \in G\}` is a set
    `\{g_\beta | g_\beta(\alpha) = \beta\}` for `\beta \in Orb`.
    Note that there may be more than one possible transversal.
    If ``pairs`` is set to ``True``, it returns the list of pairs
    `(\beta, g_\beta)`. For a proof of correctness, see [1], p.79

    if ``af`` is ``True``, the transversal elements are given in
    array form.

    If `slp` is `True`, a dictionary `{beta: slp_beta}` is returned
    for `\beta \in Orb` where `slp_beta` is a list of indices of the
    generators in `generators` s.t. if `slp_beta = [i_1 \dots i_n]`
    `g_\beta = generators[i_n] \times \dots \times generators[i_1]`.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.perm_groups import _orbit_transversal
    >>> G = DihedralGroup(6)
    >>> _orbit_transversal(G.degree, G.generators, 0, False)
    [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]
    """

    tr = [(alpha, list(range(degree)))]
    slp_dict = {alpha: []}
    used = [False]*degree
    used[alpha] = True
    gens = [x._array_form for x in generators]
    for x, px in tr:
        px_slp = slp_dict[x]
        for gen in gens:
            temp = gen[x]
            if used[temp] == False:
                slp_dict[temp] = [gens.index(gen)] + px_slp
                tr.append((temp, _af_rmul(gen, px)))
                used[temp] = True
    if pairs:
        if not af:
            tr = [(x, _af_new(y)) for x, y in tr]
        if not slp:
            return tr
        return tr, slp_dict

    if af:
        tr = [y for _, y in tr]
        if not slp:
            return tr
        return tr, slp_dict

    tr = [_af_new(y) for _, y in tr]
    if not slp:
        return tr
    return tr, slp_dict


def _stabilizer(degree, generators, alpha):
    r"""Return the stabilizer subgroup of ``alpha``.

    Explanation
    ===========

    The stabilizer of `\alpha` is the group `G_\alpha =
    \{g \in G | g(\alpha) = \alpha\}`.
    For a proof of correctness, see [1], p.79.

    degree :       degree of G
    generators :   generators of G

    Examples
    ========

    >>> from sympy.combinatorics.perm_groups import _stabilizer
    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> G = DihedralGroup(6)
    >>> _stabilizer(G.degree, G.generators, 5)
    [(5)(0 4)(1 3), (5)]

    See Also
    ========

    orbit

    """
    orb = [alpha]
    table = {alpha: list(range(degree))}
    table_inv = {alpha: list(range(degree))}
    used = [False]*degree
    used[alpha] = True
    gens = [x._array_form for x in generators]
    stab_gens = []
    for b in orb:
        for gen in gens:
            temp = gen[b]
            if used[temp] is False:
                gen_temp = _af_rmul(gen, table[b])
                orb.append(temp)
                table[temp] = gen_temp
                table_inv[temp] = _af_invert(gen_temp)
                used[temp] = True
            else:
                schreier_gen = _af_rmuln(table_inv[temp], gen, table[b])
                if schreier_gen not in stab_gens:
                    stab_gens.append(schreier_gen)
    return [_af_new(x) for x in stab_gens]


PermGroup = PermutationGroup


class SymmetricPermutationGroup(Basic):
    """
    The class defining the lazy form of SymmetricGroup.

    deg : int

    """
    def __new__(cls, deg):
        deg = _sympify(deg)
        obj = Basic.__new__(cls, deg)
        return obj

    def __init__(self, *args, **kwargs):
        self._deg = self.args[0]
        self._order = None

    def __contains__(self, i):
        """Return ``True`` if *i* is contained in SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> Permutation(1, 2, 3) in G
        True

        """
        if not isinstance(i, Permutation):
            raise TypeError("A SymmetricPermutationGroup contains only Permutations as "
                            "elements, not elements of type %s" % type(i))
        return i.size == self.degree

    def order(self):
        """
        Return the order of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.order()
        24
        """
        if self._order is not None:
            return self._order
        n = self._deg
        self._order = factorial(n)
        return self._order

    @property
    def degree(self):
        """
        Return the degree of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.degree
        4

        """
        return self._deg

    @property
    def identity(self):
        '''
        Return the identity element of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.identity()
        (3)

        '''
        return _af_new(list(range(self._deg)))


class Coset(Basic):
    """A left coset of a permutation group with respect to an element.

    Parameters
    ==========

    g : Permutation

    H : PermutationGroup

    dir : "+" or "-", If not specified by default it will be "+"
        here ``dir`` specified the type of coset "+" represent the
        right coset and "-" represent the left coset.

    G : PermutationGroup, optional
        The group which contains *H* as its subgroup and *g* as its
        element.

        If not specified, it would automatically become a symmetric
        group ``SymmetricPermutationGroup(g.size)`` and
        ``SymmetricPermutationGroup(H.degree)`` if ``g.size`` and ``H.degree``
        are matching.``SymmetricPermutationGroup`` is a lazy form of SymmetricGroup
        used for representation purpose.

    """

    def __new__(cls, g, H, G=None, dir="+"):
        g = _sympify(g)
        if not isinstance(g, Permutation):
            raise NotImplementedError

        H = _sympify(H)
        if not isinstance(H, PermutationGroup):
            raise NotImplementedError

        if G is not None:
            G = _sympify(G)
            if not isinstance(G, (PermutationGroup, SymmetricPermutationGroup)):
                raise NotImplementedError
            if not H.is_subgroup(G):
                raise ValueError("{} must be a subgroup of {}.".format(H, G))
            if g not in G:
                raise ValueError("{} must be an element of {}.".format(g, G))
        else:
            g_size = g.size
            h_degree = H.degree
            if g_size != h_degree:
                raise ValueError(
                    "The size of the permutation {} and the degree of "
                    "the permutation group {} should be matching "
                    .format(g, H))
            G = SymmetricPermutationGroup(g.size)

        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("dir must be of type basestring or "
                    "Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-'):
            raise ValueError("dir must be one of '+' or '-' not %s" % dir)
        obj = Basic.__new__(cls, g, H, G, dir)
        return obj

    def __init__(self, *args, **kwargs):
        self._dir = self.args[3]

    @property
    def is_left_coset(self):
        """
        Check if the coset is left coset that is ``gH``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="-")
        >>> cst.is_left_coset
        True

        """
        return str(self._dir) == '-'

    @property
    def is_right_coset(self):
        """
        Check if the coset is right coset that is ``Hg``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="+")
        >>> cst.is_right_coset
        True

        """
        return str(self._dir) == '+'

    def as_list(self):
        """
        Return all the elements of coset in the form of list.
        """
        g = self.args[0]
        H = self.args[1]
        cst = []
        if str(self._dir) == '+':
            for h in H.elements:
                cst.append(h*g)
        else:
            for h in H.elements:
                cst.append(g*h)
        return cst
<|endoftext|>from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base

rmul = Permutation.rmul


def _cmp_perm_lists(first, second):
    """
    Compare two lists of permutations as sets.

    Explanation
    ===========

    This is used for testing purposes. Since the array form of a
    permutation is currently a list, Permutation is not hashable
    and cannot be put into a set.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.testutil import _cmp_perm_lists
    >>> a = Permutation([0, 2, 3, 4, 1])
    >>> b = Permutation([1, 2, 0, 4, 3])
    >>> c = Permutation([3, 4, 0, 1, 2])
    >>> ls1 = [a, b, c]
    >>> ls2 = [b, c, a]
    >>> _cmp_perm_lists(ls1, ls2)
    True

    """
    return {tuple(a) for a in first} == \
           {tuple(a) for a in second}


def _naive_list_centralizer(self, other, af=False):
    from sympy.combinatorics.perm_groups import PermutationGroup
    """
    Return a list of elements for the centralizer of a subgroup/set/element.

    Explanation
    ===========

    This is a brute force implementation that goes over all elements of the
    group and checks for membership in the centralizer. It is used to
    test ``.centralizer()`` from ``sympy.combinatorics.perm_groups``.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import _naive_list_centralizer
    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> D = DihedralGroup(4)
    >>> _naive_list_centralizer(D, D)
    [Permutation([0, 1, 2, 3]), Permutation([2, 3, 0, 1])]

    See Also
    ========

    sympy.combinatorics.perm_groups.centralizer

    """
    from sympy.combinatorics.permutations import _af_commutes_with
    if hasattr(other, 'generators'):
        elements = list(self.generate_dimino(af=True))
        gens = [x._array_form for x in other.generators]
        commutes_with_gens = lambda x: all(_af_commutes_with(x, gen) for gen in gens)
        centralizer_list = []
        if not af:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(Permutation._af_new(element))
        else:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(element)
        return centralizer_list
    elif hasattr(other, 'getitem'):
        return _naive_list_centralizer(self, PermutationGroup(other), af)
    elif hasattr(other, 'array_form'):
        return _naive_list_centralizer(self, PermutationGroup([other]), af)


def _verify_bsgs(group, base, gens):
    """
    Verify the correctness of a base and strong generating set.

    Explanation
    ===========

    This is a naive implementation using the definition of a base and a strong
    generating set relative to it. There are other procedures for
    verifying a base and strong generating set, but this one will
    serve for more robust testing.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import AlternatingGroup
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> A = AlternatingGroup(4)
    >>> A.schreier_sims()
    >>> _verify_bsgs(A, A.base, A.strong_gens)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims

    """
    from sympy.combinatorics.perm_groups import PermutationGroup
    strong_gens_distr = _distribute_gens_by_base(base, gens)
    current_stabilizer = group
    for i in range(len(base)):
        candidate = PermutationGroup(strong_gens_distr[i])
        if current_stabilizer.order() != candidate.order():
            return False
        current_stabilizer = current_stabilizer.stabilizer(base[i])
    if current_stabilizer.order() != 1:
        return False
    return True


def _verify_centralizer(group, arg, centr=None):
    """
    Verify the centralizer of a group/set/element inside another group.

    This is used for testing ``.centralizer()`` from
    ``sympy.combinatorics.perm_groups``

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
    ... AlternatingGroup)
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.testutil import _verify_centralizer
    >>> S = SymmetricGroup(5)
    >>> A = AlternatingGroup(5)
    >>> centr = PermutationGroup([Permutation([0, 1, 2, 3, 4])])
    >>> _verify_centralizer(S, A, centr)
    True

    See Also
    ========

    _naive_list_centralizer,
    sympy.combinatorics.perm_groups.PermutationGroup.centralizer,
    _cmp_perm_lists

    """
    if centr is None:
        centr = group.centralizer(arg)
    centr_list = list(centr.generate_dimino(af=True))
    centr_list_naive = _naive_list_centralizer(group, arg, af=True)
    return _cmp_perm_lists(centr_list, centr_list_naive)


def _verify_normal_closure(group, arg, closure=None):
    from sympy.combinatorics.perm_groups import PermutationGroup
    """
    Verify the normal closure of a subgroup/subset/element in a group.

    This is used to test
    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
    ... AlternatingGroup)
    >>> from sympy.combinatorics.testutil import _verify_normal_closure
    >>> S = SymmetricGroup(3)
    >>> A = AlternatingGroup(3)
    >>> _verify_normal_closure(S, A, closure=A)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure

    """
    if closure is None:
        closure = group.normal_closure(arg)
    conjugates = set()
    if hasattr(arg, 'generators'):
        subgr_gens = arg.generators
    elif hasattr(arg, '__getitem__'):
        subgr_gens = arg
    elif hasattr(arg, 'array_form'):
        subgr_gens = [arg]
    for el in group.generate_dimino():
        conjugates.update(gen ^ el for gen in subgr_gens)
    naive_closure = PermutationGroup(list(conjugates))
    return closure.is_subgroup(naive_closure)


def canonicalize_naive(g, dummies, sym, *v):
    """
    Canonicalize tensor formed by tensors of the different types.

    Explanation
    ===========

    sym_i symmetry under exchange of two component tensors of type `i`
          None  no symmetry
          0     commuting
          1     anticommuting

    Parameters
    ==========

    g : Permutation representing the tensor.
    dummies : List of dummy indices.
    msym : Symmetry of the metric.
    v : A list of (base_i, gens_i, n_i, sym_i) for tensors of type `i`.
        base_i, gens_i BSGS for tensors of this type
        n_i  number of tensors of type `i`

    Returns
    =======

    Returns 0 if the tensor is zero, else returns the array form of
    the permutation representing the canonical form of the tensor.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import canonicalize_naive
    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
    >>> from sympy.combinatorics import Permutation
    >>> g = Permutation([1, 3, 2, 0, 4, 5])
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> canonicalize_naive(g, [2, 3], 0, (base2, gens2, 2, 0))
    [0, 2, 1, 3, 4, 5]
    """
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.combinatorics.tensor_can import gens_products, dummy_sgs
    from sympy.combinatorics.permutations import _af_rmul
    v1 = []
    for i in range(len(v)):
        base_i, gens_i, n_i, sym_i = v[i]
        v1.append((base_i, gens_i, [[]]*n_i, sym_i))
    size, sbase, sgens = gens_products(*v1)
    dgens = dummy_sgs(dummies, sym, size-2)
    if isinstance(sym, int):
        num_types = 1
        dummies = [dummies]
        sym = [sym]
    else:
        num_types = len(sym)
    dgens = []
    for i in range(num_types):
        dgens.extend(dummy_sgs(dummies[i], sym[i], size - 2))
    S = PermutationGroup(sgens)
    D = PermutationGroup([Permutation(x) for x in dgens])
    dlist = list(D.generate(af=True))
    g = g.array_form
    st = set()
    for s in S.generate(af=True):
        h = _af_rmul(g, s)
        for d in dlist:
            q = tuple(_af_rmul(d, h))
            st.add(q)
    a = list(st)
    a.sort()
    prev = (0,)*size
    for h in a:
        if h[:-2] == prev[:-2]:
            if h[-1] != prev[-1]:
                return 0
        prev = h
    return list(a[0])


def graph_certificate(gr):
    """
    Return a certificate for the graph

    Parameters
    ==========

    gr : adjacency list

    Explanation
    ===========

    The graph is assumed to be unoriented and without
    external lines.

    Associate to each vertex of the graph a symmetric tensor with
    number of indices equal to the degree of the vertex; indices
    are contracted when they correspond to the same line of the graph.
    The canonical form of the tensor gives a certificate for the graph.

    This is not an efficient algorithm to get the certificate of a graph.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import graph_certificate
    >>> gr1 = {0:[1, 2, 3, 5], 1:[0, 2, 4], 2:[0, 1, 3, 4], 3:[0, 2, 4], 4:[1, 2, 3, 5], 5:[0, 4]}
    >>> gr2 = {0:[1, 5], 1:[0, 2, 3, 4], 2:[1, 3, 5], 3:[1, 2, 4, 5], 4:[1, 3, 5], 5:[0, 2, 3, 4]}
    >>> c1 = graph_certificate(gr1)
    >>> c2 = graph_certificate(gr2)
    >>> c1
    [0, 2, 4, 6, 1, 8, 10, 12, 3, 14, 16, 18, 5, 9, 15, 7, 11, 17, 13, 19, 20, 21]
    >>> c1 == c2
    True
    """
    from sympy.combinatorics.permutations import _af_invert
    from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize
    items = list(gr.items())
    items.sort(key=lambda x: len(x[1]), reverse=True)
    pvert = [x[0] for x in items]
    pvert = _af_invert(pvert)

    # the indices of the tensor are twice the number of lines of the graph
    num_indices = 0
    for v, neigh in items:
        num_indices += len(neigh)
    # associate to each vertex its indices; for each line
    # between two vertices assign the
    # even index to the vertex which comes first in items,
    # the odd index to the other vertex
    vertices = [[] for i in items]
    i = 0
    for v, neigh in items:
        for v2 in neigh:
            if pvert[v] < pvert[v2]:
                vertices[pvert[v]].append(i)
                vertices[pvert[v2]].append(i+1)
                i += 2
    g = []
    for v in vertices:
        g.extend(v)
    assert len(g) == num_indices
    g += [num_indices, num_indices + 1]
    size = num_indices + 2
    assert sorted(g) == list(range(size))
    g = Permutation(g)
    vlen = [0]*(len(vertices[0])+1)
    for neigh in vertices:
        vlen[len(neigh)] += 1
    v = []
    for i in range(len(vlen)):
        n = vlen[i]
        if n:
            base, gens = get_symmetric_group_sgs(i)
            v.append((base, gens, n, 0))
    v.reverse()
    dummies = list(range(num_indices))
    can = canonicalize(g, dummies, 0, *v)
    return can
<|endoftext|>from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine

class RewritingSystem:
    '''
    A class implementing rewriting systems for `FpGroup`s.

    References
    ==========
    .. [1] Epstein, D., Holt, D. and Rees, S. (1991).
           The use of Knuth-Bendix methods to solve the word problem in automatic groups.
           Journal of Symbolic Computation, 12(4-5), pp.397-414.

    .. [2] GAP's Manual on its KBMAG package
           https://www.gap-system.org/Manuals/pkg/kbmag-1.5.3/doc/manual.pdf

    '''
    def __init__(self, group):
        self.group = group
        self.alphabet = group.generators
        self._is_confluent = None

        # these values are taken from [2]
        self.maxeqns = 32767 # max rules
        self.tidyint = 100 # rules before tidying

        # _max_exceeded is True if maxeqns is exceeded
        # at any point
        self._max_exceeded = False

        # Reduction automaton
        self.reduction_automaton = None
        self._new_rules = {}

        # dictionary of reductions
        self.rules = {}
        self.rules_cache = deque([], 50)
        self._init_rules()


        # All the transition symbols in the automaton
        generators = list(self.alphabet)
        generators += [gen**-1 for gen in generators]
        # Create a finite state machine as an instance of the StateMachine object
        self.reduction_automaton = StateMachine('Reduction automaton for '+ repr(self.group), generators)
        self.construct_automaton()

    def set_max(self, n):
        '''
        Set the maximum number of rules that can be defined

        '''
        if n > self.maxeqns:
            self._max_exceeded = False
        self.maxeqns = n
        return

    @property
    def is_confluent(self):
        '''
        Return `True` if the system is confluent

        '''
        if self._is_confluent is None:
            self._is_confluent = self._check_confluence()
        return self._is_confluent

    def _init_rules(self):
        identity = self.group.free_group.identity
        for r in self.group.relators:
            self.add_rule(r, identity)
        self._remove_redundancies()
        return

    def _add_rule(self, r1, r2):
        '''
        Add the rule r1 -> r2 with no checking or further
        deductions

        '''
        if len(self.rules) + 1 > self.maxeqns:
            self._is_confluent = self._check_confluence()
            self._max_exceeded = True
            raise RuntimeError("Too many rules were defined.")
        self.rules[r1] = r2
        # Add the newly added rule to the `new_rules` dictionary.
        if self.reduction_automaton:
            self._new_rules[r1] = r2

    def add_rule(self, w1, w2, check=False):
        new_keys = set()

        if w1 == w2:
            return new_keys

        if w1 < w2:
            w1, w2 = w2, w1

        if (w1, w2) in self.rules_cache:
            return new_keys
        self.rules_cache.append((w1, w2))

        s1, s2 = w1, w2

        # The following is the equivalent of checking
        # s1 for overlaps with the implicit reductions
        # {g*g**-1 -> <identity>} and {g**-1*g -> <identity>}
        # for any generator g without installing the
        # redundant rules that would result from processing
        # the overlaps. See [1], Section 3 for details.

        if len(s1) - len(s2) < 3:
            if s1 not in self.rules:
                new_keys.add(s1)
                if not check:
                    self._add_rule(s1, s2)
            if s2**-1 > s1**-1 and s2**-1 not in self.rules:
                new_keys.add(s2**-1)
                if not check:
                    self._add_rule(s2**-1, s1**-1)

        # overlaps on the right
        while len(s1) - len(s2) > -1:
            g = s1[len(s1)-1]
            s1 = s1.subword(0, len(s1)-1)
            s2 = s2*g**-1
            if len(s1) - len(s2) < 0:
                if s2 not in self.rules:
                    if not check:
                        self._add_rule(s2, s1)
                    new_keys.add(s2)
            elif len(s1) - len(s2) < 3:
                new = self.add_rule(s1, s2, check)
                new_keys.update(new)

        # overlaps on the left
        while len(w1) - len(w2) > -1:
            g = w1[0]
            w1 = w1.subword(1, len(w1))
            w2 = g**-1*w2
            if len(w1) - len(w2) < 0:
                if w2 not in self.rules:
                    if not check:
                        self._add_rule(w2, w1)
                    new_keys.add(w2)
            elif len(w1) - len(w2) < 3:
                new = self.add_rule(w1, w2, check)
                new_keys.update(new)

        return new_keys

    def _remove_redundancies(self, changes=False):
        '''
        Reduce left- and right-hand sides of reduction rules
        and remove redundant equations (i.e. those for which
        lhs == rhs). If `changes` is `True`, return a set
        containing the removed keys and a set containing the
        added keys

        '''
        removed = set()
        added = set()
        rules = self.rules.copy()
        for r in rules:
            v = self.reduce(r, exclude=r)
            w = self.reduce(rules[r])
            if v != r:
                del self.rules[r]
                removed.add(r)
                if v > w:
                    added.add(v)
                    self.rules[v] = w
                elif v < w:
                    added.add(w)
                    self.rules[w] = v
            else:
                self.rules[v] = w
        if changes:
            return removed, added
        return

    def make_confluent(self, check=False):
        '''
        Try to make the system confluent using the Knuth-Bendix
        completion algorithm

        '''
        if self._max_exceeded:
            return self._is_confluent
        lhs = list(self.rules.keys())

        def _overlaps(r1, r2):
            len1 = len(r1)
            len2 = len(r2)
            result = []
            for j in range(1, len1 + len2):
                if (r1.subword(len1 - j, len1 + len2 - j, strict=False)
                       == r2.subword(j - len1, j, strict=False)):
                    a = r1.subword(0, len1-j, strict=False)
                    a = a*r2.subword(0, j-len1, strict=False)
                    b = r2.subword(j-len1, j, strict=False)
                    c = r2.subword(j, len2, strict=False)
                    c = c*r1.subword(len1 + len2 - j, len1, strict=False)
                    result.append(a*b*c)
            return result

        def _process_overlap(w, r1, r2, check):
                s = w.eliminate_word(r1, self.rules[r1])
                s = self.reduce(s)
                t = w.eliminate_word(r2, self.rules[r2])
                t = self.reduce(t)
                if s != t:
                    if check:
                        # system not confluent
                        return [0]
                    try:
                        new_keys = self.add_rule(t, s, check)
                        return new_keys
                    except RuntimeError:
                        return False
                return

        added = 0
        i = 0
        while i < len(lhs):
            r1 = lhs[i]
            i += 1
            # j could be i+1 to not
            # check each pair twice but lhs
            # is extended in the loop and the new
            # elements have to be checked with the
            # preceding ones. there is probably a better way
            # to handle this
            j = 0
            while j < len(lhs):
                r2 = lhs[j]
                j += 1
                if r1 == r2:
                    continue
                overlaps = _overlaps(r1, r2)
                overlaps.extend(_overlaps(r1**-1, r2))
                if not overlaps:
                    continue
                for w in overlaps:
                    new_keys = _process_overlap(w, r1, r2, check)
                    if new_keys:
                        if check:
                            return False
                        lhs.extend(new_keys)
                        added += len(new_keys)
                    elif new_keys == False:
                        # too many rules were added so the process
                        # couldn't complete
                        return self._is_confluent

                if added > self.tidyint and not check:
                    # tidy up
                    r, a = self._remove_redundancies(changes=True)
                    added = 0
                    if r:
                        # reset i since some elements were removed
                        i = min(lhs.index(s) for s in r)
                    lhs = [l for l in lhs if l not in r]
                    lhs.extend(a)
                    if r1 in r:
                        # r1 was removed as redundant
                        break

        self._is_confluent = True
        if not check:
            self._remove_redundancies()
        return True

    def _check_confluence(self):
        return self.make_confluent(check=True)

    def reduce(self, word, exclude=None):
        '''
        Apply reduction rules to `word` excluding the reduction rule
        for the lhs equal to `exclude`

        '''
        rules = {r: self.rules[r] for r in self.rules if r != exclude}
        # the following is essentially `eliminate_words()` code from the
        # `FreeGroupElement` class, the only difference being the first
        # "if" statement
        again = True
        new = word
        while again:
            again = False
            for r in rules:
                prev = new
                if rules[r]**-1 > r**-1:
                    new = new.eliminate_word(r, rules[r], _all=True, inverse=False)
                else:
                    new = new.eliminate_word(r, rules[r], _all=True)
                if new != prev:
                    again = True
        return new

    def _compute_inverse_rules(self, rules):
        '''
        Compute the inverse rules for a given set of rules.
        The inverse rules are used in the automaton for word reduction.

        Arguments:
            rules (dictionary): Rules for which the inverse rules are to computed.

        Returns:
            Dictionary of inverse_rules.

        '''
        inverse_rules = {}
        for r in rules:
            rule_key_inverse = r**-1
            rule_value_inverse = (rules[r])**-1
            if (rule_value_inverse < rule_key_inverse):
                inverse_rules[rule_key_inverse] = rule_value_inverse
            else:
                inverse_rules[rule_value_inverse] = rule_key_inverse
        return inverse_rules

    def construct_automaton(self):
        '''
        Construct the automaton based on the set of reduction rules of the system.

        Automata Design:
        The accept states of the automaton are the proper prefixes of the left hand side of the rules.
        The complete left hand side of the rules are the dead states of the automaton.

        '''
        self._add_to_automaton(self.rules)

    def _add_to_automaton(self, rules):
        '''
        Add new states and transitions to the automaton.

        Summary:
        States corresponding to the new rules added to the system are computed and added to the automaton.
        Transitions in the previously added states are also modified if necessary.

        Arguments:
            rules (dictionary) -- Dictionary of the newly added rules.

        '''
        # Automaton variables
        automaton_alphabet = []
        proper_prefixes = {}

        # compute the inverses of all the new rules added
        all_rules = rules
        inverse_rules = self._compute_inverse_rules(all_rules)
        all_rules.update(inverse_rules)

        # Keep track of the accept_states.
        accept_states = []

        for rule in all_rules:
            # The symbols present in the new rules are the symbols to be verified at each state.
            # computes the automaton_alphabet, as the transitions solely depend upon the new states.
            automaton_alphabet += rule.letter_form_elm
            # Compute the proper prefixes for every rule.
            proper_prefixes[rule] = []
            letter_word_array = list(rule.letter_form_elm)
            len_letter_word_array = len(letter_word_array)
            for i in range (1, len_letter_word_array):
                letter_word_array[i] = letter_word_array[i-1]*letter_word_array[i]
                # Add accept states.
                elem = letter_word_array[i-1]
                if elem not in self.reduction_automaton.states:
                    self.reduction_automaton.add_state(elem, state_type='a')
                    accept_states.append(elem)
            proper_prefixes[rule] = letter_word_array
            # Check for overlaps between dead and accept states.
            if rule in accept_states:
                self.reduction_automaton.states[rule].state_type = 'd'
                self.reduction_automaton.states[rule].rh_rule = all_rules[rule]
                accept_states.remove(rule)
            # Add dead states
            if rule not in self.reduction_automaton.states:
                self.reduction_automaton.add_state(rule, state_type='d', rh_rule=all_rules[rule])

        automaton_alphabet = set(automaton_alphabet)

        # Add new transitions for every state.
        for state in self.reduction_automaton.states:
            current_state_name = state
            current_state_type = self.reduction_automaton.states[state].state_type
            # Transitions will be modified only when suffixes of the current_state
            # belongs to the proper_prefixes of the new rules.
            # The rest are ignored if they cannot lead to a dead state after a finite number of transisitons.
            if current_state_type == 's':
                for letter in automaton_alphabet:
                    if letter in self.reduction_automaton.states:
                        self.reduction_automaton.states[state].add_transition(letter, letter)
                    else:
                        self.reduction_automaton.states[state].add_transition(letter, current_state_name)
            elif current_state_type == 'a':
                # Check if the transition to any new state in possible.
                for letter in automaton_alphabet:
                    _next = current_state_name*letter
                    while len(_next) and _next not in self.reduction_automaton.states:
                        _next = _next.subword(1, len(_next))
                    if not len(_next):
                        _next = 'start'
                    self.reduction_automaton.states[state].add_transition(letter, _next)

        # Add transitions for new states. All symbols used in the automaton are considered here.
        # Ignore this if `reduction_automaton.automaton_alphabet` = `automaton_alphabet`.
        if len(self.reduction_automaton.automaton_alphabet) != len(automaton_alphabet):
            for state in accept_states:
                current_state_name = state
                for letter in self.reduction_automaton.automaton_alphabet:
                    _next = current_state_name*letter
                    while len(_next) and _next not in self.reduction_automaton.states:
                        _next = _next.subword(1, len(_next))
                    if not len(_next):
                        _next = 'start'
                    self.reduction_automaton.states[state].add_transition(letter, _next)

    def reduce_using_automaton(self, word):
        '''
        Reduce a word using an automaton.

        Summary:
        All the symbols of the word are stored in an array and are given as the input to the automaton.
        If the automaton reaches a dead state that subword is replaced and the automaton is run from the beginning.
        The complete word has to be replaced when the word is read and the automaton reaches a dead state.
        So, this process is repeated until the word is read completely and the automaton reaches the accept state.

        Arguments:
            word (instance of FreeGroupElement) -- Word that needs to be reduced.

        '''
        # Modify the automaton if new rules are found.
        if self._new_rules:
            self._add_to_automaton(self._new_rules)
            self._new_rules = {}

        flag = 1
        while flag:
            flag = 0
            current_state = self.reduction_automaton.states['start']
            for i, s in enumerate(word.letter_form_elm):
                next_state_name = current_state.transitions[s]
                next_state = self.reduction_automaton.states[next_state_name]
                if next_state.state_type == 'd':
                    subst = next_state.rh_rule
                    word = word.substituted_word(i - len(next_state_name) + 1, i+1, subst)
                    flag = 1
                    break
                current_state = next_state
        return word
<|endoftext|>from __future__ import annotations

from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int


@public
def free_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1))``.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics import free_group
    >>> F, x, y, z = free_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> x**2*y**-1
    x**2*y**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    return (_free_group,) + tuple(_free_group.generators)

@public
def xfree_group(symbols):
    """Construct a free group returning ``(FreeGroup, (f_0, f_1, ..., f_(n-1)))``.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import xfree_group
    >>> F, (x, y, z) = xfree_group("x, y, z")
    >>> F
    <free group on the generators (x, y, z)>
    >>> y**2*x**-2*z**-1
    y**2*x**-2*z**-1
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    return (_free_group, _free_group.generators)

@public
def vfree_group(symbols):
    """Construct a free group and inject ``f_0, f_1, ..., f_(n-1)`` as symbols
    into the global namespace.

    Parameters
    ==========

    symbols : str, Symbol/Expr or sequence of str, Symbol/Expr (may be empty)

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import vfree_group
    >>> vfree_group("x, y, z")
    <free group on the generators (x, y, z)>
    >>> x**2*y**-2*z # noqa: F821
    x**2*y**-2*z
    >>> type(_)
    <class 'sympy.combinatorics.free_groups.FreeGroupElement'>

    """
    _free_group = FreeGroup(symbols)
    pollute([sym.name for sym in _free_group.symbols], _free_group.generators)
    return _free_group


def _parse_symbols(symbols):
    if not symbols:
        return ()
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True)
    elif isinstance(symbols, (Expr, FreeGroupElement)):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols
    raise ValueError("The type of `symbols` must be one of the following: "
                     "a str, Symbol/Expr or a sequence of "
                     "one of these types")


##############################################################################
#                          FREE GROUP                                        #
##############################################################################

_free_group_cache: dict[int, FreeGroup] = {}

class FreeGroup(DefaultPrinting):
    """
    Free group with finite or infinite number of generators. Its input API
    is that of a str, Symbol/Expr or a sequence of one of
    these types (which may be empty)

    See Also
    ========

    sympy.polys.rings.PolyRing

    References
    ==========

    .. [1] https://www.gap-system.org/Manuals/doc/ref/chap37.html

    .. [2] https://en.wikipedia.org/wiki/Free_group

    """
    is_associative = True
    is_group = True
    is_FreeGroup = True
    is_PermutationGroup = False
    relators: list[Expr] = []

    def __new__(cls, symbols):
        symbols = tuple(_parse_symbols(symbols))
        rank = len(symbols)
        _hash = hash((cls.__name__, symbols, rank))
        obj = _free_group_cache.get(_hash)

        if obj is None:
            obj = object.__new__(cls)
            obj._hash = _hash
            obj._rank = rank
            # dtype method is used to create new instances of FreeGroupElement
            obj.dtype = type("FreeGroupElement", (FreeGroupElement,), {"group": obj})
            obj.symbols = symbols
            obj.generators = obj._generators()
            obj._gens_set = set(obj.generators)
            for symbol, generator in zip(obj.symbols, obj.generators):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if hasattr(obj, name):
                        setattr(obj, name, generator)

            _free_group_cache[_hash] = obj

        return obj

    def __getnewargs__(self):
        """Return a tuple of arguments that must be passed to __new__ in order to support pickling this object."""
        return (self.symbols,)

    def __getstate__(self):
        # Don't pickle any fields because they are regenerated within __new__
        return None

    def _generators(group):
        """Returns the generators of the FreeGroup.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> F.generators
        (x, y, z)

        """
        gens = []
        for sym in group.symbols:
            elm = ((sym, 1),)
            gens.append(group.dtype(elm))
        return tuple(gens)

    def clone(self, symbols=None):
        return self.__class__(symbols or self.symbols)

    def __contains__(self, i):
        """Return True if ``i`` is contained in FreeGroup."""
        if not isinstance(i, FreeGroupElement):
            return False
        group = i.group
        return self == group

    def __hash__(self):
        return self._hash

    def __len__(self):
        return self.rank

    def __str__(self):
        if self.rank > 30:
            str_form = "<free group with %s generators>" % self.rank
        else:
            str_form = "<free group on the generators "
            gens = self.generators
            str_form += str(gens) + ">"
        return str_form

    __repr__ = __str__

    def __getitem__(self, index):
        symbols = self.symbols[index]
        return self.clone(symbols=symbols)

    def __eq__(self, other):
        """No ``FreeGroup`` is equal to any "other" ``FreeGroup``.
        """
        return self is other

    def index(self, gen):
        """Return the index of the generator `gen` from ``(f_0, ..., f_(n-1))``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.index(y)
        1
        >>> F.index(x)
        0

        """
        if isinstance(gen, self.dtype):
            return self.generators.index(gen)
        else:
            raise ValueError("expected a generator of Free Group %s, got %s" % (self, gen))

    def order(self):
        """Return the order of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> F.order()
        oo

        >>> free_group("")[0].order()
        1

        """
        if self.rank == 0:
            return S.One
        else:
            return S.Infinity

    @property
    def elements(self):
        """
        Return the elements of the free group.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> (z,) = free_group("")
        >>> z.elements
        {<identity>}

        """
        if self.rank == 0:
            # A set containing Identity element of `FreeGroup` self is returned
            return {self.identity}
        else:
            raise ValueError("Group contains infinitely many elements"
                            ", hence cannot be represented")

    @property
    def rank(self):
        r"""
        In group theory, the `rank` of a group `G`, denoted `G.rank`,
        can refer to the smallest cardinality of a generating set
        for G, that is

        \operatorname{rank}(G)=\min\{ |X|: X\subseteq G, \left\langle X\right\rangle =G\}.

        """
        return self._rank

    @property
    def is_abelian(self):
        """Returns if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.is_abelian
        False

        """
        return self.rank in (0, 1)

    @property
    def identity(self):
        """Returns the identity element of free group."""
        return self.dtype()

    def contains(self, g):
        """Tests if Free Group element ``g`` belong to self, ``G``.

        In mathematical terms any linear combination of generators
        of a Free Group is contained in it.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> f.contains(x**3*y**2)
        True

        """
        if not isinstance(g, FreeGroupElement):
            return False
        elif self != g.group:
            return False
        else:
            return True

    def center(self):
        """Returns the center of the free group `self`."""
        return {self.identity}


############################################################################
#                          FreeGroupElement                                #
############################################################################


class FreeGroupElement(CantSympify, DefaultPrinting, tuple):
    """Used to create elements of FreeGroup. It cannot be used directly to
    create a free group element. It is called by the `dtype` method of the
    `FreeGroup` class.

    """
    __slots__ = ()
    is_assoc_word = True

    def new(self, init):
        return self.__class__(init)

    _hash = None

    def __hash__(self):
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.group, frozenset(tuple(self))))
        return _hash

    def copy(self):
        return self.new(self)

    @property
    def is_identity(self):
        return not self.array_form

    @property
    def array_form(self):
        """
        SymPy provides two different internal kinds of representation
        of associative words. The first one is called the `array_form`
        which is a tuple containing `tuples` as its elements, where the
        size of each tuple is two. At the first position the tuple
        contains the `symbol-generator`, while at the second position
        of tuple contains the exponent of that generator at the position.
        Since elements (i.e. words) do not commute, the indexing of tuple
        makes that property to stay.

        The structure in ``array_form`` of ``FreeGroupElement`` is of form:

        ``( ( symbol_of_gen, exponent ), ( , ), ... ( , ) )``

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> (x*z).array_form
        ((x, 1), (z, 1))
        >>> (x**2*z*y*x**2).array_form
        ((x, 2), (z, 1), (y, 1), (x, 2))

        See Also
        ========

        letter_repr

        """
        return tuple(self)

    @property
    def letter_form(self):
        """
        The letter representation of a ``FreeGroupElement`` is a tuple
        of generator symbols, with each entry corresponding to a group
        generator. Inverses of the generators are represented by
        negative generator symbols.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b, c, d = free_group("a b c d")
        >>> (a**3).letter_form
        (a, a, a)
        >>> (a**2*d**-2*a*b**-4).letter_form
        (a, a, -d, -d, a, -b, -b, -b, -b)
        >>> (a**-2*b**3*d).letter_form
        (-a, -a, b, b, b, d)

        See Also
        ========

        array_form

        """
        return tuple(flatten([(i,)*j if j > 0 else (-i,)*(-j)
                    for i, j in self.array_form]))

    def __getitem__(self, i):
        group = self.group
        r = self.letter_form[i]
        if r.is_Symbol:
            return group.dtype(((r, 1),))
        else:
            return group.dtype(((-r, -1),))

    def index(self, gen):
        if len(gen) != 1:
            raise ValueError()
        return (self.letter_form).index(gen.letter_form[0])

    @property
    def letter_form_elm(self):
        """
        """
        group = self.group
        r = self.letter_form
        return [group.dtype(((elm,1),)) if elm.is_Symbol \
                else group.dtype(((-elm,-1),)) for elm in r]

    @property
    def ext_rep(self):
        """This is called the External Representation of ``FreeGroupElement``
        """
        return tuple(flatten(self.array_form))

    def __contains__(self, gen):
        return gen.array_form[0][0] in tuple([r[0] for r in self.array_form])

    def __str__(self):
        if self.is_identity:
            return "<identity>"

        str_form = ""
        array_form = self.array_form
        for i in range(len(array_form)):
            if i == len(array_form) - 1:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0])
                else:
                    str_form += str(array_form[i][0]) + \
                                    "**" + str(array_form[i][1])
            else:
                if array_form[i][1] == 1:
                    str_form += str(array_form[i][0]) + "*"
                else:
                    str_form += str(array_form[i][0]) + \
                                    "**" + str(array_form[i][1]) + "*"
        return str_form

    __repr__ = __str__

    def __pow__(self, n):
        n = as_int(n)
        result = self.group.identity
        if n == 0:
            return result
        if n < 0:
            n = -n
            x = self.inverse()
        else:
            x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __mul__(self, other):
        """Returns the product of elements belonging to the same ``FreeGroup``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> x*y**2*y**-4
        x*y**-2
        >>> z*y**-2
        z*y**-2
        >>> x**2*y*y**-1*x**-2
        <identity>

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        r = list(self.array_form + other.array_form)
        zero_mul_simp(r, len(self.array_form) - 1)
        return group.dtype(tuple(r))

    def __truediv__(self, other):
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        return self*(other.inverse())

    def __rtruediv__(self, other):
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                    "be multiplied")
        return other*(self.inverse())

    def __add__(self, other):
        return NotImplemented

    def inverse(self):
        """
        Returns the inverse of a ``FreeGroupElement`` element

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> x.inverse()
        x**-1
        >>> (x*y).inverse()
        y**-1*x**-1

        """
        group = self.group
        r = tuple([(i, -j) for i, j in self.array_form[::-1]])
        return group.dtype(r)

    def order(self):
        """Find the order of a ``FreeGroupElement``.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> (x**2*y*y**-1*x**-2).order()
        1

        """
        if self.is_identity:
            return S.One
        else:
            return S.Infinity

    def commutator(self, other):
        """
        Return the commutator of `self` and `x`: ``~x*~self*x*self``

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise ValueError("commutator of only FreeGroupElement of the same "
                    "FreeGroup exists")
        else:
            return self.inverse()*other.inverse()*self*other

    def eliminate_words(self, words, _all=False, inverse=True):
        '''
        Replace each subword from the dictionary `words` by words[subword].
        If words is a list, replace the words by the identity.

        '''
        again = True
        new = self
        if isinstance(words, dict):
            while again:
                again = False
                for sub in words:
                    prev = new
                    new = new.eliminate_word(sub, words[sub], _all=_all, inverse=inverse)
                    if new != prev:
                        again = True
        else:
            while again:
                again = False
                for sub in words:
                    prev = new
                    new = new.eliminate_word(sub, _all=_all, inverse=inverse)
                    if new != prev:
                        again = True
        return new

    def eliminate_word(self, gen, by=None, _all=False, inverse=True):
        """
        For an associative word `self`, a subword `gen`, and an associative
        word `by` (identity by default), return the associative word obtained by
        replacing each occurrence of `gen` in `self` by `by`. If `_all = True`,
        the occurrences of `gen` that may appear after the first substitution will
        also be replaced and so on until no occurrences are found. This might not
        always terminate (e.g. `(x).eliminate_word(x, x**2, _all=True)`).

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y = free_group("x y")
        >>> w = x**5*y*x**2*y**-4*x
        >>> w.eliminate_word( x, x**2 )
        x**10*y*x**4*y**-4*x**2
        >>> w.eliminate_word( x, y**-1 )
        y**-11
        >>> w.eliminate_word(x**5)
        y*x**2*y**-4*x
        >>> w.eliminate_word(x*y, y)
        x**4*y*x**2*y**-4*x

        See Also
        ========
        substituted_word

        """
        if by is None:
            by = self.group.identity
        if self.is_independent(gen) or gen == by:
            return self
        if gen == self:
            return by
        if gen**-1 == by:
            _all = False
        word = self
        l = len(gen)

        try:
            i = word.subword_index(gen)
            k = 1
        except ValueError:
            if not inverse:
                return word
            try:
                i = word.subword_index(gen**-1)
                k = -1
            except ValueError:
                return word

        word = word.subword(0, i)*by**k*word.subword(i+l, len(word)).eliminate_word(gen, by)

        if _all:
            return word.eliminate_word(gen, by, _all=True, inverse=inverse)
        else:
            return word

    def __len__(self):
        """
        For an associative word `self`, returns the number of letters in it.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> len(w)
        13
        >>> len(a**17)
        17
        >>> len(w**0)
        0

        """
        return sum(abs(j) for (i, j) in self)

    def __eq__(self, other):
        """
        Two  associative words are equal if they are words over the
        same alphabet and if they are sequences of the same letters.
        This is equivalent to saying that the external representations
        of the words are equal.
        There is no "universal" empty word, every alphabet has its own
        empty word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")
        >>> f
        <free group on the generators (swapnil0, swapnil1)>
        >>> g, swap0, swap1 = free_group("swap0 swap1")
        >>> g
        <free group on the generators (swap0, swap1)>

        >>> swapnil0 == swapnil1
        False
        >>> swapnil0*swapnil1 == swapnil1/swapnil1*swapnil0*swapnil1
        True
        >>> swapnil0*swapnil1 == swapnil1*swapnil0
        False
        >>> swapnil1**0 == swap0**0
        False

        """
        group = self.group
        if not isinstance(other, group.dtype):
            return False
        return tuple.__eq__(self, other)

    def __lt__(self, other):
        """
        The  ordering  of  associative  words is defined by length and
        lexicography (this ordering is called short-lex ordering), that
        is, shorter words are smaller than longer words, and words of the
        same length are compared w.r.t. the lexicographical ordering induced
        by the ordering of generators. Generators  are  sorted  according
        to the order in which they were created. If the generators are
        invertible then each generator `g` is larger than its inverse `g^{-1}`,
        and `g^{-1}` is larger than every generator that is smaller than `g`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> b < a
        False
        >>> a < a.inverse()
        False

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                             "be compared")
        l = len(self)
        m = len(other)
        # implement lenlex order
        if l < m:
            return True
        elif l > m:
            return False
        for i in range(l):
            a = self[i].array_form[0]
            b = other[i].array_form[0]
            p = group.symbols.index(a[0])
            q = group.symbols.index(b[0])
            if p < q:
                return True
            elif p > q:
                return False
            elif a[1] < b[1]:
                return True
            elif a[1] > b[1]:
                return False
        return False

    def __le__(self, other):
        return (self == other or self < other)

    def __gt__(self, other):
        """

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, x, y, z = free_group("x y z")
        >>> y**2 > x**2
        True
        >>> y*z > z*y
        False
        >>> x > x.inverse()
        True

        """
        group = self.group
        if not isinstance(other, group.dtype):
            raise TypeError("only FreeGroup elements of same FreeGroup can "
                             "be compared")
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def exponent_sum(self, gen):
        """
        For an associative word `self` and a generator or inverse of generator
        `gen`, ``exponent_sum`` returns the number of times `gen` appears in
        `self` minus the number of times its inverse appears in `self`. If
        neither `gen` nor its inverse occur in `self` then 0 is returned.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.exponent_sum(x)
        2
        >>> w.exponent_sum(x**-1)
        -2
        >>> w = x**2*y**4*x**-3
        >>> w.exponent_sum(x)
        -1

        See Also
        ========

        generator_count

        """
        if len(gen) != 1:
            raise ValueError("gen must be a generator or inverse of a generator")
        s = gen.array_form[0]
        return s[1]*sum(i[1] for i in self.array_form if i[0] == s[0])

    def generator_count(self, gen):
        """
        For an associative word `self` and a generator `gen`,
        ``generator_count`` returns the multiplicity of generator
        `gen` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.generator_count(x)
        2
        >>> w = x**2*y**4*x**-3
        >>> w.generator_count(x)
        5

        See Also
        ========

        exponent_sum

        """
        if len(gen) != 1 or gen.array_form[0][1] < 0:
            raise ValueError("gen must be a generator")
        s = gen.array_form[0]
        return s[1]*sum(abs(i[1]) for i in self.array_form if i[0] == s[0])

    def subword(self, from_i, to_j, strict=True):
        """
        For an associative word `self` and two positive integers `from_i` and
        `to_j`, `subword` returns the subword of `self` that begins at position
        `from_i` and ends at `to_j - 1`, indexing is done with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.subword(2, 6)
        a**3*b

        """
        group = self.group
        if not strict:
            from_i = max(from_i, 0)
            to_j = min(len(self), to_j)
        if from_i < 0 or to_j > len(self):
            raise ValueError("`from_i`, `to_j` must be positive and no greater than "
                    "the length of associative word")
        if to_j <= from_i:
            return group.identity
        else:
            letter_form = self.letter_form[from_i: to_j]
            array_form = letter_form_to_array_form(letter_form, group)
            return group.dtype(array_form)

    def subword_index(self, word, start = 0):
        '''
        Find the index of `word` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**2*b*a*b**3
        >>> w.subword_index(a*b*a*b)
        1

        '''
        l = len(word)
        self_lf = self.letter_form
        word_lf = word.letter_form
        index = None
        for i in range(start,len(self_lf)-l+1):
            if self_lf[i:i+l] == word_lf:
                index = i
                break
        if index is not None:
            return index
        else:
            raise ValueError("The given word is not a subword of self")

    def is_dependent(self, word):
        """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**4*y**-3).is_dependent(x**4*y**-2)
        True
        >>> (x**2*y**-1).is_dependent(x*y)
        False
        >>> (x*y**2*x*y**2).is_dependent(x*y**2)
        True
        >>> (x**12).is_dependent(x**-4)
        True

        See Also
        ========

        is_independent

        """
        try:
            return self.subword_index(word) is not None
        except ValueError:
            pass
        try:
            return self.subword_index(word**-1) is not None
        except ValueError:
            return False

    def is_independent(self, word):
        """

        See Also
        ========

        is_dependent

        """
        return not self.is_dependent(word)

    def contains_generators(self):
        """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> (x**2*y**-1).contains_generators()
        {x, y}
        >>> (x**3*z).contains_generators()
        {x, z}

        """
        group = self.group
        gens = {group.dtype(((syllable[0], 1),)) for syllable in self.array_form}
        return gens

    def cyclic_subword(self, from_i, to_j):
        group = self.group
        l = len(self)
        letter_form = self.letter_form
        period1 = int(from_i/l)
        if from_i >= l:
            from_i -= l*period1
            to_j -= l*period1
        diff = to_j - from_i
        word = letter_form[from_i: to_j]
        period2 = int(to_j/l) - 1
        word += letter_form*period2 + letter_form[:diff-l+from_i-l*period2]
        word = letter_form_to_array_form(word, group)
        return group.dtype(word)

    def cyclic_conjugates(self):
        """Returns a words which are cyclic to the word `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x*y*x*y*x
        >>> w.cyclic_conjugates()
        {x*y*x**2*y, x**2*y*x*y, y*x*y*x**2, y*x**2*y*x, x*y*x*y*x}
        >>> s = x*y*x**2*y*x
        >>> s.cyclic_conjugates()
        {x**2*y*x**2*y, y*x**2*y*x**2, x*y*x**2*y*x}

        References
        ==========

        .. [1] https://planetmath.org/cyclicpermutation

        """
        return {self.cyclic_subword(i, i+len(self)) for i in range(len(self))}

    def is_cyclic_conjugate(self, w):
        """
        Checks whether words ``self``, ``w`` are cyclic conjugates.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w1 = x**2*y**5
        >>> w2 = x*y**5*x
        >>> w1.is_cyclic_conjugate(w2)
        True
        >>> w3 = x**-1*y**5*x**-1
        >>> w3.is_cyclic_conjugate(w2)
        False

        """
        l1 = len(self)
        l2 = len(w)
        if l1 != l2:
            return False
        w1 = self.identity_cyclic_reduction()
        w2 = w.identity_cyclic_reduction()
        letter1 = w1.letter_form
        letter2 = w2.letter_form
        str1 = ' '.join(map(str, letter1))
        str2 = ' '.join(map(str, letter2))
        if len(str1) != len(str2):
            return False

        return str1 in str2 + ' ' + str2

    def number_syllables(self):
        """Returns the number of syllables of the associative word `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, swapnil0, swapnil1 = free_group("swapnil0 swapnil1")
        >>> (swapnil1**3*swapnil0*swapnil1**-1).number_syllables()
        3

        """
        return len(self.array_form)

    def exponent_syllable(self, i):
        """
        Returns the exponent of the `i`-th syllable of the associative word
        `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.exponent_syllable( 2 )
        2

        """
        return self.array_form[i][1]

    def generator_syllable(self, i):
        """
        Returns the symbol of the generator that is involved in the
        i-th syllable of the associative word `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.generator_syllable( 3 )
        b

        """
        return self.array_form[i][0]

    def sub_syllables(self, from_i, to_j):
        """
        `sub_syllables` returns the subword of the associative word `self` that
        consists of syllables from positions `from_to` to `to_j`, where
        `from_to` and `to_j` must be positive integers and indexing is done
        with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a, b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.sub_syllables(1, 2)
        b
        >>> w.sub_syllables(3, 3)
        <identity>

        """
        if not isinstance(from_i, int) or not isinstance(to_j, int):
            raise ValueError("both arguments should be integers")
        group = self.group
        if to_j <= from_i:
            return group.identity
        else:
            r = tuple(self.array_form[from_i: to_j])
            return group.dtype(r)

    def substituted_word(self, from_i, to_j, by):
        """
        Returns the associative word obtained by replacing the subword of
        `self` that begins at position `from_i` and ends at position `to_j - 1`
        by the associative word `by`. `from_i` and `to_j` must be positive
        integers, indexing is done with origin 0. In other words,
        `w.substituted_word(w, from_i, to_j, by)` is the product of the three
        words: `w.subword(0, from_i)`, `by`, and
        `w.subword(to_j len(w))`.

        See Also
        ========

        eliminate_word

        """
        lw = len(self)
        if from_i >= to_j or from_i > lw or to_j > lw:
            raise ValueError("values should be within bounds")

        # otherwise there are four possibilities

        # first if from=1 and to=lw then
        if from_i == 0 and to_j == lw:
            return by
        elif from_i == 0:  # second if from_i=1 (and to_j < lw) then
            return by*self.subword(to_j, lw)
        elif to_j == lw:   # third if to_j=1 (and from_i > 1) then
            return self.subword(0, from_i)*by
        else:              # finally
            return self.subword(0, from_i)*by*self.subword(to_j, lw)

    def is_cyclically_reduced(self):
        r"""Returns whether the word is cyclically reduced or not.
        A word is cyclically reduced if by forming the cycle of the
        word, the word is not reduced, i.e a word w = `a_1 ... a_n`
        is called cyclically reduced if `a_1 \ne a_n^{-1}`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**-1*x**-1).is_cyclically_reduced()
        False
        >>> (y*x**2*y**2).is_cyclically_reduced()
        True

        """
        if not self:
            return True
        return self[0] != self[-1]**-1

    def identity_cyclic_reduction(self):
        """Return a unique cyclically reduced version of the word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).identity_cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).identity_cyclic_reduction()
        x**2*y**-1

        References
        ==========

        .. [1] https://planetmath.org/cyclicallyreduced

        """
        word = self.copy()
        group = self.group
        while not word.is_cyclically_reduced():
            exp1 = word.exponent_syllable(0)
            exp2 = word.exponent_syllable(-1)
            r = exp1 + exp2
            if r == 0:
                rep = word.array_form[1: word.number_syllables() - 1]
            else:
                rep = ((word.generator_syllable(0), exp1 + exp2),) + \
                        word.array_form[1: word.number_syllables() - 1]
            word = group.dtype(rep)
        return word

    def cyclic_reduction(self, removed=False):
        """Return a cyclically reduced version of the word. Unlike
        `identity_cyclic_reduction`, this will not cyclically permute
        the reduced word - just remove the "unreduced" bits on either
        side of it. Compare the examples with those of
        `identity_cyclic_reduction`.

        When `removed` is `True`, return a tuple `(word, r)` where
        self `r` is such that before the reduction the word was either
        `r*word*r**-1`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction()
        y**-1*x**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction(removed=True)
        (y**-1*x**2, x**-3)

        """
        word = self.copy()
        g = self.group.identity
        while not word.is_cyclically_reduced():
            exp1 = abs(word.exponent_syllable(0))
            exp2 = abs(word.exponent_syllable(-1))
            exp = min(exp1, exp2)
            start = word[0]**abs(exp)
            end = word[-1]**abs(exp)
            word = start**-1*word*end**-1
            g = g*start
        if removed:
            return word, g
        return word

    def power_of(self, other):
        '''
        Check if `self == other**n` for some integer n.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> ((x*y)**2).power_of(x*y)
        True
        >>> (x**-3*y**-2*x**3).power_of(x**-3*y*x**3)
        True

        '''
        if self.is_identity:
            return True

        l = len(other)
        if l == 1:
            # self has to be a power of one generator
            gens = self.contains_generators()
            s = other in gens or other**-1 in gens
            return len(gens) == 1 and s

        # if self is not cyclically reduced and it is a power of other,
        # other isn't cyclically reduced and the parts removed during
        # their reduction must be equal
        reduced, r1 = self.cyclic_reduction(removed=True)
        if not r1.is_identity:
            other, r2 = other.cyclic_reduction(removed=True)
            if r1 == r2:
                return reduced.power_of(other)
            return False

        if len(self) < l or len(self) % l:
            return False

        prefix = self.subword(0, l)
        if prefix == other or prefix**-1 == other:
            rest = self.subword(l, len(self))
            return rest.power_of(other)
        return False


def letter_form_to_array_form(array_form, group):
    """
    This method converts a list given with possible repetitions of elements in
    it. It returns a new list such that repetitions of consecutive elements is
    removed and replace with a tuple element of size two such that the first
    index contains `value` and the second index contains the number of
    consecutive repetitions of `value`.

    """
    a = list(array_form[:])
    new_array = []
    n = 1
    symbols = group.symbols
    for i in range(len(a)):
        if i == len(a) - 1:
            if a[i] == a[i - 1]:
                if (-a[i]) in symbols:
                    new_array.append((-a[i], -n))
                else:
                    new_array.append((a[i], n))
            else:
                if (-a[i]) in symbols:
                    new_array.append((-a[i], -1))
                else:
                    new_array.append((a[i], 1))
            return new_array
        elif a[i] == a[i + 1]:
            n += 1
        else:
            if (-a[i]) in symbols:
                new_array.append((-a[i], -n))
            else:
                new_array.append((a[i], n))
            n = 1


def zero_mul_simp(l, index):
    """Used to combine two reduced words."""
    while index >=0 and index < len(l) - 1 and l[index][0] == l[index + 1][0]:
        exp = l[index][1] + l[index + 1][1]
        base = l[index][0]
        l[index] = (base, exp)
        del l[index + 1]
        if l[index][1] == 0:
            del l[index]
            index -= 1
<|endoftext|>"""Compatibility interface between dense and sparse polys. """

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.polys.domains.domain import Domain
    from sympy.polys.orderings import MonomialOrder
    from sympy.polys.rings import PolyElement

from sympy.polys.densearith import dup_add_term
from sympy.polys.densearith import dmp_add_term
from sympy.polys.densearith import dup_sub_term
from sympy.polys.densearith import dmp_sub_term
from sympy.polys.densearith import dup_mul_term
from sympy.polys.densearith import dmp_mul_term
from sympy.polys.densearith import dup_add_ground
from sympy.polys.densearith import dmp_add_ground
from sympy.polys.densearith import dup_sub_ground
from sympy.polys.densearith import dmp_sub_ground
from sympy.polys.densearith import dup_mul_ground
from sympy.polys.densearith import dmp_mul_ground
from sympy.polys.densearith import dup_quo_ground
from sympy.polys.densearith import dmp_quo_ground
from sympy.polys.densearith import dup_exquo_ground
from sympy.polys.densearith import dmp_exquo_ground
from sympy.polys.densearith import dup_lshift
from sympy.polys.densearith import dup_rshift
from sympy.polys.densearith import dup_abs
from sympy.polys.densearith import dmp_abs
from sympy.polys.densearith import dup_neg
from sympy.polys.densearith import dmp_neg
from sympy.polys.densearith import dup_add
from sympy.polys.densearith import dmp_add
from sympy.polys.densearith import dup_sub
from sympy.polys.densearith import dmp_sub
from sympy.polys.densearith import dup_add_mul
from sympy.polys.densearith import dmp_add_mul
from sympy.polys.densearith import dup_sub_mul
from sympy.polys.densearith import dmp_sub_mul
from sympy.polys.densearith import dup_mul
from sympy.polys.densearith import dmp_mul
from sympy.polys.densearith import dup_sqr
from sympy.polys.densearith import dmp_sqr
from sympy.polys.densearith import dup_pow
from sympy.polys.densearith import dmp_pow
from sympy.polys.densearith import dup_pdiv
from sympy.polys.densearith import dup_prem
from sympy.polys.densearith import dup_pquo
from sympy.polys.densearith import dup_pexquo
from sympy.polys.densearith import dmp_pdiv
from sympy.polys.densearith import dmp_prem
from sympy.polys.densearith import dmp_pquo
from sympy.polys.densearith import dmp_pexquo
from sympy.polys.densearith import dup_rr_div
from sympy.polys.densearith import dmp_rr_div
from sympy.polys.densearith import dup_ff_div
from sympy.polys.densearith import dmp_ff_div
from sympy.polys.densearith import dup_div
from sympy.polys.densearith import dup_rem
from sympy.polys.densearith import dup_quo
from sympy.polys.densearith import dup_exquo
from sympy.polys.densearith import dmp_div
from sympy.polys.densearith import dmp_rem
from sympy.polys.densearith import dmp_quo
from sympy.polys.densearith import dmp_exquo
from sympy.polys.densearith import dup_max_norm
from sympy.polys.densearith import dmp_max_norm
from sympy.polys.densearith import dup_l1_norm
from sympy.polys.densearith import dmp_l1_norm
from sympy.polys.densearith import dup_l2_norm_squared
from sympy.polys.densearith import dmp_l2_norm_squared
from sympy.polys.densearith import dup_expand
from sympy.polys.densearith import dmp_expand
from sympy.polys.densebasic import dup_LC
from sympy.polys.densebasic import dmp_LC
from sympy.polys.densebasic import dup_TC
from sympy.polys.densebasic import dmp_TC
from sympy.polys.densebasic import dmp_ground_LC
from sympy.polys.densebasic import dmp_ground_TC
from sympy.polys.densebasic import dup_degree
from sympy.polys.densebasic import dmp_degree
from sympy.polys.densebasic import dmp_degree_in
from sympy.polys.densebasic import dmp_to_dict
from sympy.polys.densetools import dup_integrate
from sympy.polys.densetools import dmp_integrate
from sympy.polys.densetools import dmp_integrate_in
from sympy.polys.densetools import dup_diff
from sympy.polys.densetools import dmp_diff
from sympy.polys.densetools import dmp_diff_in
from sympy.polys.densetools import dup_eval
from sympy.polys.densetools import dmp_eval
from sympy.polys.densetools import dmp_eval_in
from sympy.polys.densetools import dmp_eval_tail
from sympy.polys.densetools import dmp_diff_eval_in
from sympy.polys.densetools import dup_trunc
from sympy.polys.densetools import dmp_trunc
from sympy.polys.densetools import dmp_ground_trunc
from sympy.polys.densetools import dup_monic
from sympy.polys.densetools import dmp_ground_monic
from sympy.polys.densetools import dup_content
from sympy.polys.densetools import dmp_ground_content
from sympy.polys.densetools import dup_primitive
from sympy.polys.densetools import dmp_ground_primitive
from sympy.polys.densetools import dup_extract
from sympy.polys.densetools import dmp_ground_extract
from sympy.polys.densetools import dup_real_imag
from sympy.polys.densetools import dup_mirror
from sympy.polys.densetools import dup_scale
from sympy.polys.densetools import dup_shift
from sympy.polys.densetools import dmp_shift
from sympy.polys.densetools import dup_transform
from sympy.polys.densetools import dup_compose
from sympy.polys.densetools import dmp_compose
from sympy.polys.densetools import dup_decompose
from sympy.polys.densetools import dmp_lift
from sympy.polys.densetools import dup_sign_variations
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.densetools import dmp_clear_denoms
from sympy.polys.densetools import dup_revert
from sympy.polys.euclidtools import dup_half_gcdex
from sympy.polys.euclidtools import dmp_half_gcdex
from sympy.polys.euclidtools import dup_gcdex
from sympy.polys.euclidtools import dmp_gcdex
from sympy.polys.euclidtools import dup_invert
from sympy.polys.euclidtools import dmp_invert
from sympy.polys.euclidtools import dup_euclidean_prs
from sympy.polys.euclidtools import dmp_euclidean_prs
from sympy.polys.euclidtools import dup_primitive_prs
from sympy.polys.euclidtools import dmp_primitive_prs
from sympy.polys.euclidtools import dup_inner_subresultants
from sympy.polys.euclidtools import dup_subresultants
from sympy.polys.euclidtools import dup_prs_resultant
from sympy.polys.euclidtools import dup_resultant
from sympy.polys.euclidtools import dmp_inner_subresultants
from sympy.polys.euclidtools import dmp_subresultants
from sympy.polys.euclidtools import dmp_prs_resultant
from sympy.polys.euclidtools import dmp_zz_modular_resultant
from sympy.polys.euclidtools import dmp_zz_collins_resultant
from sympy.polys.euclidtools import dmp_qq_collins_resultant
from sympy.polys.euclidtools import dmp_resultant
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.euclidtools import dmp_discriminant
from sympy.polys.euclidtools import dup_rr_prs_gcd
from sympy.polys.euclidtools import dup_ff_prs_gcd
from sympy.polys.euclidtools import dmp_rr_prs_gcd
from sympy.polys.euclidtools import dmp_ff_prs_gcd
from sympy.polys.euclidtools import dup_zz_heu_gcd
from sympy.polys.euclidtools import dmp_zz_heu_gcd
from sympy.polys.euclidtools import dup_qq_heu_gcd
from sympy.polys.euclidtools import dmp_qq_heu_gcd
from sympy.polys.euclidtools import dup_inner_gcd
from sympy.polys.euclidtools import dmp_inner_gcd
from sympy.polys.euclidtools import dup_gcd
from sympy.polys.euclidtools import dmp_gcd
from sympy.polys.euclidtools import dup_rr_lcm
from sympy.polys.euclidtools import dup_ff_lcm
from sympy.polys.euclidtools import dup_lcm
from sympy.polys.euclidtools import dmp_rr_lcm
from sympy.polys.euclidtools import dmp_ff_lcm
from sympy.polys.euclidtools import dmp_lcm
from sympy.polys.euclidtools import dmp_content
from sympy.polys.euclidtools import dmp_primitive
from sympy.polys.euclidtools import dup_cancel
from sympy.polys.euclidtools import dmp_cancel
from sympy.polys.factortools import dup_trial_division
from sympy.polys.factortools import dmp_trial_division
from sympy.polys.factortools import dup_zz_mignotte_bound
from sympy.polys.factortools import dmp_zz_mignotte_bound
from sympy.polys.factortools import dup_zz_hensel_step
from sympy.polys.factortools import dup_zz_hensel_lift
from sympy.polys.factortools import dup_zz_zassenhaus
from sympy.polys.factortools import dup_zz_irreducible_p
from sympy.polys.factortools import dup_cyclotomic_p
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.factortools import dup_zz_cyclotomic_factor
from sympy.polys.factortools import dup_zz_factor_sqf
from sympy.polys.factortools import dup_zz_factor
from sympy.polys.factortools import dmp_zz_wang_non_divisors
from sympy.polys.factortools import dmp_zz_wang_lead_coeffs
from sympy.polys.factortools import dup_zz_diophantine
from sympy.polys.factortools import dmp_zz_diophantine
from sympy.polys.factortools import dmp_zz_wang_hensel_lifting
from sympy.polys.factortools import dmp_zz_wang
from sympy.polys.factortools import dmp_zz_factor
from sympy.polys.factortools import dup_qq_i_factor
from sympy.polys.factortools import dup_zz_i_factor
from sympy.polys.factortools import dmp_qq_i_factor
from sympy.polys.factortools import dmp_zz_i_factor
from sympy.polys.factortools import dup_ext_factor
from sympy.polys.factortools import dmp_ext_factor
from sympy.polys.factortools import dup_gf_factor
from sympy.polys.factortools import dmp_gf_factor
from sympy.polys.factortools import dup_factor_list
from sympy.polys.factortools import dup_factor_list_include
from sympy.polys.factortools import dmp_factor_list
from sympy.polys.factortools import dmp_factor_list_include
from sympy.polys.factortools import dup_irreducible_p
from sympy.polys.factortools import dmp_irreducible_p
from sympy.polys.rootisolation import dup_sturm
from sympy.polys.rootisolation import dup_root_upper_bound
from sympy.polys.rootisolation import dup_root_lower_bound
from sympy.polys.rootisolation import dup_step_refine_real_root
from sympy.polys.rootisolation import dup_inner_refine_real_root
from sympy.polys.rootisolation import dup_outer_refine_real_root
from sympy.polys.rootisolation import dup_refine_real_root
from sympy.polys.rootisolation import dup_inner_isolate_real_roots
from sympy.polys.rootisolation import dup_inner_isolate_positive_roots
from sympy.polys.rootisolation import dup_inner_isolate_negative_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_sqf
from sympy.polys.rootisolation import dup_isolate_real_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.polys.rootisolation import dup_count_real_roots
from sympy.polys.rootisolation import dup_count_complex_roots
from sympy.polys.rootisolation import dup_isolate_complex_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots

from sympy.polys.sqfreetools import (
    dup_sqf_p, dmp_sqf_p, dmp_norm, dup_sqf_norm, dmp_sqf_norm,
    dup_gf_sqf_part, dmp_gf_sqf_part, dup_sqf_part, dmp_sqf_part,
    dup_gf_sqf_list, dmp_gf_sqf_list, dup_sqf_list, dup_sqf_list_include,
    dmp_sqf_list, dmp_sqf_list_include, dup_gff_list, dmp_gff_list)

from sympy.polys.galoistools import (
    gf_degree, gf_LC, gf_TC, gf_strip, gf_from_dict,
    gf_to_dict, gf_from_int_poly, gf_to_int_poly, gf_neg, gf_add_ground, gf_sub_ground,
    gf_mul_ground, gf_quo_ground, gf_add, gf_sub, gf_mul, gf_sqr, gf_add_mul, gf_sub_mul,
    gf_expand, gf_div, gf_rem, gf_quo, gf_exquo, gf_lshift, gf_rshift, gf_pow, gf_pow_mod,
    gf_gcd, gf_lcm, gf_cofactors, gf_gcdex, gf_monic, gf_diff, gf_eval, gf_multi_eval,
    gf_compose, gf_compose_mod, gf_trace_map, gf_random, gf_irreducible, gf_irred_p_ben_or,
    gf_irred_p_rabin, gf_irreducible_p, gf_sqf_p, gf_sqf_part, gf_Qmatrix,
    gf_berlekamp, gf_ddf_zassenhaus, gf_edf_zassenhaus, gf_ddf_shoup, gf_edf_shoup,
    gf_zassenhaus, gf_shoup, gf_factor_sqf, gf_factor)

from sympy.utilities import public

@public
class IPolys:

    gens: tuple[PolyElement, ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain
    order: MonomialOrder

    def drop(self, gen):
        pass

    def clone(self, symbols=None, domain=None, order=None):
        pass

    def to_ground(self):
        pass

    def ground_new(self, element):
        pass

    def domain_new(self, element):
        pass

    def from_dict(self, d):
        pass

    def wrap(self, element):
        from sympy.polys.rings import PolyElement
        if isinstance(element, PolyElement):
            if element.ring == self:
                return element
            else:
                raise NotImplementedError("domain conversions")
        else:
            return self.ground_new(element)

    def to_dense(self, element):
        return self.wrap(element).to_dense()

    def from_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))

    def dup_add_term(self, f, c, i):
        return self.from_dense(dup_add_term(self.to_dense(f), c, i, self.domain))
    def dmp_add_term(self, f, c, i):
        return self.from_dense(dmp_add_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))
    def dup_sub_term(self, f, c, i):
        return self.from_dense(dup_sub_term(self.to_dense(f), c, i, self.domain))
    def dmp_sub_term(self, f, c, i):
        return self.from_dense(dmp_sub_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))
    def dup_mul_term(self, f, c, i):
        return self.from_dense(dup_mul_term(self.to_dense(f), c, i, self.domain))
    def dmp_mul_term(self, f, c, i):
        return self.from_dense(dmp_mul_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))

    def dup_add_ground(self, f, c):
        return self.from_dense(dup_add_ground(self.to_dense(f), c, self.domain))
    def dmp_add_ground(self, f, c):
        return self.from_dense(dmp_add_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_sub_ground(self, f, c):
        return self.from_dense(dup_sub_ground(self.to_dense(f), c, self.domain))
    def dmp_sub_ground(self, f, c):
        return self.from_dense(dmp_sub_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_mul_ground(self, f, c):
        return self.from_dense(dup_mul_ground(self.to_dense(f), c, self.domain))
    def dmp_mul_ground(self, f, c):
        return self.from_dense(dmp_mul_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_quo_ground(self, f, c):
        return self.from_dense(dup_quo_ground(self.to_dense(f), c, self.domain))
    def dmp_quo_ground(self, f, c):
        return self.from_dense(dmp_quo_ground(self.to_dense(f), c, self.ngens-1, self.domain))
    def dup_exquo_ground(self, f, c):
        return self.from_dense(dup_exquo_ground(self.to_dense(f), c, self.domain))
    def dmp_exquo_ground(self, f, c):
        return self.from_dense(dmp_exquo_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    def dup_lshift(self, f, n):
        return self.from_dense(dup_lshift(self.to_dense(f), n, self.domain))
    def dup_rshift(self, f, n):
        return self.from_dense(dup_rshift(self.to_dense(f), n, self.domain))

    def dup_abs(self, f):
        return self.from_dense(dup_abs(self.to_dense(f), self.domain))
    def dmp_abs(self, f):
        return self.from_dense(dmp_abs(self.to_dense(f), self.ngens-1, self.domain))

    def dup_neg(self, f):
        return self.from_dense(dup_neg(self.to_dense(f), self.domain))
    def dmp_neg(self, f):
        return self.from_dense(dmp_neg(self.to_dense(f), self.ngens-1, self.domain))

    def dup_add(self, f, g):
        return self.from_dense(dup_add(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_add(self, f, g):
        return self.from_dense(dmp_add(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_sub(self, f, g):
        return self.from_dense(dup_sub(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_sub(self, f, g):
        return self.from_dense(dmp_sub(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_add_mul(self, f, g, h):
        return self.from_dense(dup_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))
    def dmp_add_mul(self, f, g, h):
        return self.from_dense(dmp_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))
    def dup_sub_mul(self, f, g, h):
        return self.from_dense(dup_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))
    def dmp_sub_mul(self, f, g, h):
        return self.from_dense(dmp_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))

    def dup_mul(self, f, g):
        return self.from_dense(dup_mul(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_mul(self, f, g):
        return self.from_dense(dmp_mul(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_sqr(self, f):
        return self.from_dense(dup_sqr(self.to_dense(f), self.domain))
    def dmp_sqr(self, f):
        return self.from_dense(dmp_sqr(self.to_dense(f), self.ngens-1, self.domain))
    def dup_pow(self, f, n):
        return self.from_dense(dup_pow(self.to_dense(f), n, self.domain))
    def dmp_pow(self, f, n):
        return self.from_dense(dmp_pow(self.to_dense(f), n, self.ngens-1, self.domain))

    def dup_pdiv(self, f, g):
        q, r = dup_pdiv(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_prem(self, f, g):
        return self.from_dense(dup_prem(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_pquo(self, f, g):
        return self.from_dense(dup_pquo(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_pexquo(self, f, g):
        return self.from_dense(dup_pexquo(self.to_dense(f), self.to_dense(g), self.domain))

    def dmp_pdiv(self, f, g):
        q, r = dmp_pdiv(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_prem(self, f, g):
        return self.from_dense(dmp_prem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_pquo(self, f, g):
        return self.from_dense(dmp_pquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_pexquo(self, f, g):
        return self.from_dense(dmp_pexquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_rr_div(self, f, g):
        q, r = dup_rr_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_rr_div(self, f, g):
        q, r = dmp_rr_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_ff_div(self, f, g):
        q, r = dup_ff_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_ff_div(self, f, g):
        q, r = dmp_ff_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))

    def dup_div(self, f, g):
        q, r = dup_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dup_rem(self, f, g):
        return self.from_dense(dup_rem(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_quo(self, f, g):
        return self.from_dense(dup_quo(self.to_dense(f), self.to_dense(g), self.domain))
    def dup_exquo(self, f, g):
        return self.from_dense(dup_exquo(self.to_dense(f), self.to_dense(g), self.domain))

    def dmp_div(self, f, g):
        q, r = dmp_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))
    def dmp_rem(self, f, g):
        return self.from_dense(dmp_rem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_quo(self, f, g):
        return self.from_dense(dmp_quo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))
    def dmp_exquo(self, f, g):
        return self.from_dense(dmp_exquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_max_norm(self, f):
        return dup_max_norm(self.to_dense(f), self.domain)
    def dmp_max_norm(self, f):
        return dmp_max_norm(self.to_dense(f), self.ngens-1, self.domain)

    def dup_l1_norm(self, f):
        return dup_l1_norm(self.to_dense(f), self.domain)
    def dmp_l1_norm(self, f):
        return dmp_l1_norm(self.to_dense(f), self.ngens-1, self.domain)

    def dup_l2_norm_squared(self, f):
        return dup_l2_norm_squared(self.to_dense(f), self.domain)
    def dmp_l2_norm_squared(self, f):
        return dmp_l2_norm_squared(self.to_dense(f), self.ngens-1, self.domain)

    def dup_expand(self, polys):
        return self.from_dense(dup_expand(list(map(self.to_dense, polys)), self.domain))
    def dmp_expand(self, polys):
        return self.from_dense(dmp_expand(list(map(self.to_dense, polys)), self.ngens-1, self.domain))

    def dup_LC(self, f):
        return dup_LC(self.to_dense(f), self.domain)
    def dmp_LC(self, f):
        LC = dmp_LC(self.to_dense(f), self.domain)
        if isinstance(LC, list):
            return self[1:].from_dense(LC)
        else:
            return LC
    def dup_TC(self, f):
        return dup_TC(self.to_dense(f), self.domain)
    def dmp_TC(self, f):
        TC = dmp_TC(self.to_dense(f), self.domain)
        if isinstance(TC, list):
            return self[1:].from_dense(TC)
        else:
            return TC

    def dmp_ground_LC(self, f):
        return dmp_ground_LC(self.to_dense(f), self.ngens-1, self.domain)
    def dmp_ground_TC(self, f):
        return dmp_ground_TC(self.to_dense(f), self.ngens-1, self.domain)

    def dup_degree(self, f):
        return dup_degree(self.to_dense(f))
    def dmp_degree(self, f):
        return dmp_degree(self.to_dense(f), self.ngens-1)
    def dmp_degree_in(self, f, j):
        return dmp_degree_in(self.to_dense(f), j, self.ngens-1)
    def dup_integrate(self, f, m):
        return self.from_dense(dup_integrate(self.to_dense(f), m, self.domain))
    def dmp_integrate(self, f, m):
        return self.from_dense(dmp_integrate(self.to_dense(f), m, self.ngens-1, self.domain))

    def dup_diff(self, f, m):
        return self.from_dense(dup_diff(self.to_dense(f), m, self.domain))
    def dmp_diff(self, f, m):
        return self.from_dense(dmp_diff(self.to_dense(f), m, self.ngens-1, self.domain))

    def dmp_diff_in(self, f, m, j):
        return self.from_dense(dmp_diff_in(self.to_dense(f), m, j, self.ngens-1, self.domain))
    def dmp_integrate_in(self, f, m, j):
        return self.from_dense(dmp_integrate_in(self.to_dense(f), m, j, self.ngens-1, self.domain))

    def dup_eval(self, f, a):
        return dup_eval(self.to_dense(f), a, self.domain)
    def dmp_eval(self, f, a):
        result = dmp_eval(self.to_dense(f), a, self.ngens-1, self.domain)
        return self[1:].from_dense(result)

    def dmp_eval_in(self, f, a, j):
        result = dmp_eval_in(self.to_dense(f), a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)
    def dmp_diff_eval_in(self, f, m, a, j):
        result = dmp_diff_eval_in(self.to_dense(f), m, a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)

    def dmp_eval_tail(self, f, A):
        result = dmp_eval_tail(self.to_dense(f), A, self.ngens-1, self.domain)
        if isinstance(result, list):
            return self[:-len(A)].from_dense(result)
        else:
            return result

    def dup_trunc(self, f, p):
        return self.from_dense(dup_trunc(self.to_dense(f), p, self.domain))
    def dmp_trunc(self, f, g):
        return self.from_dense(dmp_trunc(self.to_dense(f), self[1:].to_dense(g), self.ngens-1, self.domain))
    def dmp_ground_trunc(self, f, p):
        return self.from_dense(dmp_ground_trunc(self.to_dense(f), p, self.ngens-1, self.domain))

    def dup_monic(self, f):
        return self.from_dense(dup_monic(self.to_dense(f), self.domain))
    def dmp_ground_monic(self, f):
        return self.from_dense(dmp_ground_monic(self.to_dense(f), self.ngens-1, self.domain))

    def dup_extract(self, f, g):
        c, F, G = dup_extract(self.to_dense(f), self.to_dense(g), self.domain)
        return (c, self.from_dense(F), self.from_dense(G))
    def dmp_ground_extract(self, f, g):
        c, F, G = dmp_ground_extract(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (c, self.from_dense(F), self.from_dense(G))

    def dup_real_imag(self, f):
        p, q = dup_real_imag(self.wrap(f).drop(1).to_dense(), self.domain)
        return (self.from_dense(p), self.from_dense(q))

    def dup_mirror(self, f):
        return self.from_dense(dup_mirror(self.to_dense(f), self.domain))
    def dup_scale(self, f, a):
        return self.from_dense(dup_scale(self.to_dense(f), a, self.domain))
    def dup_shift(self, f, a):
        return self.from_dense(dup_shift(self.to_dense(f), a, self.domain))
    def dmp_shift(self, f, a):
        return self.from_dense(dmp_shift(self.to_dense(f), a, self.ngens-1, self.domain))
    def dup_transform(self, f, p, q):
        return self.from_dense(dup_transform(self.to_dense(f), self.to_dense(p), self.to_dense(q), self.domain))

    def dup_compose(self, f, g):
        return self.from_dense(dup_compose(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_compose(self, f, g):
        return self.from_dense(dmp_compose(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_decompose(self, f):
        components = dup_decompose(self.to_dense(f), self.domain)
        return list(map(self.from_dense, components))

    def dmp_lift(self, f):
        result = dmp_lift(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(result)

    def dup_sign_variations(self, f):
        return dup_sign_variations(self.to_dense(f), self.domain)

    def dup_clear_denoms(self, f, convert=False):
        c, F = dup_clear_denoms(self.to_dense(f), self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))
    def dmp_clear_denoms(self, f, convert=False):
        c, F = dmp_clear_denoms(self.to_dense(f), self.ngens-1, self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))

    def dup_revert(self, f, n):
        return self.from_dense(dup_revert(self.to_dense(f), n, self.domain))

    def dup_half_gcdex(self, f, g):
        s, h = dup_half_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(h))
    def dmp_half_gcdex(self, f, g):
        s, h = dmp_half_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(h))
    def dup_gcdex(self, f, g):
        s, t, h = dup_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))
    def dmp_gcdex(self, f, g):
        s, t, h = dmp_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))

    def dup_invert(self, f, g):
        return self.from_dense(dup_invert(self.to_dense(f), self.to_dense(g), self.domain))
    def dmp_invert(self, f, g):
        return self.from_dense(dmp_invert(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    def dup_euclidean_prs(self, f, g):
        prs = dup_euclidean_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_euclidean_prs(self, f, g):
        prs = dmp_euclidean_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))
    def dup_primitive_prs(self, f, g):
        prs = dup_primitive_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_primitive_prs(self, f, g):
        prs = dmp_primitive_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    def dup_inner_subresultants(self, f, g):
        prs, sres = dup_inner_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return (list(map(self.from_dense, prs)), sres)
    def dmp_inner_subresultants(self, f, g):
        prs, sres  = dmp_inner_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (list(map(self.from_dense, prs)), sres)

    def dup_subresultants(self, f, g):
        prs = dup_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))
    def dmp_subresultants(self, f, g):
        prs = dmp_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    def dup_prs_resultant(self, f, g):
        res, prs = dup_prs_resultant(self.to_dense(f), self.to_dense(g), self.domain)
        return (res, list(map(self.from_dense, prs)))
    def dmp_prs_resultant(self, f, g):
        res, prs = dmp_prs_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self[1:].from_dense(res), list(map(self.from_dense, prs)))

    def dmp_zz_modular_resultant(self, f, g, p):
        res = dmp_zz_modular_resultant(self.to_dense(f), self.to_dense(g), self.domain_new(p), self.ngens-1, self.domain)
        return self[1:].from_dense(res)
    def dmp_zz_collins_resultant(self, f, g):
        res = dmp_zz_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self[1:].from_dense(res)
    def dmp_qq_collins_resultant(self, f, g):
        res = dmp_qq_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self[1:].from_dense(res)

    def dup_resultant(self, f, g): #, includePRS=False):
        return dup_resultant(self.to_dense(f), self.to_dense(g), self.domain) #, includePRS=includePRS)
    def dmp_resultant(self, f, g): #, includePRS=False):
        res = dmp_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain) #, includePRS=includePRS)
        if isinstance(res, list):
            return self[1:].from_dense(res)
        else:
            return res

    def dup_discriminant(self, f):
        return dup_discriminant(self.to_dense(f), self.domain)
    def dmp_discriminant(self, f):
        disc = dmp_discriminant(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(disc, list):
            return self[1:].from_dense(disc)
        else:
            return disc

    def dup_rr_prs_gcd(self, f, g):
        H, F, G = dup_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_ff_prs_gcd(self, f, g):
        H, F, G = dup_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_rr_prs_gcd(self, f, g):
        H, F, G = dmp_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_ff_prs_gcd(self, f, g):
        H, F, G = dmp_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_zz_heu_gcd(self, f, g):
        H, F, G = dup_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_zz_heu_gcd(self, f, g):
        H, F, G = dmp_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_qq_heu_gcd(self, f, g):
        H, F, G = dup_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_qq_heu_gcd(self, f, g):
        H, F, G = dmp_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_inner_gcd(self, f, g):
        H, F, G = dup_inner_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dmp_inner_gcd(self, f, g):
        H, F, G = dmp_inner_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    def dup_gcd(self, f, g):
        H = dup_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dmp_gcd(self, f, g):
        H = dmp_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dup_rr_lcm(self, f, g):
        H = dup_rr_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dup_ff_lcm(self, f, g):
        H = dup_ff_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dup_lcm(self, f, g):
        H = dup_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)
    def dmp_rr_lcm(self, f, g):
        H = dmp_rr_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dmp_ff_lcm(self, f, g):
        H = dmp_ff_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)
    def dmp_lcm(self, f, g):
        H = dmp_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    def dup_content(self, f):
        cont = dup_content(self.to_dense(f), self.domain)
        return cont
    def dup_primitive(self, f):
        cont, prim = dup_primitive(self.to_dense(f), self.domain)
        return cont, self.from_dense(prim)

    def dmp_content(self, f):
        cont = dmp_content(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return self[1:].from_dense(cont)
        else:
            return cont
    def dmp_primitive(self, f):
        cont, prim = dmp_primitive(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return (self[1:].from_dense(cont), self.from_dense(prim))
        else:
            return (cont, self.from_dense(prim))

    def dmp_ground_content(self, f):
        cont = dmp_ground_content(self.to_dense(f), self.ngens-1, self.domain)
        return cont
    def dmp_ground_primitive(self, f):
        cont, prim = dmp_ground_primitive(self.to_dense(f), self.ngens-1, self.domain)
        return (cont, self.from_dense(prim))

    def dup_cancel(self, f, g, include=True):
        result = dup_cancel(self.to_dense(f), self.to_dense(g), self.domain, include=include)
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            F, G = result
            return (self.from_dense(F), self.from_dense(G))
    def dmp_cancel(self, f, g, include=True):
        result = dmp_cancel(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain, include=include)
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            F, G = result
            return (self.from_dense(F), self.from_dense(G))

    def dup_trial_division(self, f, factors):
        factors = dup_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_trial_division(self, f, factors):
        factors = dmp_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_zz_mignotte_bound(self, f):
        return dup_zz_mignotte_bound(self.to_dense(f), self.domain)
    def dmp_zz_mignotte_bound(self, f):
        return dmp_zz_mignotte_bound(self.to_dense(f), self.ngens-1, self.domain)

    def dup_zz_hensel_step(self, m, f, g, h, s, t):
        D = self.to_dense
        G, H, S, T = dup_zz_hensel_step(m, D(f), D(g), D(h), D(s), D(t), self.domain)
        return (self.from_dense(G), self.from_dense(H), self.from_dense(S), self.from_dense(T))
    def dup_zz_hensel_lift(self, p, f, f_list, l):
        D = self.to_dense
        polys = dup_zz_hensel_lift(p, D(f), list(map(D, f_list)), l, self.domain)
        return list(map(self.from_dense, polys))

    def dup_zz_zassenhaus(self, f):
        factors = dup_zz_zassenhaus(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_zz_irreducible_p(self, f):
        return dup_zz_irreducible_p(self.to_dense(f), self.domain)
    def dup_cyclotomic_p(self, f, irreducible=False):
        return dup_cyclotomic_p(self.to_dense(f), self.domain, irreducible=irreducible)
    def dup_zz_cyclotomic_poly(self, n):
        F = dup_zz_cyclotomic_poly(n, self.domain)
        return self.from_dense(F)
    def dup_zz_cyclotomic_factor(self, f):
        result = dup_zz_cyclotomic_factor(self.to_dense(f), self.domain)
        if result is None:
            return result
        else:
            return list(map(self.from_dense, result))

    # E: List[ZZ], cs: ZZ, ct: ZZ
    def dmp_zz_wang_non_divisors(self, E, cs, ct):
        return dmp_zz_wang_non_divisors(E, cs, ct, self.domain)

    # f: Poly, T: List[(Poly, int)], ct: ZZ, A: List[ZZ]
    #def dmp_zz_wang_test_points(f, T, ct, A):
    #   dmp_zz_wang_test_points(self.to_dense(f), T, ct, A, self.ngens-1, self.domain)

    # f: Poly, T: List[(Poly, int)], cs: ZZ, E: List[ZZ], H: List[Poly], A: List[ZZ]
    def dmp_zz_wang_lead_coeffs(self, f, T, cs, E, H, A):
        mv = self[1:]
        T = [ (mv.to_dense(t), k) for t, k in T ]
        uv = self[:1]
        H = list(map(uv.to_dense, H))
        f, HH, CC = dmp_zz_wang_lead_coeffs(self.to_dense(f), T, cs, E, H, A, self.ngens-1, self.domain)
        return self.from_dense(f), list(map(uv.from_dense, HH)), list(map(mv.from_dense, CC))

    # f: List[Poly], m: int, p: ZZ
    def dup_zz_diophantine(self, F, m, p):
        result = dup_zz_diophantine(list(map(self.to_dense, F)), m, p, self.domain)
        return list(map(self.from_dense, result))

    # f: List[Poly], c: List[Poly], A: List[ZZ], d: int, p: ZZ
    def dmp_zz_diophantine(self, F, c, A, d, p):
        result = dmp_zz_diophantine(list(map(self.to_dense, F)), self.to_dense(c), A, d, p, self.ngens-1, self.domain)
        return list(map(self.from_dense, result))

    # f: Poly, H: List[Poly], LC: List[Poly], A: List[ZZ], p: ZZ
    def dmp_zz_wang_hensel_lifting(self, f, H, LC, A, p):
        uv = self[:1]
        mv = self[1:]
        H = list(map(uv.to_dense, H))
        LC = list(map(mv.to_dense, LC))
        result = dmp_zz_wang_hensel_lifting(self.to_dense(f), H, LC, A, p, self.ngens-1, self.domain)
        return list(map(self.from_dense, result))

    def dmp_zz_wang(self, f, mod=None, seed=None):
        factors = dmp_zz_wang(self.to_dense(f), self.ngens-1, self.domain, mod=mod, seed=seed)
        return [ self.from_dense(g) for g in factors ]

    def dup_zz_factor_sqf(self, f):
        coeff, factors = dup_zz_factor_sqf(self.to_dense(f), self.domain)
        return (coeff, [ self.from_dense(g) for g in factors ])

    def dup_zz_factor(self, f):
        coeff, factors = dup_zz_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_zz_factor(self, f):
        coeff, factors = dmp_zz_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_qq_i_factor(self, f):
        coeff, factors = dup_qq_i_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_qq_i_factor(self, f):
        coeff, factors = dmp_qq_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_zz_i_factor(self, f):
        coeff, factors = dup_zz_i_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_zz_i_factor(self, f):
        coeff, factors = dmp_zz_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_ext_factor(self, f):
        coeff, factors = dup_ext_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_ext_factor(self, f):
        coeff, factors = dmp_ext_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_gf_factor(self, f):
        coeff, factors = dup_gf_factor(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_gf_factor(self, f):
        coeff, factors = dmp_gf_factor(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_factor_list(self, f):
        coeff, factors = dup_factor_list(self.to_dense(f), self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dup_factor_list_include(self, f):
        factors = dup_factor_list_include(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dmp_factor_list(self, f):
        coeff, factors = dmp_factor_list(self.to_dense(f), self.ngens-1, self.domain)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_factor_list_include(self, f):
        factors = dmp_factor_list_include(self.to_dense(f), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_irreducible_p(self, f):
        return dup_irreducible_p(self.to_dense(f), self.domain)
    def dmp_irreducible_p(self, f):
        return dmp_irreducible_p(self.to_dense(f), self.ngens-1, self.domain)

    def dup_sturm(self, f):
        seq = dup_sturm(self.to_dense(f), self.domain)
        return list(map(self.from_dense, seq))

    def dup_sqf_p(self, f):
        return dup_sqf_p(self.to_dense(f), self.domain)
    def dmp_sqf_p(self, f):
        return dmp_sqf_p(self.to_dense(f), self.ngens-1, self.domain)

    def dmp_norm(self, f):
        n = dmp_norm(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(n)

    def dup_sqf_norm(self, f):
        s, F, R = dup_sqf_norm(self.to_dense(f), self.domain)
        return (s, self.from_dense(F), self.to_ground().from_dense(R))
    def dmp_sqf_norm(self, f):
        s, F, R = dmp_sqf_norm(self.to_dense(f), self.ngens-1, self.domain)
        return (s, self.from_dense(F), self.to_ground().from_dense(R))

    def dup_gf_sqf_part(self, f):
        return self.from_dense(dup_gf_sqf_part(self.to_dense(f), self.domain))
    def dmp_gf_sqf_part(self, f):
        return self.from_dense(dmp_gf_sqf_part(self.to_dense(f), self.domain))
    def dup_sqf_part(self, f):
        return self.from_dense(dup_sqf_part(self.to_dense(f), self.domain))
    def dmp_sqf_part(self, f):
        return self.from_dense(dmp_sqf_part(self.to_dense(f), self.ngens-1, self.domain))

    def dup_gf_sqf_list(self, f, all=False):
        coeff, factors = dup_gf_sqf_list(self.to_dense(f), self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_gf_sqf_list(self, f, all=False):
        coeff, factors = dmp_gf_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_sqf_list(self, f, all=False):
        coeff, factors = dup_sqf_list(self.to_dense(f), self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dup_sqf_list_include(self, f, all=False):
        factors = dup_sqf_list_include(self.to_dense(f), self.domain, all=all)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_sqf_list(self, f, all=False):
        coeff, factors = dmp_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_sqf_list_include(self, f, all=False):
        factors = dmp_sqf_list_include(self.to_dense(f), self.ngens-1, self.domain, all=all)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_gff_list(self, f):
        factors = dup_gff_list(self.to_dense(f), self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]
    def dmp_gff_list(self, f):
        factors = dmp_gff_list(self.to_dense(f), self.ngens-1, self.domain)
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_root_upper_bound(self, f):
        return dup_root_upper_bound(self.to_dense(f), self.domain)
    def dup_root_lower_bound(self, f):
        return dup_root_lower_bound(self.to_dense(f), self.domain)

    def dup_step_refine_real_root(self, f, M, fast=False):
        return dup_step_refine_real_root(self.to_dense(f), M, self.domain, fast=fast)
    def dup_inner_refine_real_root(self, f, M, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
        return dup_inner_refine_real_root(self.to_dense(f), M, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast, mobius=mobius)
    def dup_outer_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        return dup_outer_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    def dup_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        return dup_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    def dup_inner_isolate_real_roots(self, f, eps=None, fast=False):
        return dup_inner_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, fast=fast)
    def dup_inner_isolate_positive_roots(self, f, eps=None, inf=None, sup=None, fast=False, mobius=False):
        return dup_inner_isolate_positive_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, mobius=mobius)
    def dup_inner_isolate_negative_roots(self, f, inf=None, sup=None, eps=None, fast=False, mobius=False):
        return dup_inner_isolate_negative_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, eps=eps, fast=fast, mobius=mobius)
    def dup_isolate_real_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_real_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)
    def dup_isolate_real_roots(self, f, eps=None, inf=None, sup=None, basis=False, fast=False):
        return dup_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)
    def dup_isolate_real_roots_list(self, polys, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
        return dup_isolate_real_roots_list(list(map(self.to_dense, polys)), self.domain, eps=eps, inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)
    def dup_count_real_roots(self, f, inf=None, sup=None):
        return dup_count_real_roots(self.to_dense(f), self.domain, inf=inf, sup=sup)
    def dup_count_complex_roots(self, f, inf=None, sup=None, exclude=None):
        return dup_count_complex_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, exclude=exclude)
    def dup_isolate_complex_roots_sqf(self, f, eps=None, inf=None, sup=None, blackbox=False):
        return dup_isolate_complex_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, blackbox=blackbox)
    def dup_isolate_all_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_all_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)
    def dup_isolate_all_roots(self, f, eps=None, inf=None, sup=None, fast=False):
        return dup_isolate_all_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast)

    def fateman_poly_F_1(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_1
        return tuple(map(self.from_dense, dmp_fateman_poly_F_1(self.ngens-1, self.domain)))
    def fateman_poly_F_2(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_2
        return tuple(map(self.from_dense, dmp_fateman_poly_F_2(self.ngens-1, self.domain)))
    def fateman_poly_F_3(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_3
        return tuple(map(self.from_dense, dmp_fateman_poly_F_3(self.ngens-1, self.domain)))

    def to_gf_dense(self, element):
        return gf_strip([ self.domain.dom.convert(c, self.domain) for c in self.wrap(element).to_dense() ])

    def from_gf_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain.dom))

    def gf_degree(self, f):
        return gf_degree(self.to_gf_dense(f))

    def gf_LC(self, f):
        return gf_LC(self.to_gf_dense(f), self.domain.dom)
    def gf_TC(self, f):
        return gf_TC(self.to_gf_dense(f), self.domain.dom)

    def gf_strip(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f)))
    def gf_trunc(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod))
    def gf_normal(self, f):
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_from_dict(self, f):
        return self.from_gf_dense(gf_from_dict(f, self.domain.mod, self.domain.dom))
    def gf_to_dict(self, f, symmetric=True):
        return gf_to_dict(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_from_int_poly(self, f):
        return self.from_gf_dense(gf_from_int_poly(f, self.domain.mod))
    def gf_to_int_poly(self, f, symmetric=True):
        return gf_to_int_poly(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_neg(self, f):
        return self.from_gf_dense(gf_neg(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_ground(self, f, a):
        return self.from_gf_dense(gf_add_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_sub_ground(self, f, a):
        return self.from_gf_dense(gf_sub_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_mul_ground(self, f, a):
        return self.from_gf_dense(gf_mul_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))
    def gf_quo_ground(self, f, a):
        return self.from_gf_dense(gf_quo_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_add(self, f, g):
        return self.from_gf_dense(gf_add(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_sub(self, f, g):
        return self.from_gf_dense(gf_sub(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_mul(self, f, g):
        return self.from_gf_dense(gf_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_sqr(self, f):
        return self.from_gf_dense(gf_sqr(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_mul(self, f, g, h):
        return self.from_gf_dense(gf_add_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))
    def gf_sub_mul(self, f, g, h):
        return self.from_gf_dense(gf_sub_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))

    def gf_expand(self, F):
        return self.from_gf_dense(gf_expand(list(map(self.to_gf_dense, F)), self.domain.mod, self.domain.dom))

    def gf_div(self, f, g):
        q, r = gf_div(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(q), self.from_gf_dense(r)
    def gf_rem(self, f, g):
        return self.from_gf_dense(gf_rem(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_quo(self, f, g):
        return self.from_gf_dense(gf_quo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_exquo(self, f, g):
        return self.from_gf_dense(gf_exquo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_lshift(self, f, n):
        return self.from_gf_dense(gf_lshift(self.to_gf_dense(f), n, self.domain.dom))
    def gf_rshift(self, f, n):
        return self.from_gf_dense(gf_rshift(self.to_gf_dense(f), n, self.domain.dom))

    def gf_pow(self, f, n):
        return self.from_gf_dense(gf_pow(self.to_gf_dense(f), n, self.domain.mod, self.domain.dom))
    def gf_pow_mod(self, f, n, g):
        return self.from_gf_dense(gf_pow_mod(self.to_gf_dense(f), n, self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_cofactors(self, f, g):
        h, cff, cfg = gf_cofactors(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(h), self.from_gf_dense(cff), self.from_gf_dense(cfg)
    def gf_gcd(self, f, g):
        return self.from_gf_dense(gf_gcd(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_lcm(self, f, g):
        return self.from_gf_dense(gf_lcm(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_gcdex(self, f, g):
        return self.from_gf_dense(gf_gcdex(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_monic(self, f):
        return self.from_gf_dense(gf_monic(self.to_gf_dense(f), self.domain.mod, self.domain.dom))
    def gf_diff(self, f):
        return self.from_gf_dense(gf_diff(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_eval(self, f, a):
        return gf_eval(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom)
    def gf_multi_eval(self, f, A):
        return gf_multi_eval(self.to_gf_dense(f), A, self.domain.mod, self.domain.dom)

    def gf_compose(self, f, g):
        return self.from_gf_dense(gf_compose(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_compose_mod(self, g, h, f):
        return self.from_gf_dense(gf_compose_mod(self.to_gf_dense(g), self.to_gf_dense(h), self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_trace_map(self, a, b, c, n, f):
        a = self.to_gf_dense(a)
        b = self.to_gf_dense(b)
        c = self.to_gf_dense(c)
        f = self.to_gf_dense(f)
        U, V = gf_trace_map(a, b, c, n, f, self.domain.mod, self.domain.dom)
        return self.from_gf_dense(U), self.from_gf_dense(V)

    def gf_random(self, n):
        return self.from_gf_dense(gf_random(n, self.domain.mod, self.domain.dom))
    def gf_irreducible(self, n):
        return self.from_gf_dense(gf_irreducible(n, self.domain.mod, self.domain.dom))

    def gf_irred_p_ben_or(self, f):
        return gf_irred_p_ben_or(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_irred_p_rabin(self, f):
        return gf_irred_p_rabin(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_irreducible_p(self, f):
        return gf_irreducible_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_sqf_p(self, f):
        return gf_sqf_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    def gf_sqf_part(self, f):
        return self.from_gf_dense(gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom))
    def gf_sqf_list(self, f, all=False):
        coeff, factors = gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]

    def gf_Qmatrix(self, f):
        return gf_Qmatrix(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    def gf_berlekamp(self, f):
        factors = gf_berlekamp(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_ddf_zassenhaus(self, f):
        factors = gf_ddf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]
    def gf_edf_zassenhaus(self, f, n):
        factors = gf_edf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_ddf_shoup(self, f):
        factors = gf_ddf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]
    def gf_edf_shoup(self, f, n):
        factors = gf_edf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_zassenhaus(self, f):
        factors = gf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]
    def gf_shoup(self, f):
        factors = gf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    def gf_factor_sqf(self, f, method=None):
        coeff, factors = gf_factor_sqf(self.to_gf_dense(f), self.domain.mod, self.domain.dom, method=method)
        return coeff, [ self.from_gf_dense(g) for g in factors ]
    def gf_factor(self, f):
        coeff, factors = gf_factor(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]
<|endoftext|>"""Advanced tools for dense recursive polynomials in ``K[x]`` or ``K[X]``. """


from sympy.polys.densearith import (
    dup_add_term, dmp_add_term,
    dup_lshift,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr,
    dup_div,
    dup_rem, dmp_rem,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground,
    dup_exquo_ground, dmp_exquo_ground,
)
from sympy.polys.densebasic import (
    dup_strip, dmp_strip,
    dup_convert, dmp_convert,
    dup_degree, dmp_degree,
    dmp_to_dict,
    dmp_from_dict,
    dup_LC, dmp_LC, dmp_ground_LC,
    dup_TC, dmp_TC,
    dmp_zero, dmp_ground,
    dmp_zero_p,
    dup_to_raw_dict, dup_from_raw_dict,
    dmp_zeros,
    dmp_include,
)
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    DomainError
)

from math import ceil as _ceil, log2 as _log2


def dup_integrate(f, m, K):
    """
    Computes the indefinite integral of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_integrate(x**2 + 2*x, 1)
    1/3*x**3 + x**2
    >>> R.dup_integrate(x**2 + 2*x, 2)
    1/12*x**4 + 1/3*x**3

    """
    if m <= 0 or not f:
        return f

    g = [K.zero]*m

    for i, c in enumerate(reversed(f)):
        n = i + 1

        for j in range(1, m):
            n *= i + j + 1

        g.insert(0, K.exquo(c, K(n)))

    return g


def dmp_integrate(f, m, u, K):
    """
    Computes the indefinite integral of ``f`` in ``x_0`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate(x + 2*y, 1)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate(x + 2*y, 2)
    1/6*x**3 + x**2*y

    """
    if not u:
        return dup_integrate(f, m, K)

    if m <= 0 or dmp_zero_p(f, u):
        return f

    g, v = dmp_zeros(m, u - 1, K), u - 1

    for i, c in enumerate(reversed(f)):
        n = i + 1

        for j in range(1, m):
            n *= i + j + 1

        g.insert(0, dmp_quo_ground(c, K(n), v, K))

    return g


def _rec_integrate_in(g, m, v, i, j, K):
    """Recursive helper for :func:`dmp_integrate_in`."""
    if i == j:
        return dmp_integrate(g, m, v, K)

    w, i = v - 1, i + 1

    return dmp_strip([ _rec_integrate_in(c, m, w, i, j, K) for c in g ], v)


def dmp_integrate_in(f, m, j, u, K):
    """
    Computes the indefinite integral of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate_in(x + 2*y, 1, 0)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate_in(x + 2*y, 1, 1)
    x*y + y**2

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= u expected, got u = %d, j = %d" % (u, j))

    return _rec_integrate_in(f, m, u, 0, j, K)


def dup_diff(f, m, K):
    """
    ``m``-th order derivative of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 1)
    3*x**2 + 4*x + 3
    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 2)
    6*x + 4

    """
    if m <= 0:
        return f

    n = dup_degree(f)

    if n < m:
        return []

    deriv = []

    if m == 1:
        for coeff in f[:-m]:
            deriv.append(K(n)*coeff)
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n

            for i in range(n - 1, n - m, -1):
                k *= i

            deriv.append(K(k)*coeff)
            n -= 1

    return dup_strip(deriv)


def dmp_diff(f, m, u, K):
    """
    ``m``-th order derivative in ``x_0`` of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff(f, 1)
    y**2 + 2*y + 3
    >>> R.dmp_diff(f, 2)
    0

    """
    if not u:
        return dup_diff(f, m, K)
    if m <= 0:
        return f

    n = dmp_degree(f, u)

    if n < m:
        return dmp_zero(u)

    deriv, v = [], u - 1

    if m == 1:
        for coeff in f[:-m]:
            deriv.append(dmp_mul_ground(coeff, K(n), v, K))
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n

            for i in range(n - 1, n - m, -1):
                k *= i

            deriv.append(dmp_mul_ground(coeff, K(k), v, K))
            n -= 1

    return dmp_strip(deriv, u)


def _rec_diff_in(g, m, v, i, j, K):
    """Recursive helper for :func:`dmp_diff_in`."""
    if i == j:
        return dmp_diff(g, m, v, K)

    w, i = v - 1, i + 1

    return dmp_strip([ _rec_diff_in(c, m, w, i, j, K) for c in g ], v)


def dmp_diff_in(f, m, j, u, K):
    """
    ``m``-th order derivative in ``x_j`` of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff_in(f, 1, 0)
    y**2 + 2*y + 3
    >>> R.dmp_diff_in(f, 1, 1)
    2*x*y + 2*x + 4*y + 3

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    return _rec_diff_in(f, m, u, 0, j, K)


def dup_eval(f, a, K):
    """
    Evaluate a polynomial at ``x = a`` in ``K[x]`` using Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_eval(x**2 + 2*x + 3, 2)
    11

    """
    if not a:
        return K.convert(dup_TC(f, K))

    result = K.zero

    for c in f:
        result *= a
        result += c

    return result


def dmp_eval(f, a, u, K):
    """
    Evaluate a polynomial at ``x_0 = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_eval(2*x*y + 3*x + y + 2, 2)
    5*y + 8

    """
    if not u:
        return dup_eval(f, a, K)

    if not a:
        return dmp_TC(f, K)

    result, v = dmp_LC(f, K), u - 1

    for coeff in f[1:]:
        result = dmp_mul_ground(result, a, v, K)
