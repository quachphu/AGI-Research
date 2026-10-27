        Helper function for pre-processing the information.
        """
        is_insufficient = False

        if not evaluate: # avoid pre-processing if the result is not to be evaluated
            return (True, None, None, None)

        # extracting transition matrix and state space
        trans_probs, state_index, given_condition = self._extract_information(given_condition)

        # given_condition does not have sufficient information
        # for computations
        if trans_probs is None or \
            given_condition is None:
            is_insufficient = True
        else:
            # checking transition probabilities
            if isinstance(self, DiscreteMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=1)
            elif isinstance(self, ContinuousMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=0)

            # working out state space
            state_index = self._work_out_state_index(state_index, given_condition, trans_probs)

        return is_insufficient, trans_probs, state_index, given_condition

    def replace_with_index(self, condition):
        if isinstance(condition, Relational):
            lhs, rhs = condition.lhs, condition.rhs
            if not isinstance(lhs, RandomIndexedSymbol):
                lhs, rhs = rhs, lhs
            condition = type(condition)(self.index_of.get(lhs, lhs),
                                        self.index_of.get(rhs, rhs))
        return condition

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Handles probability queries for Markov process.

        Parameters
        ==========

        condition: Relational
        given_condition: Relational/And

        Returns
        =======
        Probability
            If the information is not sufficient.
        Expr
            In all other cases.

        Note
        ====
        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.
        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """
        check, mat, state_index, new_given_condition = \
            self._preprocess(given_condition, evaluate)

        rv = list(condition.atoms(RandomIndexedSymbol))
        symbolic = False
        for sym in rv:
            if sym.key.is_symbol:
                symbolic = True
                break

        if check:
            return Probability(condition, new_given_condition)

        if isinstance(self, ContinuousMarkovChain):
            trans_probs = self.transition_probabilities(mat)
        elif isinstance(self, DiscreteMarkovChain):
            trans_probs = mat
        condition = self.replace_with_index(condition)
        given_condition = self.replace_with_index(given_condition)
        new_given_condition = self.replace_with_index(new_given_condition)

        if isinstance(condition, Relational):
            if isinstance(new_given_condition, And):
                gcs = new_given_condition.args
            else:
                gcs = (new_given_condition, )
            min_key_rv = list(new_given_condition.atoms(RandomIndexedSymbol))

            if len(min_key_rv):
                min_key_rv = min_key_rv[0]
                for r in rv:
                    if min_key_rv.key.is_symbol or r.key.is_symbol:
                        continue
                    if min_key_rv.key > r.key:
                        return Probability(condition)
            else:
                min_key_rv = None
                return Probability(condition)

            if symbolic:
                return self._symbolic_probability(condition, new_given_condition, rv, min_key_rv)

            if len(rv) > 1:
                rv[0] = condition.lhs
                rv[1] = condition.rhs
                if rv[0].key < rv[1].key:
                        rv[0], rv[1] = rv[1], rv[0]
                        if isinstance(condition, Gt):
                            condition = Lt(condition.lhs, condition.rhs)
                        elif isinstance(condition, Lt):
                            condition = Gt(condition.lhs, condition.rhs)
                        elif isinstance(condition, Ge):
                            condition = Le(condition.lhs, condition.rhs)
                        elif isinstance(condition, Le):
                            condition = Ge(condition.lhs, condition.rhs)
                s = Rational(0, 1)
                n = len(self.state_space)

                if isinstance(condition, (Eq, Ne)):
                    for i in range(0, n):
                        s += self.probability(Eq(rv[0], i), Eq(rv[1], i)) * self.probability(Eq(rv[1], i), new_given_condition)
                    return s if isinstance(condition, Eq) else 1 - s
                else:
                    upper = 0
                    greater = False
                    if isinstance(condition, (Ge, Lt)):
                        upper = 1
                    if isinstance(condition, (Ge, Gt)):
                        greater = True

                    for i in range(0, n):
                        if i <= n//2:
                            for j in range(0, i + upper):
                                s += self.probability(Eq(rv[0], i), Eq(rv[1], j)) * self.probability(Eq(rv[1], j), new_given_condition)
                        else:
                            s += self.probability(Eq(rv[0], i), new_given_condition)
                            for j in range(i + upper, n):
                                s -= self.probability(Eq(rv[0], i), Eq(rv[1], j)) * self.probability(Eq(rv[1], j), new_given_condition)
                    return s if greater else 1 - s

            rv = rv[0]
            states = condition.as_set()
            prob, gstate = {}, None
            for gc in gcs:
                if gc.has(min_key_rv):
                    if gc.has(Probability):
                        p, gp = (gc.rhs, gc.lhs) if isinstance(gc.lhs, Probability) \
                                    else (gc.lhs, gc.rhs)
                        gr = gp.args[0]
                        gset = Intersection(gr.as_set(), state_index)
                        gstate = list(gset)[0]
                        prob[gset] = p
                    else:
                        _, gstate = (gc.lhs.key, gc.rhs) if isinstance(gc.lhs, RandomIndexedSymbol) \
                                    else (gc.rhs.key, gc.lhs)

            if not all(k in self.index_set for k in (rv.key, min_key_rv.key)):
                raise IndexError("The timestamps of the process are not in it's index set.")
            states = Intersection(states, state_index) if not isinstance(self.number_of_states, Symbol) else states
            for state in Union(states, FiniteSet(gstate)):
                if not state.is_Integer or Ge(state, mat.shape[0]) is True:
                    raise IndexError("No information is available for (%s, %s) in "
                        "transition probabilities of shape, (%s, %s). "
                        "State space is zero indexed."
                        %(gstate, state, mat.shape[0], mat.shape[1]))
            if prob:
                gstates = Union(*prob.keys())
                if len(gstates) == 1:
                    gstate = list(gstates)[0]
                    gprob = list(prob.values())[0]
                    prob[gstates] = gprob
                elif len(gstates) == len(state_index) - 1:
                    gstate = list(state_index - gstates)[0]
                    gprob = S.One - sum(prob.values())
                    prob[state_index - gstates] = gprob
                else:
                    raise ValueError("Conflicting information.")
            else:
                gprob = S.One

            if min_key_rv == rv:
                return sum(prob[FiniteSet(state)] for state in states)
            if isinstance(self, ContinuousMarkovChain):
                return gprob * sum(trans_probs(rv.key - min_key_rv.key).__getitem__((gstate, state))
                                    for state in states)
            if isinstance(self, DiscreteMarkovChain):
                return gprob * sum((trans_probs**(rv.key - min_key_rv.key)).__getitem__((gstate, state))
                                    for state in states)

        if isinstance(condition, Not):
            expr = condition.args[0]
            return S.One - self.probability(expr, given_condition, evaluate, **kwargs)

        if isinstance(condition, And):
            compute_later, state2cond, conds = [], {}, condition.args
            for expr in conds:
                if isinstance(expr, Relational):
                    ris = list(expr.atoms(RandomIndexedSymbol))[0]
                    if state2cond.get(ris, None) is None:
                        state2cond[ris] = S.true
                    state2cond[ris] &= expr
                else:
                    compute_later.append(expr)
            ris = []
            for ri in state2cond:
                ris.append(ri)
                cset = Intersection(state2cond[ri].as_set(), state_index)
                if len(cset) == 0:
                    return S.Zero
                state2cond[ri] = cset.as_relational(ri)
            sorted_ris = sorted(ris, key=lambda ri: ri.key)
            prod = self.probability(state2cond[sorted_ris[0]], given_condition, evaluate, **kwargs)
            for i in range(1, len(sorted_ris)):
                ri, prev_ri = sorted_ris[i], sorted_ris[i-1]
                if not isinstance(state2cond[ri], Eq):
                    raise ValueError("The process is in multiple states at %s, unable to determine the probability."%(ri))
                mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)
                prod *= self.probability(state2cond[ri], state2cond[prev_ri]
                                 & mat_of
                                 & StochasticStateSpaceOf(self, state_index),
                                 evaluate, **kwargs)
            for expr in compute_later:
                prod *= self.probability(expr, given_condition, evaluate, **kwargs)
            return prod

        if isinstance(condition, Or):
            return sum(self.probability(expr, given_condition, evaluate, **kwargs)
                        for expr in condition.args)

        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(condition, given_condition))

    def _symbolic_probability(self, condition, new_given_condition, rv, min_key_rv):
        #Function to calculate probability for queries with symbols
        if isinstance(condition, Relational):
            curr_state = new_given_condition.rhs if isinstance(new_given_condition.lhs, RandomIndexedSymbol) \
                    else new_given_condition.lhs
            next_state = condition.rhs if isinstance(condition.lhs, RandomIndexedSymbol) \
                else condition.lhs

            if isinstance(condition, (Eq, Ne)):
                if isinstance(self, DiscreteMarkovChain):
                    P = self.transition_probabilities**(rv[0].key - min_key_rv.key)
                else:
                    P = exp(self.generator_matrix*(rv[0].key - min_key_rv.key))
                prob = P[curr_state, next_state] if isinstance(condition, Eq) else 1 - P[curr_state, next_state]
                return Piecewise((prob, rv[0].key > min_key_rv.key), (Probability(condition), True))
            else:
                upper = 1
                greater = False
                if isinstance(condition, (Ge, Lt)):
                    upper = 0
                if isinstance(condition, (Ge, Gt)):
                    greater = True
                k = Dummy('k')
                condition = Eq(condition.lhs, k) if isinstance(condition.lhs, RandomIndexedSymbol)\
                    else Eq(condition.rhs, k)
                total = Sum(self.probability(condition, new_given_condition), (k, next_state + upper, self.state_space._sup))
                return Piecewise((total, rv[0].key > min_key_rv.key), (Probability(condition), True)) if greater\
                    else Piecewise((1 - total, rv[0].key > min_key_rv.key), (Probability(condition), True))
        else:
            return Probability(condition, new_given_condition)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Handles expectation queries for markov process.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation
            Unevaluated object if computations cannot be done due to
            insufficient information.
        Expr
            In all other cases when the computations are successful.

        Note
        ====

        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.

        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """

        check, mat, state_index, condition = \
            self._preprocess(condition, evaluate)

        if check:
            return Expectation(expr, condition)

        rvs = random_symbols(expr)
        if isinstance(expr, Expr) and isinstance(condition, Eq) \
            and len(rvs) == 1:
            # handle queries similar to E(f(X[i]), Eq(X[i-m], <some-state>))
            condition=self.replace_with_index(condition)
            state_index=self.replace_with_index(state_index)
            rv = list(rvs)[0]
            lhsg, rhsg = condition.lhs, condition.rhs
            if not isinstance(lhsg, RandomIndexedSymbol):
                lhsg, rhsg = (rhsg, lhsg)
            if rhsg not in state_index:
                raise ValueError("%s state is not in the state space."%(rhsg))
            if rv.key < lhsg.key:
                raise ValueError("Incorrect given condition is given, expectation "
                    "time %s < time %s"%(rv.key, rv.key))
            mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)
            cond = condition & mat_of & \
                    StochasticStateSpaceOf(self, state_index)
            func = lambda s: self.probability(Eq(rv, s), cond) * expr.subs(rv, self._state_index[s])
            return sum(func(s) for s in state_index)

        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(expr, condition))

class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
    """
    Represents a finite discrete time-homogeneous Markov chain.

    This type of Markov Chain can be uniquely characterised by
    its (ordered) state space and its one-step transition probability
    matrix.

    Parameters
    ==========

    sym:
        The name given to the Markov Chain
    state_space:
        Optional, by default, Range(n)
    trans_probs:
        Optional, by default, MatrixSymbol('_T', n, n)

    Examples
    ========

    >>> from sympy.stats import DiscreteMarkovChain, TransitionMatrixOf, P, E
    >>> from sympy import Matrix, MatrixSymbol, Eq, symbols
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> YS = DiscreteMarkovChain("Y")

    >>> Y.state_space
    {0, 1, 2}
    >>> Y.transition_probabilities
    Matrix([
    [0.5, 0.2, 0.3],
    [0.2, 0.5, 0.3],
    [0.2, 0.3, 0.5]])
    >>> TS = MatrixSymbol('T', 3, 3)
    >>> P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TS))
    T[0, 2]*T[1, 0] + T[1, 1]*T[1, 2] + T[1, 2]*T[2, 2]
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    Probabilities will be calculated based on indexes rather
    than state names. For example, with the Sunny-Cloudy-Rainy
    model with string state names:

    >>> from sympy.core.symbol import Str
    >>> Y = DiscreteMarkovChain("Y", [Str('Sunny'), Str('Cloudy'), Str('Rainy')], T)
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    This gives the same answer as the ``[0, 1, 2]`` state space.
    Currently, there is no support for state names within probability
    and expectation statements. Here is a work-around using ``Str``:

    >>> P(Eq(Str('Rainy'), Y[3]), Eq(Y[1], Str('Cloudy'))).round(2)
    0.36

    Symbol state names can also be used:

    >>> sunny, cloudy, rainy = symbols('Sunny, Cloudy, Rainy')
    >>> Y = DiscreteMarkovChain("Y", [sunny, cloudy, rainy], T)
    >>> P(Eq(Y[3], rainy), Eq(Y[1], cloudy)).round(2)
    0.36

    Expectations will be calculated as follows:

    >>> E(Y[3], Eq(Y[1], cloudy))
    0.38*Cloudy + 0.36*Rainy + 0.26*Sunny

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    transition matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> P(Eq(Y[3], Y[1]), Eq(Y[0], 0)).round(3)
    0.409
    >>> P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2)
    0.36
    >>> P(Le(Y[15], Y[10]), Eq(Y[8], 2)).round(7)
    0.6963328

    Symbolic probability queries are also supported

    >>> a, b, c, d = symbols('a b c d')
    >>> T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1}).round(4)
    0.3096
    >>> P(Eq(Y[10], 2), Eq(Y[5], 1)).evalf().round(4)
    0.3096
    >>> query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    >>> query_gt.subs({a:21, b:0, c:5, d:0}).evalf().round(5)
    0.64705
    >>> P(Gt(Y[21], 0), Eq(Y[5], 0)).round(5)
    0.64705

    There is limited support for arbitrarily sized states:

    >>> n = symbols('n', nonnegative=True, integer=True)
    >>> T = MatrixSymbol('T', n, n)
    >>> Y = DiscreteMarkovChain("Y", trans_probs=T)
    >>> Y.state_space
    Range(0, n, 1)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1})
    (T**5)[1, 2]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Discrete-time_Markov_chain
    .. [2] https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    """
    index_set = S.Naturals0

    def __new__(cls, sym, state_space=None, trans_probs=None):
        sym = _symbol_converter(sym)

        state_space, trans_probs = MarkovProcess._sanity_checks(state_space, trans_probs)

        obj = Basic.__new__(cls, sym, state_space, trans_probs) # type: ignore
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj._state_index):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def transition_probabilities(self):
        """
        Transition probabilities of discrete Markov chain,
        either an instance of Matrix or MatrixSymbol.
        """
        return self.args[2]

    def communication_classes(self) -> list[tuple[list[Basic], Boolean, Integer]]:
        """
        Returns the list of communication classes that partition
        the states of the markov chain.

        A communication class is defined to be a set of states
        such that every state in that set is reachable from
        every other state in that set. Due to its properties
        this forms a class in the mathematical sense.
        Communication classes are also known as recurrence
        classes.

        Returns
        =======

        classes
            The ``classes`` are a list of tuples. Each
            tuple represents a single communication class
            with its properties. The first element in the
            tuple is the list of states in the class, the
            second element is whether the class is recurrent
            and the third element is the period of the
            communication class.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix
        >>> T = Matrix([[0, 1, 0],
        ...             [1, 0, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', [1, 2, 3], T)
        >>> classes = X.communication_classes()
        >>> for states, is_recurrent, period in classes:
        ...     states, is_recurrent, period
        ([1, 2], True, 2)
        ([3], False, 1)

        From this we can see that states ``1`` and ``2``
        communicate, are recurrent and have a period
        of 2. We can also see state ``3`` is transient
        with a period of 1.

        Notes
        =====

        The algorithm used is of order ``O(n**2)`` where
        ``n`` is the number of states in the markov chain.
        It uses Tarjan's algorithm to find the classes
        themselves and then it uses a breadth-first search
        algorithm to find each class's periodicity.
        Most of the algorithm's components approach ``O(n)``
        as the matrix becomes more and more sparse.

        References
        ==========

        .. [1] https://web.archive.org/web/20220207032113/https://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-MCII.pdf
        .. [2] https://cecas.clemson.edu/~shierd/Shier/markov.pdf
        .. [3] https://www.proquest.com/openview/4adc6a51d8371be5b0e4c7dff287fc70/1?pq-origsite=gscholar&cbl=2026366&diss=y
        .. [4] https://www.mathworks.com/help/econ/dtmc.classify.html
        """
        n = self.number_of_states
        T = self.transition_probabilities

        if isinstance(T, MatrixSymbol):
            raise NotImplementedError("Cannot perform the operation with a symbolic matrix.")

        # begin Tarjan's algorithm
        V = Range(n)
        # don't use state names. Rather use state
        # indexes since we use them for matrix
        # indexing here and later onward
        E = [(i, j) for i in V for j in V if T[i, j] != 0]
        classes = strongly_connected_components((V, E))
        # end Tarjan's algorithm

        recurrence = []
        periods = []
        for class_ in classes:
            # begin recurrent check (similar to self._check_trans_probs())
            submatrix = T[class_, class_]  # get the submatrix with those states
            is_recurrent = S.true
            rows = submatrix.tolist()
            for row in rows:
                if (sum(row) - 1) != 0:
                    is_recurrent = S.false
                    break
            recurrence.append(is_recurrent)
            # end recurrent check

            # begin breadth-first search
            non_tree_edge_values: set[int] = set()
            visited = {class_[0]}
            newly_visited = {class_[0]}
            level = {class_[0]: 0}
            current_level = 0
            done = False  # imitate a do-while loop
            while not done:  # runs at most len(class_) times
                done = len(visited) == len(class_)
                current_level += 1

                # this loop and the while loop above run a combined len(class_) number of times.
                # so this triple nested loop runs through each of the n states once.
                for i in newly_visited:

                    # the loop below runs len(class_) number of times
                    # complexity is around about O(n * avg(len(class_)))
                    newly_visited = {j for j in class_ if T[i, j] != 0}

                    new_tree_edges = newly_visited.difference(visited)
                    for j in new_tree_edges:
                        level[j] = current_level

                    new_non_tree_edges = newly_visited.intersection(visited)
                    new_non_tree_edge_values = {level[i]-level[j]+1 for j in new_non_tree_edges}

                    non_tree_edge_values = non_tree_edge_values.union(new_non_tree_edge_values)
                    visited = visited.union(new_tree_edges)

            # igcd needs at least 2 arguments
            positive_ntev = {val_e for val_e in non_tree_edge_values if val_e > 0}
            if len(positive_ntev) == 0:
                periods.append(len(class_))
            elif len(positive_ntev) == 1:
                periods.append(positive_ntev.pop())
            else:
                periods.append(igcd(*positive_ntev))
            # end breadth-first search

        # convert back to the user's state names
        classes = [[_sympify(self._state_index[i]) for i in class_] for class_ in classes]
        return list(zip(classes, recurrence, map(Integer,periods)))

    def fundamental_matrix(self):
        """
        Each entry fundamental matrix can be interpreted as
        the expected number of times the chains is in state j
        if it started in state i.

        References
        ==========

        .. [1] https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/

        """
        _, _, _, Q = self.decompose()

        if Q.shape[0] > 0:  # if non-ergodic
            I = eye(Q.shape[0])
            if (I - Q).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - Q).inv().as_immutable()
        else:  # if ergodic
            P = self.transition_probabilities
            I = eye(P.shape[0])
            w = self.fixed_row_vector()
            W = Matrix([list(w) for i in range(0, P.shape[0])])
            if (I - P + W).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - P + W).inv().as_immutable()

    def absorbing_probabilities(self):
        """
        Computes the absorbing probabilities, i.e.
        the ij-th entry of the matrix denotes the
        probability of Markov chain being absorbed
        in state j starting from state i.
        """
        _, _, R, _ = self.decompose()
        N = self.fundamental_matrix()
        if R is None or N is None:
            return None
        return N*R

    def absorbing_probabilites(self):
        sympy_deprecation_warning(
            """
            DiscreteMarkovChain.absorbing_probabilites() is deprecated. Use
            absorbing_probabilities() instead (note the spelling difference).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-absorbing_probabilites",
        )
        return self.absorbing_probabilities()

    def is_regular(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        classes, _, periods = list(zip(*tuples))
        return And(len(classes) == 1, periods[0] == 1)

    def is_ergodic(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        classes, _, _ = list(zip(*tuples))
        return S(len(classes) == 1)

    def is_absorbing_state(self, state):
        trans_probs = self.transition_probabilities
        if isinstance(trans_probs, ImmutableMatrix) and \
            state < trans_probs.shape[0]:
            return S(trans_probs[state, state]) is S.One

    def is_absorbing_chain(self):
        states, A, B, C = self.decompose()
        r = A.shape[0]
        return And(r > 0, A == Identity(r).as_explicit())

    def stationary_distribution(self, condition_set=False) -> ImmutableMatrix | ConditionSet | Lambda:
        r"""
        The stationary distribution is any row vector, p, that solves p = pP,
        is row stochastic and each element in p must be nonnegative.
        That means in matrix form: :math:`(P-I)^T p^T = 0` and
        :math:`(1, \dots, 1) p = 1`
        where ``P`` is the one-step transition matrix.

        All time-homogeneous Markov Chains with a finite state space
        have at least one stationary distribution. In addition, if
        a finite time-homogeneous Markov Chain is irreducible, the
        stationary distribution is unique.

        Parameters
        ==========

        condition_set : bool
            If the chain has a symbolic size or transition matrix,
            it will return a ``Lambda`` if ``False`` and return a
            ``ConditionSet`` if ``True``.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        An irreducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13, 5/13, 0]])

        A reducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [0, 0, 1]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13 - 8*tau0/13, 5/13 - 5*tau0/13, tau0]])

        >>> Y = DiscreteMarkovChain('Y')
        >>> Y.stationary_distribution()
        Lambda((wm, _T), Eq(wm*_T, wm))

        >>> Y.stationary_distribution(condition_set=True)
        ConditionSet(wm, Eq(wm*_T, wm))

        References
        ==========

        .. [1] https://www.probabilitycourse.com/chapter11/11_2_6_stationary_and_limiting_distributions.php
        .. [2] https://web.archive.org/web/20210508104430/https://galton.uchicago.edu/~yibi/teaching/stat317/2014/Lectures/Lecture4_6up.pdf

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.limiting_distribution
        """
        trans_probs = self.transition_probabilities
        n = self.number_of_states

        if n == 0:
            return ImmutableMatrix(Matrix([[]]))

        # symbolic matrix version
        if isinstance(trans_probs, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, n)
            if condition_set:
                return ConditionSet(wm, Eq(wm * trans_probs, wm))
            else:
                return Lambda((wm, trans_probs), Eq(wm * trans_probs, wm))

        # numeric matrix version
        a = Matrix(trans_probs - Identity(n)).T
        a[0, 0:n] = ones(1, n) # type: ignore
        b = zeros(n, 1)
        b[0, 0] = 1

        soln = list(linsolve((a, b)))[0]
        return ImmutableMatrix([soln])

    def fixed_row_vector(self):
        """
        A wrapper for ``stationary_distribution()``.
        """
        return self.stationary_distribution()

    @property
    def limiting_distribution(self):
        """
        The fixed row vector is the limiting
        distribution of a discrete Markov chain.
        """
        return self.fixed_row_vector()

    def decompose(self) -> tuple[list[Basic], ImmutableMatrix, ImmutableMatrix, ImmutableMatrix]:
        """
        Decomposes the transition matrix into submatrices with
        special properties.

        The transition matrix can be decomposed into 4 submatrices:
        - A - the submatrix from recurrent states to recurrent states.
        - B - the submatrix from transient to recurrent states.
        - C - the submatrix from transient to transient states.
        - O - the submatrix of zeros for recurrent to transient states.

        Returns
        =======

        states, A, B, C
            ``states`` - a list of state names with the first being
            the recurrent states and the last being
            the transient states in the order
            of the row names of A and then the row names of C.
            ``A`` - the submatrix from recurrent states to recurrent states.
            ``B`` - the submatrix from transient to recurrent states.
            ``C`` - the submatrix from transient to transient states.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        One can decompose this chain for example:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, A, B, C = X.decompose()
        >>> states
        [2, 0, 1, 3, 4]

        >>> A   # recurrent to recurrent
        Matrix([[1]])

        >>> B  # transient to recurrent
        Matrix([
        [  0],
        [2/5],
        [1/2],
        [  0]])

        >>> C  # transient to transient
        Matrix([
        [1/2, 1/2,   0,   0],
        [2/5, 1/5,   0,   0],
        [  0,   0, 1/2,   0],
        [1/2,   0,   0, 1/2]])

        This means that state 2 is the only absorbing state
        (since A is a 1x1 matrix). B is a 4x1 matrix since
        the 4 remaining transient states all merge into recurrent
        state 2. And C is the 4x4 matrix that shows how the
        transient states 0, 1, 3, 4 all interact.

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.canonical_form

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absorbing_Markov_chain
        .. [2] https://people.brandeis.edu/~igusa/Math56aS08/Math56a_S08_notes015.pdf
        """
        trans_probs = self.transition_probabilities

        classes = self.communication_classes()
        r_states = []
        t_states = []

        for states, recurrent, period in classes:
            if recurrent:
                r_states += states
            else:
                t_states += states

        states = r_states + t_states
        indexes = [self.index_of[state] for state in states] # type: ignore

        A = Matrix(len(r_states), len(r_states),
                   lambda i, j: trans_probs[indexes[i], indexes[j]])

        B = Matrix(len(t_states), len(r_states),
                   lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[j]])

        C = Matrix(len(t_states), len(t_states),
                   lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[len(r_states) + j]])

        return states, A.as_immutable(), B.as_immutable(), C.as_immutable()

    def canonical_form(self) -> tuple[list[Basic], ImmutableMatrix]:
        """
        Reorders the one-step transition matrix
        so that recurrent states appear first and transient
        states appear last. Other representations include inserting
        transient states first and recurrent states last.

        Returns
        =======

        states, P_new
            ``states`` is the list that describes the order of the
            new states in the matrix
            so that the ith element in ``states`` is the state of the
            ith row of A.
            ``P_new`` is the new transition matrix in canonical form.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        You can convert your chain into canonical form:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', list(range(1, 6)), trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        The new states are [3, 1, 2, 4, 5] and you can
        create a new chain with this and its canonical
        form will remain the same (since it is already
        in canonical form).

        >>> X = DiscreteMarkovChain('X', states, new_matrix)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        This is not limited to absorbing chains:

        >>> T = Matrix([[0, 5,  5, 0,  0],
        ...             [0, 0,  0, 10, 0],
        ...             [5, 0,  5, 0,  0],
        ...             [0, 10, 0, 0,  0],
        ...             [0, 3,  0, 3,  4]])/10
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [1, 3, 0, 2, 4]

        >>> new_matrix
        Matrix([
        [   0,    1,   0,   0,   0],
        [   1,    0,   0,   0,   0],
        [ 1/2,    0,   0, 1/2,   0],
        [   0,    0, 1/2, 1/2,   0],
        [3/10, 3/10,   0,   0, 2/5]])

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.decompose

        References
        ==========

        .. [1] https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470316887.app1
        .. [2] http://www.columbia.edu/~ww2040/6711F12/lect1023big.pdf
        """
        states, A, B, C = self.decompose()
        O = zeros(A.shape[0], C.shape[1])
        return states, BlockMatrix([[A, O], [B, C]]).as_explicit()

    def sample(self):
        """
        Returns
        =======

        sample: iterator object
            iterator object containing the sample

        """
        if not isinstance(self.transition_probabilities, (Matrix, ImmutableMatrix)):
            raise ValueError("Transition Matrix must be provided for sampling")
        Tlist = self.transition_probabilities.tolist()
        samps = [random.choice(list(self.state_space))]
        yield samps[0]
        time = 1
        densities = {}
        for state in self.state_space:
            states = list(self.state_space)
            densities[state] = {states[i]: Tlist[state][i]
                        for i in range(len(states))}
        while time < S.Infinity:
            samps.append((next(sample_iter(FiniteRV("_", densities[samps[time - 1]])))))
            yield samps[time]
            time += 1

class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
    """
    Represents continuous time Markov chain.

    Parameters
    ==========

    sym : Symbol/str
    state_space : Set
        Optional, by default, S.Reals
    gen_mat : Matrix/ImmutableMatrix/MatrixSymbol
        Optional, by default, None

    Examples
    ========

    >>> from sympy.stats import ContinuousMarkovChain, P
    >>> from sympy import Matrix, S, Eq, Gt
    >>> G = Matrix([[-S(1), S(1)], [S(1), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1], gen_mat=G)
    >>> C.limiting_distribution()
    Matrix([[1/2, 1/2]])
    >>> C.state_space
    {0, 1}
    >>> C.generator_matrix
    Matrix([
    [-1,  1],
    [ 1, -1]])

    Probability queries are supported

    >>> P(Eq(C(1.96), 0), Eq(C(0.78), 1)).round(5)
    0.45279
    >>> P(Gt(C(1.7), 0), Eq(C(0.82), 1)).round(5)
    0.58602

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    generator matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> P(Eq(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.37933
    >>> P(Gt(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.34211
    >>> P(Le(C(1.57), C(3.14)), Eq(C(1.22), 1)).round(4)
    0.7143

    Symbolic probability queries are also supported

    >>> from sympy import symbols
    >>> a,b,c,d = symbols('a b c d')
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> query = P(Eq(C(a), b), Eq(C(c), d))
    >>> query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10)
    0.4002723175
    >>> P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    0.4002723175
    >>> query_gt = P(Gt(C(a), b), Eq(C(c), d))
    >>> query_gt.subs({a:43.2, b:0, c:3.29, d:2}).evalf().round(10)
    0.6832579186
    >>> P(Gt(C(43.2), 0), Eq(C(3.29), 2)).round(10)
    0.6832579186

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain
    .. [2] https://u.math.biu.ac.il/~amirgi/CTMCnotes.pdf
    """
    index_set = S.Reals

    def __new__(cls, sym, state_space=None, gen_mat=None):
        sym = _symbol_converter(sym)
        state_space, gen_mat = MarkovProcess._sanity_checks(state_space, gen_mat)
        obj = Basic.__new__(cls, sym, state_space, gen_mat)
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj.state_space):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def generator_matrix(self):
        return self.args[2]

    @cacheit
    def transition_probabilities(self, gen_mat=None):
        t = Dummy('t')
        if isinstance(gen_mat, (Matrix, ImmutableMatrix)) and \
                gen_mat.is_diagonalizable():
            # for faster computation use diagonalized generator matrix
            Q, D = gen_mat.diagonalize()
            return Lambda(t, Q*exp(t*D)*Q.inv())
        if gen_mat != None:
            return Lambda(t, exp(t*gen_mat))

    def limiting_distribution(self):
        gen_mat = self.generator_matrix
        if gen_mat is None:
            return None
        if isinstance(gen_mat, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, gen_mat.shape[0])
            return Lambda((wm, gen_mat), Eq(wm*gen_mat, wm))
        w = IndexedBase('w')
        wi = [w[i] for i in range(gen_mat.shape[0])]
        wm = Matrix([wi])
        eqs = (wm*gen_mat).tolist()[0]
        eqs.append(sum(wi) - 1)
        soln = list(linsolve(eqs, wi))[0]
        return ImmutableMatrix([soln])


class BernoulliProcess(DiscreteTimeStochasticProcess):
    """
    The Bernoulli process consists of repeated
    independent Bernoulli process trials with the same parameter `p`.
    It's assumed that the probability `p` applies to every
    trial and that the outcomes of each trial
    are independent of all the rest. Therefore Bernoulli Process
    is Discrete State and Discrete Time Stochastic Process.

    Parameters
    ==========

    sym : Symbol/str
    success : Integer/str
            The event which is considered to be success. Default: 1.
    failure: Integer/str
            The event which is considered to be failure. Default: 0.
    p : Real Number between 0 and 1
            Represents the probability of getting success.

    Examples
    ========

    >>> from sympy.stats import BernoulliProcess, P, E
    >>> from sympy import Eq, Gt
    >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
    >>> B.state_space
    {0, 1}
    >>> B.p.round(2)
    0.70
    >>> B.success
    1
    >>> B.failure
    0
    >>> X = B[1] + B[2] + B[3]
    >>> P(Eq(X, 0)).round(2)
    0.03
    >>> P(Eq(X, 2)).round(2)
    0.44
    >>> P(Eq(X, 4)).round(2)
    0
    >>> P(Gt(X, 1)).round(2)
    0.78
    >>> P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2)
    0.04
    >>> B.joint_distribution(B[1], B[2])
    JointDistributionHandmade(Lambda((B[1], B[2]), Piecewise((0.7, Eq(B[1], 1)),
    (0.3, Eq(B[1], 0)), (0, True))*Piecewise((0.7, Eq(B[2], 1)), (0.3, Eq(B[2], 0)),
    (0, True))))
    >>> E(2*B[1] + B[2]).round(2)
    2.10
    >>> P(B[1] < 1).round(2)
    0.30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_process
    .. [2] https://mathcs.clarku.edu/~djoyce/ma217/bernoulli.pdf

    """

    index_set = S.Naturals0

    def __new__(cls, sym, p, success=1, failure=0):
        _value_check(p >= 0 and p <= 1, 'Value of p must be between 0 and 1.')
        sym = _symbol_converter(sym)
        p = _sympify(p)
        success = _sym_sympify(success)
        failure = _sym_sympify(failure)
        return Basic.__new__(cls, sym, p, success, failure)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def p(self):
        return self.args[1]

    @property
    def success(self):
        return self.args[2]

    @property
    def failure(self):
        return self.args[3]

    @property
    def state_space(self):
        return _set_converter([self.success, self.failure])

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
            return BernoulliDistribution(self.p)
        return BernoulliDistribution(self.p, self.success, self.failure)

    def simple_rv(self, rv):
        return Bernoulli(rv.name, p=self.p,
                succ=self.success, fail=self.failure)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation.

        Parameters
        ==========

        expr : RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition : Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """

        return _SubstituteRV._expectation(expr, condition, evaluate, **kwargs)

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition : Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition : Relational, Logic
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """

        return _SubstituteRV._probability(condition, given_condition, evaluate, **kwargs)

    def density(self, x):
        return Piecewise((self.p, Eq(x, self.success)),
                         (1 - self.p, Eq(x, self.failure)),
                         (S.Zero, True))

class _SubstituteRV:
    """
    Internal class to handle the queries of expectation and probability
    by substitution.
    """

    @staticmethod
    def _rvindexed_subs(expr, condition=None):
        """
        Substitutes the RandomIndexedSymbol with the RandomSymbol with
        same name, distribution and probability as RandomIndexedSymbol.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        """

        rvs_expr = random_symbols(expr)
        if len(rvs_expr) != 0:
            swapdict_expr = {}
            for rv in rvs_expr:
                if isinstance(rv, RandomIndexedSymbol):
                    newrv = rv.pspace.process.simple_rv(rv) # substitute with equivalent simple rv
                    swapdict_expr[rv] = newrv
            expr = expr.subs(swapdict_expr)
        rvs_cond = random_symbols(condition)
        if len(rvs_cond)!=0:
            swapdict_cond = {}
            for rv in rvs_cond:
                if isinstance(rv, RandomIndexedSymbol):
                    newrv = rv.pspace.process.simple_rv(rv)
                    swapdict_cond[rv] = newrv
            condition = condition.subs(swapdict_cond)
        return expr, condition

    @classmethod
    def _expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing expectation of indexed RV.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """
        new_expr, new_condition = self._rvindexed_subs(expr, condition)

        if not is_random(new_expr):
            return new_expr
        new_pspace = pspace(new_expr)
        if new_condition is not None:
            new_expr = given(new_expr, new_condition)
        if new_expr.is_Add:  # As E is Linear
            return Add(*[new_pspace.compute_expectation(
                        expr=arg, evaluate=evaluate, **kwargs)
                        for arg in new_expr.args])
        return new_pspace.compute_expectation(
                new_expr, evaluate=evaluate, **kwargs)

    @classmethod
    def _probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing probability of indexed RV

        Parameters
        ==========

        condition: Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition: Relational/And
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """
        new_condition, new_givencondition = self._rvindexed_subs(condition, given_condition)

        if isinstance(new_givencondition, RandomSymbol):
            condrv = random_symbols(new_condition)
            if len(condrv) == 1 and condrv[0] == new_givencondition:
                return BernoulliDistribution(self._probability(new_condition), 0, 1)

            if any(dependent(rv, new_givencondition) for rv in condrv):
                return Probability(new_condition, new_givencondition)
            else:
                return self._probability(new_condition)

        if new_givencondition is not None and \
                not isinstance(new_givencondition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_givencondition))
        if new_givencondition == False or new_condition == False:
            return S.Zero
        if new_condition == True:
            return S.One
        if not isinstance(new_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_condition))

        if new_givencondition is not None:  # If there is a condition
        # Recompute on new conditional expr
            return self._probability(given(new_condition, new_givencondition, **kwargs), **kwargs)
        result = pspace(new_condition).probability(new_condition, **kwargs)
        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result

def get_timerv_swaps(expr, condition):
    """
    Finds the appropriate interval for each time stamp in expr by parsing
    the given condition and returns intervals for each timestamp and
    dictionary that maps variable time-stamped Random Indexed Symbol to its
    corresponding Random Indexed variable with fixed time stamp.

    Parameters
    ==========

    expr: SymPy Expression
        Expression containing Random Indexed Symbols with variable time stamps
    condition: Relational/Boolean Expression
        Expression containing time bounds of variable time stamps in expr

    Examples
    ========

    >>> from sympy.stats.stochastic_process_types import get_timerv_swaps, PoissonProcess
    >>> from sympy import symbols, Contains, Interval
    >>> x, t, d = symbols('x t d', positive=True)
    >>> X = PoissonProcess("X", 3)
    >>> get_timerv_swaps(x*X(t), Contains(t, Interval.Lopen(0, 1)))
    ([Interval.Lopen(0, 1)], {X(t): X(1)})
    >>> get_timerv_swaps((X(t)**2 + X(d)**2), Contains(t, Interval.Lopen(0, 1))
    ... & Contains(d, Interval.Ropen(1, 4))) # doctest: +SKIP
    ([Interval.Ropen(1, 4), Interval.Lopen(0, 1)], {X(d): X(3), X(t): X(1)})

    Returns
    =======

    intervals: list
        List of Intervals/FiniteSet on which each time stamp is defined
    rv_swap: dict
        Dictionary mapping variable time Random Indexed Symbol to constant time
        Random Indexed Variable

    """

    if not isinstance(condition, (Relational, Boolean)):
        raise ValueError("%s is not a relational or combination of relationals"
            % (condition))
    expr_syms = list(expr.atoms(RandomIndexedSymbol))
    if isinstance(condition, (And, Or)):
        given_cond_args = condition.args
    else: # single condition
        given_cond_args = (condition, )
    rv_swap = {}
    intervals = []
    for expr_sym in expr_syms:
        for arg in given_cond_args:
            if arg.has(expr_sym.key) and isinstance(expr_sym.key, Symbol):
                intv = _set_converter(arg.args[1])
                diff_key = intv._sup - intv._inf
                if diff_key == oo:
                    raise ValueError("%s should have finite bounds" % str(expr_sym.name))
                elif diff_key == S.Zero: # has singleton set
                    diff_key = intv._sup
                rv_swap[expr_sym] = expr_sym.subs({expr_sym.key: diff_key})
                intervals.append(intv)
    return intervals, rv_swap


class CountingProcess(ContinuousTimeStochasticProcess):
    """
    This class handles the common methods of the Counting Processes
    such as Poisson, Wiener and Gamma Processes
    """
    index_set = _set_converter(Interval(0, oo))

    @property
    def symbol(self):
        return self.args[0]

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Expectation of the given expr

        """
        if condition is not None:
            intervals, rv_swap = get_timerv_swaps(expr, condition)
             # they are independent when they have non-overlapping intervals
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                for intv_comb in itertools.combinations(intervals, 2)):
                if expr.is_Add:
                    return Add.fromiter(self.expectation(arg, condition)
                            for arg in expr.args)
                expr = expr.subs(rv_swap)
            else:
                return Expectation(expr, condition)

        return _SubstituteRV._expectation(expr, evaluate=evaluate, **kwargs)

    def _solve_argwith_tworvs(self, arg):
        if arg.args[0].key >= arg.args[1].key or isinstance(arg, Eq):
            diff_key = abs(arg.args[0].key - arg.args[1].key)
            rv = arg.args[0]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        else:
            diff_key = arg.args[1].key - arg.args[0].key
            rv = arg.args[1]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        return arg

    def _solve_numerical(self, condition, given_condition=None):
        if isinstance(condition, And):
            args_list = list(condition.args)
        else:
            args_list = [condition]
        if given_condition is not None:
            if isinstance(given_condition, And):
                args_list.extend(list(given_condition.args))
            else:
                args_list.extend([given_condition])
        # sort the args based on timestamp to get the independent increments in
        # each segment using all the condition args as well as given_condition args
        args_list = sorted(args_list, key=lambda x: x.args[0].key)
        result = []
        cond_args = list(condition.args) if isinstance(condition, And) else [condition]
        if args_list[0] in cond_args and not (is_random(args_list[0].args[0])
                        and is_random(args_list[0].args[1])):
            result.append(_SubstituteRV._probability(args_list[0]))

        if is_random(args_list[0].args[0]) and is_random(args_list[0].args[1]):
            arg = self._solve_argwith_tworvs(args_list[0])
            result.append(_SubstituteRV._probability(arg))

        for i in range(len(args_list) - 1):
            curr, nex = args_list[i], args_list[i + 1]
            diff_key = nex.args[0].key - curr.args[0].key
            working_set = curr.args[0].pspace.process.state_space
            if curr.args[1] > nex.args[1]: #impossible condition so return 0
                result.append(0)
                break
            if isinstance(curr, Eq):
                working_set = Intersection(working_set, Interval.Lopen(curr.args[1], oo))
            else:
                working_set = Intersection(working_set, curr.as_set())
            if isinstance(nex, Eq):
                working_set = Intersection(working_set, Interval(-oo, nex.args[1]))
            else:
                working_set = Intersection(working_set, nex.as_set())
            if working_set == EmptySet:
                rv = Eq(curr.args[0].pspace.process(diff_key), 0)
                result.append(_SubstituteRV._probability(rv))
            else:
                if working_set.is_finite_set:
                    if isinstance(curr, Eq) and isinstance(nex, Eq):
                        rv = Eq(curr.args[0].pspace.process(diff_key), len(working_set))
                        result.append(_SubstituteRV._probability(rv))
                    elif isinstance(curr, Eq) ^ isinstance(nex, Eq):
                        result.append(Add.fromiter(_SubstituteRV._probability(Eq(
                        curr.args[0].pspace.process(diff_key), x))
                                for x in range(len(working_set))))
                    else:
                        n = len(working_set)
                        result.append(Add.fromiter((n - x)*_SubstituteRV._probability(Eq(
                        curr.args[0].pspace.process(diff_key), x)) for x in range(n)))
                else:
                    result.append(_SubstituteRV._probability(
                    curr.args[0].pspace.process(diff_key) <= working_set._sup - working_set._inf))
        return Mul.fromiter(result)


    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition: Relational
            Condition for which probability has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        given_condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Probability of the condition

        """
        check_numeric = True
        if isinstance(condition, (And, Or)):
            cond_args = condition.args
        else:
            cond_args = (condition, )
        # check that condition args are numeric or not
        if not all(arg.args[0].key.is_number for arg in cond_args):
            check_numeric = False
        if given_condition is not None:
            check_given_numeric = True
            if isinstance(given_condition, (And, Or)):
                given_cond_args = given_condition.args
            else:
                given_cond_args = (given_condition, )
            # check that given condition args are numeric or not
            if given_condition.has(Contains):
                check_given_numeric = False
            # Handle numerical queries
            if check_numeric and check_given_numeric:
                res = []
                if isinstance(condition, Or):
                    res.append(Add.fromiter(self._solve_numerical(arg, given_condition)
                            for arg in condition.args))
                if isinstance(given_condition, Or):
                    res.append(Add.fromiter(self._solve_numerical(condition, arg)
                            for arg in given_condition.args))
                if res:
                    return Add.fromiter(res)
                return self._solve_numerical(condition, given_condition)

            # No numeric queries, go by Contains?... then check that all the
            # given condition are in form of `Contains`
            if not all(arg.has(Contains) for arg in given_cond_args):
                raise ValueError("If given condition is passed with `Contains`, then "
                "please pass the evaluated condition with its corresponding information "
                "in terms of intervals of each time stamp to be passed in given condition.")

            intervals, rv_swap = get_timerv_swaps(condition, given_condition)
            # they are independent when they have non-overlapping intervals
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                for intv_comb in itertools.combinations(intervals, 2)):
                if isinstance(condition, And):
                    return Mul.fromiter(self.probability(arg, given_condition)
                            for arg in condition.args)
                elif isinstance(condition, Or):
                    return Add.fromiter(self.probability(arg, given_condition)
                            for arg in condition.args)
                condition = condition.subs(rv_swap)
            else:
                return Probability(condition, given_condition)
        if check_numeric:
            return self._solve_numerical(condition)
        return _SubstituteRV._probability(condition, evaluate=evaluate, **kwargs)

class PoissonProcess(CountingProcess):
    """
    The Poisson process is a counting process. It is usually used in scenarios
    where we are counting the occurrences of certain events that appear
    to happen at a certain rate, but completely at random.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Rate of the process, ``lambda > 0``

    Examples
    ========

    >>> from sympy.stats import PoissonProcess, P, E
    >>> from sympy import symbols, Eq, Ne, Contains, Interval
    >>> X = PoissonProcess("X", lamda=3)
    >>> X.state_space
    Naturals0
    >>> X.lamda
    3
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 4)
    (9*t1**3/2 + 9*t1**2/2 + 3*t1 + 1)*exp(-3*t1)
    >>> P(Eq(X(t1), 2) | Ne(X(t1), 4), Contains(t1, Interval.Ropen(2, 4)))
    1 - 36*exp(-6)
    >>> P(Eq(X(t1), 2) & Eq(X(t2), 3), Contains(t1, Interval.Lopen(0, 2))
    ... & Contains(t2, Interval.Lopen(2, 4)))
    648*exp(-12)
    >>> E(X(t1))
    3*t1
    >>> E(X(t1)**2 + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    18
    >>> P(X(3) < 1, Eq(X(1), 0))
    exp(-6)
    >>> P(Eq(X(4), 3), Eq(X(2), 3))
    exp(-6)
    >>> P(X(2) <= 3, X(1) > 1)
    5*exp(-3)

    Merging two Poisson Processes

    >>> Y = PoissonProcess("Y", lamda=4)
    >>> Z = X + Y
    >>> Z.lamda
    7

    Splitting a Poisson Process into two independent Poisson Processes

    >>> N, M = Z.split(l1=2, l2=5)
    >>> N.lamda, M.lamda
    (2, 5)

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_0_0_intro.php
    .. [2] https://en.wikipedia.org/wiki/Poisson_point_process

    """

    def __new__(cls, sym, lamda):
        _value_check(lamda > 0, 'lamda should be a positive number.')
        sym = _symbol_converter(sym)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def state_space(self):
        return S.Naturals0

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return PoissonDistribution(self.lamda*key.key)
        return PoissonDistribution(self.lamda*key)

    def density(self, x):
        return (self.lamda*x.key)**x / factorial(x) * exp(-(self.lamda*x.key))

    def simple_rv(self, rv):
        return Poisson(rv.name, lamda=self.lamda*rv.key)

    def __add__(self, other):
        if not isinstance(other, PoissonProcess):
            raise ValueError("Only instances of Poisson Process can be merged")
        return PoissonProcess(Dummy(self.symbol.name + other.symbol.name),
                self.lamda + other.lamda)

    def split(self, l1, l2):
        if _sympify(l1 + l2) != self.lamda:
            raise ValueError("Sum of l1 and l2 should be %s" % str(self.lamda))
        return PoissonProcess(Dummy("l1"), l1), PoissonProcess(Dummy("l2"), l2)

class WienerProcess(CountingProcess):
    """
    The Wiener process is a real valued continuous-time stochastic process.
    In physics it is used to study Brownian motion and it is often also called
    Brownian motion due to its historical connection with physical process of the
    same name originally observed by Scottish botanist Robert Brown.

    Parameters
    ==========

    sym : Symbol/str

    Examples
    ========

    >>> from sympy.stats import WienerProcess, P, E
    >>> from sympy import symbols, Contains, Interval
    >>> X = WienerProcess("X")
    >>> X.state_space
    Reals
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 7).simplify()
    erf(7*sqrt(2)/(2*sqrt(t1)))/2 + 1/2
    >>> P((X(t1) > 2) | (X(t1) < 4), Contains(t1, Interval.Ropen(2, 4))).simplify()
    -erf(1)/2 + erf(2)/2 + 1
    >>> E(X(t1))
    0
    >>> E(X(t1) + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    0

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_4_0_brownian_motion_wiener_process.php
    .. [2] https://en.wikipedia.org/wiki/Wiener_process

    """
    def __new__(cls, sym):
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym)

    @property
    def state_space(self):
        return S.Reals

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return NormalDistribution(0, sqrt(key.key))
        return NormalDistribution(0, sqrt(key))

    def density(self, x):
        return exp(-x**2/(2*x.key)) / (sqrt(2*pi)*sqrt(x.key))

    def simple_rv(self, rv):
        return Normal(rv.name, 0, sqrt(rv.key))


class GammaProcess(CountingProcess):
    r"""
    A Gamma process is a random process with independent gamma distributed
    increments. It is a pure-jump increasing Levy process.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Jump size of the process, ``lamda > 0``
    gamma : Positive number
        Rate of jump arrivals, `\gamma > 0`

    Examples
    ========

    >>> from sympy.stats import GammaProcess, E, P, variance
    >>> from sympy import symbols, Contains, Interval, Not
    >>> t, d, x, l, g = symbols('t d x l g', positive=True)
    >>> X = GammaProcess("X", l, g)
    >>> E(X(t))
    g*t/l
    >>> variance(X(t)).simplify()
    g*t/l**2
    >>> X = GammaProcess('X', 1, 2)
    >>> P(X(t) < 1).simplify()
    lowergamma(2*t, 1)/gamma(2*t)
    >>> P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
    ... Contains(d, Interval.Lopen(7, 8))).simplify()
    -4*exp(-3) + 472*exp(-8)/3 + 1
    >>> E(X(2) + x*E(X(5)))
    10*x + 4

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_process

    """
    def __new__(cls, sym, lamda, gamma):
        _value_check(lamda > 0, 'lamda should be a positive number')
        _value_check(gamma > 0, 'gamma should be a positive number')
        sym = _symbol_converter(sym)
        gamma = _sympify(gamma)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda, gamma)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def gamma(self):
        return self.args[2]

    @property
    def state_space(self):
        return _set_converter(Interval(0, oo))

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return GammaDistribution(self.gamma*key.key, 1/self.lamda)
        return GammaDistribution(self.gamma*key, 1/self.lamda)

    def density(self, x):
        k = self.gamma*x.key
        theta = 1/self.lamda
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    def simple_rv(self, rv):
        return Gamma(rv.name, self.gamma*rv.key, 1/self.lamda)
<|endoftext|>import itertools

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.stats.rv import RandomSymbol, is_random
from sympy.core.sympify import _sympify
from sympy.stats.symbolic_probability import Variance, Covariance, Expectation


class ExpectationMatrix(Expectation, MatrixExpr):
    """
    Expectation of a random matrix expression.

    Examples
    ========

    >>> from sympy.stats import ExpectationMatrix, Normal
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> ExpectationMatrix(X)
    ExpectationMatrix(X)
    >>> ExpectationMatrix(A*X).shape
    (k, 1)

    To expand the expectation in its expression, use ``expand()``:

    >>> ExpectationMatrix(A*X + B*Y).expand()
    A*ExpectationMatrix(X) + B*ExpectationMatrix(Y)
    >>> ExpectationMatrix((X + Y)*(X - Y).T).expand()
    ExpectationMatrix(X*X.T) - ExpectationMatrix(X*Y.T) + ExpectationMatrix(Y*X.T) - ExpectationMatrix(Y*Y.T)

    To evaluate the ``ExpectationMatrix``, use ``doit()``:

    >>> N11, N12 = Normal('N11', 11, 1), Normal('N12', 12, 1)
    >>> N21, N22 = Normal('N21', 21, 1), Normal('N22', 22, 1)
    >>> M11, M12 = Normal('M11', 1, 1), Normal('M12', 2, 1)
    >>> M21, M22 = Normal('M21', 3, 1), Normal('M22', 4, 1)
    >>> x1 = Matrix([[N11, N12], [N21, N22]])
    >>> x2 = Matrix([[M11, M12], [M21, M22]])
    >>> ExpectationMatrix(x1 + x2).doit()
    Matrix([
    [12, 14],
    [24, 26]])

    """
    def __new__(cls, expr, condition=None):
        expr = _sympify(expr)
        if condition is None:
            if not is_random(expr):
                return expr
            obj = Expr.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)

        obj._shape = expr.shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        expr = self.args[0]
        condition = self._condition
        if not is_random(expr):
            return expr

        if isinstance(expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expr.args)

        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expand_expr.args)

        elif isinstance(expr, (Mul, MatMul)):
            rv = []
            nonrv = []
            postnon = []

            for a in expr.args:
                if is_random(a):
                    if rv:
                        rv.extend(postnon)
                    else:
                        nonrv.extend(postnon)
                    postnon = []
                    rv.append(a)
                elif a.is_Matrix:
                    postnon.append(a)
                else:
                    nonrv.append(a)

            # In order to avoid infinite-looping (MatMul may call .doit() again),
            # do not rebuild
            if len(nonrv) == 0:
                return self
            return Mul.fromiter(nonrv)*Expectation(Mul.fromiter(rv),
                    condition=condition)*Mul.fromiter(postnon)

        return self

class VarianceMatrix(Variance, MatrixExpr):
    """
    Variance of a random matrix probability expression. Also known as
    Covariance matrix, auto-covariance matrix, dispersion matrix,
    or variance-covariance matrix.

    Examples
    ========

    >>> from sympy.stats import VarianceMatrix
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> VarianceMatrix(X)
    VarianceMatrix(X)
    >>> VarianceMatrix(X).shape
    (k, k)

    To expand the variance in its expression, use ``expand()``:

    >>> VarianceMatrix(A*X).expand()
    A*VarianceMatrix(X)*A.T
    >>> VarianceMatrix(A*X + B*Y).expand()
    2*A*CrossCovarianceMatrix(X, Y)*B.T + A*VarianceMatrix(X)*A.T + B*VarianceMatrix(Y)*B.T
    """
    def __new__(cls, arg, condition=None):
        arg = _sympify(arg)

        if 1 not in arg.shape:
            raise ShapeError("Expression is not a vector")

        shape = (arg.shape[0], arg.shape[0]) if arg.shape[1] == 1 else (arg.shape[1], arg.shape[1])

        if condition:
            obj = Expr.__new__(cls, arg, condition)
        else:
            obj = Expr.__new__(cls, arg)

        obj._shape = shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        arg = self.args[0]
        condition = self._condition

        if not is_random(arg):
            return ZeroMatrix(*self.shape)

        if isinstance(arg, RandomSymbol):
            return self
        elif isinstance(arg, Add):
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            return variances + covariances
        elif isinstance(arg, (Mul, MatMul)):
            nonrv = []
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            if len(rv) == 0:
                return ZeroMatrix(*self.shape)
            # Avoid possible infinite loops with MatMul:
            if len(nonrv) == 0:
                return self
            # Variance of many multiple matrix products is not implemented:
            if len(rv) > 1:
                return self
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv),
                            condition)*(Mul.fromiter(nonrv)).transpose()

        # this expression contains a RandomSymbol somehow:
        return self

class CrossCovarianceMatrix(Covariance, MatrixExpr):
    """
    Covariance of a random matrix probability expression.

    Examples
    ========

    >>> from sympy.stats import CrossCovarianceMatrix
    >>> from sympy.stats.rv import RandomMatrixSymbol
    >>> from sympy import symbols, MatrixSymbol
    >>> k = symbols("k")
    >>> A, B = MatrixSymbol("A", k, k), MatrixSymbol("B", k, k)
    >>> C, D = MatrixSymbol("C", k, k), MatrixSymbol("D", k, k)
    >>> X, Y = RandomMatrixSymbol("X", k, 1), RandomMatrixSymbol("Y", k, 1)
    >>> Z, W = RandomMatrixSymbol("Z", k, 1), RandomMatrixSymbol("W", k, 1)
    >>> CrossCovarianceMatrix(X, Y)
    CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(X, Y).shape
    (k, k)

    To expand the covariance in its expression, use ``expand()``:

    >>> CrossCovarianceMatrix(X + Y, Z).expand()
    CrossCovarianceMatrix(X, Z) + CrossCovarianceMatrix(Y, Z)
    >>> CrossCovarianceMatrix(A*X, Y).expand()
    A*CrossCovarianceMatrix(X, Y)
    >>> CrossCovarianceMatrix(A*X, B.T*Y).expand()
    A*CrossCovarianceMatrix(X, Y)*B
    >>> CrossCovarianceMatrix(A*X + B*Y, C.T*Z + D.T*W).expand()
    A*CrossCovarianceMatrix(X, W)*D + A*CrossCovarianceMatrix(X, Z)*C + B*CrossCovarianceMatrix(Y, W)*D + B*CrossCovarianceMatrix(Y, Z)*C

    """
    def __new__(cls, arg1, arg2, condition=None):
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)

        if (1 not in arg1.shape) or (1 not in arg2.shape) or (arg1.shape[1] != arg2.shape[1]):
            raise ShapeError("Expression is not a vector")

        shape = (arg1.shape[0], arg2.shape[0]) if arg1.shape[1] == 1 and arg2.shape[1] == 1 \
                    else (1, 1)

        if condition:
            obj = Expr.__new__(cls, arg1, arg2, condition)
        else:
            obj = Expr.__new__(cls, arg1, arg2)

        obj._shape = shape
        obj._condition = condition
        return obj

    @property
    def shape(self):
        return self._shape

    def expand(self, **hints):
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition

        if arg1 == arg2:
            return VarianceMatrix(arg1, condition).expand()

        if not is_random(arg1) or not is_random(arg2):
            return ZeroMatrix(*self.shape)

        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return CrossCovarianceMatrix(arg1, arg2, condition)

        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())

        addends = [a*CrossCovarianceMatrix(r1, r2, condition=condition)*b.transpose()
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    def _expand_single_argument(cls, expr):
        # return (coefficient, random_symbol) pairs:
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        elif isinstance(expr, Add):
            outval = []
            for a in expr.args:
                if isinstance(a, (Mul, MatMul)):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                elif is_random(a):
                    outval.append((S.One, a))

            return outval
        elif isinstance(expr, (Mul, MatMul)):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        for a in m.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))
<|endoftext|>"""Tools for arithmetic error propagation."""

from itertools import repeat, combinations

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.simplify.simplify import simplify
from sympy.stats.symbolic_probability import RandomSymbol, Variance, Covariance
from sympy.stats.rv import is_random

_arg0_or_var = lambda var: var.args[0] if len(var.args) > 0 else var


def variance_prop(expr, consts=(), include_covar=False):
    r"""Symbolically propagates variance (`\sigma^2`) for expressions.
    This is computed as as seen in [1]_.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to compute the variance for.
    consts : sequence of Symbols, optional
        Represents symbols that are known constants in the expr,
        and thus have zero variance. All symbols not in consts are
        assumed to be variant.
    include_covar : bool, optional
        Flag for whether or not to include covariances, default=False.

    Returns
    =======

    var_expr : Expr
        An expression for the total variance of the expr.
        The variance for the original symbols (e.g. x) are represented
        via instance of the Variance symbol (e.g. Variance(x)).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.stats.error_prop import variance_prop
    >>> x, y = symbols('x y')

    >>> variance_prop(x + y)
    Variance(x) + Variance(y)

    >>> variance_prop(x * y)
    x**2*Variance(y) + y**2*Variance(x)

    >>> variance_prop(exp(2*x))
    4*exp(4*x)*Variance(x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    """
    args = expr.args
    if len(args) == 0:
        if expr in consts:
            return S.Zero
        elif is_random(expr):
            return Variance(expr).doit()
        elif isinstance(expr, Symbol):
            return Variance(RandomSymbol(expr)).doit()
        else:
            return S.Zero
    nargs = len(args)
    var_args = list(map(variance_prop, args, repeat(consts, nargs),
                        repeat(include_covar, nargs)))
    if isinstance(expr, Add):
        var_expr = Add(*var_args)
        if include_covar:
            terms = [2 * Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand() \
                     for x, y in combinations(var_args, 2)]
            var_expr += Add(*terms)
    elif isinstance(expr, Mul):
        terms = [v/a**2 for a, v in zip(args, var_args)]
        var_expr = simplify(expr**2 * Add(*terms))
        if include_covar:
            terms = [2*Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand()/(a*b) \
                     for (a, b), (x, y) in zip(combinations(args, 2),
                                               combinations(var_args, 2))]
            var_expr += Add(*terms)
    elif isinstance(expr, Pow):
        b = args[1]
        v = var_args[0] * (expr * b / args[0])**2
        var_expr = simplify(v)
    elif isinstance(expr, exp):
        var_expr = simplify(var_args[0] * expr**2)
    else:
        # unknown how to proceed, return variance of whole expr.
        var_expr = Variance(expr)
    return var_expr
<|endoftext|>import itertools
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Not
from sympy.core.parameters import global_parameters
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.stats import variance, covariance
from sympy.stats.rv import (RandomSymbol, pspace, dependent,
                            given, sampling_E, RandomIndexedSymbol, is_random,
                            PSpace, sampling_P, random_symbols)

__all__ = ['Probability', 'Expectation', 'Variance', 'Covariance']


@is_random.register(Expr)
def _(x):
    atoms = x.free_symbols
    if len(atoms) == 1 and next(iter(atoms)) == x:
        return False
    return any(is_random(i) for i in atoms)

@is_random.register(RandomSymbol)  # type: ignore
def _(x):
    return True


class Probability(Expr):
    """
    Symbolic expression for the probability.

    Examples
    ========

    >>> from sympy.stats import Probability, Normal
    >>> from sympy import Integral
    >>> X = Normal("X", 0, 1)
    >>> prob = Probability(X > 1)
    >>> prob
    Probability(X > 1)

    Integral representation:

    >>> prob.rewrite(Integral)
    Integral(sqrt(2)*exp(-_z**2/2)/(2*sqrt(pi)), (_z, 1, oo))

    Evaluation of the integral:

    >>> prob.evaluate_integral()
    sqrt(2)*(-sqrt(2)*sqrt(pi)*erf(sqrt(2)/2) + sqrt(2)*sqrt(pi))/(4*sqrt(pi))
    """

    is_commutative = True

    def __new__(cls, prob, condition=None, **kwargs):
        prob = _sympify(prob)
        if condition is None:
            obj = Expr.__new__(cls, prob)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, prob, condition)
        obj._condition = condition
        return obj

    def doit(self, **hints):
        condition = self.args[0]
        given_condition = self._condition
        numsamples = hints.get('numsamples', False)
        evaluate = hints.get('evaluate', True)

        if isinstance(condition, Not):
            return S.One - self.func(condition.args[0], given_condition,
                                    evaluate=evaluate).doit(**hints)

        if condition.has(RandomIndexedSymbol):
            return pspace(condition).probability(condition, given_condition,
                                             evaluate=evaluate)

        if isinstance(given_condition, RandomSymbol):
            condrv = random_symbols(condition)
            if len(condrv) == 1 and condrv[0] == given_condition:
                from sympy.stats.frv_types import BernoulliDistribution
                return BernoulliDistribution(self.func(condition).doit(**hints), 0, 1)
            if any(dependent(rv, given_condition) for rv in condrv):
                return Probability(condition, given_condition)
            else:
                return Probability(condition).doit()

        if given_condition is not None and \
                not isinstance(given_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (given_condition))

        if given_condition == False or condition is S.false:
            return S.Zero
        if not isinstance(condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (condition))
        if condition is S.true:
            return S.One

        if numsamples:
            return sampling_P(condition, given_condition, numsamples=numsamples)
        if given_condition is not None:  # If there is a condition
            # Recompute on new conditional expr
            return Probability(given(condition, given_condition)).doit()

        # Otherwise pass work off to the ProbabilitySpace
        if pspace(condition) == PSpace():
            return Probability(condition, given_condition)

        result = pspace(condition).probability(condition)
        if hasattr(result, 'doit') and evaluate:
            return result.doit()
        else:
            return result

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return self.func(arg, condition=condition).doit(evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Expectation(Expr):
    """
    Symbolic expression for the expectation.

    Examples
    ========

    >>> from sympy.stats import Expectation, Normal, Probability, Poisson
    >>> from sympy import symbols, Integral, Sum
    >>> mu = symbols("mu")
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Expectation(X)
    Expectation(X)
    >>> Expectation(X).evaluate_integral().simplify()
    mu

    To get the integral expression of the expectation:

    >>> Expectation(X).rewrite(Integral)
    Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    The same integral expression, in more abstract terms:

    >>> Expectation(X).rewrite(Probability)
    Integral(x*Probability(Eq(X, x)), (x, -oo, oo))

    To get the Summation expression of the expectation for discrete random variables:

    >>> lamda = symbols('lamda', positive=True)
    >>> Z = Poisson('Z', lamda)
    >>> Expectation(Z).rewrite(Sum)
    Sum(Z*lamda**Z*exp(-lamda)/factorial(Z), (Z, 0, oo))

    This class is aware of some properties of the expectation:

    >>> from sympy.abc import a
    >>> Expectation(a*X)
    Expectation(a*X)
    >>> Y = Normal("Y", 1, 2)
    >>> Expectation(X + Y)
    Expectation(X + Y)

    To expand the ``Expectation`` into its expression, use ``expand()``:

    >>> Expectation(X + Y).expand()
    Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y).expand()
    a*Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y)
    Expectation(a*X + Y)
    >>> Expectation((X + Y)*(X - Y)).expand()
    Expectation(X**2) - Expectation(Y**2)

    To evaluate the ``Expectation``, use ``doit()``:

    >>> Expectation(X + Y).doit()
    mu + 1
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit()
    3*mu + 1

    To prevent evaluating nested ``Expectation``, use ``doit(deep=False)``

    >>> Expectation(X + Expectation(Y)).doit(deep=False)
    mu + Expectation(Expectation(Y))
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False)
    mu + Expectation(Expectation(Expectation(2*X) + Y))

    """

    def __new__(cls, expr, condition=None, **kwargs):
        expr = _sympify(expr)
        if expr.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import ExpectationMatrix
            return ExpectationMatrix(expr, condition)
        if condition is None:
            if not is_random(expr):
                return expr
            obj = Expr.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return(self.args[0].is_commutative)

    def expand(self, **hints):
        expr = self.args[0]
        condition = self._condition

        if not is_random(expr):
            return expr

        if isinstance(expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expr.args)

        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expand_expr.args)

        elif isinstance(expr, Mul):
            rv = []
            nonrv = []
            for a in expr.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            return Mul.fromiter(nonrv)*Expectation(Mul.fromiter(rv), condition=condition)

        return self

    def doit(self, **hints):
        deep = hints.get('deep', True)
        condition = self._condition
        expr = self.args[0]
        numsamples = hints.get('numsamples', False)
        evaluate = hints.get('evaluate', True)

        if deep:
            expr = expr.doit(**hints)

        if not is_random(expr) or isinstance(expr, Expectation):  # expr isn't random?
            return expr
        if numsamples:  # Computing by monte carlo sampling?
            evalf = hints.get('evalf', True)
            return sampling_E(expr, condition, numsamples=numsamples, evalf=evalf)

        if expr.has(RandomIndexedSymbol):
            return pspace(expr).compute_expectation(expr, condition)

        # Create new expr and recompute E
        if condition is not None:  # If there is a condition
            return self.func(given(expr, condition)).doit(**hints)

        # A few known statements for efficiency

        if expr.is_Add:  # We know that E is Linear
            return Add(*[self.func(arg, condition).doit(**hints)
                    if not isinstance(arg, Expectation) else self.func(arg, condition)
                         for arg in expr.args])
        if expr.is_Mul:
            if expr.atoms(Expectation):
                return expr

        if pspace(expr) == PSpace():
            return self.func(expr)
        # Otherwise case is simple, pass work off to the ProbabilitySpace
        result = pspace(expr).compute_expectation(expr, evaluate=evaluate)
        if hasattr(result, 'doit') and evaluate:
            return result.doit(**hints)
        else:
            return result


    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        rvs = arg.atoms(RandomSymbol)
        if len(rvs) > 1:
            raise NotImplementedError()
        if len(rvs) == 0:
            return arg

        rv = rvs.pop()
        if rv.pspace is None:
            raise ValueError("Probability space not known")

        symbol = rv.symbol
        if symbol.name[0].isupper():
            symbol = Symbol(symbol.name.lower())
        else :
            symbol = Symbol(symbol.name + "_1")

        if rv.pspace.is_Continuous:
            return Integral(arg.replace(rv, symbol)*Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.domain.set.sup))
        else:
            if rv.pspace.is_Finite:
                raise NotImplementedError
            else:
                return Sum(arg.replace(rv, symbol)*Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.set.sup))

    def _eval_rewrite_as_Integral(self, arg, condition=None, evaluate=False, **kwargs):
        return self.func(arg, condition=condition).doit(deep=False, evaluate=evaluate)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral # For discrete this will be Sum

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()

    evaluate_sum = evaluate_integral

class Variance(Expr):
    """
    Symbolic expression for the variance.

    Examples
    ========

    >>> from sympy import symbols, Integral
    >>> from sympy.stats import Normal, Expectation, Variance, Probability
    >>> mu = symbols("mu", positive=True)
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Variance(X)
    Variance(X)
    >>> Variance(X).evaluate_integral()
    sigma**2

    Integral representation of the underlying calculations:

    >>> Variance(X).rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**2*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    Integral representation, without expanding the PDF:

    >>> Variance(X).rewrite(Probability)
    -Integral(x*Probability(Eq(X, x)), (x, -oo, oo))**2 + Integral(x**2*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the variance in terms of the expectation

    >>> Variance(X).rewrite(Expectation)
    -Expectation(X)**2 + Expectation(X**2)

    Some transformations based on the properties of the variance may happen:

    >>> from sympy.abc import a
    >>> Y = Normal("Y", 0, 1)
    >>> Variance(a*X)
    Variance(a*X)

    To expand the variance in its expression, use ``expand()``:

    >>> Variance(a*X).expand()
    a**2*Variance(X)
    >>> Variance(X + Y)
    Variance(X + Y)
    >>> Variance(X + Y).expand()
    2*Covariance(X, Y) + Variance(X) + Variance(Y)

    """
    def __new__(cls, arg, condition=None, **kwargs):
        arg = _sympify(arg)

        if arg.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import VarianceMatrix
            return VarianceMatrix(arg, condition)
        if condition is None:
            obj = Expr.__new__(cls, arg)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def expand(self, **hints):
        arg = self.args[0]
        condition = self._condition

        if not is_random(arg):
            return S.Zero

        if isinstance(arg, RandomSymbol):
            return self
        elif isinstance(arg, Add):
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            return variances + covariances
        elif isinstance(arg, Mul):
            nonrv = []
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a**2)
            if len(rv) == 0:
                return S.Zero
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv), condition)

        # this expression contains a RandomSymbol somehow:
        return self

    def _eval_rewrite_as_Expectation(self, arg, condition=None, **kwargs):
            e1 = Expectation(arg**2, condition)
            e2 = Expectation(arg, condition)**2
            return e1 - e2

    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return variance(self.args[0], self._condition, evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Covariance(Expr):
    """
    Symbolic expression for the covariance.

    Examples
    ========

    >>> from sympy.stats import Covariance
    >>> from sympy.stats import Normal
    >>> X = Normal("X", 3, 2)
    >>> Y = Normal("Y", 0, 1)
    >>> Z = Normal("Z", 0, 1)
    >>> W = Normal("W", 0, 1)
    >>> cexpr = Covariance(X, Y)
    >>> cexpr
    Covariance(X, Y)

    Evaluate the covariance, `X` and `Y` are independent,
    therefore zero is the result:

    >>> cexpr.evaluate_integral()
    0

    Rewrite the covariance expression in terms of expectations:

    >>> from sympy.stats import Expectation
    >>> cexpr.rewrite(Expectation)
    Expectation(X*Y) - Expectation(X)*Expectation(Y)

    In order to expand the argument, use ``expand()``:

    >>> from sympy.abc import a, b, c, d
    >>> Covariance(a*X + b*Y, c*Z + d*W)
    Covariance(a*X + b*Y, c*Z + d*W)
    >>> Covariance(a*X + b*Y, c*Z + d*W).expand()
    a*c*Covariance(X, Z) + a*d*Covariance(W, X) + b*c*Covariance(Y, Z) + b*d*Covariance(W, Y)

    This class is aware of some properties of the covariance:

    >>> Covariance(X, X).expand()
    Variance(X)
    >>> Covariance(a*X, b*Y).expand()
    a*b*Covariance(X, Y)
    """

    def __new__(cls, arg1, arg2, condition=None, **kwargs):
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)

        if arg1.is_Matrix or arg2.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import CrossCovarianceMatrix
            return CrossCovarianceMatrix(arg1, arg2, condition)

        if kwargs.pop('evaluate', global_parameters.evaluate):
            arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        if condition is None:
            obj = Expr.__new__(cls, arg1, arg2)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg1, arg2, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def expand(self, **hints):
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition

        if arg1 == arg2:
            return Variance(arg1, condition).expand()

        if not is_random(arg1):
            return S.Zero
        if not is_random(arg2):
            return S.Zero

        arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return Covariance(arg1, arg2, condition)

        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())

        addends = [a*b*Covariance(*sorted([r1, r2], key=default_sort_key), condition=condition)
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    def _expand_single_argument(cls, expr):
        # return (coefficient, random_symbol) pairs:
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        elif isinstance(expr, Add):
            outval = []
            for a in expr.args:
                if isinstance(a, Mul):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                elif is_random(a):
                    outval.append((S.One, a))

            return outval
        elif isinstance(expr, Mul):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        for a in m.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))

    def _eval_rewrite_as_Expectation(self, arg1, arg2, condition=None, **kwargs):
        e1 = Expectation(arg1*arg2, condition)
        e2 = Expectation(arg1, condition)*Expectation(arg2, condition)
        return e1 - e2

    def _eval_rewrite_as_Probability(self, arg1, arg2, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, arg1, arg2, condition=None, **kwargs):
        return covariance(self.args[0], self.args[1], self._condition, evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Moment(Expr):
    """
    Symbolic class for Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, Moment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> M = Moment(X, 3, 1)

    To evaluate the result of Moment use `doit`:

    >>> M.doit()
    mu**3 - 3*mu**2 + 3*mu*sigma**2 + 3*mu - 3*sigma**2 - 1

    Rewrite the Moment expression in terms of Expectation:

    >>> M.rewrite(Expectation)
    Expectation((X - 1)**3)

    Rewrite the Moment expression in terms of Probability:

    >>> M.rewrite(Probability)
    Integral((x - 1)**3*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the Moment expression in terms of Integral:

    >>> M.rewrite(Integral)
    Integral(sqrt(2)*(X - 1)**3*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    def __new__(cls, X, n, c=0, condition=None, **kwargs):
        X = _sympify(X)
        n = _sympify(n)
        c = _sympify(c)
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, c, condition)
        else:
            return super().__new__(cls, X, n, c)

    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, c=0, condition=None, **kwargs):
        return Expectation((X - c)**n, condition)

    def _eval_rewrite_as_Probability(self, X, n, c=0, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, c=0, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Integral)


class CentralMoment(Expr):
    """
    Symbolic class Central Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, CentralMoment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> CM = CentralMoment(X, 4)

    To evaluate the result of CentralMoment use `doit`:

    >>> CM.doit().simplify()
    3*sigma**4

    Rewrite the CentralMoment expression in terms of Expectation:

    >>> CM.rewrite(Expectation)
    Expectation((-Expectation(X) + X)**4)

    Rewrite the CentralMoment expression in terms of Probability:

    >>> CM.rewrite(Probability)
    Integral((x - Integral(x*Probability(True), (x, -oo, oo)))**4*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the CentralMoment expression in terms of Integral:

    >>> CM.rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**4*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    def __new__(cls, X, n, condition=None, **kwargs):
        X = _sympify(X)
        n = _sympify(n)
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, condition)
        else:
            return super().__new__(cls, X, n)

    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, condition=None, **kwargs):
        mu = Expectation(X, condition, **kwargs)
        return Moment(X, n, mu, condition, **kwargs).rewrite(Expectation)

    def _eval_rewrite_as_Probability(self, X, n, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Integral)
<|endoftext|>from sympy.sets import FiniteSet
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import FallingFactorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.integrals.integrals import Integral
from sympy.solvers.solveset import solveset
from .rv import (probability, expectation, density, where, given, pspace, cdf, PSpace,
                 characteristic_function, sample, sample_iter, random_symbols, independent, dependent,
                 sampling_density, moment_generating_function, quantile, is_random,
                 sample_stochastic_process)


__all__ = ['P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf',
        'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std',
        'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'median',
        'independent', 'random_symbols', 'correlation', 'factorial_moment',
        'moment', 'cmoment', 'sampling_density', 'moment_generating_function',
        'smoment', 'quantile', 'sample_stochastic_process']



def moment(X, n, c=0, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth moment of a random expression about c.

    .. math::
        moment(X, c, n) = E((X-c)^{n})

    Default value of c is 0.

    Examples
    ========

    >>> from sympy.stats import Die, moment, E
    >>> X = Die('X', 6)
    >>> moment(X, 1, 6)
    -5/2
    >>> moment(X, 2)
    91/6
    >>> moment(X, 1) == E(X)
    True
    """
    from sympy.stats.symbolic_probability import Moment
    if evaluate:
        return Moment(X, n, c, condition).doit()
    return Moment(X, n, c, condition).rewrite(Integral)


def variance(X, condition=None, **kwargs):
    """
    Variance of a random expression.

    .. math::
        variance(X) = E((X-E(X))^{2})

    Examples
    ========

    >>> from sympy.stats import Die, Bernoulli, variance
    >>> from sympy import simplify, Symbol

    >>> X = Die('X', 6)
    >>> p = Symbol('p')
    >>> B = Bernoulli('B', p, 1, 0)

    >>> variance(2*X)
    35/3

    >>> simplify(variance(B))
    p*(1 - p)
    """
    if is_random(X) and pspace(X) == PSpace():
        from sympy.stats.symbolic_probability import Variance
        return Variance(X, condition)

    return cmoment(X, 2, condition, **kwargs)


def standard_deviation(X, condition=None, **kwargs):
    r"""
    Standard Deviation of a random expression

    .. math::
        std(X) = \sqrt(E((X-E(X))^{2}))

    Examples
    ========

    >>> from sympy.stats import Bernoulli, std
    >>> from sympy import Symbol, simplify

    >>> p = Symbol('p')
    >>> B = Bernoulli('B', p, 1, 0)

    >>> simplify(std(B))
    sqrt(p*(1 - p))
    """
    return sqrt(variance(X, condition, **kwargs))
std = standard_deviation

def entropy(expr, condition=None, **kwargs):
    """
    Calculates entropy of a probability distribution.

    Parameters
    ==========

    expression : the random expression whose entropy is to be calculated
    condition : optional, to specify conditions on random expression
    b: base of the logarithm, optional
       By default, it is taken as Euler's number

    Returns
    =======

    result : Entropy of the expression, a constant

    Examples
    ========

    >>> from sympy.stats import Normal, Die, entropy
    >>> X = Normal('X', 0, 1)
    >>> entropy(X)
    log(2)/2 + 1/2 + log(pi)/2

    >>> D = Die('D', 4)
    >>> entropy(D)
    log(4)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
    .. [2] https://www.crmarsh.com/static/pdf/Charles_Marsh_Continuous_Entropy.pdf
    .. [3] https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf
    """
    pdf = density(expr, condition, **kwargs)
    base = kwargs.get('b', exp(1))
    if isinstance(pdf, dict):
            return sum(-prob*log(prob, base) for prob in pdf.values())
    return expectation(-log(pdf(expr), base))

def covariance(X, Y, condition=None, **kwargs):
    """
    Covariance of two random expressions.

    Explanation
    ===========

    The expectation that the two variables will rise and fall together

    .. math::
        covariance(X,Y) = E((X-E(X)) (Y-E(Y)))

    Examples
    ========

    >>> from sympy.stats import Exponential, covariance
    >>> from sympy import Symbol

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> X = Exponential('X', rate)
    >>> Y = Exponential('Y', rate)

    >>> covariance(X, X)
    lambda**(-2)
    >>> covariance(X, Y)
    0
    >>> covariance(X, Y + rate*X)
    1/lambda
    """
    if (is_random(X) and pspace(X) == PSpace()) or (is_random(Y) and pspace(Y) == PSpace()):
        from sympy.stats.symbolic_probability import Covariance
        return Covariance(X, Y, condition)

    return expectation(
        (X - expectation(X, condition, **kwargs)) *
        (Y - expectation(Y, condition, **kwargs)),
        condition, **kwargs)


def correlation(X, Y, condition=None, **kwargs):
    r"""
    Correlation of two random expressions, also known as correlation
    coefficient or Pearson's correlation.

    Explanation
    ===========

    The normalized expectation that the two variables will rise
    and fall together

    .. math::
        correlation(X,Y) = E((X-E(X))(Y-E(Y)) / (\sigma_x  \sigma_y))

    Examples
    ========

    >>> from sympy.stats import Exponential, correlation
    >>> from sympy import Symbol

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> X = Exponential('X', rate)
    >>> Y = Exponential('Y', rate)

    >>> correlation(X, X)
    1
    >>> correlation(X, Y)
    0
    >>> correlation(X, Y + rate*X)
    1/sqrt(1 + lambda**(-2))
    """
    return covariance(X, Y, condition, **kwargs)/(std(X, condition, **kwargs)
     * std(Y, condition, **kwargs))


def cmoment(X, n, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth central moment of a random expression about its mean.

    .. math::
        cmoment(X, n) = E((X - E(X))^{n})

    Examples
    ========

    >>> from sympy.stats import Die, cmoment, variance
    >>> X = Die('X', 6)
    >>> cmoment(X, 3)
    0
    >>> cmoment(X, 2)
    35/12
    >>> cmoment(X, 2) == variance(X)
    True
    """
    from sympy.stats.symbolic_probability import CentralMoment
    if evaluate:
        return CentralMoment(X, n, condition).doit()
    return CentralMoment(X, n, condition).rewrite(Integral)


def smoment(X, n, condition=None, **kwargs):
    r"""
    Return the nth Standardized moment of a random expression.

    .. math::
        smoment(X, n) = E(((X - \mu)/\sigma_X)^{n})

    Examples
    ========

    >>> from sympy.stats import skewness, Exponential, smoment
    >>> from sympy import Symbol
    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> smoment(Y, 4)
    9
    >>> smoment(Y, 4) == smoment(3*Y, 4)
    True
    >>> smoment(Y, 3) == skewness(Y)
    True
    """
    sigma = std(X, condition, **kwargs)
    return (1/sigma)**n*cmoment(X, n, condition, **kwargs)

def skewness(X, condition=None, **kwargs):
    r"""
    Measure of the asymmetry of the probability distribution.

    Explanation
    ===========

    Positive skew indicates that most of the values lie to the right of
    the mean.

    .. math::
        skewness(X) = E(((X - E(X))/\sigma_X)^{3})

    Parameters
    ==========

    condition : Expr containing RandomSymbols
            A conditional expression. skewness(X, X>0) is skewness of X given X > 0

    Examples
    ========

    >>> from sympy.stats import skewness, Exponential, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> skewness(X)
    0
    >>> skewness(X, X > 0) # find skewness given X > 0
    (-sqrt(2)/sqrt(pi) + 4*sqrt(2)/pi**(3/2))/(1 - 2/pi)**(3/2)

    >>> rate = Symbol('lambda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> skewness(Y)
    2
    """
    return smoment(X, 3, condition=condition, **kwargs)

def kurtosis(X, condition=None, **kwargs):
    r"""
    Characterizes the tails/outliers of a probability distribution.

    Explanation
    ===========

    Kurtosis of any univariate normal distribution is 3. Kurtosis less than
    3 means that the distribution produces fewer and less extreme outliers
    than the normal distribution.

    .. math::
        kurtosis(X) = E(((X - E(X))/\sigma_X)^{4})

    Parameters
    ==========

    condition : Expr containing RandomSymbols
            A conditional expression. kurtosis(X, X>0) is kurtosis of X given X > 0

    Examples
    ========

    >>> from sympy.stats import kurtosis, Exponential, Normal
    >>> from sympy import Symbol
    >>> X = Normal('X', 0, 1)
    >>> kurtosis(X)
    3
    >>> kurtosis(X, X > 0) # find kurtosis given X > 0
    (-4/pi - 12/pi**2 + 3)/(1 - 2/pi)**2

    >>> rate = Symbol('lamda', positive=True, real=True)
    >>> Y = Exponential('Y', rate)
    >>> kurtosis(Y)
    9

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kurtosis
    .. [2] https://mathworld.wolfram.com/Kurtosis.html
    """
    return smoment(X, 4, condition=condition, **kwargs)


def factorial_moment(X, n, condition=None, **kwargs):
    """
    The factorial moment is a mathematical quantity defined as the expectation
    or average of the falling factorial of a random variable.

    .. math::
        factorial-moment(X, n) = E(X(X - 1)(X - 2)...(X - n + 1))

    Parameters
    ==========

    n: A natural number, n-th factorial moment.

    condition : Expr containing RandomSymbols
            A conditional expression.

    Examples
    ========

    >>> from sympy.stats import factorial_moment, Poisson, Binomial
    >>> from sympy import Symbol, S
    >>> lamda = Symbol('lamda')
    >>> X = Poisson('X', lamda)
    >>> factorial_moment(X, 2)
    lamda**2
    >>> Y = Binomial('Y', 2, S.Half)
    >>> factorial_moment(Y, 2)
    1/2
    >>> factorial_moment(Y, 2, Y > 1) # find factorial moment for Y > 1
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Factorial_moment
    .. [2] https://mathworld.wolfram.com/FactorialMoment.html
    """
    return expectation(FallingFactorial(X, n), condition=condition, **kwargs)

def median(X, evaluate=True, **kwargs):
    r"""
    Calculates the median of the probability distribution.

    Explanation
    ===========

    Mathematically, median of Probability distribution is defined as all those
    values of `m` for which the following condition is satisfied

    .. math::
        P(X\leq m) \geq  \frac{1}{2} \text{ and} \text{ } P(X\geq m)\geq \frac{1}{2}

    Parameters
    ==========

    X: The random expression whose median is to be calculated.

    Returns
    =======

    The FiniteSet or an Interval which contains the median of the
    random expression.

    Examples
    ========

    >>> from sympy.stats import Normal, Die, median
    >>> N = Normal('N', 3, 1)
    >>> median(N)
    {3}
    >>> D = Die('D')
    >>> median(D)
    {3, 4}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Median#Probability_distributions

    """
    if not is_random(X):
        return X

    from sympy.stats.crv import ContinuousPSpace
    from sympy.stats.drv import DiscretePSpace
    from sympy.stats.frv import FinitePSpace

    if isinstance(pspace(X), FinitePSpace):
        cdf = pspace(X).compute_cdf(X)
        result = []
        for key, value in cdf.items():
            if value>= Rational(1, 2) and (1 - value) + \
            pspace(X).probability(Eq(X, key)) >= Rational(1, 2):
                result.append(key)
        return FiniteSet(*result)
    if isinstance(pspace(X), (ContinuousPSpace, DiscretePSpace)):
        cdf = pspace(X).compute_cdf(X)
        x = Dummy('x')
        result = solveset(piecewise_fold(cdf(x) - Rational(1, 2)), x, pspace(X).set)
        return result
    raise NotImplementedError("The median of %s is not implemented."%str(pspace(X)))


def coskewness(X, Y, Z, condition=None, **kwargs):
    r"""
    Calculates the co-skewness of three random variables.

    Explanation
    ===========

    Mathematically Coskewness is defined as

    .. math::
        coskewness(X,Y,Z)=\frac{E[(X-E[X]) * (Y-E[Y]) * (Z-E[Z])]} {\sigma_{X}\sigma_{Y}\sigma_{Z}}

    Parameters
    ==========

    X : RandomSymbol
            Random Variable used to calculate coskewness
    Y : RandomSymbol
            Random Variable used to calculate coskewness
    Z : RandomSymbol
            Random Variable used to calculate coskewness
    condition : Expr containing RandomSymbols
            A conditional expression

    Examples
    ========

    >>> from sympy.stats import coskewness, Exponential, skewness
    >>> from sympy import symbols
    >>> p = symbols('p', positive=True)
    >>> X = Exponential('X', p)
    >>> Y = Exponential('Y', 2*p)
    >>> coskewness(X, Y, Y)
    0
    >>> coskewness(X, Y + X, Y + 2*X)
    16*sqrt(85)/85
    >>> coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3)
    9*sqrt(170)/85
    >>> coskewness(Y, Y, Y) == skewness(Y)
    True
    >>> coskewness(X, Y + p*X, Y + 2*p*X)
    4/(sqrt(1 + 1/(4*p**2))*sqrt(4 + 1/(4*p**2)))

    Returns
    =======

    coskewness : The coskewness of the three random variables

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coskewness

    """
    num = expectation((X - expectation(X, condition, **kwargs)) \
         * (Y - expectation(Y, condition, **kwargs)) \
         * (Z - expectation(Z, condition, **kwargs)), condition, **kwargs)
    den = std(X, condition, **kwargs) * std(Y, condition, **kwargs) \
         * std(Z, condition, **kwargs)
    return num/den


P = probability
E = expectation
H = entropy
<|endoftext|>"""
Continuous Random Variables Module

See Also
========
sympy.stats.crv_types
sympy.stats.rv
sympy.stats.frv
"""


from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import (I, nan, oo)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Union)
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import (RandomDomain, SingleDomain, ConditionalDomain, is_random,
        ProductDomain, PSpace, SinglePSpace, random_symbols, NamedArgsMixin, Distribution)


class ContinuousDomain(RandomDomain):
    """
    A domain with continuous support

    Represented using symbols and Intervals.
    """
    is_Continuous = True

    def as_boolean(self):
        raise NotImplementedError("Not Implemented for generic Domains")


class SingleContinuousDomain(ContinuousDomain, SingleDomain):
    """
    A univariate domain with continuous support

    Represented using a single symbol and interval.
    """
    def compute_expectation(self, expr, variables=None, **kwargs):
        if variables is None:
            variables = self.symbols
        if not variables:
            return expr
        if frozenset(variables) != frozenset(self.symbols):
            raise ValueError("Values should be equal")
        # assumes only intervals
        return Integral(expr, (self.symbol, self.set), **kwargs)

    def as_boolean(self):
        return self.set.as_relational(self.symbol)


class ProductContinuousDomain(ProductDomain, ContinuousDomain):
    """
    A collection of independent domains with continuous support
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if variables is None:
            variables = self.symbols
        for domain in self.domains:
            domain_vars = frozenset(variables) & frozenset(domain.symbols)
            if domain_vars:
                expr = domain.compute_expectation(expr, domain_vars, **kwargs)
        return expr

    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])


class ConditionalContinuousDomain(ContinuousDomain, ConditionalDomain):
    """
    A domain with continuous support that has been further restricted by a
    condition such as $x > 3$.
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if variables is None:
            variables = self.symbols
        if not variables:
            return expr
        # Extract the full integral
        fullintgrl = self.fulldomain.compute_expectation(expr, variables)
        # separate into integrand and limits
        integrand, limits = fullintgrl.function, list(fullintgrl.limits)

        conditions = [self.condition]
        while conditions:
            cond = conditions.pop()
            if cond.is_Boolean:
                if isinstance(cond, And):
                    conditions.extend(cond.args)
                elif isinstance(cond, Or):
                    raise NotImplementedError("Or not implemented here")
            elif cond.is_Relational:
                if cond.is_Equality:
                    # Add the appropriate Delta to the integrand
                    integrand *= DiracDelta(cond.lhs - cond.rhs)
                else:
                    symbols = cond.free_symbols & set(self.symbols)
                    if len(symbols) != 1:  # Can't handle x > y
                        raise NotImplementedError(
                            "Multivariate Inequalities not yet implemented")
                    # Can handle x > 0
                    symbol = symbols.pop()
                    # Find the limit with x, such as (x, -oo, oo)
                    for i, limit in enumerate(limits):
                        if limit[0] == symbol:
                            # Make condition into an Interval like [0, oo]
                            cintvl = reduce_rational_inequalities_wrap(
                                cond, symbol)
                            # Make limit into an Interval like [-oo, oo]
                            lintvl = Interval(limit[1], limit[2])
                            # Intersect them to get [0, oo]
                            intvl = cintvl.intersect(lintvl)
                            # Put back into limits list
                            limits[i] = (symbol, intvl.left, intvl.right)
            else:
                raise TypeError(
                    "Condition %s is not a relational or Boolean" % cond)

        return Integral(integrand, *limits, **kwargs)

    def as_boolean(self):
        return And(self.fulldomain.as_boolean(), self.condition)

    @property
    def set(self):
        if len(self.symbols) == 1:
            return (self.fulldomain.set & reduce_rational_inequalities_wrap(
                self.condition, tuple(self.symbols)[0]))
        else:
            raise NotImplementedError(
                "Set of Conditional Domain not Implemented")


class ContinuousDistribution(Distribution):
    def __call__(self, *args):
        return self.pdf(*args)


class SingleContinuousDistribution(ContinuousDistribution, NamedArgsMixin):
    """ Continuous distribution of a single variable.

    Explanation
    ===========

    Serves as superclass for Normal/Exponential/UniformDistribution etc....

    Represented by parameters for each of the specific classes.  E.g
    NormalDistribution is represented by a mean and standard deviation.

    Provides methods for pdf, cdf, and sampling.

    See Also
    ========

    sympy.stats.crv_types.*
    """

    set = Interval(-oo, oo)

    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.set.start

        # CDF is integral of PDF from left bound to z
        pdf = self.pdf(x)
        cdf = integrate(pdf.doit(), (x, left_bound, z), **kwargs)
        # CDF Ensure that CDF left of left_bound is zero
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    def _cdf(self, x):
        return None

    def cdf(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)

    @cacheit
    def compute_characteristic_function(self, **kwargs):
        """ Compute the characteristic function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        cf = integrate(exp(I*t*x)*pdf, (x, self.set))
        return Lambda(t, cf)

    def _characteristic_function(self, t):
        return None

    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
        if len(kwargs) == 0:
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        return self.compute_characteristic_function(**kwargs)(t)

    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        """ Compute the moment generating function from the PDF.

        Returns a Lambda.
        """
        x, t = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        mgf = integrate(exp(t * x) * pdf, (x, self.set))
        return Lambda(t, mgf)

    def _moment_generating_function(self, t):
        return None

    def moment_generating_function(self, t, **kwargs):
        """ Moment generating function """
        if not kwargs:
                mgf = self._moment_generating_function(t)
                if mgf is not None:
                    return mgf
        return self.compute_moment_generating_function(**kwargs)(t)

    def expectation(self, expr, var, evaluate=True, **kwargs):
        """ Expectation of expression over distribution """
        if evaluate:
            try:
                p = poly(expr, var)
                if p.is_zero:
                    return S.Zero
                t = Dummy('t', real=True)
                mgf = self._moment_generating_function(t)
                if mgf is None:
                    return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
                deg = p.degree()
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                for k in range(deg+1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)
                return result
            except PolynomialError:
                return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
        else:
            return Integral(expr * self.pdf(var), (var, self.set), **kwargs)

    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
        x, p = symbols('x, p', real=True, cls=Dummy)
        left_bound = self.set.start

        pdf = self.pdf(x)
        cdf = integrate(pdf, (x, left_bound, x), **kwargs)
        quantile = solveset(cdf - p, x, self.set)
        return Lambda(p, Piecewise((quantile, (p >= 0) & (p <= 1) ), (nan, True)))

    def _quantile(self, x):
        return None

    def quantile(self, x, **kwargs):
        """ Cumulative density function """
        if len(kwargs) == 0:
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        return self.compute_quantile(**kwargs)(x)


class ContinuousPSpace(PSpace):
    """ Continuous Probability Space

    Represents the likelihood of an event space defined over a continuum.

    Represented with a ContinuousDomain and a PDF (Lambda-Like)
    """

    is_Continuous = True
    is_real = True

    @property
    def pdf(self):
        return self.density(*self.domain.symbols)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        if rvs is None:
            rvs = self.values
        else:
            rvs = frozenset(rvs)

        expr = expr.xreplace({rv: rv.symbol for rv in rvs})

        domain_symbols = frozenset(rv.symbol for rv in rvs)

        return self.domain.compute_expectation(self.pdf * expr,
                domain_symbols, **kwargs)

    def compute_density(self, expr, **kwargs):
        # Common case Density(X) where X in self.values
        if expr in self.values:
            # Marginalize all other random symbols out of the density
            randomsymbols = tuple(set(self.values) - frozenset([expr]))
            symbols = tuple(rs.symbol for rs in randomsymbols)
            pdf = self.domain.compute_expectation(self.pdf, symbols, **kwargs)
            return Lambda(expr.symbol, pdf)

        z = Dummy('z', real=True)
        return Lambda(z, self.compute_expectation(DiracDelta(expr - z), **kwargs))

    @cacheit
    def compute_cdf(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError(
                "CDF not well defined on multivariate expressions")

        d = self.compute_density(expr, **kwargs)
        x, z = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.domain.set.start

        # CDF is integral of PDF from left bound to z
        cdf = integrate(d(x), (x, left_bound, z), **kwargs)
        # CDF Ensure that CDF left of left_bound is zero
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    @cacheit
    def compute_characteristic_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError("Characteristic function of multivariate expressions not implemented")

        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        cf = integrate(exp(I*t*x)*d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, cf)

    @cacheit
    def compute_moment_generating_function(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise NotImplementedError("Moment generating function of multivariate expressions not implemented")

        d = self.compute_density(expr, **kwargs)
        x, t = symbols('x, t', real=True, cls=Dummy)
        mgf = integrate(exp(t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, mgf)

    @cacheit
    def compute_quantile(self, expr, **kwargs):
        if not self.domain.set.is_Interval:
            raise ValueError(
                "Quantile not well defined on multivariate expressions")

        d = self.compute_cdf(expr, **kwargs)
        x = Dummy('x', real=True)
        p = Dummy('p', positive=True)

        quantile = solveset(d(x) - p, x, self.set)

        return Lambda(p, quantile)

    def probability(self, condition, **kwargs):
        z = Dummy('z', real=True)
        cond_inv = False
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        # Univariate case can be handled by where
        try:
            domain = self.where(condition)
            rv = [rv for rv in self.values if rv.symbol == domain.symbol][0]
            # Integrate out all other random variables
            pdf = self.compute_density(rv, **kwargs)
            # return S.Zero if `domain` is empty set
            if domain.set is S.EmptySet or isinstance(domain.set, FiniteSet):
                return S.Zero if not cond_inv else S.One
            if isinstance(domain.set, Union):
                return sum(
                     Integral(pdf(z), (z, subset), **kwargs) for subset in
                     domain.set.args if isinstance(subset, Interval))
            # Integrate out the last variable over the special domain
            return Integral(pdf(z), (z, domain.set), **kwargs)

        # Other cases can be turned into univariate case
        # by computing a density handled by density computation
        except NotImplementedError:
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            if not is_random(expr):
                dens = self.density
                comp = condition.rhs
            else:
                dens = density(expr, **kwargs)
                comp = 0
            if not isinstance(dens, ContinuousDistribution):
                from sympy.stats.crv_types import ContinuousDistributionHandmade
                dens = ContinuousDistributionHandmade(dens, set=self.domain.set)
            # Turn problem into univariate case
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, comp))
            return result if not cond_inv else S.One - result

    def where(self, condition):
        rvs = frozenset(random_symbols(condition))
        if not (len(rvs) == 1 and rvs.issubset(self.values)):
            raise NotImplementedError(
                "Multiple continuous random variables not supported")
        rv = tuple(rvs)[0]
        interval = reduce_rational_inequalities_wrap(condition, rv)
        interval = interval.intersect(self.domain.set)
        return SingleContinuousDomain(rv.symbol, interval)

    def conditional_space(self, condition, normalize=True, **kwargs):
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        domain = ConditionalContinuousDomain(self.domain, condition)
        if normalize:
            # create a clone of the variable to
            # make sure that variables in nested integrals are different
            # from the variables outside the integral
            # this makes sure that they are evaluated separately
            # and in the correct order
            replacement  = {rv: Dummy(str(rv)) for rv in self.symbols}
            norm = domain.compute_expectation(self.pdf, **kwargs)
            pdf = self.pdf / norm.xreplace(replacement)
            # XXX: Converting set to tuple. The order matters to Lambda though
            # so we shouldn't be starting with a set here...
            density = Lambda(tuple(domain.symbols), pdf)

        return ContinuousPSpace(domain, density)


class SingleContinuousPSpace(ContinuousPSpace, SinglePSpace):
    """
    A continuous probability space over a single univariate variable.

    These consist of a Symbol and a SingleContinuousDistribution

    This class is normally accessed through the various random variable
    functions, Normal, Exponential, Uniform, etc....
    """

    @property
    def set(self):
        return self.distribution.set

    @property
    def domain(self):
        return SingleContinuousDomain(sympify(self.symbol), self.set)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        rvs = rvs or (self.value,)
        if self.value not in rvs:
            return expr

        expr = _sympify(expr)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})

        x = self.value.symbol
        try:
            return self.distribution.expectation(expr, x, evaluate=evaluate, **kwargs)
        except PoleError:
            return Integral(expr * self.pdf, (x, self.set), **kwargs)

    def compute_cdf(self, expr, **kwargs):
        if expr == self.value:
            z = Dummy("z", real=True)
            return Lambda(z, self.distribution.cdf(z, **kwargs))
        else:
            return ContinuousPSpace.compute_cdf(self, expr, **kwargs)

    def compute_characteristic_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_characteristic_function(self, expr, **kwargs)

    def compute_moment_generating_function(self, expr, **kwargs):
        if expr == self.value:
            t = Dummy("t", real=True)
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_moment_generating_function(self, expr, **kwargs)

    def compute_density(self, expr, **kwargs):
        # https://en.wikipedia.org/wiki/Random_variable#Functions_of_random_variables
        if expr == self.value:
            return self.density
        y = Dummy('y', real=True)

        gs = solveset(expr - y, self.value, S.Reals)

        if isinstance(gs, Intersection):
            if len(gs.args) == 2 and gs.args[0] is S.Reals:
                gs = gs.args[1]
        if not gs.is_FiniteSet:
            raise ValueError("Can not solve %s for %s" % (expr, self.value))
        fx = self.compute_density(self.value)
        fy = sum(fx(g) * abs(g.diff(y)) for g in gs)
        return Lambda(y, fy)

    def compute_quantile(self, expr, **kwargs):

        if expr == self.value:
            p = Dummy("p", real=True)
            return Lambda(p, self.distribution.quantile(p, **kwargs))
        else:
            return ContinuousPSpace.compute_quantile(self, expr, **kwargs)

def _reduce_inequalities(conditions, var, **kwargs):
    try:
        return reduce_rational_inequalities(conditions, var, **kwargs)
    except PolynomialError:
        raise ValueError("Reduction of condition failed %s\n" % conditions[0])


def reduce_rational_inequalities_wrap(condition, var):
    if condition.is_Relational:
        return _reduce_inequalities([[condition]], var, relational=False)
    if isinstance(condition, Or):
        return Union(*[_reduce_inequalities([[arg]], var, relational=False)
            for arg in condition.args])
    if isinstance(condition, And):
        intervals = [_reduce_inequalities([[arg]], var, relational=False)
            for arg in condition.args]
        I = intervals[0]
        for i in intervals:
            I = I.intersect(i)
        return I
<|endoftext|>from math import prod

from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
                            MatrixSymbol, MatrixBase, Transpose, MatrixSet,
                            matrix2numpy)
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
                            _symbol_converter, MatrixDomain, Distribution)
from sympy.external import import_module


################################################################################
#------------------------Matrix Probability Space------------------------------#
################################################################################
class MatrixPSpace(PSpace):
    """
    Represents probability space for
    Matrix Distributions.
    """
    def __new__(cls, sym, distribution, dim_n, dim_m):
        sym = _symbol_converter(sym)
        dim_n, dim_m = _sympify(dim_n), _sympify(dim_m)
        if not (dim_n.is_integer and dim_m.is_integer):
            raise ValueError("Dimensions should be integers")
        return Basic.__new__(cls, sym, distribution, dim_n, dim_m)

    distribution = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])

    @property
    def domain(self):
        return MatrixDomain(self.symbol, self.distribution.set)

    @property
    def value(self):
        return RandomMatrixSymbol(self.symbol, self.args[2], self.args[3], self)

    @property
    def values(self):
        return {self.value}

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 1 or (not isinstance(expr, RandomMatrixSymbol)):
            raise NotImplementedError("Currently, no algorithm has been "
                    "implemented to handle general expressions containing "
                    "multiple matrix distributions.")
        return self.distribution.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomMatrixSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}


def rv(symbol, cls, args):
    args = list(map(sympify, args))
    dist = cls(*args)
    dist.check(*args)
    dim = dist.dimension
    pspace = MatrixPSpace(symbol, dist, dim[0], dim[1])
    return pspace.value


class SampleMatrixScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        from scipy import stats as scipy_stats
        import numpy
        scipy_rv_map = {
            'WishartDistribution': lambda dist, size, rand_state: scipy_stats.wishart.rvs(
                df=int(dist.n), scale=matrix2numpy(dist.scale_matrix, float), size=size),
            'MatrixNormalDistribution': lambda dist, size, rand_state: scipy_stats.matrix_normal.rvs(
                mean=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float), size=size, random_state=rand_state)
        }

        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = scipy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        samp = scipy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixNumpy:
    """Returns the sample from numpy of the given distribution"""

    ### TODO: Add tests after adding matrix distributions in numpy_rv_map
    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        numpy_rv_map = {
        }

        sample_shape = {
        }

        dist_list = numpy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        samp = numpy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {
            'MatrixNormalDistribution': lambda dist: pymc.MatrixNormal('X',
                mu=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float),
                shape=dist.location_matrix.shape),
            'WishartDistribution': lambda dist: pymc.WishartBartlett('X',
                nu=int(dist.n), S=matrix2numpy(dist.scale_matrix, float))
        }

        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = pymc_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None
        import logging
        logging.getLogger("pymc").setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samps = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)['X']
        return samps.reshape(size + sample_shape[dist.__class__.__name__](dist))

_get_sample_class_matrixrv = {
    'scipy': SampleMatrixScipy,
    'pymc3': SampleMatrixPymc,
    'pymc': SampleMatrixPymc,
    'numpy': SampleMatrixNumpy
}

################################################################################
#-------------------------Matrix Distribution----------------------------------#
################################################################################

class MatrixDistribution(Distribution, NamedArgsMixin):
    """
    Abstract class for Matrix Distribution.
    """
    def __new__(cls, *args):
        args = [ImmutableMatrix(arg) if isinstance(arg, list)
                else _sympify(arg) for arg in args]
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    def __call__(self, expr):
        if isinstance(expr, list):
            expr = ImmutableMatrix(expr)
        return self.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """

        libraries = ['scipy', 'numpy', 'pymc3', 'pymc']
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        samps = _get_sample_class_matrixrv[library](self, size, seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

################################################################################
#------------------------Matrix Distribution Types-----------------------------#
################################################################################

#-------------------------------------------------------------------------------
# Matrix Gamma distribution ----------------------------------------------------

class MatrixGammaDistribution(MatrixDistribution):

    _argnames = ('alpha', 'beta', 'scale_matrix')

    @staticmethod
    def check(alpha, beta, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        _value_check(beta.is_positive, "Scale parameter should be positive.")

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        alpha, beta, scale_matrix = self.alpha, self.beta, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        sigma_inv_x = - Inverse(scale_matrix)*x / beta
        term1 = exp(Trace(sigma_inv_x))/((beta**(p*alpha)) * multigamma(alpha, p))
        term2 = (Determinant(scale_matrix))**(-alpha)
        term3 = (Determinant(x))**(alpha - S(p + 1)/2)
        return term1 * term2 * term3

def MatrixGamma(symbol, alpha, beta, scale_matrix):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    alpha: Positive Real number
        Shape Parameter
    beta: Positive Real number
        Scale Parameter
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MatrixGamma
    >>> from sympy import MatrixSymbol, symbols
    >>> a, b = symbols('a b', positive=True)
    >>> M = MatrixGamma('M', a, b, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(M)(X).doit()
    exp(Trace(Matrix([
    [-2/3,  1/3],
    [ 1/3, -2/3]])*X)/b)*Determinant(X)**(a - 3/2)/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))
    >>> density(M)([[1, 0], [0, 1]]).doit()
    exp(-4/(3*b))/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_gamma_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, MatrixGammaDistribution, (alpha, beta, scale_matrix))

#-------------------------------------------------------------------------------
# Wishart Distribution ---------------------------------------------------------

class WishartDistribution(MatrixDistribution):

    _argnames = ('n', 'scale_matrix')

    @staticmethod
    def check(n, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(n.is_positive, "Shape parameter should be positive.")

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        n, scale_matrix = self.n, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        sigma_inv_x = - Inverse(scale_matrix)*x / S(2)
        term1 = exp(Trace(sigma_inv_x))/((2**(p*n/S(2))) * multigamma(n/S(2), p))
        term2 = (Determinant(scale_matrix))**(-n/S(2))
        term3 = (Determinant(x))**(S(n - p - 1)/2)
        return term1 * term2 * term3

def Wishart(symbol, n, scale_matrix):
    """
    Creates a random variable with Wishart Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n: Positive Real number
        Represents degrees of freedom
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Wishart
    >>> from sympy import MatrixSymbol, symbols
    >>> n = symbols('n', positive=True)
    >>> W = Wishart('W', n, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(W)(X).doit()
    exp(Trace(Matrix([
    [-1/3,  1/6],
    [ 1/6, -1/3]])*X))*Determinant(X)**(n/2 - 3/2)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))
    >>> density(W)([[1, 0], [0, 1]]).doit()
    exp(-2/3)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wishart_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, WishartDistribution, (n, scale_matrix))

#-------------------------------------------------------------------------------
# Matrix Normal distribution ---------------------------------------------------

class MatrixNormalDistribution(MatrixDistribution):

    _argnames = ('location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix_1.is_square, "Scale matrix 1 should be "
        "be square matrix")
        _value_check(scale_matrix_2.is_square, "Scale matrix 2 should be "
        "be square matrix")
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == n, "Scale matrix 1 should be"
        " of shape %s x %s"% (str(n), str(n)))
        _value_check(scale_matrix_2.shape[0] == p, "Scale matrix 2 should be"
        " of shape %s x %s"% (str(p), str(p)))

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        M, U, V = self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        term1 = Inverse(V)*Transpose(x - M)*Inverse(U)*(x - M)
        num = exp(-Trace(term1)/S(2))
        den = (2*pi)**(S(n*p)/2) * Determinant(U)**(S(p)/2) * Determinant(V)**(S(n)/2)
        return num/den

def MatrixNormal(symbol, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Normal Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    location_matrix: Real ``n x p`` matrix
        Represents degrees of freedom
    scale_matrix_1: Positive definite matrix
        Scale Matrix of shape ``n x n``
    scale_matrix_2: Positive definite matrix
        Scale Matrix of shape ``p x p``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> from sympy.stats import density, MatrixNormal
    >>> M = MatrixNormal('M', [[1, 2]], [1], [[1, 0], [0, 1]])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X).doit()
    exp(-Trace((Matrix([
    [-1],
    [-2]]) + X.T)*(Matrix([[-1, -2]]) + X))/2)/(2*pi)
    >>> density(M)([[3, 4]]).doit()
    exp(-4)/(2*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_normal_distribution

    """
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    args = (location_matrix, scale_matrix_1, scale_matrix_2)
    return rv(symbol, MatrixNormalDistribution, args)

#-------------------------------------------------------------------------------
# Matrix Student's T distribution ---------------------------------------------------

class MatrixStudentTDistribution(MatrixDistribution):

    _argnames = ('nu', 'location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(nu, location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        _value_check(scale_matrix_1.is_square != False, "Scale matrix 1 should be "
                                               "be square matrix")
        _value_check(scale_matrix_2.is_square != False, "Scale matrix 2 should be "
                                               "be square matrix")
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == p, "Scale matrix 1 should be"
                                                   " of shape %s x %s" % (str(p), str(p)))
        _value_check(scale_matrix_2.shape[0] == n, "Scale matrix 2 should be"
                                                   " of shape %s x %s" % (str(n), str(n)))
        _value_check(nu.is_positive != False, "Degrees of freedom must be positive")

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        from sympy.matrices.dense import eye
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                             "or MatrixSymbol" % str(x))
        nu, M, Omega, Sigma = self.nu, self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape

        K = multigamma((nu + n + p - 1)/2, p) * Determinant(Omega)**(-n/2) * Determinant(Sigma)**(-p/2) \
            / ((pi)**(n*p/2) * multigamma((nu + p - 1)/2, p))
        return K * (Determinant(eye(n) + Inverse(Sigma)*(x - M)*Inverse(Omega)*Transpose(x - M))) \
               **(-(nu + n + p -1)/2)



def MatrixStudentT(symbol, nu, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    nu: Positive Real number
        degrees of freedom
    location_matrix: Positive definite real square matrix
        Location Matrix of shape ``n x p``
    scale_matrix_1: Positive definite real square matrix
        Scale Matrix of shape ``p x p``
    scale_matrix_2: Positive definite real square matrix
        Scale Matrix of shape ``n x n``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol,symbols
    >>> from sympy.stats import density, MatrixStudentT
    >>> v = symbols('v',positive=True)
    >>> M = MatrixStudentT('M', v, [[1, 2]], [[1, 0], [0, 1]], [1])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X)
    gamma(v/2 + 1)*Determinant((Matrix([[-1, -2]]) + X)*(Matrix([
    [-1],
    [-2]]) + X.T) + Matrix([[1]]))**(-v/2 - 1)/(pi**1.0*gamma(v/2)*Determinant(Matrix([[1]]))**1.0*Determinant(Matrix([
    [1, 0],
    [0, 1]]))**0.5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_t-distribution

    """
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    args = (nu, location_matrix, scale_matrix_1, scale_matrix_2)
    return rv(symbol, MatrixStudentTDistribution, args)
<|endoftext|>from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besselk
from sympy.functions.special.gamma_functions import gamma
from sympy.matrices.dense import (Matrix, ones)
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import ImmutableMatrix, MatrixSymbol
from sympy.matrices.expressions.determinant import det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats.joint_rv import JointDistribution, JointPSpace, MarginalDistribution
from sympy.stats.rv import _value_check, random_symbols

__all__ = ['JointRV',
'MultivariateNormal',
'MultivariateLaplace',
'Dirichlet',
'GeneralizedMultivariateLogGamma',
'GeneralizedMultivariateLogGammaOmega',
'Multinomial',
'MultivariateBeta',
'MultivariateEwens',
'MultivariateT',
'NegativeMultinomial',
'NormalGamma'
]

def multivariate_rv(cls, sym, *args):
    args = list(map(sympify, args))
    dist = cls(*args)
    args = dist.args
    dist.check(*args)
    return JointPSpace(sym, dist).value


def marginal_distribution(rv, *indices):
    """
    Marginal distribution function of a joint random variable.

    Parameters
    ==========

    rv : A random variable with a joint probability distribution.
    indices : Component indices or the indexed random symbol
        for which the joint distribution is to be calculated

    Returns
    =======

    A Lambda expression in `sym`.

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, marginal_distribution
    >>> m = MultivariateNormal('X', [1, 2], [[2, 1], [1, 2]])
    >>> marginal_distribution(m, m[0])(1)
    1/(2*sqrt(pi))

    """
    indices = list(indices)
    for i in range(len(indices)):
        if isinstance(indices[i], Indexed):
            indices[i] = indices[i].args[1]
    prob_space = rv.pspace
    if not indices:
        raise ValueError(
            "At least one component for marginal density is needed.")
    if hasattr(prob_space.distribution, '_marginal_distribution'):
        return prob_space.distribution._marginal_distribution(indices, rv.symbol)
    return prob_space.marginal_distribution(*indices)


class JointDistributionHandmade(JointDistribution):

    _argnames = ('pdf',)
    is_Continuous = True

    @property
    def set(self):
        return self.args[1]


def JointRV(symbol, pdf, _set=None):
    """
    Create a Joint Random Variable where each of its component is continuous,
    given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    pdf : A PDF in terms of indexed symbols of the symbol given
        as the first argument

    NOTE
    ====

    As of now, the set for each component for a ``JointRV`` is
    equal to the set of all integers, which cannot be changed.

    Examples
    ========

    >>> from sympy import exp, pi, Indexed, S
    >>> from sympy.stats import density, JointRV
    >>> x1, x2 = (Indexed('x', i) for i in (1, 2))
    >>> pdf = exp(-x1**2/2 + x1 - x2**2/2 - S(1)/2)/(2*pi)
    >>> N1 = JointRV('x', pdf) #Multivariate Normal distribution
    >>> density(N1)(1, 2)
    exp(-2)/(2*pi)

    Returns
    =======

    RandomSymbol

    """
    #TODO: Add support for sets provided by the user
    symbol = sympify(symbol)
    syms = [i for i in pdf.free_symbols if isinstance(i, Indexed)
        and i.base == IndexedBase(symbol)]
    syms = tuple(sorted(syms, key = lambda index: index.args[1]))
    _set = S.Reals**len(syms)
    pdf = Lambda(syms, pdf)
    dist = JointDistributionHandmade(pdf, _set)
    jrv = JointPSpace(symbol, dist).value
    rvs = random_symbols(pdf)
    if len(rvs) != 0:
        dist = MarginalDistribution(dist, (jrv,))
        return JointPSpace(symbol, dist).value
    return jrv

#-------------------------------------------------------------------------------
# Multivariate Normal distribution ---------------------------------------------

class MultivariateNormalDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')

    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
            "Size of the mean vector and covariance matrix are incorrect.")
        #check if covariance matrix is positive semi definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_semidefinite,
            "The covariance matrix must be positive semi definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        k = mu.shape[0]
        if len(args) == 1 and args[0].is_Matrix:
            args = args[0]
        else:
            args = ImmutableMatrix(args)
        x = args - mu
        density = S.One/sqrt((2*pi)**(k)*det(sigma))*exp(
            Rational(-1, 2)*x.transpose()*(sigma.inv()*x))
        return MatrixElement(density, 0, 0)

    def _marginal_distribution(self, indices, sym):
        sym = ImmutableMatrix([Indexed(sym, i) for i in indices])
        _mu, _sigma = self.mu, self.sigma
        k = self.mu.shape[0]
        for i in range(k):
            if i not in indices:
                _mu = _mu.row_del(i)
                _sigma = _sigma.col_del(i)
                _sigma = _sigma.row_del(i)
        return Lambda(tuple(sym), S.One/sqrt((2*pi)**(len(_mu))*det(_sigma))*exp(
            Rational(-1, 2)*(_mu - sym).transpose()*(_sigma.inv()*\
                (_mu - sym)))[0])

def MultivariateNormal(name, mu, sigma):
    r"""
    Creates a continuous random variable with Multivariate Normal
    Distribution.

    The density of the multivariate normal distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
    sigma : Positive semidefinite square matrix
        Represents covariance Matrix.
        If `\sigma` is noninvertible then only sampling is supported currently

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateNormal, density, marginal_distribution
    >>> from sympy import symbols, MatrixSymbol
    >>> X = MultivariateNormal('X', [3, 4], [[2, 1], [1, 2]])
    >>> y, z = symbols('y z')
    >>> density(X)(y, z)
    sqrt(3)*exp(-y**2/3 + y*z/3 + 2*y/3 - z**2/3 + 5*z/3 - 13/3)/(6*pi)
    >>> density(X)(1, 2)
    sqrt(3)*exp(-4/3)/(6*pi)
    >>> marginal_distribution(X, X[1])(y)
    exp(-(y - 4)**2/4)/(2*sqrt(pi))
    >>> marginal_distribution(X, X[0])(y)
    exp(-(y - 3)**2/4)/(2*sqrt(pi))

    The example below shows that it is also possible to use
    symbolic parameters to define the MultivariateNormal class.

    >>> n = symbols('n', integer=True, positive=True)
    >>> Sg = MatrixSymbol('Sg', n, n)
    >>> mu = MatrixSymbol('mu', n, 1)
    >>> obs = MatrixSymbol('obs', n, 1)
    >>> X = MultivariateNormal('X', mu, Sg)

    The density of a multivariate normal can be
    calculated using a matrix argument, as shown below.

    >>> density(X)(obs)
    (exp(((1/2)*mu.T - (1/2)*obs.T)*Sg**(-1)*(-mu + obs))/sqrt((2*pi)**n*Determinant(Sg)))[0, 0]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    """
    return multivariate_rv(MultivariateNormalDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# Multivariate Laplace distribution --------------------------------------------

class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ('mu', 'sigma')
    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the mean vector and covariance matrix are incorrect.")
        # check if covariance matrix is positive definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The covariance matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.sigma
        mu_T = mu.transpose()
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        args_T = args.transpose()
        x = (mu_T*sigma_inv*mu)[0]
        y = (args_T*sigma_inv*args)[0]
        v = 1 - k/2
        return (2 * (y/(2 + x))**(v/2) * besselk(v, sqrt((2 + x)*y)) *
                exp((args_T * sigma_inv * mu)[0]) /
                ((2 * pi)**(k/2) * sqrt(det(sigma))))


def MultivariateLaplace(name, mu, sigma):
    """
    Creates a continuous random variable with Multivariate Laplace
    Distribution.

    The density of the multivariate Laplace distribution can be found at [1].

    Parameters
    ==========

    mu : List representing the mean or the mean vector
    sigma : Positive definite square matrix
        Represents covariance Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import MultivariateLaplace, density
    >>> from sympy import symbols
    >>> y, z = symbols('y z')
    >>> X = MultivariateLaplace('X', [2, 4], [[3, 1], [1, 3]])
    >>> density(X)(y, z)
    sqrt(2)*exp(y/4 + 5*z/4)*besselk(0, sqrt(15*y*(3*y/8 - z/8)/2 + 15*z*(-y/8 + 3*z/8)/2))/(4*pi)
    >>> density(X)(1, 2)
    sqrt(2)*exp(11/4)*besselk(0, sqrt(165)/4)/(4*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution

    """
    return multivariate_rv(MultivariateLaplaceDistribution, name, mu, sigma)

#-------------------------------------------------------------------------------
# Multivariate StudentT distribution -------------------------------------------

class MultivariateTDistribution(JointDistribution):
    _argnames = ('mu', 'shape_mat', 'dof')
    is_Continuous=True

    @property
    def set(self):
        k = self.mu.shape[0]
        return S.Reals**k

    @staticmethod
    def check(mu, sigma, v):
        _value_check(mu.shape[0] == sigma.shape[0],
                     "Size of the location vector and shape matrix are incorrect.")
        # check if covariance matrix is positive definite or not.
        if not isinstance(sigma, MatrixSymbol):
            _value_check(sigma.is_positive_definite,
                         "The shape matrix must be positive definite. ")

    def pdf(self, *args):
        mu, sigma = self.mu, self.shape_mat
        v = S(self.dof)
        k = S(mu.shape[0])
        sigma_inv = sigma.inv()
        args = ImmutableMatrix(args)
        x = args - mu
        return gamma((k + v)/2)/(gamma(v/2)*(v*pi)**(k/2)*sqrt(det(sigma)))\
        *(1 + 1/v*(x.transpose()*sigma_inv*x)[0])**((-v - k)/2)

def MultivariateT(syms, mu, sigma, v):
    """
    Creates a joint random variable with multivariate T-distribution.

    Parameters
    ==========

    syms : A symbol/str
        For identifying the random variable.
    mu : A list/matrix
        Representing the location vector
    sigma : The shape matrix for the distribution

    Examples
    ========

    >>> from sympy.stats import density, MultivariateT
    >>> from sympy import Symbol

    >>> x = Symbol("x")
    >>> X = MultivariateT("x", [1, 1], [[1, 0], [0, 1]], 2)

    >>> density(X)(1, 2)
    2/(9*pi)

    Returns
    =======

    RandomSymbol

    """
    return multivariate_rv(MultivariateTDistribution, syms, mu, sigma, v)


#-------------------------------------------------------------------------------
# Multivariate Normal Gamma distribution ---------------------------------------

class NormalGammaDistribution(JointDistribution):

    _argnames = ('mu', 'lamda', 'alpha', 'beta')
    is_Continuous=True

    @staticmethod
    def check(mu, lamda, alpha, beta):
        _value_check(mu.is_real, "Location must be real.")
        _value_check(lamda > 0, "Lambda must be positive")
        _value_check(alpha > 0, "alpha must be positive")
        _value_check(beta > 0, "beta must be positive")

    @property
    def set(self):
        return S.Reals*Interval(0, S.Infinity)

    def pdf(self, x, tau):
        beta, alpha, lamda = self.beta, self.alpha, self.lamda
        mu = self.mu

        return beta**alpha*sqrt(lamda)/(gamma(alpha)*sqrt(2*pi))*\
        tau**(alpha - S.Half)*exp(-1*beta*tau)*\
        exp(-1*(lamda*tau*(x - mu)**2)/S(2))

    def _marginal_distribution(self, indices, *sym):
        if len(indices) == 2:
            return self.pdf(*sym)
        if indices[0] == 0:
            #For marginal over `x`, return non-standardized Student-T's
            #distribution
            x = sym[0]
            v, mu, sigma = self.alpha - S.Half, self.mu, \
                S(self.beta)/(self.lamda * self.alpha)
            return Lambda(sym, gamma((v + 1)/2)/(gamma(v/2)*sqrt(pi*v)*sigma)*\
                (1 + 1/v*((x - mu)/sigma)**2)**((-v -1)/2))
        #For marginal over `tau`, return Gamma distribution as per construction
        from sympy.stats.crv_types import GammaDistribution
        return Lambda(sym, GammaDistribution(self.alpha, self.beta)(sym[0]))

def NormalGamma(sym, mu, lamda, alpha, beta):
    """
    Creates a bivariate joint random variable with multivariate Normal gamma
    distribution.

    Parameters
    ==========

    sym : A symbol/str
        For identifying the random variable.
    mu : A real number
        The mean of the normal distribution
    lamda : A positive integer
        Parameter of joint distribution
    alpha : A positive integer
        Parameter of joint distribution
    beta : A positive integer
        Parameter of joint distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NormalGamma
    >>> from sympy import symbols

    >>> X = NormalGamma('x', 0, 1, 2, 3)
    >>> y, z = symbols('y z')

    >>> density(X)(y, z)
    9*sqrt(2)*z**(3/2)*exp(-3*z)*exp(-y**2*z/2)/(2*sqrt(pi))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal-gamma_distribution

    """
    return multivariate_rv(NormalGammaDistribution, sym, mu, lamda, alpha, beta)

#-------------------------------------------------------------------------------
# Multivariate Beta/Dirichlet distribution -------------------------------------

class MultivariateBetaDistribution(JointDistribution):

    _argnames = ('alpha',)
    is_Continuous = True

    @staticmethod
    def check(alpha):
        _value_check(len(alpha) >= 2, "At least two categories should be passed.")
        for a_k in alpha:
            _value_check((a_k > 0) != False, "Each concentration parameter"
                                            " should be positive.")

    @property
    def set(self):
        k = len(self.alpha)
        return Interval(0, 1)**k

    def pdf(self, *syms):
        alpha = self.alpha
        B = Mul.fromiter(map(gamma, alpha))/gamma(Add(*alpha))
        return Mul.fromiter(sym**(a_k - 1) for a_k, sym in zip(alpha, syms))/B

def MultivariateBeta(syms, *alpha):
    """
    Creates a continuous random variable with Dirichlet/Multivariate Beta
    Distribution.

    The density of the Dirichlet distribution can be found at [1].

    Parameters
    ==========

    alpha : Positive real numbers
        Signifies concentration numbers.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MultivariateBeta, marginal_distribution
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> B = MultivariateBeta('B', [a1, a2])
    >>> C = MultivariateBeta('C', a1, a2)
    >>> x = Symbol('x')
    >>> y = Symbol('y')
    >>> density(B)(x, y)
    x**(a1 - 1)*y**(a2 - 1)*gamma(a1 + a2)/(gamma(a1)*gamma(a2))
    >>> marginal_distribution(C, C[0])(x)
    x**(a1 - 1)*gamma(a1 + a2)/(a2*gamma(a1)*gamma(a2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_distribution
    .. [2] https://mathworld.wolfram.com/DirichletDistribution.html

    """
    if not isinstance(alpha[0], list):
        alpha = (list(alpha),)
    return multivariate_rv(MultivariateBetaDistribution, syms, alpha[0])

Dirichlet = MultivariateBeta

#-------------------------------------------------------------------------------
# Multivariate Ewens distribution ----------------------------------------------

class MultivariateEwensDistribution(JointDistribution):

    _argnames = ('n', 'theta')
    is_Discrete = True
    is_Continuous = False

    @staticmethod
    def check(n, theta):
        _value_check((n > 0),
                        "sample size should be positive integer.")
        _value_check(theta.is_positive, "mutation rate should be positive.")

    @property
    def set(self):
        if not isinstance(self.n, Integer):
            i = Symbol('i', integer=True, positive=True)
            return Product(Intersection(S.Naturals0, Interval(0, self.n//i)),
                                    (i, 1, self.n))
        prod_set = Range(0, self.n + 1)
        for i in range(2, self.n + 1):
            prod_set *= Range(0, self.n//i + 1)
        return prod_set.flatten()

    def pdf(self, *syms):
        n, theta = self.n, self.theta
        condi = isinstance(self.n, Integer)
        if not (isinstance(syms[0], IndexedBase) or condi):
            raise ValueError("Please use IndexedBase object for syms as "
                                "the dimension is symbolic")
        term_1 = factorial(n)/rf(theta, n)
        if condi:
            term_2 = Mul.fromiter(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j]))
                                    for j in range(n))
            cond = Eq(sum((k + 1)*syms[k] for k in range(n)), n)
            return Piecewise((term_1 * term_2, cond), (0, True))
        syms = syms[0]
        j, k = symbols('j, k', positive=True, integer=True)
        term_2 = Product(theta**syms[j]/((j+1)**syms[j]*factorial(syms[j])),
                            (j, 0, n - 1))
        cond = Eq(Sum((k + 1)*syms[k], (k, 0, n - 1)), n)
        return Piecewise((term_1 * term_2, cond), (0, True))


def MultivariateEwens(syms, n, theta):
    """
    Creates a discrete random variable with Multivariate Ewens
    Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Size of the sample or the integer whose partitions are considered
    theta : Positive real number
        Denotes Mutation rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, marginal_distribution, MultivariateEwens
    >>> from sympy import Symbol
    >>> a1 = Symbol('a1', positive=True)
    >>> a2 = Symbol('a2', positive=True)
    >>> ed = MultivariateEwens('E', 2, 1)
    >>> density(ed)(a1, a2)
    Piecewise((1/(2**a2*factorial(a1)*factorial(a2)), Eq(a1 + 2*a2, 2)), (0, True))
    >>> marginal_distribution(ed, ed[0])(a1)
    Piecewise((1/factorial(a1), Eq(a1, 2)), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ewens%27s_sampling_formula
    .. [2] https://www.jstor.org/stable/24780825
    """
    return multivariate_rv(MultivariateEwensDistribution, syms, n, theta)

#-------------------------------------------------------------------------------
# Generalized Multivariate Log Gamma distribution ------------------------------

class GeneralizedMultivariateLogGammaDistribution(JointDistribution):

    _argnames = ('delta', 'v', 'lamda', 'mu')
    is_Continuous=True

    def check(self, delta, v, l, mu):
        _value_check((delta >= 0, delta <= 1), "delta must be in range [0, 1].")
        _value_check((v > 0), "v must be positive")
        for lk in l:
            _value_check((lk > 0), "lamda must be a positive vector.")
        for muk in mu:
            _value_check((muk > 0), "mu must be a positive vector.")
        _value_check(len(l) > 1,"the distribution should have at least"
                                " two random variables.")

    @property
    def set(self):
        return S.Reals**len(self.lamda)

    def pdf(self, *y):
        d, v, l, mu = self.delta, self.v, self.lamda, self.mu
        n = Symbol('n', negative=False, integer=True)
        k = len(l)
        sterm1 = Pow((1 - d), n)/\
                ((gamma(v + n)**(k - 1))*gamma(v)*gamma(n + 1))
        sterm2 = Mul.fromiter(mui*li**(-v - n) for mui, li in zip(mu, l))
        term1 = sterm1 * sterm2
        sterm3 = (v + n) * sum(mui * yi for mui, yi in zip(mu, y))
        sterm4 = sum(exp(mui * yi)/li for (mui, yi, li) in zip(mu, y, l))
        term2 = exp(sterm3 - sterm4)
        return Pow(d, v) * Sum(term1 * term2, (n, 0, S.Infinity))

def GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu):
    """
    Creates a joint random variable with generalized multivariate log gamma
    distribution.

    The joint pdf can be found at [1].

    Parameters
    ==========

    syms : list/tuple/set of symbols for identifying each component
    delta : A constant in range $[0, 1]$
    v : Positive real number
    lamda : List of positive real numbers
    mu : List of positive real numbers

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma
    >>> from sympy import symbols, S
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> d = S.Half
    >>> y = symbols('y_1:4', positive=True)
    >>> Gd = GeneralizedMultivariateLogGamma('G', d, v, l, mu)
    >>> density(Gd)(y[0], y[1], y[2])
    Sum(exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) - exp(y_2) -
    exp(y_3))/(2**n*gamma(n + 1)**3), (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Note
    ====

    If the GeneralizedMultivariateLogGamma is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGamma as GMVLG
    >>> Gd = GMVLG('G', d, v, l, mu)

    If you want to pass the matrix omega instead of the constant delta, then use
    ``GeneralizedMultivariateLogGammaOmega``.

    """
    return multivariate_rv(GeneralizedMultivariateLogGammaDistribution,
                            syms, delta, v, lamda, mu)

def GeneralizedMultivariateLogGammaOmega(syms, omega, v, lamda, mu):
    """
    Extends GeneralizedMultivariateLogGamma.

    Parameters
    ==========

    syms : list/tuple/set of symbols
        For identifying each component
    omega : A square matrix
           Every element of square matrix must be absolute value of
           square root of correlation coefficient
    v : Positive real number
    lamda : List of positive real numbers
    mu : List of positive real numbers

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density
    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega
    >>> from sympy import Matrix, symbols, S
    >>> omega = Matrix([[1, S.Half, S.Half], [S.Half, 1, S.Half], [S.Half, S.Half, 1]])
    >>> v = 1
    >>> l, mu = [1, 1, 1], [1, 1, 1]
    >>> G = GeneralizedMultivariateLogGammaOmega('G', omega, v, l, mu)
    >>> y = symbols('y_1:4', positive=True)
    >>> density(G)(y[0], y[1], y[2])
    sqrt(2)*Sum((1 - sqrt(2)/2)**n*exp((n + 1)*(y_1 + y_2 + y_3) - exp(y_1) -
    exp(y_2) - exp(y_3))/gamma(n + 1)**3, (n, 0, oo))/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalized_multivariate_log-gamma_distribution
    .. [2] https://www.researchgate.net/publication/234137346_On_a_multivariate_log-gamma_distribution_and_the_use_of_the_distribution_in_the_Bayesian_analysis

    Notes
    =====

    If the GeneralizedMultivariateLogGammaOmega is too long to type use,

    >>> from sympy.stats.joint_rv_types import GeneralizedMultivariateLogGammaOmega as GMVLGO
    >>> G = GMVLGO('G', omega, v, l, mu)

    """
    _value_check((omega.is_square, isinstance(omega, Matrix)), "omega must be a"
                                                            " square matrix")
    for val in omega.values():
        _value_check((val >= 0, val <= 1),
            "all values in matrix must be between 0 and 1(both inclusive).")
    _value_check(omega.diagonal().equals(ones(1, omega.shape[0])),
                    "all the elements of diagonal should be 1.")
    _value_check((omega.shape[0] == len(lamda), len(lamda) == len(mu)),
                    "lamda, mu should be of same length and omega should "
                    " be of shape (length of lamda, length of mu)")
    _value_check(len(lamda) > 1,"the distribution should have at least"
                            " two random variables.")
    delta = Pow(Rational(omega.det()), Rational(1, len(lamda) - 1))
    return GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu)


#-------------------------------------------------------------------------------
# Multinomial distribution -----------------------------------------------------

class MultinomialDistribution(JointDistribution):

    _argnames = ('n', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(n, p):
        _value_check(n > 0,
                        "number of trials must be a positive integer")
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1]")
        _value_check(Eq(sum(p), 1),
                        "probabilities must sum to 1")

    @property
    def set(self):
        return Intersection(S.Naturals0, Interval(0, self.n))**len(self.p)

    def pdf(self, *x):
        n, p = self.n, self.p
        term_1 = factorial(n)/Mul.fromiter(factorial(x_k) for x_k in x)
        term_2 = Mul.fromiter(p_k**x_k for p_k, x_k in zip(p, x))
        return Piecewise((term_1 * term_2, Eq(sum(x), n)), (0, True))

def Multinomial(syms, n, *p):
    """
    Creates a discrete random variable with Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n : Positive integer
        Represents number of trials
    p : List of event probabilities
        Must be in the range of $[0, 1]$.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Multinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> M = Multinomial('M', 3, p1, p2, p3)
    >>> density(M)(x1, x2, x3)
    Piecewise((6*p1**x1*p2**x2*p3**x3/(factorial(x1)*factorial(x2)*factorial(x3)),
    Eq(x1 + x2 + x3, 3)), (0, True))
    >>> marginal_distribution(M, M[0])(x1).subs(x1, 1)
    3*p1*p2**2 + 6*p1*p2*p3 + 3*p1*p3**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multinomial_distribution
    .. [2] https://mathworld.wolfram.com/MultinomialDistribution.html

    """
    if not isinstance(p[0], list):
        p = (list(p), )
    return multivariate_rv(MultinomialDistribution, syms, n, p[0])

#-------------------------------------------------------------------------------
# Negative Multinomial Distribution --------------------------------------------

class NegativeMultinomialDistribution(JointDistribution):

    _argnames = ('k0', 'p')
    is_Continuous=False
    is_Discrete = True

    @staticmethod
    def check(k0, p):
        _value_check(k0 > 0,
                        "number of failures must be a positive integer")
        for p_k in p:
            _value_check((p_k >= 0, p_k <= 1),
                        "probability must be in range [0, 1].")
        _value_check(sum(p) <= 1,
                        "success probabilities must not be greater than 1.")

    @property
    def set(self):
        return Range(0, S.Infinity)**len(self.p)

    def pdf(self, *k):
        k0, p = self.k0, self.p
        term_1 = (gamma(k0 + sum(k))*(1 - sum(p))**k0)/gamma(k0)
        term_2 = Mul.fromiter(pi**ki/factorial(ki) for pi, ki in zip(p, k))
        return term_1 * term_2

def NegativeMultinomial(syms, k0, *p):
    """
    Creates a discrete random variable with Negative Multinomial Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    k0 : positive integer
        Represents number of failures before the experiment is stopped
    p : List of event probabilities
        Must be in the range of $[0, 1]$

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, NegativeMultinomial, marginal_distribution
    >>> from sympy import symbols
    >>> x1, x2, x3 = symbols('x1, x2, x3', nonnegative=True, integer=True)
    >>> p1, p2, p3 = symbols('p1, p2, p3', positive=True)
    >>> N = NegativeMultinomial('M', 3, p1, p2, p3)
    >>> N_c = NegativeMultinomial('M', 3, 0.1, 0.1, 0.1)
    >>> density(N)(x1, x2, x3)
    p1**x1*p2**x2*p3**x3*(-p1 - p2 - p3 + 1)**3*gamma(x1 + x2 +
    x3 + 3)/(2*factorial(x1)*factorial(x2)*factorial(x3))
    >>> marginal_distribution(N_c, N_c[0])(1).evalf().round(2)
    0.25


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_multinomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    if not isinstance(p[0], list):
        p = (list(p), )
    return multivariate_rv(NegativeMultinomialDistribution, syms, k0, p[0])
<|endoftext|>"""

Contains
========
FlorySchulz
Geometric
Hermite
Logarithmic
NegativeBinomial
Poisson
Skellam
YuleSimon
Zeta
"""



from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial, FallingFactorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import (polylog, zeta)
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random


__all__ = ['FlorySchulz',
'Geometric',
'Hermite',
'Logarithmic',
'NegativeBinomial',
'Poisson',
'Skellam',
'YuleSimon',
'Zeta'
]


def rv(symbol, cls, *args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleDiscretePSpace(symbol, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value


class DiscreteDistributionHandmade(SingleDiscreteDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=S.Integers):
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        x = Dummy('x')
        val = Sum(pdf(x), (x, set._inf, set._sup)).doit()
        _value_check(Eq(val, 1) != S.false, "The pdf is incorrect on the given set.")



def DiscreteRV(symbol, density, set=S.Integers, **kwargs):
    """
    Create a Discrete Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    density : Expression containing symbol
        Represents probability density function.
    set : set
        Represents the region where the pdf is valid, by default is real line.
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import DiscreteRV, P, E
    >>> from sympy import Rational, Symbol
    >>> x = Symbol('x')
    >>> n = 10
    >>> density = Rational(1, 10)
    >>> X = DiscreteRV(x, density, set=set(range(n)))
    >>> E(X)
    9/2
    >>> P(X>3)
    3/5

    Returns
    =======

    RandomSymbol

    """
    set = sympify(set)
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, DiscreteDistributionHandmade, pdf, set, **kwargs)


#-------------------------------------------------------------------------------
# Flory-Schulz distribution ------------------------------------------------------------

class FlorySchulzDistribution(SingleDiscreteDistribution):
    _argnames = ('a',)
    set = S.Naturals

    @staticmethod
    def check(a):
        _value_check((0 < a, a < 1), "a must be between 0 and 1")

    def pdf(self, k):
        a = self.a
        return (a**2 * k * (1 - a)**(k - 1))

    def _characteristic_function(self, t):
        a = self.a
        return a**2*exp(I*t)/((1 + (a - 1)*exp(I*t))**2)

    def _moment_generating_function(self, t):
        a = self.a
        return a**2*exp(t)/((1 + (a - 1)*exp(t))**2)


def FlorySchulz(name, a):
    r"""
    Create a discrete random variable with a FlorySchulz distribution.

    The density of the FlorySchulz distribution is given by

    .. math::
        f(k) := (a^2) k (1 - a)^{k-1}

    Parameters
    ==========

    a : A real number between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, E, variance, FlorySchulz
