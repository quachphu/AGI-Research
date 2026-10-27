        grp = []

        for i in reals:
            if len(grp) == 0:
                grp.append([i])
                continue
            for j in grp:
                if int_valued(j[0] - i):
                    j.append(i)
                    break
            else:
                grp.append([i])

        # True if none of the roots differ by an integer i.e.
        # each element in group have only one member
        independent = all(len(i) == 1 for i in grp)

        allpos = all(i >= 0 for i in reals)
        allint = all(int_valued(i) for i in reals)

        # if initial conditions are provided
        # then use them.
        if self.is_singularics() == True:
            rootstoconsider = []
            for i in ordered(self.y0.keys()):
                for j in ordered(indicialroots.keys()):
                    if equal_valued(j, i):
                        rootstoconsider.append(i)

        elif allpos and allint:
            rootstoconsider = [min(reals)]

        elif independent:
            rootstoconsider = [i[0] for i in grp] + [j[0] for j in compl]

        elif not allint:
            rootstoconsider = [i for i in reals if not int(i) == i]

        elif not allpos:

            if not self._have_init_cond() or S(self.y0[0]).is_finite ==  False:
                rootstoconsider = [min(reals)]

            else:
                posroots = [i for i in reals if i >= 0]
                rootstoconsider = [min(posroots)]

        n = Symbol('n', integer=True)
        dom = self.annihilator.parent.base.dom
        R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')

        finalsol = []
        char = ord('C')

        for p in rootstoconsider:
            dict1 = {}

            for i, j in enumerate(self.annihilator.listofpoly):

                listofdmp = j.all_coeffs()
                degree = len(listofdmp) - 1

                for k in range(degree + 1):
                    coeff = listofdmp[degree - k]

                    if coeff == 0:
                        continue

                    if (i - k, k - i) in dict1:
                        dict1[(i - k, k - i)] += (dom.to_sympy(coeff) * rf(n - k + 1 + p, i))
                    else:
                        dict1[(i - k, k - i)] = (dom.to_sympy(coeff) * rf(n - k + 1 + p, i))

            sol = []
            keylist = [i[0] for i in dict1]
            lower = min(keylist)
            upper = max(keylist)
            degree = max(i[1] for i in dict1)
            degree2 = min(i[1] for i in dict1)

            smallest_n = lower + degree
            dummys = {}
            eqs = []
            unknowns = []

            for j in range(lower, upper + 1):
                if j in keylist:
                    temp = sum((v.subs(n, n - lower)
                               for k, v in dict1.items() if k[0] == j),
                               start=S.Zero)
                    sol.append(temp)
                else:
                    sol.append(S.Zero)

            # the recurrence relation
            sol = RecurrenceOperator(sol, R)

            # computing the initial conditions for recurrence
            order = sol.order
            all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
            all_roots = all_roots.keys()

            if all_roots:
                max_root = max(all_roots) + 1
                smallest_n = max(max_root, smallest_n)
            order += smallest_n

            u0 = []

            if self.is_singularics() == True:
                u0 = self.y0[p]

            elif self.is_singularics() == False and p >= 0 and int(p) == p and len(rootstoconsider) == 1:
                y0 = _extend_y0(self, order + int(p))
                # u(n) = y^n(0)/factorial(n)
                if len(y0) > int(p):
                    u0 = [y0[i] / factorial(i) for i in range(int(p), len(y0))]

            if len(u0) < order:

                for i in range(degree2, degree):
                    eq = S.Zero

                    for j in dict1:
                        if i + j[0] < 0:
                            dummys[i + j[0]] = S.Zero

                        elif i + j[0] < len(u0):
                            dummys[i + j[0]] = u0[i + j[0]]

                        elif not i + j[0] in dummys:
                            letter = chr(char) + '_%s' %(i + j[0])
                            dummys[i + j[0]] = Symbol(letter)
                            unknowns.append(dummys[i + j[0]])

                        if j[1] <= i:
                            eq += dict1[j].subs(n, i) * dummys[i + j[0]]

                    eqs.append(eq)

                # solve the system of equations formed
                soleqs = solve(eqs, *unknowns)

                if isinstance(soleqs, dict):

                    for i in range(len(u0), order):

                        if i not in dummys:
                            letter = chr(char) + '_%s' %i
                            dummys[i] = Symbol(letter)

                        if dummys[i] in soleqs:
                            u0.append(soleqs[dummys[i]])

                        else:
                            u0.append(dummys[i])

                    if lb:
                        finalsol.append((HolonomicSequence(sol, u0), p, smallest_n))
                        continue
                    else:
                        finalsol.append((HolonomicSequence(sol, u0), p))
                        continue

                for i in range(len(u0), order):

                    if i not in dummys:
                        letter = chr(char) + '_%s' %i
                        dummys[i] = Symbol(letter)

                    s = False
                    for j in soleqs:
                        if dummys[i] in j:
                            u0.append(j[dummys[i]])
                            s = True
                    if not s:
                        u0.append(dummys[i])
            if lb:
                finalsol.append((HolonomicSequence(sol, u0), p, smallest_n))

            else:
                finalsol.append((HolonomicSequence(sol, u0), p))
            char += 1
        return finalsol

    def series(self, n=6, coefficient=False, order=True, _recur=None):
        r"""
        Finds the power series expansion of given holonomic function about :math:`x_0`.

        Explanation
        ===========

        A list of series might be returned if :math:`x_0` is a regular point with
        multiple roots of the indicial equation.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).series()  # e^x
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).series(n=8)  # sin(x)
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**8)

        See Also
        ========

        HolonomicFunction.to_sequence
        """

        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            constantpower = recurrence[1]
            recurrence = recurrence[0]

        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            return [self.series(_recur=i) for i in recurrence]

        n = n - int(constantpower)
        l = len(recurrence.u0) - 1
        k = recurrence.recurrence.order
        x = self.x
        x0 = self.x0
        seq_dmp = recurrence.recurrence.listofpoly
        R = recurrence.recurrence.parent.base
        K = R.get_field()
        seq = [K.new(j.to_list()) for j in seq_dmp]
        sub = [-seq[i] / seq[k] for i in range(k)]
        sol = list(recurrence.u0)

        if l + 1 < n:
            # use the initial conditions to find the next term
            for i in range(l + 1 - k, n - k):
                coeff = sum((DMFsubs(sub[j], i) * sol[i + j]
                            for j in range(k) if i + j >= 0), start=S.Zero)
                sol.append(coeff)

        if coefficient:
            return sol

        ser = sum((x**(i + constantpower) * j for i, j in enumerate(sol)),
                  start=S.Zero)
        if order:
            ser += Order(x**(n + int(constantpower)), x)
        if x0 != 0:
            return ser.subs(x, x - x0)
        return ser

    def _indicial(self):
        """
        Computes roots of the Indicial equation.
        """

        if self.x0 != 0:
            return self.shift_x(self.x0)._indicial()

        list_coeff = self.annihilator.listofpoly
        R = self.annihilator.parent.base
        x = self.x
        s = R.zero
        y = R.one

        def _pole_degree(poly):
            root_all = roots(R.to_sympy(poly), x, filter='Z')
            if 0 in root_all.keys():
                return root_all[0]
            else:
                return 0

        degree = max(j.degree() for j in list_coeff)
        inf = 10 * (max(1, degree) + max(1, self.annihilator.order))

        deg = lambda q: inf if q.is_zero else _pole_degree(q)
        b = min(deg(q) - j for j, q in enumerate(list_coeff))

        for i, j in enumerate(list_coeff):
            listofdmp = j.all_coeffs()
            degree = len(listofdmp) - 1
            if 0 <= i + b <= degree:
                s = s + listofdmp[degree - i - b] * y
            y *= R.from_sympy(x - i)

        return roots(R.to_sympy(s), x)

    def evalf(self, points, method='RK4', h=0.05, derivatives=False):
        r"""
        Finds numerical value of a holonomic function using numerical methods.
        (RK4 by default). A set of points (real or complex) must be provided
        which will be the path for the numerical integration.

        Explanation
        ===========

        The path should be given as a list :math:`[x_1, x_2, \dots x_n]`. The numerical
        values will be computed at each point in this order
        :math:`x_1 \rightarrow x_2 \rightarrow x_3 \dots \rightarrow x_n`.

        Returns values of the function at :math:`x_1, x_2, \dots x_n` in a list.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

        A straight line on the real axis from (0 to 1)

        >>> r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        Runge-Kutta 4th order on e^x from 0.1 to 1.
        Exact solution at 1 is 2.71828182845905

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r)
        [1.10517083333333, 1.22140257085069, 1.34985849706254, 1.49182424008069,
        1.64872063859684, 1.82211796209193, 2.01375162659678, 2.22553956329232,
        2.45960141378007, 2.71827974413517]

        Euler's method for the same

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r, method='Euler')
        [1.1, 1.21, 1.331, 1.4641, 1.61051, 1.771561, 1.9487171, 2.14358881,
        2.357947691, 2.5937424601]

        One can also observe that the value obtained using Runge-Kutta 4th order
        is much more accurate than Euler's method.
        """

        from sympy.holonomic.numerical import _evalf
        lp = False

        # if a point `b` is given instead of a mesh
        if not hasattr(points, "__iter__"):
            lp = True
            b = S(points)
            if self.x0 == b:
                return _evalf(self, [b], method=method, derivatives=derivatives)[-1]

            if not b.is_Number:
                raise NotImplementedError

            a = self.x0
            if a > b:
                h = -h
            n = int((b - a) / h)
            points = [a + h]
            for i in range(n - 1):
                points.append(points[-1] + h)

        for i in roots(self.annihilator.parent.base.to_sympy(self.annihilator.listofpoly[-1]), self.x):
            if i == self.x0 or i in points:
                raise SingularityError(self, i)

        if lp:
            return _evalf(self, points, method=method, derivatives=derivatives)[-1]
        return _evalf(self, points, method=method, derivatives=derivatives)

    def change_x(self, z):
        """
        Changes only the variable of Holonomic Function, for internal
        purposes. For composition use HolonomicFunction.composition()
        """

        dom = self.annihilator.parent.base.dom
        R = dom.old_poly_ring(z)
        parent, _ = DifferentialOperators(R, 'Dx')
        sol = [R(j.to_list()) for j in self.annihilator.listofpoly]
        sol =  DifferentialOperator(sol, parent)
        return HolonomicFunction(sol, z, self.x0, self.y0)

    def shift_x(self, a):
        """
        Substitute `x + a` for `x`.
        """

        x = self.x
        listaftershift = self.annihilator.listofpoly
        base = self.annihilator.parent.base

        sol = [base.from_sympy(base.to_sympy(i).subs(x, x + a)) for i in listaftershift]
        sol = DifferentialOperator(sol, self.annihilator.parent)
        x0 = self.x0 - a
        if not self._have_init_cond():
            return HolonomicFunction(sol, x)
        return HolonomicFunction(sol, x, x0, self.y0)

    def to_hyper(self, as_list=False, _recur=None):
        r"""
        Returns a hypergeometric function (or linear combination of them)
        representing the given holonomic function.

        Explanation
        ===========

        Returns an answer of the form:
        `a_1 \cdot x^{b_1} \cdot{hyper()} + a_2 \cdot x^{b_2} \cdot{hyper()} \dots`

        This is very useful as one can now use ``hyperexpand`` to find the
        symbolic expressions/functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> # sin(x)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).to_hyper()
        x*hyper((), (3/2,), -x**2/4)
        >>> # exp(x)
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_hyper()
        hyper((), (), x)

        See Also
        ========

        from_hyper, from_meijerg
        """

        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            smallest_n = recurrence[1]
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            smallest_n = recurrence[2]
            constantpower = recurrence[1]
            recurrence = recurrence[0]
        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            smallest_n = recurrence[0][1]
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            smallest_n = recurrence[0][2]
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            sol = self.to_hyper(as_list=as_list, _recur=recurrence[0])
            for i in recurrence[1:]:
                sol += self.to_hyper(as_list=as_list, _recur=i)
            return sol

        u0 = recurrence.u0
        r = recurrence.recurrence
        x = self.x
        x0 = self.x0

        # order of the recurrence relation
        m = r.order

        # when no recurrence exists, and the power series have finite terms
        if m == 0:
            nonzeroterms = roots(r.parent.base.to_sympy(r.listofpoly[0]), recurrence.n, filter='R')

            sol = S.Zero
            for j, i in enumerate(nonzeroterms):

                if i < 0 or not int_valued(i):
                    continue

                i = int(i)
                if i < len(u0):
                    if isinstance(u0[i], (PolyElement, FracElement)):
                        u0[i] = u0[i].as_expr()
                    sol += u0[i] * x**i

                else:
                    sol += Symbol('C_%s' %j) * x**i

            if isinstance(sol, (PolyElement, FracElement)):
                sol = sol.as_expr() * x**constantpower
            else:
                sol = sol * x**constantpower
            if as_list:
                if x0 != 0:
                    return [(sol.subs(x, x - x0), )]
                return [(sol, )]
            if x0 != 0:
                return sol.subs(x, x - x0)
            return sol

        if smallest_n + m > len(u0):
            raise NotImplementedError("Can't compute sufficient Initial Conditions")

        # check if the recurrence represents a hypergeometric series
        if any(i != r.parent.base.zero for i in r.listofpoly[1:-1]):
            raise NotHyperSeriesError(self, self.x0)

        a = r.listofpoly[0]
        b = r.listofpoly[-1]

        # the constant multiple of argument of hypergeometric function
        if isinstance(a.LC(), (PolyElement, FracElement)):
            c = - (S(a.LC().as_expr()) * m**(a.degree())) / (S(b.LC().as_expr()) * m**(b.degree()))
        else:
            c = - (S(a.LC()) * m**(a.degree())) / (S(b.LC()) * m**(b.degree()))

        sol = 0

        arg1 = roots(r.parent.base.to_sympy(a), recurrence.n)
        arg2 = roots(r.parent.base.to_sympy(b), recurrence.n)

        # iterate through the initial conditions to find
        # the hypergeometric representation of the given
        # function.
        # The answer will be a linear combination
        # of different hypergeometric series which satisfies
        # the recurrence.
        if as_list:
            listofsol = []
        for i in range(smallest_n + m):

            # if the recurrence relation doesn't hold for `n = i`,
            # then a Hypergeometric representation doesn't exist.
            # add the algebraic term a * x**i to the solution,
            # where a is u0[i]
            if i < smallest_n:
                if as_list:
                    listofsol.append(((S(u0[i]) * x**(i+constantpower)).subs(x, x-x0), ))
                else:
                    sol += S(u0[i]) * x**i
                continue

            # if the coefficient u0[i] is zero, then the
            # independent hypergeomtric series starting with
            # x**i is not a part of the answer.
            if S(u0[i]) == 0:
                continue

            ap = []
            bq = []

            # substitute m * n + i for n
            for k in ordered(arg1.keys()):
                ap.extend([nsimplify((i - k) / m)] * arg1[k])

            for k in ordered(arg2.keys()):
                bq.extend([nsimplify((i - k) / m)] * arg2[k])

            # convention of (k + 1) in the denominator
            if 1 in bq:
                bq.remove(1)
            else:
                ap.append(1)
            if as_list:
                listofsol.append(((S(u0[i])*x**(i+constantpower)).subs(x, x-x0), (hyper(ap, bq, c*x**m)).subs(x, x-x0)))
            else:
                sol += S(u0[i]) * hyper(ap, bq, c * x**m) * x**i
        if as_list:
            return listofsol
        sol = sol * x**constantpower
        if x0 != 0:
            return sol.subs(x, x - x0)

        return sol

    def to_expr(self):
        """
        Converts a Holonomic Function back to elementary functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(x**2*Dx**2 + x*Dx + (x**2 - 1), x, 0, [0, S(1)/2]).to_expr()
        besselj(1, x)
        >>> HolonomicFunction((1 + x)*Dx**3 + Dx**2, x, 0, [1, 1, 1]).to_expr()
        x*log(x + 1) + log(x + 1) + 1

        """

        return hyperexpand(self.to_hyper()).simplify()

    def change_ics(self, b, lenics=None):
        """
        Changes the point `x0` to ``b`` for initial conditions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import symbols, sin, exp
        >>> x = symbols('x')

        >>> expr_to_holonomic(sin(x)).change_ics(1)
        HolonomicFunction((1) + (1)*Dx**2, x, 1, [sin(1), cos(1)])

        >>> expr_to_holonomic(exp(x)).change_ics(2)
        HolonomicFunction((-1) + (1)*Dx, x, 2, [exp(2)])
        """

        symbolic = True

        if lenics is None and len(self.y0) > self.annihilator.order:
            lenics = len(self.y0)
        dom = self.annihilator.parent.base.domain

        try:
            sol = expr_to_holonomic(self.to_expr(), x=self.x, x0=b, lenics=lenics, domain=dom)
        except (NotPowerSeriesError, NotHyperSeriesError):
            symbolic = False

        if symbolic and sol.x0 == b:
            return sol

        y0 = self.evalf(b, derivatives=True)
        return HolonomicFunction(self.annihilator, self.x, b, y0)

    def to_meijerg(self):
        """
        Returns a linear combination of Meijer G-functions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import sin, cos, hyperexpand, log, symbols
        >>> x = symbols('x')
        >>> hyperexpand(expr_to_holonomic(cos(x) + sin(x)).to_meijerg())
        sin(x) + cos(x)
        >>> hyperexpand(expr_to_holonomic(log(x)).to_meijerg()).simplify()
        log(x)

        See Also
        ========

        to_hyper
        """

        # convert to hypergeometric first
        rep = self.to_hyper(as_list=True)
        sol = S.Zero

        for i in rep:
            if len(i) == 1:
                sol += i[0]

            elif len(i) == 2:
                sol += i[0] * _hyper_to_meijerg(i[1])

        return sol


def from_hyper(func, x0=0, evalf=False):
    r"""
    Converts a hypergeometric function to holonomic.
    ``func`` is the Hypergeometric Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_hyper
    >>> from sympy import symbols, hyper, S
    >>> x = symbols('x')
    >>> from_hyper(hyper([], [S(3)/2], x**2/4))
    HolonomicFunction((-x) + (2)*Dx + (x)*Dx**2, x, 1, [sinh(1), -sinh(1) + cosh(1)])
    """

    a = func.ap
    b = func.bq
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # generalized hypergeometric differential equation
    xDx = x*Dx
    r1 = 1
    for ai in a:  # XXX gives sympify error if Mul is used with list of all factors
        r1 *= xDx + ai
    xDx_1 = xDx - 1
    # r2 = Mul(*([Dx] + [xDx_1 + bi for bi in b]))  # XXX gives sympify error
    r2 = Dx
    for bi in b:
        r2 *= xDx_1 + bi
    sol = r1 - r2

    simp = hyperexpand(func)

    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # if the function is known symbolically
    if not isinstance(simp, hyper):
        y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)
        while not y0:
            # if values don't exist at 0, then try to find initial
            # conditions at 1. If it doesn't exist at 1 too then
            # try 2 and so on.
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    if isinstance(simp, hyper):
        x0 = 1
        # use evalf if the function can't be simplified
        y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    return HolonomicFunction(sol, x).composition(z)


def from_meijerg(func, x0=0, evalf=False, initcond=True, domain=QQ):
    """
    Converts a Meijer G-function to Holonomic.
    ``func`` is the G-Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_meijerg
    >>> from sympy import symbols, meijerg, S
    >>> x = symbols('x')
    >>> from_meijerg(meijerg(([], []), ([S(1)/2], [0]), x**2/4))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1/sqrt(pi)])
    """

    a = func.ap
    b = func.bq
    n = len(func.an)
    m = len(func.bm)
    p = len(a)
    z = func.args[2]
    x = z.atoms(Symbol).pop()
    R, Dx = DifferentialOperators(domain.old_poly_ring(x), 'Dx')

    # compute the differential equation satisfied by the
    # Meijer G-function.
    xDx = x*Dx
    xDx1 = xDx + 1
    r1 = x*(-1)**(m + n - p)
    for ai in a:  # XXX gives sympify error if args given in list
        r1 *= xDx1 - ai
    # r2 = Mul(*[xDx - bi for bi in b])  # gives sympify error
    r2 = 1
    for bi in b:
        r2 *= xDx - bi
    sol = r1 - r2

    if not initcond:
        return HolonomicFunction(sol, x).composition(z)

    simp = hyperexpand(func)

    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # computing initial conditions
    if not isinstance(simp, meijerg):
        y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    if isinstance(simp, meijerg):
        x0 = 1
        y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf, use_limit=False)

        return HolonomicFunction(sol, x).composition(z, x0, y0)

    return HolonomicFunction(sol, x).composition(z)


x_1 = Dummy('x_1')
_lookup_table = None
domain_for_table = None
from sympy.integrals.meijerint import _mytype


def expr_to_holonomic(func, x=None, x0=0, y0=None, lenics=None, domain=None, initcond=True):
    """
    Converts a function or an expression to a holonomic function.

    Parameters
    ==========

    func:
        The expression to be converted.
    x:
        variable for the function.
    x0:
        point at which initial condition must be computed.
    y0:
        One can optionally provide initial condition if the method
        is not able to do it automatically.
    lenics:
        Number of terms in the initial condition. By default it is
        equal to the order of the annihilator.
    domain:
        Ground domain for the polynomials in ``x`` appearing as coefficients
        in the annihilator.
    initcond:
        Set it false if you do not want the initial conditions to be computed.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import expr_to_holonomic
    >>> from sympy import sin, exp, symbols
    >>> x = symbols('x')
    >>> expr_to_holonomic(sin(x))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1])
    >>> expr_to_holonomic(exp(x))
    HolonomicFunction((-1) + (1)*Dx, x, 0, [1])

    See Also
    ========

    sympy.integrals.meijerint._rewrite1, _convert_poly_rat_alg, _create_table
    """
    func = sympify(func)
    syms = func.free_symbols

    if not x:
        if len(syms) == 1:
            x= syms.pop()
        else:
            raise ValueError("Specify the variable for the function")
    elif x in syms:
        syms.remove(x)

    extra_syms = list(syms)

    if domain is None:
        if func.has(Float):
            domain = RR
        else:
            domain = QQ
        if len(extra_syms) != 0:
            domain = domain[extra_syms].get_field()

    # try to convert if the function is polynomial or rational
    solpoly = _convert_poly_rat_alg(func, x, x0=x0, y0=y0, lenics=lenics, domain=domain, initcond=initcond)
    if solpoly:
        return solpoly

    # create the lookup table
    global _lookup_table, domain_for_table
    if not _lookup_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)
    elif domain != domain_for_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)

    # use the table directly to convert to Holonomic
    if func.is_Function:
        f = func.subs(x, x_1)
        t = _mytype(f, x_1)
        if t in _lookup_table:
            l = _lookup_table[t]
            sol = l[0][1].change_x(x)
        else:
            sol = _convert_meijerint(func, x, initcond=False, domain=domain)
            if not sol:
                raise NotImplementedError
            if y0:
                sol.y0 = y0
            if y0 or not initcond:
                sol.x0 = x0
                return sol
            if not lenics:
                lenics = sol.annihilator.order
            _y0 = _find_conditions(func, x, x0, lenics)
            while not _y0:
                x0 += 1
                _y0 = _find_conditions(func, x, x0, lenics)
            return HolonomicFunction(sol.annihilator, x, x0, _y0)

        if y0 or not initcond:
            sol = sol.composition(func.args[0])
            if y0:
                sol.y0 = y0
            sol.x0 = x0
            return sol
        if not lenics:
            lenics = sol.annihilator.order

        _y0 = _find_conditions(func, x, x0, lenics)
        while not _y0:
            x0 += 1
            _y0 = _find_conditions(func, x, x0, lenics)
        return sol.composition(func.args[0], x0, _y0)

    # iterate through the expression recursively
    args = func.args
    f = func.func
    sol = expr_to_holonomic(args[0], x=x, initcond=False, domain=domain)

    if f is Add:
        for i in range(1, len(args)):
            sol += expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    elif f is Mul:
        for i in range(1, len(args)):
            sol *= expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    elif f is Pow:
        sol = sol**args[1]
    sol.x0 = x0
    if not sol:
        raise NotImplementedError
    if y0:
        sol.y0 = y0
    if y0 or not initcond:
        return sol
    if sol.y0:
        return sol
    if not lenics:
        lenics = sol.annihilator.order
    if sol.annihilator.is_singular(x0):
        r = sol._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            return HolonomicFunction(sol.annihilator, x, x0, y0)

    _y0 = _find_conditions(func, x, x0, lenics)
    while not _y0:
        x0 += 1
        _y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol.annihilator, x, x0, _y0)


## Some helper functions ##

def _normalize(list_of, parent, negative=True):
    """
    Normalize a given annihilator
    """

    num = []
    denom = []
    base = parent.base
    K = base.get_field()
    lcm_denom = base.from_sympy(S.One)
    list_of_coeff = []

    # convert polynomials to the elements of associated
    # fraction field
    for i, j in enumerate(list_of):
        if isinstance(j, base.dtype):
            list_of_coeff.append(K.new(j.to_list()))
        elif not isinstance(j, K.dtype):
            list_of_coeff.append(K.from_sympy(sympify(j)))
        else:
            list_of_coeff.append(j)

        # corresponding numerators of the sequence of polynomials
        num.append(list_of_coeff[i].numer())

        # corresponding denominators
        denom.append(list_of_coeff[i].denom())

    # lcm of denominators in the coefficients
    for i in denom:
        lcm_denom = i.lcm(lcm_denom)

    if negative:
        lcm_denom = -lcm_denom

    lcm_denom = K.new(lcm_denom.to_list())

    # multiply the coefficients with lcm
    for i, j in enumerate(list_of_coeff):
        list_of_coeff[i] = j * lcm_denom

    gcd_numer = base((list_of_coeff[-1].numer() / list_of_coeff[-1].denom()).to_list())

    # gcd of numerators in the coefficients
    for i in num:
        gcd_numer = i.gcd(gcd_numer)

    gcd_numer = K.new(gcd_numer.to_list())

    # divide all the coefficients by the gcd
    for i, j in enumerate(list_of_coeff):
        frac_ans = j / gcd_numer
        list_of_coeff[i] = base((frac_ans.numer() / frac_ans.denom()).to_list())

    return DifferentialOperator(list_of_coeff, parent)


def _derivate_diff_eq(listofpoly, K):
    """
    Let a differential equation a0(x)y(x) + a1(x)y'(x) + ... = 0
    where a0, a1,... are polynomials or rational functions. The function
    returns b0, b1, b2... such that the differential equation
    b0(x)y(x) + b1(x)y'(x) +... = 0 is formed after differentiating the
    former equation.
    """

    sol = []
    a = len(listofpoly) - 1
    sol.append(DMFdiff(listofpoly[0], K))

    for i, j in enumerate(listofpoly[1:]):
        sol.append(DMFdiff(j, K) + listofpoly[i])

    sol.append(listofpoly[a])
    return sol


def _hyper_to_meijerg(func):
    """
    Converts a `hyper` to meijerg.
    """
    ap = func.ap
    bq = func.bq

    if any(i <= 0 and int(i) == i for i in ap):
        return hyperexpand(func)

    z = func.args[2]

    # parameters of the `meijerg` function.
    an = (1 - i for i in ap)
    anp = ()
    bm = (S.Zero, )
    bmq = (1 - i for i in bq)

    k = S.One

    for i in bq:
        k = k * gamma(i)

    for i in ap:
        k = k / gamma(i)

    return k * meijerg(an, anp, bm, bmq, -z)


def _add_lists(list1, list2):
    """Takes polynomial sequences of two annihilators a and b and returns
    the list of polynomials of sum of a and b.
    """
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for a, b in zip(list1, list2)] + list1[len(list2):]
    return sol


def _extend_y0(Holonomic, n):
    """
    Tries to find more initial conditions by substituting the initial
    value point in the differential equation.
    """

    if Holonomic.annihilator.is_singular(Holonomic.x0) or Holonomic.is_singularics() == True:
        return Holonomic.y0

    annihilator = Holonomic.annihilator
    a = annihilator.order

    listofpoly = []

    y0 = Holonomic.y0
    R = annihilator.parent.base
    K = R.get_field()

    for j in annihilator.listofpoly:
        if isinstance(j, annihilator.parent.base.dtype):
            listofpoly.append(K.new(j.to_list()))

    if len(y0) < a or n <= len(y0):
        return y0
    list_red = [-listofpoly[i] / listofpoly[a]
                for i in range(a)]
    y1 = y0[:min(len(y0), a)]
    for _ in range(n - a):
        sol = 0
        for a, b in zip(y1, list_red):
            r = DMFsubs(b, Holonomic.x0)
            if not getattr(r, 'is_finite', True):
                return y0
            if isinstance(r, (PolyElement, FracElement)):
                r = r.as_expr()
            sol += a * r
        y1.append(sol)
        list_red = _derivate_diff_eq(list_red, K)
    return y0 + y1[len(y0):]


def DMFdiff(frac, K):
    # differentiate a DMF object represented as p/q
    if not isinstance(frac, DMF):
        return frac.diff()

    p = K.numer(frac)
    q = K.denom(frac)
    sol_num = - p * q.diff() + q * p.diff()
    sol_denom = q**2
    return K((sol_num.to_list(), sol_denom.to_list()))


def DMFsubs(frac, x0, mpm=False):
    # substitute the point x0 in DMF object of the form p/q
    if not isinstance(frac, DMF):
        return frac

    p = frac.num
    q = frac.den
    sol_p = S.Zero
    sol_q = S.Zero

    if mpm:
        from mpmath import mp

    for i, j in enumerate(reversed(p)):
        if mpm:
            j = sympify(j)._to_mpmath(mp.prec)
        sol_p += j * x0**i

    for i, j in enumerate(reversed(q)):
        if mpm:
            j = sympify(j)._to_mpmath(mp.prec)
        sol_q += j * x0**i

    if isinstance(sol_p, (PolyElement, FracElement)):
        sol_p = sol_p.as_expr()
    if isinstance(sol_q, (PolyElement, FracElement)):
        sol_q = sol_q.as_expr()

    return sol_p / sol_q


def _convert_poly_rat_alg(func, x, x0=0, y0=None, lenics=None, domain=QQ, initcond=True):
    """
    Converts polynomials, rationals and algebraic functions to holonomic.
    """

    ispoly = func.is_polynomial()
    if not ispoly:
        israt = func.is_rational_function()
    else:
        israt = True

    if not (ispoly or israt):
        basepoly, ratexp = func.as_base_exp()
        if basepoly.is_polynomial() and ratexp.is_Number:
            if isinstance(ratexp, Float):
                ratexp = nsimplify(ratexp)
            m, n = ratexp.p, ratexp.q
            is_alg = True
        else:
            is_alg = False
    else:
        is_alg = True

    if not (ispoly or israt or is_alg):
        return None

    R = domain.old_poly_ring(x)
    _, Dx = DifferentialOperators(R, 'Dx')

    # if the function is constant
    if not func.has(x):
        return HolonomicFunction(Dx, x, 0, [func])

    if ispoly:
        # differential equation satisfied by polynomial
        sol = func * Dx - func.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)
        is_singular = sol.is_singular(x0)

        # try to compute the conditions for singular points
        if y0 is None and x0 == 0 and is_singular:
            rep = R.from_sympy(func).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                coeff = list(reversed(rep))[i:]
                indicial = i
                break
            for i, j in enumerate(coeff):
                if isinstance(j, (PolyElement, FracElement)):
                    coeff[i] = j.as_expr()
            y0 = {indicial: S(coeff)}

    elif israt:
        p, q = func.as_numer_denom()
        # differential equation satisfied by rational
        sol = p * q * Dx + p * q.diff(x) - q * p.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)

    elif is_alg:
        sol = n * (x / m) * Dx - 1
        sol = HolonomicFunction(sol, x).composition(basepoly).annihilator
        is_singular = sol.is_singular(x0)

        # try to compute the conditions for singular points
        if y0 is None and x0 == 0 and is_singular and \
            (lenics is None or lenics <= 1):
            rep = R.from_sympy(basepoly).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                if isinstance(j, (PolyElement, FracElement)):
                    j = j.as_expr()

                coeff = S(j)**ratexp
                indicial = S(i) * ratexp
                break
            if isinstance(coeff, (PolyElement, FracElement)):
                coeff = coeff.as_expr()
            y0 = {indicial: S([coeff])}

    if y0 or not initcond:
        return HolonomicFunction(sol, x, x0, y0)

    if not lenics:
        lenics = sol.order

    if sol.is_singular(x0):
        r = HolonomicFunction(sol, x, x0)._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            return HolonomicFunction(sol, x, x0, y0)

    y0 = _find_conditions(func, x, x0, lenics)
    while not y0:
        x0 += 1
        y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol, x, x0, y0)


def _convert_meijerint(func, x, initcond=True, domain=QQ):
    args = meijerint._rewrite1(func, x)

    if args:
        fac, po, g, _ = args
    else:
        return None

    # lists for sum of meijerg functions
    fac_list = [fac * i[0] for i in g]
    t = po.as_base_exp()
    s = t[1] if t[0] == x else S.Zero
    po_list = [s + i[1] for i in g]
    G_list = [i[2] for i in g]

    # finds meijerg representation of x**s * meijerg(a1 ... ap, b1 ... bq, z)
    def _shift(func, s):
        z = func.args[-1]
        if z.has(I):
            z = z.subs(exp_polar, exp)

        d = z.collect(x, evaluate=False)
        b = list(d)[0]
        a = d[b]

        t = b.as_base_exp()
        b = t[1] if t[0] == x else S.Zero
        r = s / b
        an = (i + r for i in func.args[0][0])
        ap = (i + r for i in func.args[0][1])
        bm = (i + r for i in func.args[1][0])
        bq = (i + r for i in func.args[1][1])

        return a**-r, meijerg((an, ap), (bm, bq), z)

    coeff, m = _shift(G_list[0], po_list[0])
    sol = fac_list[0] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    # add all the meijerg functions after converting to holonomic
    for i in range(1, len(G_list)):
        coeff, m = _shift(G_list[i], po_list[i])
        sol += fac_list[i] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    return sol


def _create_table(table, domain=QQ):
    """
    Creates the look-up table. For a similar implementation
    see meijerint._create_lookup_table.
    """

    def add(formula, annihilator, arg, x0=0, y0=()):
        """
        Adds a formula in the dictionary
        """
        table.setdefault(_mytype(formula, x_1), []).append((formula,
            HolonomicFunction(annihilator, arg, x0, y0)))

    R = domain.old_poly_ring(x_1)
    _, Dx = DifferentialOperators(R, 'Dx')

    # add some basic functions
    add(sin(x_1), Dx**2 + 1, x_1, 0, [0, 1])
    add(cos(x_1), Dx**2 + 1, x_1, 0, [1, 0])
    add(exp(x_1), Dx - 1, x_1, 0, 1)
    add(log(x_1), Dx + x_1*Dx**2, x_1, 1, [0, 1])

    add(erf(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])
    add(erfc(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [1, -2/sqrt(pi)])
    add(erfi(x_1), -2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])

    add(sinh(x_1), Dx**2 - 1, x_1, 0, [0, 1])
    add(cosh(x_1), Dx**2 - 1, x_1, 0, [1, 0])

    add(sinc(x_1), x_1 + 2*Dx + x_1*Dx**2, x_1)

    add(Si(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)
    add(Ci(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)

    add(Shi(x_1), -x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)


def _find_conditions(func, x, x0, order, evalf=False, use_limit=True):
    y0 = []
    for _ in range(order):
        val = func.subs(x, x0)
        if evalf:
            val = val.evalf()
        if use_limit and isinstance(val, NaN):
            val = limit(func, x, x0)
        if val.is_finite is False or isinstance(val, NaN):
            return None
        y0.append(val)
        func = func.diff(x)
    return y0
<|endoftext|>r"""
The :py:mod:`~sympy.holonomic` module is intended to deal with holonomic functions along
with various operations on them like addition, multiplication, composition,
integration and differentiation. The module also implements various kinds of
conversions such as converting holonomic functions to a different form and the
other way around.
"""

from .holonomic import (DifferentialOperator, HolonomicFunction, DifferentialOperators,
    from_hyper, from_meijerg, expr_to_holonomic)
from .recurrence import RecurrenceOperators, RecurrenceOperator, HolonomicSequence

__all__ = [
    'DifferentialOperator', 'HolonomicFunction', 'DifferentialOperators',
    'from_hyper', 'from_meijerg', 'expr_to_holonomic',

    'RecurrenceOperators', 'RecurrenceOperator', 'HolonomicSequence',
]
<|endoftext|>"""Recurrence Operators"""

from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify


def RecurrenceOperators(base, generator):
    """
    Returns an Algebra of Recurrence Operators and the operator for
    shifting i.e. the `Sn` operator.
    The first argument needs to be the base polynomial ring for the algebra
    and the second argument must be a generator which can be either a
    noncommutative Symbol or a string.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    """

    ring = RecurrenceOperatorAlgebra(base, generator)
    return (ring, ring.shift_operator)


class RecurrenceOperatorAlgebra:
    """
    A Recurrence Operator Algebra is a set of noncommutative polynomials
    in intermediate `Sn` and coefficients in a base ring A. It follows the
    commutation rule:
    Sn * a(n) = a(n + 1) * Sn

    This class represents a Recurrence Operator Algebra and serves as the parent ring
    for Recurrence Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    >>> R
    Univariate Recurrence Operator Algebra in intermediate Sn over the base ring
    ZZ[n]

    See Also
    ========

    RecurrenceOperator
    """

    def __init__(self, base, generator):
        # the base ring for the algebra
        self.base = base
        # the operator representing shift i.e. `Sn`
        self.shift_operator = RecurrenceOperator(
            [base.zero, base.one], self)

        if generator is None:
            self.gen_symbol = symbols('Sn', commutative=False)
        else:
            if isinstance(generator, str):
                self.gen_symbol = symbols(generator, commutative=False)
            elif isinstance(generator, Symbol):
                self.gen_symbol = generator

    def __str__(self):
        string = 'Univariate Recurrence Operator Algebra in intermediate '\
            + sstr(self.gen_symbol) + ' over the base ring ' + \
            (self.base).__str__()

        return string

    __repr__ = __str__

    def __eq__(self, other):
        if self.base == other.base and self.gen_symbol == other.gen_symbol:
            return True
        else:
            return False


def _add_lists(list1, list2):
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for a, b in zip(list1, list2)] + list1[len(list2):]
    return sol


class RecurrenceOperator:
    """
    The Recurrence Operators are defined by a list of polynomials
    in the base ring and the parent ring of the Operator.

    Explanation
    ===========

    Takes a list of polynomials for each power of Sn and the
    parent ring which must be an instance of RecurrenceOperatorAlgebra.

    A Recurrence Operator can be created easily using
    the operator `Sn`. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.recurrence import RecurrenceOperator, RecurrenceOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n),'Sn')

    >>> RecurrenceOperator([0, 1, n**2], R)
    (1)Sn + (n**2)Sn**2

    >>> Sn*n
    (n + 1)Sn

    >>> n*Sn*n + 1 - Sn**2*n
    (1) + (n**2 + n)Sn + (-n - 2)Sn**2

    See Also
    ========

    DifferentialOperatorAlgebra
    """

    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        # the parent ring for this operator
        # must be an RecurrenceOperatorAlgebra object
        self.parent = parent
        # sequence of polynomials in n for each power of Sn
        # represents the operator
        # convert the expressions into ring elements using from_sympy
        if isinstance(list_of_poly, list):
            for i, j in enumerate(list_of_poly):
                if isinstance(j, int):
                    list_of_poly[i] = self.parent.base.from_sympy(S(j))
                elif not isinstance(j, self.parent.base.dtype):
                    list_of_poly[i] = self.parent.base.from_sympy(j)

            self.listofpoly = list_of_poly
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        """
        Multiplies two Operators and returns another
        RecurrenceOperator instance using the commutation rule
        Sn * a(n) = a(n + 1) * Sn
        """

        listofself = self.listofpoly
        base = self.parent.base

        if not isinstance(other, RecurrenceOperator):
            if not isinstance(other, self.parent.base.dtype):
                listofother = [self.parent.base.from_sympy(sympify(other))]

            else:
                listofother = [other]
        else:
            listofother = other.listofpoly
        # multiply a polynomial `b` with a list of polynomials

        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]

        sol = _mul_dmp_diffop(listofself[0], listofother)

        # compute Sn^i * b
        def _mul_Sni_b(b):
            sol = [base.zero]

            if isinstance(b, list):
                for i in b:
                    j = base.to_sympy(i).subs(base.gens[0], base.gens[0] + S.One)
                    sol.append(base.from_sympy(j))

            else:
                j = b.subs(base.gens[0], base.gens[0] + S.One)
                sol.append(base.from_sympy(j))

            return sol

        for i in range(1, len(listofself)):
            # find Sn^i * b in ith iteration
            listofother = _mul_Sni_b(listofother)
            # solution = solution + listofself[i] * (Sn^i * b)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))

        return RecurrenceOperator(sol, self.parent)

    def __rmul__(self, other):
        if not isinstance(other, RecurrenceOperator):

            if isinstance(other, int):
                other = S(other)

            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(other)

            sol = [other * j for j in self.listofpoly]
            return RecurrenceOperator(sol, self.parent)

    def __add__(self, other):
        if isinstance(other, RecurrenceOperator):

            sol = _add_lists(self.listofpoly, other.listofpoly)
            return RecurrenceOperator(sol, self.parent)

        else:

            if isinstance(other, int):
                other = S(other)
            list_self = self.listofpoly
            if not isinstance(other, self.parent.base.dtype):
                list_other = [((self.parent).base).from_sympy(other)]
            else:
                list_other = [other]
            sol = [list_self[0] + list_other[0]] + list_self[1:]

            return RecurrenceOperator(sol, self.parent)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __pow__(self, n):
        if n == 1:
            return self
        result = RecurrenceOperator([self.parent.base.one], self.parent)
        if n == 0:
            return result
        # if self is `Sn`
        if self.listofpoly == self.parent.shift_operator.listofpoly:
            sol = [self.parent.base.zero] * n + [self.parent.base.one]
            return RecurrenceOperator(sol, self.parent)
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __str__(self):
        listofpoly = self.listofpoly
        print_str = ''

        for i, j in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue

            j = self.parent.base.to_sympy(j)

            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue

            if print_str:
                print_str += ' + '

            if i == 1:
                print_str += '(' + sstr(j) + ')Sn'
                continue

            print_str += '(' + sstr(j) + ')' + 'Sn**' + sstr(i)

        return print_str

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, RecurrenceOperator):
            if self.listofpoly == other.listofpoly and self.parent == other.parent:
                return True
            else:
                return False
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])


class HolonomicSequence:
    """
    A Holonomic Sequence is a type of sequence satisfying a linear homogeneous
    recurrence relation with Polynomial coefficients. Alternatively, A sequence
    is Holonomic if and only if its generating function is a Holonomic Function.
    """

    def __init__(self, recurrence, u0=[]):
        self.recurrence = recurrence
        if not isinstance(u0, list):
            self.u0 = [u0]
        else:
            self.u0 = u0

        if len(self.u0) == 0:
            self._have_init_cond = False
        else:
            self._have_init_cond = True
        self.n = recurrence.parent.base.gens[0]

    def __repr__(self):
        str_sol = 'HolonomicSequence(%s, %s)' % ((self.recurrence).__repr__(), sstr(self.n))
        if not self._have_init_cond:
            return str_sol
        else:
            cond_str = ''
            seq_str = 0
            for i in self.u0:
                cond_str += ', u(%s) = %s' % (sstr(seq_str), sstr(i))
                seq_str += 1

            sol = str_sol + cond_str
            return sol

    __str__ = __repr__

    def __eq__(self, other):
        if self.recurrence != other.recurrence or self.n != other.n:
            return False
        if self._have_init_cond and other._have_init_cond:
            return self.u0 == other.u0
        return True
<|endoftext|>"""Numerical Methods for Holonomic Functions"""

from sympy.core.sympify import sympify
from sympy.holonomic.holonomic import DMFsubs

from mpmath import mp


def _evalf(func, points, derivatives=False, method='RK4'):
    """
    Numerical methods for numerical integration along a given set of
    points in the complex plane.
    """

    ann = func.annihilator
    a = ann.order
    R = ann.parent.base
    K = R.get_field()

    if method == 'Euler':
        meth = _euler
    else:
        meth = _rk4

    dmf = [K.new(j.to_list()) for j in ann.listofpoly]
    red = [-dmf[i] / dmf[a] for i in range(a)]

    y0 = func.y0
    if len(y0) < a:
        raise TypeError("Not Enough Initial Conditions")
    x0 = func.x0
    sol = [meth(red, x0, points[0], y0, a)]

    for i, j in enumerate(points[1:]):
        sol.append(meth(red, points[i], j, sol[-1], a))

    if not derivatives:
        return [sympify(i[0]) for i in sol]
    else:
        return sympify(sol)


def _euler(red, x0, x1, y0, a):
    """
    Euler's method for numerical integration.
    From x0 to x1 with initial values given at x0 as vector y0.
    """

    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A
    f_0 = y_0[1:]
    f_0_n = 0

    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    return [y_0[i] + h * f_0[i] for i in range(a)]


def _rk4(red, x0, x1, y0, a):
    """
    Runge-Kutta 4th order numerical method.
    """

    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A

    f_0_n = 0
    f_1_n = 0
    f_2_n = 0
    f_3_n = 0

    f_0 = y_0[1:]
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)

    f_1 = [y_0[i] + f_0[i]*h/2 for i in range(1, a)]
    for i in range(a):
        f_1_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_0[i]*h/2)
    f_1.append(f_1_n)

    f_2 = [y_0[i] + f_1[i]*h/2 for i in range(1, a)]
    for i in range(a):
        f_2_n += sympify(DMFsubs(red[i], A + h/2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_1[i]*h/2)
    f_2.append(f_2_n)

    f_3 = [y_0[i] + f_2[i]*h for i in range(1, a)]
    for i in range(a):
        f_3_n += sympify(DMFsubs(red[i], A + h, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_2[i]*h)
    f_3.append(f_3_n)

    return [y_0[i] + h*(f_0[i]+2*f_1[i]+2*f_2[i]+f_3[i])/6 for i in range(a)]
<|endoftext|>from sympy.core.basic import Basic
from sympy.printing import pprint

import random

def interactive_traversal(expr):
    """Traverse a tree asking a user which branch to choose. """

    RED, BRED = '\033[0;31m', '\033[1;31m'
    GREEN, BGREEN = '\033[0;32m', '\033[1;32m'
    YELLOW, BYELLOW = '\033[0;33m', '\033[1;33m'  # noqa
    BLUE, BBLUE = '\033[0;34m', '\033[1;34m'      # noqa
    MAGENTA, BMAGENTA = '\033[0;35m', '\033[1;35m'# noqa
    CYAN, BCYAN = '\033[0;36m', '\033[1;36m'      # noqa
    END = '\033[0m'

    def cprint(*args):
        print("".join(map(str, args)) + END)

    def _interactive_traversal(expr, stage):
        if stage > 0:
            print()

        cprint("Current expression (stage ", BYELLOW, stage, END, "):")
        print(BCYAN)
        pprint(expr)
        print(END)

        if isinstance(expr, Basic):
            if expr.is_Add:
                args = expr.as_ordered_terms()
            elif expr.is_Mul:
                args = expr.as_ordered_factors()
            else:
                args = expr.args
        elif hasattr(expr, "__iter__"):
            args = list(expr)
        else:
            return expr

        n_args = len(args)

        if not n_args:
            return expr

        for i, arg in enumerate(args):
            cprint(GREEN, "[", BGREEN, i, GREEN, "] ", BLUE, type(arg), END)
            pprint(arg)
            print()

        if n_args == 1:
            choices = '0'
        else:
            choices = '0-%d' % (n_args - 1)

        try:
            choice = input("Your choice [%s,f,l,r,d,?]: " % choices)
        except EOFError:
            result = expr
            print()
        else:
            if choice == '?':
                cprint(RED, "%s - select subexpression with the given index" %
                       choices)
                cprint(RED, "f - select the first subexpression")
                cprint(RED, "l - select the last subexpression")
                cprint(RED, "r - select a random subexpression")
                cprint(RED, "d - done\n")

                result = _interactive_traversal(expr, stage)
            elif choice in ('d', ''):
                result = expr
            elif choice == 'f':
                result = _interactive_traversal(args[0], stage + 1)
            elif choice == 'l':
                result = _interactive_traversal(args[-1], stage + 1)
            elif choice == 'r':
                result = _interactive_traversal(random.choice(args), stage + 1)
            else:
                try:
                    choice = int(choice)
                except ValueError:
                    cprint(BRED,
                           "Choice must be a number in %s range\n" % choices)
                    result = _interactive_traversal(expr, stage)
                else:
                    if choice < 0 or choice >= n_args:
                        cprint(BRED, "Choice must be in %s range\n" % choices)
                        result = _interactive_traversal(expr, stage)
                    else:
                        result = _interactive_traversal(args[choice], stage + 1)

        return result

    return _interactive_traversal(expr, 0)
<|endoftext|>"""Helper module for setting up interactive SymPy sessions. """

from .printing import init_printing
from .session import init_session
from .traversal import interactive_traversal


__all__ = ['init_printing', 'init_session', 'interactive_traversal']
<|endoftext|>"""Tools for setting up interactive sessions. """

from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple

from sympy.interactive.printing import init_printing

from sympy.utilities.misc import ARCH

preexec_source = """\
from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)
init_printing()
"""

verbose_message = """\
These commands were executed:
%(source)s
Documentation can be found at https://docs.sympy.org/%(version)s
"""

no_ipython = """\
Could not locate IPython. Having IPython installed is greatly recommended.
See http://ipython.scipy.org for more details. If you use Debian/Ubuntu,
just install the 'ipython' package and start isympy again.
"""


def _make_message(ipython=True, quiet=False, source=None):
    """Create a banner for an interactive session. """
    from sympy import __version__ as sympy_version
    from sympy import SYMPY_DEBUG

    import sys
    import os

    if quiet:
        return ""

    python_version = "%d.%d.%d" % sys.version_info[:3]

    if ipython:
        shell_name = "IPython"
    else:
        shell_name = "Python"

    info = ['ground types: %s' % GROUND_TYPES]

    cache = os.getenv('SYMPY_USE_CACHE')

    if cache is not None and cache.lower() == 'no':
        info.append('cache: off')

    if SYMPY_DEBUG:
        info.append('debugging: on')

    args = shell_name, sympy_version, python_version, ARCH, ', '.join(info)
    message = "%s console for SymPy %s (Python %s-%s) (%s)\n" % args

    if source is None:
        source = preexec_source

    _source = ""

    for line in source.split('\n')[:-1]:
        if not line:
            _source += '\n'
        else:
            _source += '>>> ' + line + '\n'

    doc_version = sympy_version
    if 'dev' in doc_version:
        doc_version = "dev"
    else:
        doc_version = "%s/" % doc_version

    message += '\n' + verbose_message % {'source': _source,
                                         'version': doc_version}

    return message


def int_to_Integer(s):
    """
    Wrap integer literals with Integer.

    This is based on the decistmt example from
    https://docs.python.org/3/library/tokenize.html.

    Only integer literals are converted.  Float literals are left alone.

    Examples
    ========

    >>> from sympy import Integer # noqa: F401
    >>> from sympy.interactive.session import int_to_Integer
    >>> s = '1.2 + 1/2 - 0x12 + a1'
    >>> int_to_Integer(s)
    '1.2 +Integer (1 )/Integer (2 )-Integer (0x12 )+a1 '
    >>> s = 'print (1/2)'
    >>> int_to_Integer(s)
    'print (Integer (1 )/Integer (2 ))'
    >>> exec(s)
    0.5
    >>> exec(int_to_Integer(s))
    1/2
    """
    from tokenize import generate_tokens, untokenize, NUMBER, NAME, OP
    from io import StringIO

    def _is_int(num):
        """
        Returns true if string value num (with token NUMBER) represents an integer.
        """
        # XXX: Is there something in the standard library that will do this?
        if '.' in num or 'j' in num.lower() or 'e' in num.lower():
            return False
        return True

    result = []
    g = generate_tokens(StringIO(s).readline)  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        if toknum == NUMBER and _is_int(tokval):  # replace NUMBER tokens
            result.extend([
                (NAME, 'Integer'),
                (OP, '('),
                (NUMBER, tokval),
                (OP, ')')
            ])
        else:
            result.append((toknum, tokval))
    return untokenize(result)


def enable_automatic_int_sympification(shell):
    """
    Allow IPython to automatically convert integer literals to Integer.
    """
    import ast
    old_run_cell = shell.run_cell

    def my_run_cell(cell, *args, **kwargs):
        try:
            # Check the cell for syntax errors.  This way, the syntax error
            # will show the original input, not the transformed input.  The
            # downside here is that IPython magic like %timeit will not work
            # with transformed input (but on the other hand, IPython magic
            # that doesn't expect transformed input will continue to work).
            ast.parse(cell)
        except SyntaxError:
            pass
        else:
            cell = int_to_Integer(cell)
        return old_run_cell(cell, *args, **kwargs)

    shell.run_cell = my_run_cell


def enable_automatic_symbols(shell):
    """Allow IPython to automatically create symbols (``isympy -a``). """
    # XXX: This should perhaps use tokenize, like int_to_Integer() above.
    # This would avoid re-executing the code, which can lead to subtle
    # issues.  For example:
    #
    # In [1]: a = 1
    #
    # In [2]: for i in range(10):
    #    ...:     a += 1
    #    ...:
    #
    # In [3]: a
    # Out[3]: 11
    #
    # In [4]: a = 1
    #
    # In [5]: for i in range(10):
    #    ...:     a += 1
    #    ...:     print b
    #    ...:
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    #
    # In [6]: a
    # Out[6]: 12
    #
    # Note how the for loop is executed again because `b` was not defined, but `a`
    # was already incremented once, so the result is that it is incremented
    # multiple times.

    import re
    re_nameerror = re.compile(
        "name '(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)' is not defined")

    def _handler(self, etype, value, tb, tb_offset=None):
        """Handle :exc:`NameError` exception and allow injection of missing symbols. """
        if etype is NameError and tb.tb_next and not tb.tb_next.tb_next:
            match = re_nameerror.match(str(value))

            if match is not None:
                # XXX: Make sure Symbol is in scope. Otherwise you'll get infinite recursion.
                self.run_cell("%(symbol)s = Symbol('%(symbol)s')" %
                              {'symbol': match.group("symbol")}, store_history=False)

                try:
                    code = self.user_ns['In'][-1]
                except (KeyError, IndexError):
                    pass
                else:
                    self.run_cell(code, store_history=False)
                    return None
                finally:
                    self.run_cell("del %s" % match.group("symbol"),
                                  store_history=False)

        stb = self.InteractiveTB.structured_traceback(
            etype, value, tb, tb_offset=tb_offset)
        self._showtraceback(etype, value, stb)

    shell.set_custom_exc((NameError,), _handler)


def init_ipython_session(shell=None, argv=[], auto_symbols=False, auto_int_to_Integer=False):
    """Construct new IPython session. """
    import IPython

    if version_tuple(IPython.__version__) >= version_tuple('0.11'):
        if not shell:
            # use an app to parse the command line, and init config
            # IPython 1.0 deprecates the frontend module, so we import directly
            # from the terminal module to prevent a deprecation message from being
            # shown.
            if version_tuple(IPython.__version__) >= version_tuple('1.0'):
                from IPython.terminal import ipapp
            else:
                from IPython.frontend.terminal import ipapp
            app = ipapp.TerminalIPythonApp()

            # don't draw IPython banner during initialization:
            app.display_banner = False
            app.initialize(argv)

            shell = app.shell

        if auto_symbols:
            enable_automatic_symbols(shell)
        if auto_int_to_Integer:
            enable_automatic_int_sympification(shell)

        return shell
    else:
        from IPython.Shell import make_IPython
        return make_IPython(argv)


def init_python_session():
    """Construct new Python session. """
    from code import InteractiveConsole

    class SymPyConsole(InteractiveConsole):
        """An interactive console with readline support. """

        def __init__(self):
            ns_locals = {}
            InteractiveConsole.__init__(self, locals=ns_locals)
            try:
                import rlcompleter
                import readline
            except ImportError:
                pass
            else:
                import os
                import atexit

                readline.set_completer(rlcompleter.Completer(ns_locals).complete)
                readline.parse_and_bind('tab: complete')

                if hasattr(readline, 'read_history_file'):
                    history = os.path.expanduser('~/.sympy-history')

                    try:
                        readline.read_history_file(history)
                    except OSError:
                        pass

                    atexit.register(readline.write_history_file, history)

    return SymPyConsole()


def init_session(ipython=None, pretty_print=True, order=None,
                 use_unicode=None, use_latex=None, quiet=False, auto_symbols=False,
                 auto_int_to_Integer=False, str_printer=None, pretty_printer=None,
                 latex_printer=None, argv=[]):
    """
    Initialize an embedded IPython or Python session. The IPython session is
    initiated with the --pylab option, without the numpy imports, so that
    matplotlib plotting can be interactive.

    Parameters
    ==========

    pretty_print: boolean
        If True, use pretty_print to stringify;
        if False, use sstrrepr to stringify.
    order: string or None
        There are a few different settings for this parameter:
        lex (default), which is lexographic order;
        grlex, which is graded lexographic order;
        grevlex, which is reversed graded lexographic order;
        old, which is used for compatibility reasons and for long expressions;
        None, which sets it to lex.
    use_unicode: boolean or None
        If True, use unicode characters;
        if False, do not use unicode characters.
    use_latex: boolean or None
        If True, use latex rendering if IPython GUI's;
        if False, do not use latex rendering.
    quiet: boolean
        If True, init_session will not print messages regarding its status;
        if False, init_session will print messages regarding its status.
    auto_symbols: boolean
        If True, IPython will automatically create symbols for you.
        If False, it will not.
        The default is False.
    auto_int_to_Integer: boolean
        If True, IPython will automatically wrap int literals with Integer, so
        that things like 1/2 give Rational(1, 2).
        If False, it will not.
        The default is False.
    ipython: boolean or None
        If True, printing will initialize for an IPython console;
        if False, printing will initialize for a normal console;
        The default is None, which automatically determines whether we are in
        an ipython instance or not.
    str_printer: function, optional, default=None
        A custom string printer function. This should mimic
        sympy.printing.sstrrepr().
    pretty_printer: function, optional, default=None
        A custom pretty printer. This should mimic sympy.printing.pretty().
    latex_printer: function, optional, default=None
        A custom LaTeX printer. This should mimic sympy.printing.latex()
        This should mimic sympy.printing.latex().
    argv: list of arguments for IPython
        See sympy.bin.isympy for options that can be used to initialize IPython.

    See Also
    ========

    sympy.interactive.printing.init_printing: for examples and the rest of the parameters.


    Examples
    ========

    >>> from sympy import init_session, Symbol, sin, sqrt
    >>> sin(x) #doctest: +SKIP
    NameError: name 'x' is not defined
    >>> init_session() #doctest: +SKIP
    >>> sin(x) #doctest: +SKIP
    sin(x)
    >>> sqrt(5) #doctest: +SKIP
      ___
    \\/ 5
    >>> init_session(pretty_print=False) #doctest: +SKIP
    >>> sqrt(5) #doctest: +SKIP
    sqrt(5)
    >>> y + x + y**2 + x**2 #doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_session(order='grlex') #doctest: +SKIP
    >>> y + x + y**2 + x**2 #doctest: +SKIP
    x**2 + y**2 + x + y
    >>> init_session(order='grevlex') #doctest: +SKIP
    >>> y * x**2 + x * y**2 #doctest: +SKIP
    x**2*y + x*y**2
    >>> init_session(order='old') #doctest: +SKIP
    >>> x**2 + y**2 + x + y #doctest: +SKIP
    x + y + x**2 + y**2
    >>> theta = Symbol('theta') #doctest: +SKIP
    >>> theta #doctest: +SKIP
    theta
    >>> init_session(use_unicode=True) #doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    """
    import sys

    in_ipython = False

    if ipython is not False:
        try:
            import IPython
        except ImportError:
            if ipython is True:
                raise RuntimeError("IPython is not available on this system")
            ip = None
        else:
            try:
                from IPython import get_ipython
                ip = get_ipython()
            except ImportError:
                ip = None
        in_ipython = bool(ip)
        if ipython is None:
            ipython = in_ipython

    if ipython is False:
        ip = init_python_session()
        mainloop = ip.interact
    else:
        ip = init_ipython_session(ip, argv=argv, auto_symbols=auto_symbols,
                                  auto_int_to_Integer=auto_int_to_Integer)

        if version_tuple(IPython.__version__) >= version_tuple('0.11'):
            # runsource is gone, use run_cell instead, which doesn't
            # take a symbol arg.  The second arg is `store_history`,
            # and False means don't add the line to IPython's history.
            ip.runsource = lambda src, symbol='exec': ip.run_cell(src, False)

            # Enable interactive plotting using pylab.
            try:
                ip.enable_pylab(import_all=False)
            except Exception:
                # Causes an import error if matplotlib is not installed.
                # Causes other errors (depending on the backend) if there
                # is no display, or if there is some problem in the
                # backend, so we have a bare "except Exception" here
                pass
        if not in_ipython:
            mainloop = ip.mainloop

    if auto_symbols and (not ipython or version_tuple(IPython.__version__) < version_tuple('0.11')):
        raise RuntimeError("automatic construction of symbols is possible only in IPython 0.11 or above")
    if auto_int_to_Integer and (not ipython or version_tuple(IPython.__version__) < version_tuple('0.11')):
        raise RuntimeError("automatic int to Integer transformation is possible only in IPython 0.11 or above")

    _preexec_source = preexec_source

    ip.runsource(_preexec_source, symbol='exec')
    init_printing(pretty_print=pretty_print, order=order,
                  use_unicode=use_unicode, use_latex=use_latex, ip=ip,
                  str_printer=str_printer, pretty_printer=pretty_printer,
                  latex_printer=latex_printer)

    message = _make_message(ipython, quiet, _preexec_source)

    if not in_ipython:
        print(message)
        mainloop()
        sys.exit('Exiting ...')
    else:
        print(message)
        import atexit
        atexit.register(lambda: print("Exiting ...\n"))
<|endoftext|>"""Tools for setting up printing in interactive sessions. """

from io import BytesIO

from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
from sympy.external import import_module


def _init_python_printing(stringify_func, **settings):
    """Setup printing in Python interactive session. """
    import sys
    import builtins

    def _displayhook(arg):
        """Python's pretty-printer display hook.

           This function was adapted from:

            https://www.python.org/dev/peps/pep-0217/

        """
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg, **settings))
            builtins._ = arg

    sys.displayhook = _displayhook


def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor,
                           backcolor, fontsize, latex_mode, print_builtin,
                           latex_printer, scale, **settings):
    """Setup printing in IPython interactive session. """
    IPython = import_module("IPython", min_module_version="1.0")
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass

    # Guess best font color if none was given based on the ip.colors string.
    # From the IPython documentation:
    #   It has four case-insensitive values: 'nocolor', 'neutral', 'linux',
    #   'lightbg'. The default is neutral, which should be legible on either
    #   dark or light terminal backgrounds. linux is optimised for dark
    #   backgrounds and lightbg for light ones.
    if forecolor is None:
        color = ip.colors.lower()
        if color == 'lightbg':
            forecolor = 'Black'
        elif color == 'linux':
            forecolor = 'White'
        else:
            # No idea, go with gray.
            forecolor = 'Gray'
        debug("init_printing: Automatic foreground color:", forecolor)

    if use_latex == "svg":
        extra_preamble = "\n\\special{color %s}" % forecolor
    else:
        extra_preamble = ""

    imagesize = 'tight'
    offset = "0cm,0cm"
    resolution = round(150*scale)
    dvi = r"-T %s -D %d -bg %s -fg %s -O %s" % (
        imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()

    svg_scale = 150/72*scale
    dvioptions_svg = ["--no-fonts", "--scale={}".format(svg_scale)]

    debug("init_printing: DVIOPTIONS:", dvioptions)
    debug("init_printing: DVIOPTIONS_SVG:", dvioptions_svg)

    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        """caller for pretty, for use in IPython 0.11"""
        if _can_print(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))

    def _preview_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='png', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions, fontsize=fontsize)
        except Exception as e:
            # IPython swallows exceptions
            debug("png printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue()

    def _svg_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='svg', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions_svg, fontsize=fontsize)
        except Exception as e:
            # IPython swallows exceptions
            debug("svg printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue().decode('utf-8')

    def _matplotlib_wrapper(o):
        # mathtext can't render some LaTeX commands. For example, it can't
        # render any LaTeX environments such as array or matrix. So here we
        # ensure that if mathtext fails to render, we return None.
        try:
            try:
                return latex_to_png(o, color=forecolor, scale=scale)
            except TypeError: #  Old IPython version without color and scale
                return latex_to_png(o)
        except ValueError as e:
            debug('matplotlib exception caught:', repr(e))
            return None


    # Hook methods for builtin SymPy printers
    printing_hooks = ('_latex', '_sympystr', '_pretty', '_sympyrepr')


    def _can_print(o):
        """Return True if type o can be printed with one of the SymPy printers.

        If o is a container type, this is True if and only if every element of
        o can be printed in this way.
        """

        try:
            # If you're adding another type, make sure you add it to printable_types
            # later in this file as well

            builtin_types = (list, tuple, set, frozenset)
            if isinstance(o, builtin_types):
                # If the object is a custom subclass with a custom str or
                # repr, use that instead.
                if (type(o).__str__ not in (i.__str__ for i in builtin_types) or
                    type(o).__repr__ not in (i.__repr__ for i in builtin_types)):
                    return False
                return all(_can_print(i) for i in o)
            elif isinstance(o, dict):
                return all(_can_print(i) and _can_print(o[i]) for i in o)
            elif isinstance(o, bool):
                return False
            elif isinstance(o, Printable):
                # types known to SymPy
                return True
            elif any(hasattr(o, hook) for hook in printing_hooks):
                # types which add support themselves
                return True
            elif isinstance(o, (float, int)) and print_builtin:
                return True
            return False
        except RuntimeError:
            return False
            # This is in case maximum recursion depth is reached.
            # Since RecursionError is for versions of Python 3.5+
            # so this is to guard against RecursionError for older versions.

    def _print_latex_png(o):
        """
        A function that returns a png rendered by an external latex
        distribution, falling back to matplotlib rendering
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _preview_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' Falling back to matplotlib backend')
                if latex_mode != 'inline':
                    s = latex(o, mode='inline', **settings)
                return _matplotlib_wrapper(s)

    def _print_latex_svg(o):
        """
        A function that returns a svg rendered by an external latex
        distribution, no fallback available.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _svg_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' No fallback available.')

    def _print_latex_matplotlib(o):
        """
        A function that returns a png rendered by mathtext
        """
        if _can_print(o):
            s = latex(o, mode='inline', **settings)
            return _matplotlib_wrapper(s)

    def _print_latex_text(o):
        """
        A function to generate the latex representation of SymPy expressions.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                return '$\\displaystyle %s$' % s
            return s

    # Printable is our own type, so we handle it with methods instead of
    # the approach required by builtin types. This allows downstream
    # packages to override the methods in their own subclasses of Printable,
    # which avoids the effects of gh-16002.
    printable_types = [float, tuple, list, set, frozenset, dict, int]

    plaintext_formatter = ip.display_formatter.formatters['text/plain']

    # Exception to the rule above: IPython has better dispatching rules
    # for plaintext printing (xref ipython/ipython#8938), and we can't
    # use `_repr_pretty_` without hitting a recursion error in _print_plain.
    for cls in printable_types + [Printable]:
        plaintext_formatter.for_type(cls, _print_plain)

    svg_formatter = ip.display_formatter.formatters['image/svg+xml']
    if use_latex in ('svg', ):
        debug("init_printing: using svg formatter")
        for cls in printable_types:
            svg_formatter.for_type(cls, _print_latex_svg)
        Printable._repr_svg_ = _print_latex_svg
    else:
        debug("init_printing: not using any svg formatter")
        for cls in printable_types:
            # Better way to set this, but currently does not work in IPython
            #png_formatter.for_type(cls, None)
            if cls in svg_formatter.type_printers:
                svg_formatter.type_printers.pop(cls)
        Printable._repr_svg_ = Printable._repr_disabled

    png_formatter = ip.display_formatter.formatters['image/png']
    if use_latex in (True, 'png'):
        debug("init_printing: using png formatter")
        for cls in printable_types:
            png_formatter.for_type(cls, _print_latex_png)
        Printable._repr_png_ = _print_latex_png
    elif use_latex == 'matplotlib':
        debug("init_printing: using matplotlib formatter")
        for cls in printable_types:
            png_formatter.for_type(cls, _print_latex_matplotlib)
        Printable._repr_png_ = _print_latex_matplotlib
    else:
        debug("init_printing: not using any png formatter")
        for cls in printable_types:
            # Better way to set this, but currently does not work in IPython
            #png_formatter.for_type(cls, None)
            if cls in png_formatter.type_printers:
                png_formatter.type_printers.pop(cls)
        Printable._repr_png_ = Printable._repr_disabled

    latex_formatter = ip.display_formatter.formatters['text/latex']
    if use_latex in (True, 'mathjax'):
        debug("init_printing: using mathjax formatter")
        for cls in printable_types:
            latex_formatter.for_type(cls, _print_latex_text)
        Printable._repr_latex_ = _print_latex_text
    else:
        debug("init_printing: not using text/latex formatter")
        for cls in printable_types:
            # Better way to set this, but currently does not work in IPython
            #latex_formatter.for_type(cls, None)
            if cls in latex_formatter.type_printers:
                latex_formatter.type_printers.pop(cls)
        Printable._repr_latex_ = Printable._repr_disabled

def _is_ipython(shell):
    """Is a shell instance an IPython shell?"""
    # shortcut, so we don't import IPython if we don't have to
    from sys import modules
    if 'IPython' not in modules:
        return False
    try:
        from IPython.core.interactiveshell import InteractiveShell
    except ImportError:
        # IPython < 0.11
        try:
            from IPython.iplib import InteractiveShell
        except ImportError:
            # Reaching this points means IPython has changed in a backward-incompatible way
            # that we don't know about. Warn?
            return False
    return isinstance(shell, InteractiveShell)

# Used by the doctester to override the default for no_global
NO_GLOBAL = False

def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    r"""
    Initializes pretty-printer depending on the environment.

    Parameters
    ==========

    pretty_print : bool, default=True
        If ``True``, use :func:`~.pretty_print` to stringify or the provided pretty
        printer; if ``False``, use :func:`~.sstrrepr` to stringify or the provided string
        printer.
    order : string or None, default='lex'
        There are a few different settings for this parameter:
        ``'lex'`` (default), which is lexographic order;
        ``'grlex'``, which is graded lexographic order;
        ``'grevlex'``, which is reversed graded lexographic order;
        ``'old'``, which is used for compatibility reasons and for long expressions;
        ``None``, which sets it to lex.
    use_unicode : bool or None, default=None
        If ``True``, use unicode characters;
        if ``False``, do not use unicode characters;
        if ``None``, make a guess based on the environment.
    use_latex : string, bool, or None, default=None
        If ``True``, use default LaTeX rendering in GUI interfaces (png and
        mathjax);
        if ``False``, do not use LaTeX rendering;
        if ``None``, make a guess based on the environment;
        if ``'png'``, enable LaTeX rendering with an external LaTeX compiler,
        falling back to matplotlib if external compilation fails;
        if ``'matplotlib'``, enable LaTeX rendering with matplotlib;
        if ``'mathjax'``, enable LaTeX text generation, for example MathJax
        rendering in IPython notebook or text rendering in LaTeX documents;
        if ``'svg'``, enable LaTeX rendering with an external latex compiler,
        no fallback
    wrap_line : bool
        If True, lines will wrap at the end; if False, they will not wrap
        but continue as one line. This is only relevant if ``pretty_print`` is
        True.
    num_columns : int or None, default=None
        If ``int``, number of columns before wrapping is set to num_columns; if
        ``None``, number of columns before wrapping is set to terminal width.
        This is only relevant if ``pretty_print`` is ``True``.
    no_global : bool, default=False
        If ``True``, the settings become system wide;
        if ``False``, use just for this console/session.
    ip : An interactive console
        This can either be an instance of IPython,
        or a class that derives from code.InteractiveConsole.
    euler : bool, optional, default=False
        Loads the euler package in the LaTeX preamble for handwritten style
        fonts (https://www.ctan.org/pkg/euler).
    forecolor : string or None, optional, default=None
        DVI setting for foreground color. ``None`` means that either ``'Black'``,
        ``'White'``, or ``'Gray'`` will be selected based on a guess of the IPython
        terminal color setting. See notes.
    backcolor : string, optional, default='Transparent'
        DVI setting for background color. See notes.
    fontsize : string or int, optional, default='10pt'
        A font size to pass to the LaTeX documentclass function in the
        preamble. Note that the options are limited by the documentclass.
        Consider using scale instead.
    latex_mode : string, optional, default='plain'
        The mode used in the LaTeX printer. Can be one of:
        ``{'inline'|'plain'|'equation'|'equation*'}``.
    print_builtin : boolean, optional, default=True
        If ``True`` then floats and integers will be printed. If ``False`` the
        printer will only print SymPy types.
    str_printer : function, optional, default=None
        A custom string printer function. This should mimic
        :func:`~.sstrrepr`.
    pretty_printer : function, optional, default=None
        A custom pretty printer. This should mimic :func:`~.pretty`.
    latex_printer : function, optional, default=None
        A custom LaTeX printer. This should mimic :func:`~.latex`.
    scale : float, optional, default=1.0
        Scale the LaTeX output when using the ``'png'`` or ``'svg'`` backends.
        Useful for high dpi screens.
    settings :
        Any additional settings for the ``latex`` and ``pretty`` commands can
        be used to fine-tune the output.

    Examples
    ========

    >>> from sympy.interactive import init_printing
    >>> from sympy import Symbol, sqrt
    >>> from sympy.abc import x, y
    >>> sqrt(5)
    sqrt(5)
    >>> init_printing(pretty_print=True) # doctest: +SKIP
    >>> sqrt(5) # doctest: +SKIP
      ___
    \/ 5
    >>> theta = Symbol('theta') # doctest: +SKIP
    >>> init_printing(use_unicode=True) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    >>> init_printing(use_unicode=False) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    theta
    >>> init_printing(order='lex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grlex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grevlex') # doctest: +SKIP
    >>> str(y * x**2 + x * y**2) # doctest: +SKIP
    x**2*y + x*y**2
    >>> init_printing(order='old') # doctest: +SKIP
    >>> str(x**2 + y**2 + x + y) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(num_columns=10) # doctest: +SKIP
    >>> x**2 + x + y**2 + y # doctest: +SKIP
    x + y +
    x**2 + y**2

    Notes
    =====

    The foreground and background colors can be selected when using ``'png'`` or
    ``'svg'`` LaTeX rendering. Note that before the ``init_printing`` command is
    executed, the LaTeX rendering is handled by the IPython console and not SymPy.

    The colors can be selected among the 68 standard colors known to ``dvips``,
    for a list see [1]_. In addition, the background color can be
    set to  ``'Transparent'`` (which is the default value).

    When using the ``'Auto'`` foreground color, the guess is based on the
    ``colors`` variable in the IPython console, see [2]_. Hence, if
    that variable is set correctly in your IPython console, there is a high
    chance that the output will be readable, although manual settings may be
    needed.


    References
    ==========

    .. [1] https://en.wikibooks.org/wiki/LaTeX/Colors#The_68_standard_colors_known_to_dvips

    .. [2] https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors

    See Also
    ========

    sympy.printing.latex
    sympy.printing.pretty

    """
    import sys
    from sympy.printing.printer import Printer

    if pretty_print:
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            from sympy.printing import pretty as stringify_func
    else:
        if str_printer is not None:
            stringify_func = str_printer
        else:
            from sympy.printing import sstrrepr as stringify_func

    # Even if ip is not passed, double check that not in IPython shell
    in_ipython = False
    if ip is None:
        try:
            ip = get_ipython()
        except NameError:
            pass
        else:
            in_ipython = (ip is not None)

    if ip and not in_ipython:
        in_ipython = _is_ipython(ip)

    if in_ipython and pretty_print:
        try:
            from IPython.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            # This will be True if we are in the qtconsole or notebook
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) \
                    and 'ipython-console' not in ''.join(sys.argv):
                if use_unicode is None:
                    debug("init_printing: Setting use_unicode to True")
                    use_unicode = True
                if use_latex is None:
                    debug("init_printing: Setting use_latex to True")
                    use_latex = True

    if not NO_GLOBAL and not no_global:
        Printer.set_global_settings(order=order, use_unicode=use_unicode,
                                    wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func

        if pretty_print:
            stringify_func = lambda expr, **settings: \
                             _stringify_func(expr, order=order,
                                             use_unicode=use_unicode,
                                             wrap_line=wrap_line,
                                             num_columns=num_columns,
                                             **settings)
        else:
            stringify_func = \
                lambda expr, **settings: _stringify_func(
                    expr, order=order, **settings)

    if in_ipython:
        mode_in_settings = settings.pop("mode", None)
        if mode_in_settings:
            debug("init_printing: Mode is not able to be set due to internals"
                  "of IPython printing")
        _init_ipython_printing(ip, stringify_func, use_latex, euler,
                               forecolor, backcolor, fontsize, latex_mode,
                               print_builtin, latex_printer, scale,
                               **settings)
    else:
        _init_python_printing(stringify_func, **settings)
<|endoftext|>from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.utilities.misc import as_int

from mpmath.libmp.libmpf import prec_to_dps


def _check_norm(elements, norm):
    """validate if input norm is consistent"""
    if norm is not None and norm.is_number:
        if norm.is_positive is False:
            raise ValueError("Input norm must be positive.")

        numerical = all(i.is_number and i.is_real is True for i in elements)
        if numerical and is_eq(norm**2, sum(i**2 for i in elements)) is False:
            raise ValueError("Incompatible value for norm.")


def _is_extrinsic(seq):
    """validate seq and return True if seq is lowercase and False if uppercase"""
    if type(seq) != str:
        raise ValueError('Expected seq to be a string.')
    if len(seq) != 3:
        raise ValueError("Expected 3 axes, got `{}`.".format(seq))

    intrinsic = seq.isupper()
    extrinsic = seq.islower()
    if not (intrinsic or extrinsic):
        raise ValueError("seq must either be fully uppercase (for extrinsic "
                         "rotations), or fully lowercase, for intrinsic "
                         "rotations).")

    i, j, k = seq.lower()
    if (i == j) or (j == k):
        raise ValueError("Consecutive axes must be different")

    bad = set(seq) - set('xyzXYZ')
    if bad:
        raise ValueError("Expected axes from `seq` to be from "
                         "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                         "got {}".format(''.join(bad)))

    return extrinsic


class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as ``Quaternion(a, b, c, d)``
    as in $q = a + bi + cj + dk$.

    Parameters
    ==========

    norm : None or number
        Pre-defined quaternion norm. If a value is given, Quaternion.norm
        returns this pre-defined value instead of calculating the norm

    Examples
    ========

    >>> from sympy import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as:

    >>> from sympy import Quaternion
    >>> from sympy import symbols, I
    >>> x = symbols('x')
    >>> q1 = Quaternion(x, x**3, x, x**2, real_field = False)
    >>> q2 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    >>> q1
    x + x**3*i + x*j + x**2*k
    >>> q2
    (3 + 4*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

    Defining symbolic unit quaternions:

    >>> from sympy import Quaternion
    >>> from sympy.abc import w, x, y, z
    >>> q = Quaternion(w, x, y, z, norm=1)
    >>> q
    w + x*i + y*j + z*k
    >>> q.norm()
    1

    References
    ==========

    .. [1] https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    .. [2] https://en.wikipedia.org/wiki/Quaternion

    """
    _op_priority = 11.0

    is_commutative = False

    def __new__(cls, a=0, b=0, c=0, d=0, real_field=True, norm=None):
        a, b, c, d = map(sympify, (a, b, c, d))

        if any(i.is_commutative is False for i in [a, b, c, d]):
            raise ValueError("arguments have to be commutative")
        obj = super().__new__(cls, a, b, c, d)
        obj._real_field = real_field
        obj.set_norm(norm)
        return obj

    def set_norm(self, norm):
        """Sets norm of an already instantiated quaternion.

        Parameters
        ==========

        norm : None or number
            Pre-defined quaternion norm. If a value is given, Quaternion.norm
            returns this pre-defined value instead of calculating the norm

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        Setting the norm:

        >>> q.set_norm(1)
        >>> q.norm()
        1

        Removing set norm:

        >>> q.set_norm(None)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        """
        norm = sympify(norm)
        _check_norm(self.args, norm)
        self._norm = norm

    @property
    def a(self):
        return self.args[0]

    @property
    def b(self):
        return self.args[1]

    @property
    def c(self):
        return self.args[2]

    @property
    def d(self):
        return self.args[3]

    @property
    def real_field(self):
        return self._real_field

    @property
    def product_matrix_left(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        left. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  &-d  & c \\
                                  c  & d  & a  &-b \\
                                  d  &-c  & b  & a \end{bmatrix}

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(1, 0, 0, 1)
        >>> q2 = Quaternion(a, b, c, d)
        >>> q1.product_matrix_left
        Matrix([
        [1, 0,  0, -1],
        [0, 1, -1,  0],
        [0, 1,  1,  0],
        [1, 0,  0,  1]])

        >>> q1.product_matrix_left * q2.to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])
        """
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, -self.d, self.c],
                [self.c, self.d, self.a, -self.b],
                [self.d, -self.c, self.b, self.a]])

    @property
    def product_matrix_right(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        right. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  & d  &-c \\
                                  c  &-d  & a  & b \\
                                  d  & c  &-b  & a \end{bmatrix}


        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(a, b, c, d)
        >>> q2 = Quaternion(1, 0, 0, 1)
        >>> q2.product_matrix_right
        Matrix([
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [1, 0, 0, 1]])

        Note the switched arguments: the matrix represents the quaternion on
        the right, but is still considered as a matrix multiplication from the
        left.

        >>> q2.product_matrix_right * q1.to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])
        """
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, self.d, -self.c],
                [self.c, -self.d, self.a, self.b],
                [self.d, self.c, -self.b, self.a]])

    def to_Matrix(self, vector_only=False):
        """Returns elements of quaternion as a column vector.
        By default, a ``Matrix`` of length 4 is returned, with the real part as the
        first element.
        If ``vector_only`` is ``True``, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        vector_only : bool
            If True, only imaginary part is returned.
            Default value: False

        Returns
        =======

        Matrix
            A column vector constructed by the elements of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q
        a + b*i + c*j + d*k

        >>> q.to_Matrix()
        Matrix([
        [a],
        [b],
        [c],
        [d]])


        >>> q.to_Matrix(vector_only=True)
        Matrix([
        [b],
        [c],
        [d]])

        """
        if vector_only:
            return Matrix(self.args[1:])
        else:
            return Matrix(self.args)

    @classmethod
    def from_Matrix(cls, elements):
        """Returns quaternion from elements of a column vector`.
        If vector_only is True, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        elements : Matrix, list or tuple of length 3 or 4. If length is 3,
            assume real part is zero.
            Default value: False

        Returns
        =======

        Quaternion
            A quaternion created from the input elements.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion.from_Matrix([a, b, c, d])
        >>> q
        a + b*i + c*j + d*k

        >>> q = Quaternion.from_Matrix([b, c, d])
        >>> q
        0 + b*i + c*j + d*k

        """
        length = len(elements)
        if length != 3 and length != 4:
            raise ValueError("Input elements must have length 3 or 4, got {} "
                             "elements".format(length))

        if length == 3:
            return Quaternion(0, *elements)
        else:
            return Quaternion(*elements)

    @classmethod
    def from_euler(cls, angles, seq):
        """Returns quaternion equivalent to rotation represented by the Euler
        angles, in the sequence defined by ``seq``.

        Parameters
        ==========

        angles : list, tuple or Matrix of 3 numbers
            The Euler angles (in radians).
        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the Euler angles
            in the given sequence.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi
        >>> q = Quaternion.from_euler([pi/2, 0, 0], 'xyz')
        >>> q
        sqrt(2)/2 + sqrt(2)/2*i + 0*j + 0*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'zyz')
        >>> q
        0 + (-sqrt(2)/2)*i + 0*j + sqrt(2)/2*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'ZYZ')
        >>> q
        0 + sqrt(2)/2*i + 0*j + sqrt(2)/2*k

        """

        if len(angles) != 3:
            raise ValueError("3 angles must be given.")

        extrinsic = _is_extrinsic(seq)
        i, j, k = seq.lower()

        # get elementary basis vectors
        ei = [1 if n == i else 0 for n in 'xyz']
        ej = [1 if n == j else 0 for n in 'xyz']
        ek = [1 if n == k else 0 for n in 'xyz']

        # calculate distinct quaternions
        qi = cls.from_axis_angle(ei, angles[0])
        qj = cls.from_axis_angle(ej, angles[1])
        qk = cls.from_axis_angle(ek, angles[2])

        if extrinsic:
            return trigsimp(qk * qj * qi)
        else:
            return trigsimp(qi * qj * qk)

    def to_euler(self, seq, angle_addition=True, avoid_square_root=False):
        r"""Returns Euler angles representing same rotation as the quaternion,
        in the sequence given by ``seq``. This implements the method described
        in [1]_.

        For degenerate cases (gymbal lock cases), the third angle is
        set to zero.

        Parameters
        ==========

        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        angle_addition : bool
            When True, first and third angles are given as an addition and
            subtraction of two simpler ``atan2`` expressions. When False, the
            first and third angles are each given by a single more complicated
            ``atan2`` expression. This equivalent expression is given by:

            .. math::

                \operatorname{atan_2} (b,a) \pm \operatorname{atan_2} (d,c) =
                \operatorname{atan_2} (bc\pm ad, ac\mp bd)

            Default value: True

        avoid_square_root : bool
            When True, the second angle is calculated with an expression based
            on ``acos``, which is slightly more complicated but avoids a square
            root. When False, second angle is calculated with ``atan2``, which
            is simpler and can be better for numerical reasons (some
            numerical implementations of ``acos`` have problems near zero).
            Default value: False


        Returns
        =======

        Tuple
            The Euler angles calculated from the quaternion

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> euler = Quaternion(a, b, c, d).to_euler('zyz')
        >>> euler
        (-atan2(-b, c) + atan2(d, a),
         2*atan2(sqrt(b**2 + c**2), sqrt(a**2 + d**2)),
         atan2(-b, c) + atan2(d, a))


        References
        ==========

        .. [1] https://doi.org/10.1371/journal.pone.0276302

        """
        if self.is_zero_quaternion():
            raise ValueError('Cannot convert a quaternion with norm 0.')

        angles = [0, 0, 0]

        extrinsic = _is_extrinsic(seq)
        i, j, k = seq.lower()

        # get index corresponding to elementary basis vectors
        i = 'xyz'.index(i) + 1
        j = 'xyz'.index(j) + 1
        k = 'xyz'.index(k) + 1

        if not extrinsic:
            i, k = k, i

        # check if sequence is symmetric
        symmetric = i == k
        if symmetric:
            k = 6 - i - j

        # parity of the permutation
        sign = (i - j) * (j - k) * (k - i) // 2

        # permutate elements
        elements = [self.a, self.b, self.c, self.d]
        a = elements[0]
        b = elements[i]
        c = elements[j]
        d = elements[k] * sign

        if not symmetric:
            a, b, c, d = a - c, b + d, c + a, d - b

        if avoid_square_root:
            if symmetric:
                n2 = self.norm()**2
                angles[1] = acos((a * a + b * b - c * c - d * d) / n2)
            else:
                n2 = 2 * self.norm()**2
                angles[1] = asin((c * c + d * d - a * a - b * b) / n2)
        else:
            angles[1] = 2 * atan2(sqrt(c * c + d * d), sqrt(a * a + b * b))
            if not symmetric:
                angles[1] -= S.Pi / 2

        # Check for singularities in numerical cases
        case = 0
        if is_eq(c, S.Zero) and is_eq(d, S.Zero):
            case = 1
        if is_eq(a, S.Zero) and is_eq(b, S.Zero):
            case = 2

        if case == 0:
            if angle_addition:
                angles[0] = atan2(b, a) + atan2(d, c)
                angles[2] = atan2(b, a) - atan2(d, c)
            else:
                angles[0] = atan2(b*c + a*d, a*c - b*d)
                angles[2] = atan2(b*c - a*d, a*c + b*d)

        else:  # any degenerate case
            angles[2 * (not extrinsic)] = S.Zero
            if case == 1:
                angles[2 * extrinsic] = 2 * atan2(b, a)
            else:
                angles[2 * extrinsic] = 2 * atan2(d, c)
                angles[2 * extrinsic] *= (-1 if extrinsic else 1)

        # for Tait-Bryan angles
        if not symmetric:
            angles[0] *= sign

        if extrinsic:
            return tuple(angles[::-1])
        else:
            return tuple(angles)

    @classmethod
    def from_axis_angle(cls, vector, angle):
        """Returns a rotation quaternion given the axis and the angle of rotation.

        Parameters
        ==========

        vector : tuple of three numbers
            The vector representation of the given axis.
        angle : number
            The angle by which axis is rotated (in radians).

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the given axis and the angle of rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi, sqrt
        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)
        >>> q
        1/2 + 1/2*i + 1/2*j + 1/2*k

        """
        (x, y, z) = vector
        norm = sqrt(x**2 + y**2 + z**2)
        (x, y, z) = (x / norm, y / norm, z / norm)
        s = sin(angle * S.Half)
        a = cos(angle * S.Half)
        b = x * s
        c = y * s
        d = z * s

        # note that this quaternion is already normalized by construction:
        # c^2 + (s*x)^2 + (s*y)^2 + (s*z)^2 = c^2 + s^2*(x^2 + y^2 + z^2) = c^2 + s^2 * 1 = c^2 + s^2 = 1
        # so, what we return is a normalized quaternion

        return cls(a, b, c, d)

    @classmethod
    def from_rotation_matrix(cls, M):
        """Returns the equivalent quaternion of a matrix. The quaternion will be normalized
        only if the matrix is special orthogonal (orthogonal and det(M) = 1).

        Parameters
        ==========

        M : Matrix
            Input matrix to be converted to equivalent quaternion. M must be special
            orthogonal (orthogonal and det(M) = 1) for the quaternion to be normalized.

        Returns
        =======

        Quaternion
            The quaternion equivalent to given matrix.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Matrix, symbols, cos, sin, trigsimp
        >>> x = symbols('x')
        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])
        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))
        >>> q
        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(2 - 2*cos(x))*sign(sin(x))/2*k

        """

        absQ = M.det()**Rational(1, 3)

        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2

        b = b * sign(M[2, 1] - M[1, 2])
        c = c * sign(M[0, 2] - M[2, 0])
        d = d * sign(M[1, 0] - M[0, 1])

        return Quaternion(a, b, c, d)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other*-1)

    def __mul__(self, other):
        return self._generic_mul(self, _sympify(other))

    def __rmul__(self, other):
        return self._generic_mul(_sympify(other), self)

    def __pow__(self, p):
        return self.pow(p)

    def __neg__(self):
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    def __truediv__(self, other):
        return self * sympify(other)**-1

    def __rtruediv__(self, other):
        return sympify(other) * self**-1

    def _eval_Integral(self, *args):
        return self.integrate(*args)

    def diff(self, *symbols, **kwargs):
        kwargs.setdefault('evaluate', True)
        return self.func(*[a.diff(*symbols, **kwargs) for a  in self.args])

    def add(self, other):
        """Adds quaternions.

        Parameters
        ==========

        other : Quaternion
            The quaternion to add to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after adding self to other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.add(q2)
        6 + 8*i + 10*j + 12*k
        >>> q1 + 5
        6 + 2*i + 3*j + 4*k
        >>> x = symbols('x', real = True)
        >>> q1.add(x)
        (x + 1) + 2*i + 3*j + 4*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.add(2 + 3*I)
        (5 + 7*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

        """
        q1 = self
        q2 = sympify(other)

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
            elif q2.is_commutative:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)
            else:
                raise ValueError("Only commutative expressions can be added with a Quaternion.")

        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d
                          + q2.d)

    def mul(self, other):
        """Multiplies quaternions.

        Parameters
        ==========

        other : Quaternion or symbol
            The quaternion to multiply to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying self with other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.mul(q2)
        (-60) + 12*i + 30*j + 24*k
        >>> q1.mul(2)
        2 + 4*i + 6*j + 8*k
        >>> x = symbols('x', real = True)
        >>> q1.mul(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        return self._generic_mul(self, _sympify(other))

    @staticmethod
    def _generic_mul(q1, q2):
        """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
        q2 : Quaternion or symbol

        It is important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Symbol, S
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, S(2))
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real = True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        # None is a Quaternion:
        if not isinstance(q1, Quaternion) and not isinstance(q2, Quaternion):
            return q1 * q2

        # If q1 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q1, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If any of the quaternions has a fixed norm, pre-compute norm
        if q1._norm is None and q2._norm is None:
            norm = None
        else:
            norm = q1.norm() * q2.norm()

        return Quaternion(-q1.b*q2.b - q1.c*q2.c - q1.d*q2.d + q1.a*q2.a,
                          q1.b*q2.a + q1.c*q2.d - q1.d*q2.c + q1.a*q2.b,
                          -q1.b*q2.d + q1.c*q2.a + q1.d*q2.b + q1.a*q2.c,
                          q1.b*q2.c - q1.c*q2.b + q1.d*q2.a + q1.a * q2.d,
                          norm=norm)

    def _eval_conjugate(self):
        """Returns the conjugate of the quaternion."""
        q = self
        return Quaternion(q.a, -q.b, -q.c, -q.d, norm=q._norm)

    def norm(self):
        """Returns the norm of the quaternion."""
        if self._norm is None:  # check if norm is pre-defined
            q = self
            # trigsimp is used to simplify sin(x)^2 + cos(x)^2 (these terms
            # arise when from_axis_angle is used).
            return sqrt(trigsimp(q.a**2 + q.b**2 + q.c**2 + q.d**2))

        return self._norm

    def normalize(self):
        """Returns the normalized form of the quaternion."""
        q = self
        return q * (1/q.norm())

    def inverse(self):
        """Returns the inverse of the quaternion."""
        q = self
        if not q.norm():
            raise ValueError("Cannot compute inverse for a quaternion with zero norm")
        return conjugate(q) * (1/q.norm()**2)

    def pow(self, p):
        """Finds the pth power of the quaternion.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            Returns the p-th power of the current quaternion.
            Returns the inverse if p = -1.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k

        """
        try:
            q, p = self, as_int(p)
        except ValueError:
            return NotImplemented

        if p < 0:
            q, p = q.inverse(), -p

        if p == 1:
            return q

        res = Quaternion(1, 0, 0, 0)
        while p > 0:
            if p & 1:
                res *= q
            q *= q
            p >>= 1

        return res

    def exp(self):
        """Returns the exponential of $q$, given by $e^q$.

        Returns
        =======

        Quaternion
            The exponential of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.exp()
        E*cos(sqrt(29))
        + 2*sqrt(29)*E*sin(sqrt(29))/29*i
        + 3*sqrt(29)*E*sin(sqrt(29))/29*j
        + 4*sqrt(29)*E*sin(sqrt(29))/29*k

        """
        # exp(q) = e^a(cos||v|| + v/||v||*sin||v||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        a = exp(q.a) * cos(vector_norm)
        b = exp(q.a) * sin(vector_norm) * q.b / vector_norm
        c = exp(q.a) * sin(vector_norm) * q.c / vector_norm
        d = exp(q.a) * sin(vector_norm) * q.d / vector_norm

        return Quaternion(a, b, c, d)

    def log(self):
        r"""Returns the logarithm of the quaternion, given by $\log q$.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.log()
        log(sqrt(30))
        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i
        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j
        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k

        """
        # log(q) = log||q|| + v/||v||*arccos(a/||q||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        q_norm = q.norm()
        a = ln(q_norm)
        b = q.b * acos(q.a / q_norm) / vector_norm
        c = q.c * acos(q.a / q_norm) / vector_norm
        d = q.d * acos(q.a / q_norm) / vector_norm

        return Quaternion(a, b, c, d)

    def _eval_subs(self, *args):
        elements = [i.subs(*args) for i in self.args]
        norm = self._norm
        if norm is not None:
            norm = norm.subs(*args)
        _check_norm(elements, norm)
        return Quaternion(*elements, norm=norm)

    def _eval_evalf(self, prec):
        """Returns the floating point approximations (decimal numbers) of the quaternion.

        Returns
        =======

        Quaternion
            Floating point approximations of quaternion(self)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import sqrt
        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
        >>> q.evalf()
        1.00000000000000
        + 0.707106781186547*i
        + 0.577350269189626*j
        + 0.500000000000000*k

        """
        nprec = prec_to_dps(prec)
        return Quaternion(*[arg.evalf(n=nprec) for arg in self.args])

    def pow_cos_sin(self, p):
        """Computes the pth power in the cos-sin form.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            The p-th power in the cos-sin form.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow_cos_sin(4)
        900*cos(4*acos(sqrt(30)/30))
        + 1800*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*i
        + 2700*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*j
        + 3600*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*k

        """
        # q = ||q||*(cos(a) + u*sin(a))
        # q^p = ||q||^p * (cos(p*a) + u*sin(p*a))

        q = self
        (v, angle) = q.to_axis_angle()
        q2 = Quaternion.from_axis_angle(v, p * angle)
        return q2 * (q.norm()**p)

    def integrate(self, *args):
        """Computes integration of quaternion.

        Returns
        =======

        Quaternion
            Integration of the quaternion(self) with the given variable.

        Examples
        ========

        Indefinite Integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Definite integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate((x, 1, 5))
        4 + 8*i + 12*j + 16*k

        """
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args),
                          integrate(self.c, *args), integrate(self.d, *args))

    @staticmethod
    def rotate_point(pin, r):
        """Returns the coordinates of the point pin (a 3 tuple) after rotation.

        Parameters
        ==========

        pin : tuple
            A 3-element tuple of coordinates of a point which needs to be
            rotated.
        r : Quaternion or tuple
            Axis and angle of rotation.

            It's important to note that when r is a tuple, it must be of the form
            (axis, angle)

        Returns
        =======

        tuple
            The coordinates of the point after rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)

        """
        if isinstance(r, tuple):
            # if r is of the form (vector, angle)
            q = Quaternion.from_axis_angle(r[0], r[1])
        else:
            # if r is a quaternion
            q = r.normalize()
        pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
        return (pout.b, pout.c, pout.d)

    def to_axis_angle(self):
        """Returns the axis and angle of rotation of a quaternion.

        Returns
        =======

        tuple
            Tuple of (axis, angle)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> axis
        (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)
        >>> angle
        2*pi/3

        """
        q = self
        if q.a.is_negative:
            q = q * -1

        q = q.normalize()
        angle = trigsimp(2 * acos(q.a))

        # Since quaternion is normalised, q.a is less than 1.
        s = sqrt(1 - q.a*q.a)

        x = trigsimp(q.b / s)
        y = trigsimp(q.c / s)
        z = trigsimp(q.d / s)

        v = (x, y, z)
        t = (v, angle)

        return t

    def to_rotation_matrix(self, v=None, homogeneous=True):
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if ``v`` is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None
        homogeneous : bool
            When True, gives an expression that may be more efficient for
            symbolic calculations but less so for direct evaluation. Both
            formulas are mathematically equivalent.
            Default value: True

        Returns
        =======

        tuple
            Returns the equivalent rotation transformation matrix of the quaternion
            which represents rotation about the origin if v is not passed.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.
        """

        q = self
        s = q.norm()**-2

        # diagonal elements are different according to parameter normal
        if homogeneous:
            m00 = s*(q.a**2 + q.b**2 - q.c**2 - q.d**2)
            m11 = s*(q.a**2 - q.b**2 + q.c**2 - q.d**2)
            m22 = s*(q.a**2 - q.b**2 - q.c**2 + q.d**2)
        else:
            m00 = 1 - 2*s*(q.c**2 + q.d**2)
            m11 = 1 - 2*s*(q.b**2 + q.d**2)
            m22 = 1 - 2*s*(q.b**2 + q.c**2)

        m01 = 2*s*(q.b*q.c - q.d*q.a)
        m02 = 2*s*(q.b*q.d + q.c*q.a)

        m10 = 2*s*(q.b*q.c + q.d*q.a)
        m12 = 2*s*(q.c*q.d - q.b*q.a)

        m20 = 2*s*(q.b*q.d - q.c*q.a)
        m21 = 2*s*(q.c*q.d + q.b*q.a)

        if not v:
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        else:
            (x, y, z) = v

            m03 = x - x*m00 - y*m01 - z*m02
            m13 = y - x*m10 - y*m11 - z*m12
            m23 = z - x*m20 - y*m21 - z*m22
            m30 = m31 = m32 = 0
            m33 = 1

            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13],
                          [m20, m21, m22, m23], [m30, m31, m32, m33]])

    def scalar_part(self):
        r"""Returns scalar part($\mathbf{S}(q)$) of the quaternion q.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{S}(q) = a$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.scalar_part()
        4

        """

        return self.a

    def vector_part(self):
        r"""
        Returns $\mathbf{V}(q)$, the vector part of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{V}(q) = bi + cj + dk$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.vector_part()
        0 + 1*i + 1*j + 1*k

        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.vector_part()
        0 + 8*i + 13*j + 12*k

        """

        return Quaternion(0, self.b, self.c, self.d)

    def axis(self):
        r"""
        Returns $\mathbf{Ax}(q)$, the axis of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{Ax}(q)$  i.e., the versor of the vector part of that quaternion
        equal to $\mathbf{U}[\mathbf{V}(q)]$.
        The axis is always an imaginary unit with square equal to $-1 + 0i + 0j + 0k$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.axis()
        0 + sqrt(3)/3*i + sqrt(3)/3*j + sqrt(3)/3*k

        See Also
        ========

        vector_part

        """
        axis = self.vector_part().normalize()

        return Quaternion(0, axis.b, axis.c, axis.d)

    def is_pure(self):
        """
        Returns true if the quaternion is pure, false if the quaternion is not pure
        or returns none if it is unknown.

        Explanation
        ===========

        A pure quaternion (also a vector quaternion) is a quaternion with scalar
        part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 8, 13, 12)
        >>> q.is_pure()
        True

        See Also
        ========
        scalar_part

        """

        return self.a.is_zero

    def is_zero_quaternion(self):
        """
        Returns true if the quaternion is a zero quaternion or false if it is not a zero quaternion
        and None if the value is unknown.

        Explanation
        ===========

        A zero quaternion is a quaternion with both scalar part and
        vector part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q.is_zero_quaternion()
        False

        >>> q = Quaternion(0, 0, 0, 0)
        >>> q.is_zero_quaternion()
        True

        See Also
        ========
        scalar_part
        vector_part

        """

        return self.norm().is_zero

    def angle(self):
        r"""
        Returns the angle of the quaternion measured in the real-axis plane.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$ where $a$, $b$, $c$ and $d$
        are real numbers, returns the angle of the quaternion given by

        .. math::
            \theta := 2 \operatorname{atan_2}\left(\sqrt{b^2 + c^2 + d^2}, {a}\right)

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 4, 4, 4)
        >>> q.angle()
        2*atan(4*sqrt(3))

        """

        return 2 * atan2(self.vector_part().norm(), self.scalar_part())


    def arc_coplanar(self, other):
        """
        Returns True if the transformation arcs represented by the input quaternions happen in the same plane.

        Explanation
        ===========

        Two quaternions are said to be coplanar (in this arc sense) when their axes are parallel.
        The plane of a quaternion is the one normal to its axis.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the planes of the two quaternions are the same, apart from its orientation/sign.
        False : if the planes of the two quaternions are not the same, apart from its orientation/sign.
        None : if plane of either of the quaternion is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(1, 4, 4, 4)
        >>> q2 = Quaternion(3, 8, 8, 8)
        >>> Quaternion.arc_coplanar(q1, q2)
        True

        >>> q1 = Quaternion(2, 8, 13, 12)
        >>> Quaternion.arc_coplanar(q1, q2)
        False

        See Also
        ========

        vector_coplanar
        is_pure

        """
        if (self.is_zero_quaternion()) or (other.is_zero_quaternion()):
            raise ValueError('Neither of the given quaternions can be 0')

        return fuzzy_or([(self.axis() - other.axis()).is_zero_quaternion(), (self.axis() + other.axis()).is_zero_quaternion()])

    @classmethod
    def vector_coplanar(cls, q1, q2, q3):
        r"""
        Returns True if the axis of the pure quaternions seen as 3D vectors
        ``q1``, ``q2``, and ``q3`` are coplanar.

        Explanation
        ===========

        Three pure quaternions are vector coplanar if the quaternions seen as 3D vectors are coplanar.

        Parameters
        ==========

        q1
            A pure Quaternion.
        q2
            A pure Quaternion.
        q3
            A pure Quaternion.

        Returns
        =======

        True : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar.
        False : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are not coplanar.
        None : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(0, 4, 4, 4)
        >>> q2 = Quaternion(0, 8, 8, 8)
        >>> q3 = Quaternion(0, 24, 24, 24)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        True

        >>> q1 = Quaternion(0, 8, 16, 8)
        >>> q2 = Quaternion(0, 8, 3, 12)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        False

        See Also
        ========

        axis
        is_pure

        """

        if fuzzy_not(q1.is_pure()) or fuzzy_not(q2.is_pure()) or fuzzy_not(q3.is_pure()):
            raise ValueError('The given quaternions must be pure')

        M = Matrix([[q1.b, q1.c, q1.d], [q2.b, q2.c, q2.d], [q3.b, q3.c, q3.d]]).det()
        return M.is_zero

    def parallel(self, other):
        """
        Returns True if the two pure quaternions seen as 3D vectors are parallel.

        Explanation
        ===========

        Two pure quaternions are called parallel when their vector product is commutative which
        implies that the quaternions seen as 3D vectors have same direction.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are parallel.
        False : if the two pure quaternions seen as 3D vectors are not parallel.
        None : if the two pure quaternions seen as 3D vectors are parallel is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.parallel(q1)
        True

        >>> q1 = Quaternion(0, 8, 13, 12)
        >>> q.parallel(q1)
        False

        """

        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The provided quaternions must be pure')

        return (self*other - other*self).is_zero_quaternion()

    def orthogonal(self, other):
        """
        Returns the orthogonality of two quaternions.

        Explanation
        ===========

        Two pure quaternions are called orthogonal when their product is anti-commutative.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are orthogonal.
        False : if the two pure quaternions seen as 3D vectors are not orthogonal.
        None : if the two pure quaternions seen as 3D vectors are orthogonal is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.orthogonal(q1)
        False

        >>> q1 = Quaternion(0, 2, 2, 0)
        >>> q = Quaternion(0, 2, -2, 0)
        >>> q.orthogonal(q1)
        True

        """

        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The given quaternions must be pure')

        return (self*other + other*self).is_zero_quaternion()

    def index_vector(self):
        r"""
        Returns the index vector of the quaternion.

        Explanation
        ===========

        The index vector is given by $\mathbf{T}(q)$, the norm (or magnitude) of
        the quaternion $q$, multiplied by $\mathbf{Ax}(q)$, the axis of $q$.

        Returns
        =======

        Quaternion: representing index vector of the provided quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.index_vector()
        0 + 4*sqrt(10)/3*i + 2*sqrt(10)/3*j + 4*sqrt(10)/3*k

        See Also
        ========

        axis
        norm

        """

        return self.norm() * self.axis()

    def mensor(self):
        """
        Returns the natural logarithm of the norm(magnitude) of the quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.mensor()
        log(2*sqrt(10))
        >>> q.norm()
        2*sqrt(10)

        See Also
        ========

        norm

        """

        return ln(self.norm())
<|endoftext|>"""
A Printer which converts an expression into its LaTeX equivalent.
"""
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING

import itertools

from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse


# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str

from sympy.utilities.iterables import has_variety, sift

import re

if TYPE_CHECKING:
    from sympy.tensor.array import NDimArray
    from sympy.vector.basisdependent import BasisDependent

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at
# https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which SymPy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                            'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec',
                            'csc', 'cot', 'coth', 're', 'im', 'frac', 'root',
                            'arg',
                            ]

tex_greek_dictionary = {
    'Alpha': r'\mathrm{A}',
    'Beta': r'\mathrm{B}',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': r'\mathrm{E}',
    'Zeta': r'\mathrm{Z}',
    'Eta': r'\mathrm{H}',
    'Theta': r'\Theta',
    'Iota': r'\mathrm{I}',
    'Kappa': r'\mathrm{K}',
    'Lambda': r'\Lambda',
    'Mu': r'\mathrm{M}',
    'Nu': r'\mathrm{N}',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': r'\mathrm{O}',
    'Pi': r'\Pi',
    'Rho': r'\mathrm{P}',
    'Sigma': r'\Sigma',
    'Tau': r'\mathrm{T}',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': r'\mathrm{X}',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'\mathrm{X}',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

other_symbols = {'aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp'}

# Variable name modifiers
modifier_dict: dict[str, Callable[[str], str]] = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
    'ddot': lambda s: r'\ddot{'+s+r'}',
    'dot': lambda s: r'\dot{'+s+r'}',
    'check': lambda s: r'\check{'+s+r'}',
    'breve': lambda s: r'\breve{'+s+r'}',
    'acute': lambda s: r'\acute{'+s+r'}',
    'grave': lambda s: r'\grave{'+s+r'}',
    'tilde': lambda s: r'\tilde{'+s+r'}',
    'hat': lambda s: r'\hat{'+s+r'}',
    'bar': lambda s: r'\bar{'+s+r'}',
    'vec': lambda s: r'\vec{'+s+r'}',
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'\boldsymbol{'+s+r'}',
    'bm': lambda s: r'\boldsymbol{'+s+r'}',
    'cal': lambda s: r'\mathcal{'+s+r'}',
    'scr': lambda s: r'\mathscr{'+s+r'}',
    'frak': lambda s: r'\mathfrak{'+s+r'}',
    # Brackets
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',
    'abs': lambda s: r'\left|{'+s+r'}\right|',
    'mag': lambda s: r'\left|{'+s+r'}\right|',
}

greek_letters_set = frozenset(greeks)

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # search
    re.compile(r'(\d|\\frac{\d+}{\d+})'),  # match
)


def latex_escape(s: str) -> str:
    """
    Escape a string such that latex interprets it as plaintext.

    We cannot use verbatim easily with mathjax, so escaping is easier.
    Rules from https://tex.stackexchange.com/a/34586/41112.
    """
    s = s.replace('\\', r'\textbackslash')
    for c in '&%$#_{}':
        s = s.replace(c, '\\' + c)
    s = s.replace('~', r'\textasciitilde')
    s = s.replace('^', r'\textasciicircum')
    return s


class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings: dict[str, Any] = {
        "full_prec": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "itex": False,
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_str": None,
        "mode": "plain",
        "mul_symbol": None,
        "order": None,
        "symbol_names": {},
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "gothic_re_im": False,
        "decimal_separator": "period",
        "perm_cyclic": True,
        "parenthesize_super": True,
        "min": None,
        "max": None,
        "diff_operator": "d",
        "adjoint_style": "dagger",
        "disable_split_super_sub": False,
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation',
                           'equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', "
                                 "'equation' or 'equation*'")

        if self._settings['fold_short_frac'] is None and \
                self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        mul_symbol_table = {
            None: r" ",
            "ldot": r" \,.\, ",
            "dot": r" \cdot ",
            "times": r" \times "
        }
        try:
            self._settings['mul_symbol_latex'] = \
                mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            self._settings['mul_symbol_latex'] = \
                self._settings['mul_symbol']
        try:
            self._settings['mul_symbol_latex_numbers'] = \
                mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            if (self._settings['mul_symbol'].strip() in
                    ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']):
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol']

        self._delim_dict = {'(': ')', '[': ']'}

        imaginary_unit_table = {
            None: r"i",
            "i": r"i",
            "ri": r"\mathrm{i}",
            "ti": r"\text{i}",
            "j": r"j",
            "rj": r"\mathrm{j}",
            "tj": r"\text{j}",
        }
        imag_unit = self._settings['imaginary_unit']
        self._settings['imaginary_unit_latex'] = imaginary_unit_table.get(imag_unit, imag_unit)

        diff_operator_table = {
            None: r"d",
            "d": r"d",
            "rd": r"\mathrm{d}",
            "td": r"\text{d}",
        }
        diff_operator = self._settings['diff_operator']
        self._settings["diff_operator_latex"] = diff_operator_table.get(diff_operator, diff_operator)

    def _add_parens(self, s) -> str:
        return r"\left({}\right)".format(s)

    # TODO: merge this with the above, which requires a lot of test changes
    def _add_parens_lspace(self, s) -> str:
        return r"\left( {}\right)".format(s)

    def parenthesize(self, item, level, is_neg=False, strict=False) -> str:
        prec_val = precedence_traditional(item)
        if is_neg and strict:
            return self._add_parens(self._print(item))

        if (prec_val < level) or ((not strict) and prec_val <= level):
            return self._add_parens(self._print(item))
        else:
            return self._print(item)

    def parenthesize_super(self, s):
        """
        Protect superscripts in s

        If the parenthesize_super option is set, protect with parentheses, else
        wrap in braces.
        """
        if "^" in s:
            if self._settings['parenthesize_super']:
                return self._add_parens(s)
            else:
                return "{{{}}}".format(s)
        return s

    def doprint(self, expr) -> str:
        tex = Printer.doprint(self, expr)

        if self._settings['mode'] == 'plain':
            return tex
        elif self._settings['mode'] == 'inline':
            return r"$%s$" % tex
        elif self._settings['itex']:
            return r"$$%s$$" % tex
        else:
            env_str = self._settings['mode']
            return r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)

    def _needs_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))

    def _needs_function_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        if not self._needs_brackets(expr):
            return False
        else:
            # Muls of the form a*b*c... can be folded
            if expr.is_Mul and not self._mul_is_clean(expr):
                return True
            # Pows which don't need brackets can be folded
            elif expr.is_Pow and not self._pow_is_clean(expr):
                return True
            # Add and Function always need brackets
            elif expr.is_Add or expr.is_Function:
                return True
            else:
                return False

    def _needs_mul_brackets(self, expr, first=False, last=False) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        ``first=True`` specifies that this expr is the first to appear in
        a Mul.
        """
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral

        if expr.is_Mul:
            if not first and expr.could_extract_minus_sign():
                return True
        elif precedence_traditional(expr) < PRECEDENCE["Mul"]:
            return True
        elif expr.is_Relational:
            return True
        if expr.is_Piecewise:
            return True
        if any(expr.has(x) for x in (Mod,)):
            return True
        if (not last and
                any(expr.has(x) for x in (Integral, Product, Sum))):
            return True

        return False

    def _needs_add_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of an Add, False otherwise.  This is False for most
        things.
        """
        if expr.is_Relational:
            return True
        if any(expr.has(x) for x in (Mod,)):
            return True
        if expr.is_Add:
            return True
        return False

    def _mul_is_clean(self, expr) -> bool:
        for arg in expr.args:
            if arg.is_Function:
                return False
        return True

    def _pow_is_clean(self, expr) -> bool:
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr: str, exp):
        if exp is not None:
            return r"\left(%s\right)^{%s}" % (expr, exp)
        else:
            return expr

    def _print_Basic(self, expr):
        name = self._deal_with_super_sub(expr.__class__.__name__)
        if expr.args:
            ls = [self._print(o) for o in expr.args]
            s = r"\operatorname{{{}}}\left({}\right)"
            return s.format(name, ", ".join(ls))
        else:
            return r"\text{{{}}}".format(name)

    def _print_bool(self, e: bool | BooleanTrue | BooleanFalse):
        return r"\text{%s}" % e

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        return r"\text{%s}" % e

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif term.could_extract_minus_sign():
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex

        return tex

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation
        from sympy.utilities.exceptions import sympy_deprecation_warning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=8,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            return self._print_Cycle(expr)

        if expr.size == 0:
            return r"\left( \right)"

        lower = [self._print(arg) for arg in expr.array_form]
        upper = [self._print(arg) for arg in range(len(lower))]

        row1 = " & ".join(upper)
        row2 = " & ".join(lower)
        mat = r" \\ ".join((row1, row2))
        return r"\begin{pmatrix} %s \end{pmatrix}" % mat


    def _print_AppliedPermutation(self, expr):
        perm, var = expr.args
        return r"\sigma_{%s}(%s)" % (self._print(perm), self._print(var))

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        strip = False if self._settings['full_prec'] else True
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]
            if self._settings['decimal_separator'] == 'comma':
                mant = mant.replace('.','{,}')

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"\infty"
        elif str_real == "-inf":
            return r"- \infty"
        else:
            if self._settings['decimal_separator'] == 'comma':
                str_real = str_real.replace('.','{,}')
            return str_real

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                 self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return r"\nabla %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Laplacian(self, expr):
        func = expr._expr
        return r"\Delta %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Mul(self, expr: Expr):
        from sympy.simplify import fraction
        separator: str = self._settings['mul_symbol_latex']
        numbersep: str = self._settings['mul_symbol_latex_numbers']

        def convert(expr) -> str:
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)

                # If there are quantities or prefixes, append them at the back.
                units, nonunits = sift(args, lambda x: (hasattr(x, "_scale_factor") or hasattr(x, "is_physical_constant")) or
                              (isinstance(x, Pow) and
                               hasattr(x.base, "is_physical_constant")), binary=True)
                prefixes, units = sift(units, lambda x: hasattr(x, "_scale_factor"), binary=True)
                return convert_args(nonunits + prefixes + units)

        def convert_args(args) -> str:
            _tex = last_term_tex = ""

            for i, term in enumerate(args):
                term_tex = self._print(term)
                if not (hasattr(term, "_scale_factor") or hasattr(term, "is_physical_constant")):
                    if self._needs_mul_brackets(term, first=(i == 0),
                                                last=(i == len(args) - 1)):
                        term_tex = r"\left(%s\right)" % term_tex

                    if  _between_two_numbers_p[0].search(last_term_tex) and \
                        _between_two_numbers_p[1].match(term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator
                elif _tex:
                    _tex += separator

                _tex += term_tex
                last_term_tex = term_tex
            return _tex

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        # XXX: _print_Pow calls this routine with instances of Pow...
        if isinstance(expr, Mul):
            args = expr.args
            if args[0] is S.One or any(isinstance(arg, Number) for arg in args[1:]):
                return convert_args(args)

        include_parens = False
        if expr.could_extract_minus_sign():
            expr = -expr
            tex = "- "
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        numer, denom = fraction(expr, exact=True)

        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)

        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] and ldenom <= 2 and \
                    "^" not in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            elif ratio is not None and \
                    len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"\frac{1}{%s}%s\left(%s\right)" \
                        % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    # split a long numerator
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or \
                                len(convert(a*x).split()) > ratio*ldenom or \
                                (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"\frac{%s}{%s}%s\left(%s\right)" \
                            % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += r"\frac{%s}{%s}%s%s" \
                            % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += r"\frac{1}{%s}%s%s" % (sdenom, separator, snumer)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_PrimeIdeal(self, expr):
        p = self._print(expr.p)
        if expr.is_inert:
            return rf'\left({p}\right)'
        alpha = self._print(expr.alpha.as_expr())
        return rf'\left({p}, {alpha}\right)'

    def _print_Pow(self, expr: Pow):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational:
            p: int = expr.exp.p  # type: ignore
            q: int = expr.exp.q  # type: ignore
            if abs(p) == 1 and q != 1 and self._settings['root_notation']:
                base = self._print(expr.base)
                if q == 2:
                    tex = r"\sqrt{%s}" % base
                elif self._settings['itex']:
                    tex = r"\root{%d}{%s}" % (q, base)
                else:
                    tex = r"\sqrt[%d]{%s}" % (q, base)
                if expr.exp.is_negative:
                    return r"\frac{1}{%s}" % tex
                else:
                    return tex
            elif self._settings['fold_frac_powers'] and q != 1:
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                # issue #12886: add parentheses for superscripts raised to powers
                if expr.base.is_Symbol:
                    base = self.parenthesize_super(base)
                if expr.base.is_Function:
                    return self._print(expr.base, exp="%s/%s" % (p, q))
                return r"%s^{%s/%s}" % (base, p, q)
            elif expr.exp.is_negative and expr.base.is_commutative:
                # special case for 1^(-x), issue 9216
                if expr.base == 1:
                    return r"%s^{%s}" % (expr.base, expr.exp)
                # special case for (1/x)^(-y) and (-1/-x)^(-y), issue 20252
                if expr.base.is_Rational:
                    base_p: int = expr.base.p  # type: ignore
                    base_q: int = expr.base.q  # type: ignore
                    if base_p * base_q == abs(base_q):
                        if expr.exp == -1:
                            return r"\frac{1}{\frac{%s}{%s}}" % (base_p, base_q)
                        else:
                            return r"\frac{1}{(\frac{%s}{%s})^{%s}}" % (base_p, base_q, abs(expr.exp))
                # things like 1/x
                return self._print_Mul(expr)
        if expr.base.is_Function:
            return self._print(expr.base, exp=self._print(expr.exp))
        tex = r"%s^{%s}"
        return self._helper_print_standard_power(expr, tex)

    def _helper_print_standard_power(self, expr, template: str) -> str:
        exp = self._print(expr.exp)
        # issue #12886: add parentheses around superscripts raised
        # to powers
        base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
        if expr.base.is_Symbol:
            base = self.parenthesize_super(base)
        elif expr.base.is_Float:
            base = r"{%s}" % base
        elif (isinstance(expr.base, Derivative)
            and base.startswith(r'\left(')
            and re.match(r'\\left\(\\d?d?dot', base)
            and base.endswith(r'\right)')):
            # don't use parentheses around dotted derivative
            base = base[6: -7]  # remove outermost added parens
        return template % (base, exp)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex

    def _print_Product(self, expr):
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex

    def _print_BasisDependent(self, expr: 'BasisDependent'):
        from sympy.vector import Vector

        o1: list[str] = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = r'\left(' + self._print(v) + r'\right)'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)
        tex = '{'+tex_base+'}'+'_{%s}' % ','.join(
            map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Idx(self, expr):
        label = self._print(expr.label)
        if expr.upper is not None:
            upper = self._print(expr.upper)
            if expr.lower is not None:
                lower = self._print(expr.lower)
            else:
                lower = self._print(S.Zero)
            interval = '{lower}\\mathrel{{..}}\\nobreak {upper}'.format(
                    lower = lower, upper = upper)
            return '{{{label}}}_{{{interval}}}'.format(
                label = label, interval = interval)
        #if no bounds are defined this just prints the label
        return label

    def _print_Derivative(self, expr):
        if requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = self._settings["diff_operator_latex"]

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol,
                                        self.parenthesize_super(self._print(x)),
                                        self._print(num))

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, self._print(dim), tex)

        if any(i.could_extract_minus_sign() for i in expr.args):
            return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=True,
                                                  strict=True))

        return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=False,
                                                  strict=True))

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr,
                                                         latex_subs)

    def _print_Integral(self, expr):
        tex, symbols = "", []
        diff_symbol = self._settings["diff_operator_latex"]

        # Only up to \iiiint exists
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            symbols = [r"\, %s%s" % (diff_symbol, self._print(symbol[0]))
                       for symbol in expr.limits]

        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"

                if len(lim) > 1:
                    if self._settings['mode'] != 'inline' \
                            and not self._settings['itex']:
                        tex += r"\limits"

                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]),
                                               self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % (self._print(lim[1]))

                symbols.insert(0, r"\, %s%s" % (diff_symbol, self._print(symbol)))

        return r"%s %s%s" % (tex, self.parenthesize(expr.function,
                                                    PRECEDENCE["Mul"],
                                                    is_neg=any(i.could_extract_minus_sign() for i in expr.args),
                                                    strict=True),
                             "".join(symbols))

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))

    def _hprint_Function(self, func: str) -> str:
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, excluding sub- and superscripts, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        func = self._deal_with_super_sub(func)
        superscriptidx = func.find("^")
        subscriptidx = func.find("_")
        if func in accepted_latex_functions:
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or superscriptidx == 1:
            name = func
        else:
            if superscriptidx > 0 and subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:min(subscriptidx,superscriptidx)],
                    func[min(subscriptidx,superscriptidx):])
            elif superscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:superscriptidx],
                    func[superscriptidx:])
            elif subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:subscriptidx],
                    func[subscriptidx:])
            else:
                name = r"\operatorname{%s}" % func
        return name

    def _print_Function(self, expr: Function, exp=None) -> str:
        r'''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and \
                not isinstance(expr, AppliedUndef):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [str(self._print(arg)) for arg in expr.args]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = [
                "asin", "acos", "atan",
                "acsc", "asec", "acot",
                "asinh", "acosh", "atanh",
                "acsch", "asech", "acoth",
            ]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    pass
                elif inv_trig_style == "full":
                    func = ("ar" if func[-1] == "h" else "arc") + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        can_fold_brackets = False

            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = r"\%s^{-1}" % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func
            elif exp is not None:
                func_tex = self._hprint_Function(func)
                func_tex = self.parenthesize_super(func_tex)
                name = r'%s^{%s}' % (func_tex, exp)
            else:
                name = self._hprint_Function(func)

            if can_fold_brackets:
                if func in accepted_latex_functions:
                    # Wrap argument safely to avoid parse-time conflicts
                    # with the function name itself
                    name += r" {%s}"
                else:
                    name += r"%s"
            else:
                name += r"{\left(%s \right)}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    def _print_ElementwiseApplyFunction(self, expr):
        return r"{%s}_{\circ}\left({%s}\right)" % (
            self._print(expr.function),
            self._print(expr.expr),
        )

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _print_IdentityFunction(self, expr):
        return r"\left( x \mapsto x \right)"

    def _hprint_variadic_function(self, expr, exp=None) -> str:
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % (str(expr.func).lower(),
                                       ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = r"\left\lfloor{%s}\right\rfloor" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\left\lceil{%s}\right\rceil" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings["ln_notation"]:
            tex = r"\log{\left(%s \right)}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = r"\left|{%s}\right|" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_re(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{re}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{im}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_Not(self, e):
        from sympy.logic.boolalg import (Equivalent, Implies)
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg \left(%s\right)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])

    def _print_LogOp(self, args, char):
        arg = args[0]
        if arg.is_Boolean and not arg.is_Not:
            tex = r"\left(%s\right)" % self._print(arg)
        else:
            tex = r"%s" % self._print(arg)

        for arg in args[1:]:
            if arg.is_Boolean and not arg.is_Not:
                tex += r" %s \left(%s\right)" % (char, self._print(arg))
            else:
                tex += r" %s %s" % (char, self._print(arg))

        return tex

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\wedge")

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\vee")

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\veebar")

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or r"\Rightarrow")

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    def _print_conjugate(self, expr, exp=None):
        tex = r"\overline{%s}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (func, exp, arg)
        else:
            return r"%s%s" % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        tex = r"e^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    def _print_Exp1(self, expr, exp=None):
        return "e"

    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]),
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex

    def _print_beta(self, expr, exp=None):
        x = expr.args[0]
        # Deal with unevaluated single argument beta
        y = expr.args[0] if len(expr.args) == 1 else expr.args[1]
        tex = rf"\left({x}, {y}\right)"

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_betainc(self, expr, exp=None, operator='B'):
        largs = [self._print(arg) for arg in expr.args]
        tex = r"\left(%s, %s\right)" % (largs[0], largs[1])

        if exp is not None:
            return r"\operatorname{%s}_{(%s, %s)}^{%s}%s" % (operator, largs[2], largs[3], exp, tex)
        else:
            return r"\operatorname{%s}_{(%s, %s)}%s" % (operator, largs[2], largs[3], tex)

    def _print_betainc_regularized(self, expr, exp=None):
        return self._print_betainc(expr, exp, operator='I')

    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    def _hprint_one_arg_func(self, expr, exp=None) -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"\left(%s\right)^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial2(self, expr, exp=None):
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_binomial(self, expr, exp=None):
        tex = r"{\binom{%s}{%s}}" % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        return self._do_exponent(tex, exp)

    def _hprint_BesselBase(self, expr, exp, sym: str) -> str:
        tex = r"%s" % (sym)

        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, exp)
            else:
                need_exp = True

        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex

    def _hprint_vec(self, vec) -> str:
        if not vec:
            return ""
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')

    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (exp, tex)
        return r"\eta%s" % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\zeta^{%s}%s" % (exp, tex)
        return r"\zeta%s" % tex

    def _print_stieltjes(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_{%s}\left(%s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"_{%s}" % self._print(expr.args[0])
