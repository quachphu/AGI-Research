        return mrow

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        if expr == expr.zero:
            # Not clear if this is ever called
            return self._print(expr.zero)
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        mrow = self.dom.createElement('mrow')
        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x:x[0].__str__())
            for i, (k, v) in enumerate(inneritems):
                if v == 1:
                    if i: # No + for first item
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
                elif v == -1:
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('-'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
                else:
                    if i: # No + for first item
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mbrac = self.dom.createElement('mrow')
                    mbrac.appendChild(self._l_paren())
                    mbrac.appendChild(self._print(v))
                    mbrac.appendChild(self._r_paren())
                    mrow.appendChild(mbrac)
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
        return mrow


    def _print_And(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x2227;')

    def _print_Or(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x2228;')

    def _print_Xor(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x22BB;')

    def _print_Implies(self, expr):
        return self._print_LogOp(expr.args, '&#x21D2;')

    def _print_Equivalent(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x21D4;')

    def _print_Not(self, e):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xAC;'))
        mrow.appendChild(mo)
        if (e.args[0].is_Boolean):
            x = self.dom.createElement('mrow')
            x.appendChild(self._l_paren())
            x.appendChild(self._print(e.args[0]))
            x.appendChild(self._r_paren())
        else:
            x = self._print(e.args[0])
        mrow.appendChild(x)
        return mrow

    def _print_bool(self, e):
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        return mi

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        return mi

    def _print_Range(self, s):
        dots = "\u2026"
        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            it = iter(s)
            printset = next(it), next(it), dots
        elif len(s) > 4:
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            printset = tuple(s)
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_brace())
        for i, el in enumerate(printset):
            if i:
                brac.appendChild(self._comma())
            if el == dots:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(dots))
                brac.appendChild(mi)
            else:
                brac.appendChild(self._print(el))
        brac.appendChild(self._r_brace())
        return brac

    def _hprint_variadic_function(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode((str(expr.func)).lower()))
        mrow.appendChild(mo)
        mrow.appendChild(self._paren_comma_separated(*args))
        return mrow

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_exp(self, expr):
        msup = self.dom.createElement('msup')
        msup.appendChild(self._print_Exp1(None))
        msup.appendChild(self._print(expr.args[0]))
        return msup

    def _print_Relational(self, e):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.lhs))
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(x)
        mrow.appendChild(self._print(e.rhs))
        return mrow

    def _print_int(self, p):
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element

    def _print_BaseScalar(self, e):
        msub = self.dom.createElement('msub')
        index, system = e._id
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._variable_names[index]))
        msub.appendChild(mi)
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._name))
        msub.appendChild(mi)
        return msub

    def _print_BaseVector(self, e):
        msub = self.dom.createElement('msub')
        index, system = e._id
        mover = self.dom.createElement('mover')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._vector_names[index]))
        mover.appendChild(mi)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('^'))
        mover.appendChild(mo)
        msub.appendChild(mover)
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._name))
        msub.appendChild(mi)
        return msub

    def _print_VectorZero(self, e):
        mover = self.dom.createElement('mover')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode("0"))
        mover.appendChild(mi)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('^'))
        mover.appendChild(mo)
        return mover

    def _print_Cross(self, expr):
        mrow = self.dom.createElement('mrow')
        vec1 = expr._expr1
        vec2 = expr._expr2
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xD7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        return mrow

    def _print_Curl(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xD7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Divergence(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Dot(self, expr):
        mrow = self.dom.createElement('mrow')
        vec1 = expr._expr1
        vec2 = expr._expr2
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        return mrow

    def _print_Gradient(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Laplacian(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2206;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Integers(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2124;'))
        return x

    def _print_Complexes(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2102;'))
        return x

    def _print_Reals(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x211D;'))
        return x

    def _print_Naturals(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        return x

    def _print_Naturals0(self, e):
        sub = self.dom.createElement('msub')
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        sub.appendChild(x)
        sub.appendChild(self._print(S.Zero))
        return sub

    def _print_SingularityFunction(self, expr):
        shift = expr.args[0] - expr.args[1]
        power = expr.args[2]
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u27e8'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u27e9'))
        brac = self.dom.createElement('mrow')
        brac.appendChild(left)
        brac.appendChild(self._print(shift))
        brac.appendChild(right)
        sup = self.dom.createElement('msup')
        sup.appendChild(brac)
        sup.appendChild(self._print(power))
        return sup

    def _print_NaN(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('NaN'))
        return x

    def _print_number_function(self, e, name):
        # Print name_arg[0] for one argument or name_arg[0](arg[1])
        # for more than one argument
        sub = self.dom.createElement('msub')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(name))
        sub.appendChild(mi)
        sub.appendChild(self._print(e.args[0]))
        if len(e.args) == 1:
            return sub
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(sub)
        mrow.appendChild(self._paren_comma_separated(*e.args[1:]))
        return mrow

    def _print_bernoulli(self, e):
        return self._print_number_function(e, 'B')

    _print_bell = _print_bernoulli

    def _print_catalan(self, e):
        return self._print_number_function(e, 'C')

    def _print_euler(self, e):
        return self._print_number_function(e, 'E')

    def _print_fibonacci(self, e):
        return self._print_number_function(e, 'F')

    def _print_lucas(self, e):
        return self._print_number_function(e, 'L')

    def _print_stieltjes(self, e):
        return self._print_number_function(e, '&#x03B3;')

    def _print_tribonacci(self, e):
        return self._print_number_function(e, 'T')

    def _print_ComplexInfinity(self, e):
        x = self.dom.createElement('mover')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x221E;'))
        x.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('~'))
        x.appendChild(mo)
        return x

    def _print_EmptySet(self, e):
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x2205;'))
        return x

    def _print_UniversalSet(self, e):
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x1D54C;'))
        return x

    def _print_Adjoint(self, expr):
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(mat))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2020;'))
        sup.appendChild(mo)
        return sup

    def _print_Transpose(self, expr):
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(mat))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        sup.appendChild(mo)
        return sup

    def _print_Inverse(self, expr):
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(mat))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        sup.appendChild(self._print(-1))
        return sup

    def _print_MatMul(self, expr):
        from sympy.matrices.expressions.matmul import MatMul

        x = self.dom.createElement('mrow')
        args = expr.args
        if isinstance(args[0], Mul):
            args = args[0].as_ordered_factors() + list(args[1:])
        else:
            args = list(args)

        if isinstance(expr, MatMul) and expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            x.appendChild(mo)

        for arg in args[:-1]:
            x.appendChild(self.parenthesize(arg, precedence_traditional(expr),
                                            False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
            x.appendChild(mo)
        x.appendChild(self.parenthesize(args[-1], precedence_traditional(expr),
                                        False))
        return x

    def _print_MatPow(self, expr):
        from sympy.matrices import MatrixSymbol
        base, exp = expr.base, expr.exp
        sup = self.dom.createElement('msup')
        if not isinstance(base, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(base))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(base))
        sup.appendChild(self._print(exp))
        return sup

    def _print_HadamardProduct(self, expr):
        x = self.dom.createElement('mrow')
        args = expr.args
        for arg in args[:-1]:
            x.appendChild(
                self.parenthesize(arg, precedence_traditional(expr), False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&#x2218;'))
            x.appendChild(mo)
        x.appendChild(
            self.parenthesize(args[-1], precedence_traditional(expr), False))
        return x

    def _print_ZeroMatrix(self, Z):
        x = self.dom.createElement('mn')
        x.appendChild(self.dom.createTextNode('&#x1D7D8'))
        return x

    def _print_OneMatrix(self, Z):
        x = self.dom.createElement('mn')
        x.appendChild(self.dom.createTextNode('&#x1D7D9'))
        return x

    def _print_Identity(self, I):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x1D540;'))
        return x

    def _print_floor(self, e):
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u230A'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u230B'))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(self._print(e.args[0]))
        mrow.appendChild(right)
        return mrow

    def _print_ceiling(self, e):
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u2308'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u2309'))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(self._print(e.args[0]))
        mrow.appendChild(right)
        return mrow

    def _print_Lambda(self, e):
        mrow = self.dom.createElement('mrow')
        symbols = e.args[0]
        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(symbols)
        mrow.appendChild(self._l_paren())
        mrow.appendChild(symbols)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x21A6;'))
        mrow.appendChild(mo)
        mrow.appendChild(self._print(e.args[1]))
        mrow.appendChild(self._r_paren())
        return mrow

    def _print_tuple(self, e):
        return self._paren_comma_separated(*e)

    def _print_IndexedBase(self, e):
        return self._print(e.label)

    def _print_Indexed(self, e):
        x = self.dom.createElement('msub')
        x.appendChild(self._print(e.base))
        if len(e.indices) == 1:
            x.appendChild(self._print(e.indices[0]))
            return x
        x.appendChild(self._print(e.indices))
        return x

    def _print_MatrixElement(self, e):
        x = self.dom.createElement('msub')
        x.appendChild(self.parenthesize(e.parent, PRECEDENCE["Atom"], strict = True))
        brac = self.dom.createElement('mrow')
        for i, arg in enumerate(e.indices):
            if i:
                brac.appendChild(self._comma())
            brac.appendChild(self._print(arg))
        x.appendChild(brac)
        return x

    def _print_elliptic_f(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a5;'))
        x.appendChild(mi)
        x.appendChild(self._paren_bar_separated(*e.args))
        return x

    def _print_elliptic_e(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a4;'))
        x.appendChild(mi)
        x.appendChild(self._paren_bar_separated(*e.args))
        return x

    def _print_elliptic_pi(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d6f1;'))
        x.appendChild(mi)
        y = self.dom.createElement('mrow')
        y.appendChild(self._l_paren())
        if len(e.args) == 2:
            n, m = e.args
            y.appendChild(self._print(n))
            y.appendChild(self._bar())
            y.appendChild(self._print(m))
        else:
            n, m, z = e.args
            y.appendChild(self._print(n))
            y.appendChild(self._semicolon())
            y.appendChild(self._print(m))
            y.appendChild(self._bar())
            y.appendChild(self._print(z))
        y.appendChild(self._r_paren())
        x.appendChild(y)
        return x

    def _print_Ei(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Ei'))
        x.appendChild(mi)
        x.appendChild(self._print(e.args))
        return x

    def _print_expint(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('E'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_jacobi(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:3]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[3:]))
        return x

    def _print_gegenbauer(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('C'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_chebyshevt(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_chebyshevu(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('U'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_legendre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_assoc_legendre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_laguerre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('L'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_assoc_laguerre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('L'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_hermite(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('H'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x


@print_function(MathMLPrinterBase)
def mathml(expr, printer='content', **settings):
    """Returns the MathML representation of expr. If printer is presentation
    then prints Presentation MathML else prints content MathML.
    """
    if printer == 'presentation':
        return MathMLPresentationPrinter(settings).doprint(expr)
    else:
        return MathMLContentPrinter(settings).doprint(expr)


def print_mathml(expr, printer='content', **settings):
    """
    Prints a pretty representation of the MathML code for expr. If printer is
    presentation then prints Presentation MathML else prints content MathML.

    Examples
    ========

    >>> ##
    >>> from sympy import print_mathml
    >>> from sympy.abc import x
    >>> print_mathml(x+1) #doctest: +NORMALIZE_WHITESPACE
    <apply>
        <plus/>
        <ci>x</ci>
        <cn>1</cn>
    </apply>
    >>> print_mathml(x+1, printer='presentation')
    <mrow>
        <mi>x</mi>
        <mo>+</mo>
        <mn>1</mn>
    </mrow>

    """
    if printer == 'presentation':
        s = MathMLPresentationPrinter(settings)
    else:
        s = MathMLContentPrinter(settings)
    xml = s._print(sympify(expr))
    pretty_xml = xml.toprettyxml()

    print(pretty_xml)


# For backward compatibility
MathMLPrinter = MathMLContentPrinter
<|endoftext|>def pprint_nodes(subtrees):
    """
    Prettyprints systems of nodes.

    Examples
    ========

    >>> from sympy.printing.tree import pprint_nodes
    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))
    +-a
    +-b1
    | b2
    +-c

    """
    def indent(s, type=1):
        x = s.split("\n")
        r = "+-%s\n" % x[0]
        for a in x[1:]:
            if a == "":
                continue
            if type == 1:
                r += "| %s\n" % a
            else:
                r += "  %s\n" % a
        return r
    if not subtrees:
        return ""
    f = ""
    for a in subtrees[:-1]:
        f += indent(a)
    f += indent(subtrees[-1], 2)
    return f


def print_node(node, assumptions=True):
    """
    Returns information about the "node".

    This includes class name, string representation and assumptions.

    Parameters
    ==========

    assumptions : bool, optional
        See the ``assumptions`` keyword in ``tree``
    """
    s = "%s: %s\n" % (node.__class__.__name__, str(node))

    if assumptions:
        d = node._assumptions
    else:
        d = None

    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += "%s: %s\n" % (a, v)

    return s


def tree(node, assumptions=True):
    """
    Returns a tree representation of "node" as a string.

    It uses print_node() together with pprint_nodes() on node.args recursively.

    Parameters
    ==========

    assumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer`, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    See Also
    ========

    print_tree

    """
    subtrees = []
    for arg in node.args:
        subtrees.append(tree(arg, assumptions=assumptions))
    s = print_node(node, assumptions=assumptions) + pprint_nodes(subtrees)
    return s


def print_tree(node, assumptions=True):
    """
    Prints a tree representation of "node".

    Parameters
    ==========

    assumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer`, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    Examples
    ========

    >>> from sympy.printing import print_tree
    >>> from sympy import Symbol
    >>> x = Symbol('x', odd=True)
    >>> y = Symbol('y', even=True)

    Printing with full assumptions information:

    >>> print_tree(y**x)
    Pow: y**x
    +-Symbol: y
    | algebraic: True
    | commutative: True
    | complex: True
    | even: True
    | extended_real: True
    | finite: True
    | hermitian: True
    | imaginary: False
    | infinite: False
    | integer: True
    | irrational: False
    | noninteger: False
    | odd: False
    | rational: True
    | real: True
    | transcendental: False
    +-Symbol: x
      algebraic: True
      commutative: True
      complex: True
      even: False
      extended_nonzero: True
      extended_real: True
      finite: True
      hermitian: True
      imaginary: False
      infinite: False
      integer: True
      irrational: False
      noninteger: False
      nonzero: True
      odd: True
      rational: True
      real: True
      transcendental: False
      zero: False

    Hiding the assumptions:

    >>> print_tree(y**x, assumptions=False)
    Pow: y**x
    +-Symbol: y
    +-Symbol: x

    See Also
    ========

    tree

    """
    print(tree(node, assumptions=assumptions))
<|endoftext|>"""
Julia code printer

The `JuliaCodePrinter` converts SymPy expressions into Julia expressions.

A complete code generator, which uses `julia_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

from __future__ import annotations
from typing import Any

from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search

# List of known functions.  First, those that have the same name in
# SymPy and Julia. This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "atan", "acot", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "sech", "csch",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "atan2", "sign", "floor", "log", "exp",
                   "cbrt", "sqrt", "erf", "erfc", "erfi",
                   "factorial", "gamma", "digamma", "trigamma",
                   "polygamma", "beta",
                   "airyai", "airyaiprime", "airybi", "airybiprime",
                   "besselj", "bessely", "besseli", "besselk",
                   "erfinv", "erfcinv"]
# These functions have different names ("SymPy": "Julia"), more
# generally a mapping to (argument_conditions, julia_function).
known_fcns_src2 = {
    "Abs": "abs",
    "ceiling": "ceil",
    "conjugate": "conj",
    "hankel1": "hankelh1",
    "hankel2": "hankelh2",
    "im": "imag",
    "re": "real"
}


class JuliaCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Julia code.
    """
    printmethod = "_julia"
    language = "Julia"

    _operators = {
        'and': '&&',
        'or': '||',
        'not': '!',
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'inline': True,
    })
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Julia.

    def __init__(self, settings={}):
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s" % codestring


    def _get_comment(self, text):
        return "# {}".format(text)


    def _declare_number_const(self, name, value):
        return "const {} = {}".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Julia uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Julia arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines


    def _print_Mul(self, expr):
        # print complex numbers nicely in Julia
        if (expr.is_number and expr.is_imaginary and
                expr.as_coeff_Mul()[0].is_integer):
            return "%sim" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity and item.p == 1:
                # Save the Rational type in julia Unless the numerator is 1.
                # For example:
                # julia_code(Rational(3, 7)*x) --> (3 // 7) * x
                # julia_code(x/3) --> x / 3 but not x * (1 // 3)
                b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = "%s %s %s" % (r, mulsym, a_str[i])
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return "%s %s %s" % (sign+multjoin(a, a_str), divsym, b_str[0])
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return "%s %s (%s)" % (sign + multjoin(a, a_str), divsym, multjoin(b, b_str))

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Pow(self, expr):
        powsymbol = '^' if all(x.is_number for x in expr.args) else '.^'

        PREC = precedence(expr)

        if equal_valued(expr.exp, 0.5):
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return "1 %s sqrt(%s)" % (sym, self._print(expr.base))
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return  "1 %s %s" % (sym, self.parenthesize(expr.base, PREC))

        return '%s %s %s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s ^ %s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))


    def _print_Pi(self, expr):
        if self._settings["inline"]:
            return "pi"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_ImaginaryUnit(self, expr):
        return "im"


    def _print_Exp1(self, expr):
        if self._settings["inline"]:
            return "e"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_EulerGamma(self, expr):
        if self._settings["inline"]:
            return "eulergamma"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_Catalan(self, expr):
        if self._settings["inline"]:
            return "catalan"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_GoldenRatio(self, expr):
        if self._settings["inline"]:
            return "golden"
        else:
            return super()._print_NumberSymbol(expr)


    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        # Copied from codeprinter, but remove special MatrixSymbol treatment
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))


    def _print_Infinity(self, expr):
        return 'Inf'


    def _print_NegativeInfinity(self, expr):
        return '-Inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return 'Any[' + ', '.join(self._print(a) for a in expr) + ']'


    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")
    _print_Tuple = _print_tuple


    def _print_BooleanTrue(self, expr):
        return "true"


    def _print_BooleanFalse(self, expr):
        return "false"


    def _print_bool(self, expr):
        return str(expr).lower()


    # Could generate quadrature code for definite Integrals?
    #_print_Integral = _print_not_supported


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            return "[%s]" % A[0, 0]
        elif A.rows == 1:
            return "[%s]" % A.table(self, rowstart='', rowend='', colsep=' ')
        elif A.cols == 1:
            # note .table would unnecessarily equispace the rows
            return "[%s]" % ", ".join([self._print(a) for a in A])
        return "[%s]" % A.table(self, rowstart='', rowend='',
                                rowsep=';\n', colsep=' ')


    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()
        # make row vectors of the indices and entries
        I = Matrix([k[0] + 1 for k in L])
        J = Matrix([k[1] + 1 for k in L])
        AIJ = Matrix([k[2] for k in L])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s,%s]' % (expr.i + 1, expr.j + 1)


    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return (self._print(expr.parent) + '[' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ',' +
                strslice(expr.colslice, expr.parent.shape[1]) + ']')


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s[%s]" % (self._print(expr.base.label), ",".join(inds))

    def _print_Identity(self, expr):
        return "eye(%s)" % self._print(expr.shape[0])

    def _print_HadamardProduct(self, expr):
        return ' .* '.join([self.parenthesize(arg, precedence(expr))
                          for arg in expr.args])

    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([
            self.parenthesize(expr.base, PREC),
            self.parenthesize(expr.exp, PREC)
            ])

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        return "%s // %s" % (expr.p, expr.q)

    # Note: as of 2022, Julia doesn't have spherical Bessel functions
    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*besselj(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*bessely(expr.order + S.Half, x)
        return self._print(expr2)

    def _print_sinc(self, expr):
        # Julia has the normalized sinc function
        return "sinc({})".format(self._print(expr.args[0] / S.Pi))

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if self._settings["inline"]:
            # Express each (cond, expr) pair in a nested Horner form:
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # Expressions that result in multiple statements won't work here.
            ecpairs = ["({}) ? ({}) :".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = " (%s)" % self._print(expr.args[-1].expr)
            pw = "\n".join(ecpairs) + elast
            # Note: current need these outer brackets for 2*pw.  Would be
            # nicer to teach parenthesize() to do this for us when needed!
            return "(" + pw + ")"
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append("end")
            return "\n".join(lines)

    def _print_MatMul(self, expr):
        c, m = expr.as_coeff_mmul()

        sign = ""
        if c.is_number:
            re, im = c.as_real_imag()
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = "-"

        return sign + ' * '.join(
            (self.parenthesize(arg, precedence(expr)) for arg in expr.args)
        )


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')

        # pre-strip left-space from the code
        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(search(re, line) for re in inc_regex))
                     for line in code ]
        decrease = [ int(any(search(re, line) for re in dec_regex))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty


def julia_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Julia code.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    Examples
    ========

    >>> from sympy import julia_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> julia_code(sin(x).series(x).removeO())
    'x .^ 5 / 120 - x .^ 3 / 6 + x'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> julia_code((2*tau)**Rational(7, 2))
    '8 * sqrt(2) * tau .^ (7 // 2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its possible in Julia to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> julia_code(sin(pi*x*y), assign_to="s")
    's = sin(pi * x .* y)'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> julia_code(3*pi*A**3)
    '(3 * pi) * A ^ 3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> julia_code(x**2*y*A**3)
    '(x .^ 2 .* y) * A ^ 3'

    Matrices are supported using Julia inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> julia_code(mat, assign_to='A')
    'A = [x .^ 2 sin(x) ceil(x)]'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> julia_code(pw, assign_to=tau)
    'tau = ((x > 0) ? (x + 1) : (x))'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> julia_code(mat, assign_to='A')
    'A = [x .^ 2 ((x > 0) ? (x + 1) : (x)) sin(x)]'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Julia function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_julia_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> julia_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_julia_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> julia_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i]) ./ (t[i + 1] - t[i])'
    """
    return JuliaCodePrinter(settings).doprint(expr, assign_to)


def print_julia_code(expr, **settings):
    """Prints the Julia representation of the given expression.

    See `julia_code` for the meaning of the optional arguments.
    """
    print(julia_code(expr, **settings))
<|endoftext|>"""Symbolic primitives + unicode/ASCII abstraction for pretty.py"""

import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata

unicode_warnings = ''

def U(name):
    """
    Get a unicode character by name or, None if not found.

    This exists because older versions of Python use older unicode databases.
    """
    try:
        return unicodedata.lookup(name)
    except KeyError:
        global unicode_warnings
        unicode_warnings += 'No \'%s\' in unicodedata\n' % name
        return None

from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning

# prefix conventions when constructing tables
# L   - LATIN     i
# G   - GREEK     beta
# D   - DIGIT     0
# S   - SYMBOL    +


__all__ = ['greek_unicode', 'sub', 'sup', 'xsym', 'vobj', 'hobj', 'pretty_symbol',
           'annotated', 'center_pad', 'center']


_use_unicode = False


def pretty_use_unicode(flag=None):
    """Set whether pretty-printer should use unicode by default"""
    global _use_unicode, unicode_warnings
    if flag is None:
        return _use_unicode

    if flag and unicode_warnings:
        # print warnings (if any) on first unicode usage
        warnings.warn(unicode_warnings)
        unicode_warnings = ''

    use_unicode_prev = _use_unicode
    _use_unicode = flag
    return use_unicode_prev


def pretty_try_use_unicode():
    """See if unicode output is available and leverage it if possible"""

    encoding = getattr(sys.stdout, 'encoding', None)

    # this happens when e.g. stdout is redirected through a pipe, or is
    # e.g. a cStringIO.StringO
    if encoding is None:
        return  # sys.stdout has no encoding

    symbols = []

    # see if we can represent greek alphabet
    symbols += greek_unicode.values()

    # and atoms
    symbols += atoms_table.values()

    for s in symbols:
        if s is None:
            return  # common symbols not present!

        try:
            s.encode(encoding)
        except UnicodeEncodeError:
            return

    # all the characters were present and encodable
    pretty_use_unicode(True)


def xstr(*args):
    sympy_deprecation_warning(
        """
        The sympy.printing.pretty.pretty_symbology.xstr() function is
        deprecated. Use str() instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-pretty-printing-functions"
    )
    return str(*args)

# GREEK
g = lambda l: U('GREEK SMALL LETTER %s' % l.upper())
G = lambda l: U('GREEK CAPITAL LETTER %s' % l.upper())

greek_letters = list(greeks) # make a copy
# deal with Unicode's funny spelling of lambda
greek_letters[greek_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_unicode = {L: g(L) for L in greek_letters}
greek_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_letters)

# aliases
greek_unicode['lambda'] = greek_unicode['lamda']
greek_unicode['Lambda'] = greek_unicode['Lamda']
greek_unicode['varsigma'] = '\N{GREEK SMALL LETTER FINAL SIGMA}'

# BOLD
b = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
B = lambda l: U('MATHEMATICAL BOLD CAPITAL %s' % l.upper())

bold_unicode = {l: b(l) for l in ascii_lowercase}
bold_unicode.update((L, B(L)) for L in ascii_uppercase)

# GREEK BOLD
gb = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
GB = lambda l: U('MATHEMATICAL BOLD CAPITAL  %s' % l.upper())

greek_bold_letters = list(greeks) # make a copy, not strictly required here
# deal with Unicode's funny spelling of lambda
greek_bold_letters[greek_bold_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_bold_unicode = {L: g(L) for L in greek_bold_letters}
greek_bold_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_bold_letters)
greek_bold_unicode['lambda'] = greek_unicode['lamda']
greek_bold_unicode['Lambda'] = greek_unicode['Lamda']
greek_bold_unicode['varsigma'] = '\N{MATHEMATICAL BOLD SMALL FINAL SIGMA}'

digit_2txt = {
    '0':    'ZERO',
    '1':    'ONE',
    '2':    'TWO',
    '3':    'THREE',
    '4':    'FOUR',
    '5':    'FIVE',
    '6':    'SIX',
    '7':    'SEVEN',
    '8':    'EIGHT',
    '9':    'NINE',
}

symb_2txt = {
    '+':    'PLUS SIGN',
    '-':    'MINUS',
    '=':    'EQUALS SIGN',
    '(':    'LEFT PARENTHESIS',
    ')':    'RIGHT PARENTHESIS',
    '[':    'LEFT SQUARE BRACKET',
    ']':    'RIGHT SQUARE BRACKET',
    '{':    'LEFT CURLY BRACKET',
    '}':    'RIGHT CURLY BRACKET',

    # non-std
    '{}':   'CURLY BRACKET',
    'sum':  'SUMMATION',
    'int':  'INTEGRAL',
}

# SUBSCRIPT & SUPERSCRIPT
LSUB = lambda letter: U('LATIN SUBSCRIPT SMALL LETTER %s' % letter.upper())
GSUB = lambda letter: U('GREEK SUBSCRIPT SMALL LETTER %s' % letter.upper())
DSUB = lambda digit:  U('SUBSCRIPT %s' % digit_2txt[digit])
SSUB = lambda symb:   U('SUBSCRIPT %s' % symb_2txt[symb])

LSUP = lambda letter: U('SUPERSCRIPT LATIN SMALL LETTER %s' % letter.upper())
DSUP = lambda digit:  U('SUPERSCRIPT %s' % digit_2txt[digit])
SSUP = lambda symb:   U('SUPERSCRIPT %s' % symb_2txt[symb])

sub = {}    # symb -> subscript symbol
sup = {}    # symb -> superscript symbol

# latin subscripts
for l in 'aeioruvxhklmnpst':
    sub[l] = LSUB(l)

for l in 'in':
    sup[l] = LSUP(l)

for gl in ['beta', 'gamma', 'rho', 'phi', 'chi']:
    sub[gl] = GSUB(gl)

for d in [str(i) for i in range(10)]:
    sub[d] = DSUB(d)
    sup[d] = DSUP(d)

for s in '+-=()':
    sub[s] = SSUB(s)
    sup[s] = SSUP(s)

# Variable modifiers
# TODO: Make brackets adjust to height of contents
modifier_dict = {
    # Accents
    'mathring': lambda s: center_accent(s, '\N{COMBINING RING ABOVE}'),
    'ddddot': lambda s: center_accent(s, '\N{COMBINING FOUR DOTS ABOVE}'),
    'dddot': lambda s: center_accent(s, '\N{COMBINING THREE DOTS ABOVE}'),
    'ddot': lambda s: center_accent(s, '\N{COMBINING DIAERESIS}'),
    'dot': lambda s: center_accent(s, '\N{COMBINING DOT ABOVE}'),
    'check': lambda s: center_accent(s, '\N{COMBINING CARON}'),
    'breve': lambda s: center_accent(s, '\N{COMBINING BREVE}'),
    'acute': lambda s: center_accent(s, '\N{COMBINING ACUTE ACCENT}'),
    'grave': lambda s: center_accent(s, '\N{COMBINING GRAVE ACCENT}'),
    'tilde': lambda s: center_accent(s, '\N{COMBINING TILDE}'),
    'hat': lambda s: center_accent(s, '\N{COMBINING CIRCUMFLEX ACCENT}'),
    'bar': lambda s: center_accent(s, '\N{COMBINING OVERLINE}'),
    'vec': lambda s: center_accent(s, '\N{COMBINING RIGHT ARROW ABOVE}'),
    'prime': lambda s: s+'\N{PRIME}',
    'prm': lambda s: s+'\N{PRIME}',
    # # Faces -- these are here for some compatibility with latex printing
    # 'bold': lambda s: s,
    # 'bm': lambda s: s,
    # 'cal': lambda s: s,
    # 'scr': lambda s: s,
    # 'frak': lambda s: s,
    # Brackets
    'norm': lambda s: '\N{DOUBLE VERTICAL LINE}'+s+'\N{DOUBLE VERTICAL LINE}',
    'avg': lambda s: '\N{MATHEMATICAL LEFT ANGLE BRACKET}'+s+'\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
    'abs': lambda s: '\N{VERTICAL LINE}'+s+'\N{VERTICAL LINE}',
    'mag': lambda s: '\N{VERTICAL LINE}'+s+'\N{VERTICAL LINE}',
}

# VERTICAL OBJECTS
HUP = lambda symb: U('%s UPPER HOOK' % symb_2txt[symb])
CUP = lambda symb: U('%s UPPER CORNER' % symb_2txt[symb])
MID = lambda symb: U('%s MIDDLE PIECE' % symb_2txt[symb])
EXT = lambda symb: U('%s EXTENSION' % symb_2txt[symb])
HLO = lambda symb: U('%s LOWER HOOK' % symb_2txt[symb])
CLO = lambda symb: U('%s LOWER CORNER' % symb_2txt[symb])
TOP = lambda symb: U('%s TOP' % symb_2txt[symb])
BOT = lambda symb: U('%s BOTTOM' % symb_2txt[symb])

# {} '('  ->  (extension, start, end, middle) 1-character
_xobj_unicode = {

    # vertical symbols
    #                       (( ext, top, bot, mid ), c1)
    '(':                    (( EXT('('), HUP('('), HLO('(') ), '('),
    ')':                    (( EXT(')'), HUP(')'), HLO(')') ), ')'),
    '[':                    (( EXT('['), CUP('['), CLO('[') ), '['),
    ']':                    (( EXT(']'), CUP(']'), CLO(']') ), ']'),
    '{':                    (( EXT('{}'), HUP('{'), HLO('{'), MID('{') ), '{'),
    '}':                    (( EXT('{}'), HUP('}'), HLO('}'), MID('}') ), '}'),
    '|':                    U('BOX DRAWINGS LIGHT VERTICAL'),
    'Tee':                  U('BOX DRAWINGS LIGHT UP AND HORIZONTAL'),
    'UpTack':               U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL'),
    'corner_up_centre'
    '(_ext':                U('LEFT PARENTHESIS EXTENSION'),
    ')_ext':                U('RIGHT PARENTHESIS EXTENSION'),
    '(_lower_hook':         U('LEFT PARENTHESIS LOWER HOOK'),
    ')_lower_hook':         U('RIGHT PARENTHESIS LOWER HOOK'),
    '(_upper_hook':         U('LEFT PARENTHESIS UPPER HOOK'),
    ')_upper_hook':         U('RIGHT PARENTHESIS UPPER HOOK'),
    '<':                  ((U('BOX DRAWINGS LIGHT VERTICAL'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')), '<'),

    '>':                  ((U('BOX DRAWINGS LIGHT VERTICAL'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), '>'),

    'lfloor':               (( EXT('['), EXT('['), CLO('[') ), U('LEFT FLOOR')),
    'rfloor':               (( EXT(']'), EXT(']'), CLO(']') ), U('RIGHT FLOOR')),
    'lceil':                (( EXT('['), CUP('['), EXT('[') ), U('LEFT CEILING')),
    'rceil':                (( EXT(']'), CUP(']'), EXT(']') ), U('RIGHT CEILING')),

    'int':                  (( EXT('int'), U('TOP HALF INTEGRAL'), U('BOTTOM HALF INTEGRAL') ), U('INTEGRAL')),
    'sum':                  (( U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'), '_', U('OVERLINE'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), U('N-ARY SUMMATION')),

    # horizontal objects
    #'-':   '-',
    '-':    U('BOX DRAWINGS LIGHT HORIZONTAL'),
    '_':    U('LOW LINE'),
    # We used to use this, but LOW LINE looks better for roots, as it's a
    # little lower (i.e., it lines up with the / perfectly.  But perhaps this
    # one would still be wanted for some cases?
    # '_':    U('HORIZONTAL SCAN LINE-9'),

    # diagonal objects '\' & '/' ?
    '/':    U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
    '\\':   U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
}

_xobj_ascii = {
    # vertical symbols
    #       (( ext, top, bot, mid ), c1)
    '(':    (( '|', '/', '\\' ), '('),
    ')':    (( '|', '\\', '/' ), ')'),

# XXX this looks ugly
#   '[':    (( '|', '-', '-' ), '['),
#   ']':    (( '|', '-', '-' ), ']'),
# XXX not so ugly :(
    '[':    (( '[', '[', '[' ), '['),
    ']':    (( ']', ']', ']' ), ']'),

    '{':    (( '|', '/', '\\', '<' ), '{'),
    '}':    (( '|', '\\', '/', '>' ), '}'),
    '|':    '|',

    '<':    (( '|', '/', '\\' ), '<'),
    '>':    (( '|', '\\', '/' ), '>'),

    'int':  ( ' | ', '  /', '/  ' ),

    # horizontal objects
    '-':    '-',
    '_':    '_',

    # diagonal objects '\' & '/' ?
    '/':    '/',
    '\\':   '\\',
}


def xobj(symb, length):
    """Construct spatial object of given length.

    return: [] of equal-length strings
    """

    if length <= 0:
        raise ValueError("Length should be greater than 0")

    # TODO robustify when no unicodedat available
    if _use_unicode:
        _xobj = _xobj_unicode
    else:
        _xobj = _xobj_ascii

    vinfo = _xobj[symb]

    c1 = top = bot = mid = None

    if not isinstance(vinfo, tuple):        # 1 entry
        ext = vinfo
    else:
        if isinstance(vinfo[0], tuple):     # (vlong), c1
            vlong = vinfo[0]
            c1 = vinfo[1]
        else:                               # (vlong), c1
            vlong = vinfo

        ext = vlong[0]

        try:
            top = vlong[1]
            bot = vlong[2]
            mid = vlong[3]
        except IndexError:
            pass

    if c1 is None:
        c1 = ext
    if top is None:
        top = ext
    if bot is None:
        bot = ext
    if mid is not None:
        if (length % 2) == 0:
            # even height, but we have to print it somehow anyway...
            # XXX is it ok?
            length += 1

    else:
        mid = ext

    if length == 1:
        return c1

    res = []
    next = (length - 2)//2
    nmid = (length - 2) - next*2

    res += [top]
    res += [ext]*next
    res += [mid]*nmid
    res += [ext]*next
    res += [bot]

    return res


def vobj(symb, height):
    """Construct vertical object of a given height

       see: xobj
    """
    return '\n'.join( xobj(symb, height) )


def hobj(symb, width):
    """Construct horizontal object of a given width

       see: xobj
    """
    return ''.join( xobj(symb, width) )

# RADICAL
# n -> symbol
root = {
    2: U('SQUARE ROOT'),   # U('RADICAL SYMBOL BOTTOM')
    3: U('CUBE ROOT'),
    4: U('FOURTH ROOT'),
}


# RATIONAL
VF = lambda txt: U('VULGAR FRACTION %s' % txt)

# (p,q) -> symbol
frac = {
    (1, 2): VF('ONE HALF'),
    (1, 3): VF('ONE THIRD'),
    (2, 3): VF('TWO THIRDS'),
    (1, 4): VF('ONE QUARTER'),
    (3, 4): VF('THREE QUARTERS'),
    (1, 5): VF('ONE FIFTH'),
    (2, 5): VF('TWO FIFTHS'),
    (3, 5): VF('THREE FIFTHS'),
    (4, 5): VF('FOUR FIFTHS'),
    (1, 6): VF('ONE SIXTH'),
    (5, 6): VF('FIVE SIXTHS'),
    (1, 8): VF('ONE EIGHTH'),
    (3, 8): VF('THREE EIGHTHS'),
    (5, 8): VF('FIVE EIGHTHS'),
    (7, 8): VF('SEVEN EIGHTHS'),
}


# atom symbols
_xsym = {
    '==':  ('=', '='),
    '<':   ('<', '<'),
    '>':   ('>', '>'),
    '<=':  ('<=', U('LESS-THAN OR EQUAL TO')),
    '>=':  ('>=', U('GREATER-THAN OR EQUAL TO')),
    '!=':  ('!=', U('NOT EQUAL TO')),
    ':=':  (':=', ':='),
    '+=':  ('+=', '+='),
    '-=':  ('-=', '-='),
    '*=':  ('*=', '*='),
    '/=':  ('/=', '/='),
    '%=':  ('%=', '%='),
    '*':   ('*', U('DOT OPERATOR')),
    '-->': ('-->', U('EM DASH') + U('EM DASH') +
            U('BLACK RIGHT-POINTING TRIANGLE') if U('EM DASH')
            and U('BLACK RIGHT-POINTING TRIANGLE') else None),
    '==>': ('==>', U('BOX DRAWINGS DOUBLE HORIZONTAL') +
            U('BOX DRAWINGS DOUBLE HORIZONTAL') +
            U('BLACK RIGHT-POINTING TRIANGLE') if
            U('BOX DRAWINGS DOUBLE HORIZONTAL') and
            U('BOX DRAWINGS DOUBLE HORIZONTAL') and
            U('BLACK RIGHT-POINTING TRIANGLE') else None),
    '.':   ('*', U('RING OPERATOR')),
}


def xsym(sym):
    """get symbology for a 'character'"""
    op = _xsym[sym]

    if _use_unicode:
        return op[1]
    else:
        return op[0]


# SYMBOLS

atoms_table = {
    # class                    how-to-display
    'Exp1':                    U('SCRIPT SMALL E'),
    'Pi':                      U('GREEK SMALL LETTER PI'),
    'Infinity':                U('INFINITY'),
    'NegativeInfinity':        U('INFINITY') and ('-' + U('INFINITY')),  # XXX what to do here
    #'ImaginaryUnit':          U('GREEK SMALL LETTER IOTA'),
    #'ImaginaryUnit':          U('MATHEMATICAL ITALIC SMALL I'),
    'ImaginaryUnit':           U('DOUBLE-STRUCK ITALIC SMALL I'),
    'EmptySet':                U('EMPTY SET'),
    'Naturals':                U('DOUBLE-STRUCK CAPITAL N'),
    'Naturals0':              (U('DOUBLE-STRUCK CAPITAL N') and
                              (U('DOUBLE-STRUCK CAPITAL N') +
                               U('SUBSCRIPT ZERO'))),
    'Integers':                U('DOUBLE-STRUCK CAPITAL Z'),
    'Rationals':               U('DOUBLE-STRUCK CAPITAL Q'),
    'Reals':                   U('DOUBLE-STRUCK CAPITAL R'),
    'Complexes':               U('DOUBLE-STRUCK CAPITAL C'),
    'Universe':                U('MATHEMATICAL DOUBLE-STRUCK CAPITAL U'),
    'IdentityMatrix':          U('MATHEMATICAL DOUBLE-STRUCK CAPITAL I'),
    'ZeroMatrix':              U('MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO'),
    'OneMatrix':               U('MATHEMATICAL DOUBLE-STRUCK DIGIT ONE'),
    'Differential':            U('DOUBLE-STRUCK ITALIC SMALL D'),
    'Union':                   U('UNION'),
    'ElementOf':               U('ELEMENT OF'),
    'SmallElementOf':          U('SMALL ELEMENT OF'),
    'SymmetricDifference':     U('INCREMENT'),
    'Intersection':            U('INTERSECTION'),
    'Ring':                    U('RING OPERATOR'),
    'Multiplication':          U('MULTIPLICATION SIGN'),
    'TensorProduct':           U('N-ARY CIRCLED TIMES OPERATOR'),
    'Dots':                    U('HORIZONTAL ELLIPSIS'),
    'Modifier Letter Low Ring':U('Modifier Letter Low Ring'),
    'EmptySequence':           'EmptySequence',
    'SuperscriptPlus':         U('SUPERSCRIPT PLUS SIGN'),
    'SuperscriptMinus':        U('SUPERSCRIPT MINUS'),
    'Dagger':                  U('DAGGER'),
    'Degree':                  U('DEGREE SIGN'),
    #Logic Symbols
    'And':                     U('LOGICAL AND'),
    'Or':                      U('LOGICAL OR'),
    'Not':                     U('NOT SIGN'),
    'Nor':                     U('NOR'),
    'Nand':                    U('NAND'),
    'Xor':                     U('XOR'),
    'Equiv':                   U('LEFT RIGHT DOUBLE ARROW'),
    'NotEquiv':                U('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    'Implies':                 U('LEFT RIGHT DOUBLE ARROW'),
    'NotImplies':              U('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    'Arrow':                   U('RIGHTWARDS ARROW'),
    'ArrowFromBar':            U('RIGHTWARDS ARROW FROM BAR'),
    'NotArrow':                U('RIGHTWARDS ARROW WITH STROKE'),
    'Tautology':               U('BOX DRAWINGS LIGHT UP AND HORIZONTAL'),
    'Contradiction':           U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL')
}


def pretty_atom(atom_name, default=None, printer=None):
    """return pretty representation of an atom"""
    if _use_unicode:
        if printer is not None and atom_name == 'ImaginaryUnit' and printer._settings['imaginary_unit'] == 'j':
            return U('DOUBLE-STRUCK ITALIC SMALL J')
        else:
            return atoms_table[atom_name]
    else:
        if default is not None:
            return default

        raise KeyError('only unicode')  # send it default printer


def pretty_symbol(symb_name, bold_name=False):
    """return pretty representation of a symbol"""
    # let's split symb_name into symbol + index
    # UC: beta1
    # UC: f_beta

    if not _use_unicode:
        return symb_name

    name, sups, subs = split_super_sub(symb_name)

    def translate(s, bold_name) :
        if bold_name:
            gG = greek_bold_unicode.get(s)
        else:
            gG = greek_unicode.get(s)
        if gG is not None:
            return gG
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True) :
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)], bold_name))
        if bold_name:
            return ''.join([bold_unicode[c] for c in s])
        return s

    name = translate(name, bold_name)

    # Let's prettify sups/subs. If it fails at one of them, pretty sups/subs are
    # not used at all.
    def pretty_list(l, mapping):
        result = []
        for s in l:
            pretty = mapping.get(s)
            if pretty is None:
                try:  # match by separate characters
                    pretty = ''.join([mapping[c] for c in s])
                except (TypeError, KeyError):
                    return None
            result.append(pretty)
        return result

    pretty_sups = pretty_list(sups, sup)
    if pretty_sups is not None:
        pretty_subs = pretty_list(subs, sub)
    else:
        pretty_subs = None

    # glue the results into one string
    if pretty_subs is None:  # nice formatting of sups/subs did not work
        if subs:
            name += '_'+'_'.join([translate(s, bold_name) for s in subs])
        if sups:
            name += '__'+'__'.join([translate(s, bold_name) for s in sups])
        return name
    else:
        sups_result = ' '.join(pretty_sups)
        subs_result = ' '.join(pretty_subs)

    return ''.join([name, sups_result, subs_result])


def annotated(letter):
    """
    Return a stylised drawing of the letter ``letter``, together with
    information on how to put annotations (super- and subscripts to the
    left and to the right) on it.

    See pretty.py functions _print_meijerg, _print_hyper on how to use this
    information.
    """
    ucode_pics = {
        'F': (2, 0, 2, 0, '\N{BOX DRAWINGS LIGHT DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                          '\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                          '\N{BOX DRAWINGS LIGHT UP}'),
        'G': (3, 0, 3, 1, '\N{BOX DRAWINGS LIGHT ARC DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{BOX DRAWINGS LIGHT ARC DOWN AND LEFT}\n'
                          '\N{BOX DRAWINGS LIGHT VERTICAL}\N{BOX DRAWINGS LIGHT RIGHT}\N{BOX DRAWINGS LIGHT DOWN AND LEFT}\n'
                          '\N{BOX DRAWINGS LIGHT ARC UP AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{BOX DRAWINGS LIGHT ARC UP AND LEFT}')
    }
    ascii_pics = {
        'F': (3, 0, 3, 0, ' _\n|_\n|\n'),
        'G': (3, 0, 3, 1, ' __\n/__\n\\_|')
    }

    if _use_unicode:
        return ucode_pics[letter]
    else:
        return ascii_pics[letter]

_remove_combining = dict.fromkeys(list(range(ord('\N{COMBINING GRAVE ACCENT}'), ord('\N{COMBINING LATIN SMALL LETTER X}')))
                            + list(range(ord('\N{COMBINING LEFT HARPOON ABOVE}'), ord('\N{COMBINING ASTERISK ABOVE}'))))

def is_combining(sym):
    """Check whether symbol is a unicode modifier. """

    return ord(sym) in _remove_combining


def center_accent(string, accent):
    """
    Returns a string with accent inserted on the middle character. Useful to
    put combining accents on symbol names, including multi-character names.

    Parameters
    ==========

    string : string
        The string to place the accent in.
    accent : string
        The combining accent to insert

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combining_character
    .. [2] https://en.wikipedia.org/wiki/Combining_Diacritical_Marks

    """

    # Accent is placed on the previous character, although it may not always look
    # like that depending on console
    midpoint = len(string) // 2 + 1
    firstpart = string[:midpoint]
    secondpart = string[midpoint:]
    return firstpart + accent + secondpart


def line_width(line):
    """Unicode combining symbols (modifiers) are not ever displayed as
    separate symbols and thus should not be counted
    """
    return len(line.translate(_remove_combining))


def is_subscriptable_in_unicode(subscript):
    """
    Checks whether a string is subscriptable in unicode or not.

    Parameters
    ==========

    subscript: the string which needs to be checked

    Examples
    ========

    >>> from sympy.printing.pretty.pretty_symbology import is_subscriptable_in_unicode
    >>> is_subscriptable_in_unicode('abc')
    False
    >>> is_subscriptable_in_unicode('123')
    True

    """
    return all(character in sub for character in subscript)


def center_pad(wstring, wtarget, fillchar=' '):
    """
    Return the padding strings necessary to center a string of
    wstring characters wide in a wtarget wide space.

    The line_width wstring should always be less or equal to wtarget
    or else a ValueError will be raised.
    """
    if wstring > wtarget:
        raise ValueError('not enough space for string')
    wdelta = wtarget - wstring

    wleft = wdelta // 2  # favor left '1 '
    wright = wdelta - wleft

    left = fillchar * wleft
    right = fillchar * wright

    return left, right


def center(string, width, fillchar=' '):
    """Return a centered string of length determined by `line_width`
    that uses `fillchar` for padding.
    """
    left, right = center_pad(line_width(string), width, fillchar)
    return ''.join([left, string, right])
<|endoftext|>"""Prettyprinter by Jurjen Bos.
(I hate spammers: mail me at pietjepuk314 at the reverse of ku.oc.oohay).
All objects have a method that create a "stringPict",
that can be used in the str method for pretty printing.

Updates by Jason Gedge (email <my last name> at cs mun ca)
    - terminal_string() method
    - minor fixes and changes (mostly to prettyForm)

TODO:
    - Allow left/center/right alignment options for above/below and
      top/center/bottom alignment options for left/right
"""

import shutil

from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width, center
from sympy.utilities.exceptions import sympy_deprecation_warning

_GLOBAL_WRAP_LINE = None

class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    #special value for stringPict.below
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        self.s = s
        #picture is a string that just can be printed
        self.picture = stringPict.equalLengths(s.splitlines())
        #baseline is the line number of the "base line"
        self.baseline = baseline
        self.binding = None

    @staticmethod
    def equalLengths(lines):
        # empty lines
        if not lines:
            return ['']

        width = max(line_width(line) for line in lines)
        return [center(line, width) for line in lines]

    def height(self):
        """The height of the picture in characters."""
        return len(self.picture)

    def width(self):
        """The width of the picture in characters."""
        return line_width(self.picture[0])

    @staticmethod
    def next(*args):
        """Put a string of stringPicts next to each other.
        Returns string, baseline arguments for stringPict.
        """
        #convert everything to stringPicts
        objects = []
        for arg in args:
            if isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        #make a list of pictures, with equal height and baseline
        newBaseline = max(obj.baseline for obj in objects)
        newHeightBelowBaseline = max(
            obj.height() - obj.baseline
            for obj in objects)
        newHeight = newBaseline + newHeightBelowBaseline

        pictures = []
        for obj in objects:
            oneEmptyLine = [' '*obj.width()]
            basePadding = newBaseline - obj.baseline
            totalPadding = newHeight - obj.height()
            pictures.append(
                oneEmptyLine * basePadding +
                obj.picture +
                oneEmptyLine * (totalPadding - basePadding))

        result = [''.join(lines) for lines in zip(*pictures)]
        return '\n'.join(result), newBaseline

    def right(self, *args):
        r"""Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("10").right(" + ",stringPict("1\r-\r2",1))[0])
             1
        10 + -
             2

        """
        return stringPict.next(self, *args)

    def left(self, *args):
        """Put pictures (left to right) at left.
        Returns string, baseline arguments for stringPict.
        """
        return stringPict.next(*(args + (self,)))

    @staticmethod
    def stack(*args):
        """Put pictures on top of each other,
        from top to bottom.
        Returns string, baseline arguments for stringPict.
        The baseline is the baseline of the second picture.
        Everything is centered.
        Baseline is the baseline of the second picture.
        Strings are allowed.
        The special value stringPict.LINE is a row of '-' extended to the width.
        """
        #convert everything to stringPicts; keep LINE
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        #compute new width
        newWidth = max(
            obj.width()
            for obj in objects
            if obj is not stringPict.LINE)

        lineObj = stringPict(hobj('-', newWidth))

        #replace LINE with proper lines
        for i, obj in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj

        #stack the pictures, and center the result
        newPicture = [center(line, newWidth) for obj in objects for line in obj.picture]
        newBaseline = objects[0].height() + objects[1].baseline
        return '\n'.join(newPicture), newBaseline

    def below(self, *args):
        """Put pictures under this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of top picture

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("x+3").below(
        ...       stringPict.LINE, '3')[0]) #doctest: +NORMALIZE_WHITESPACE
        x+3
        ---
         3

        """
        s, baseline = stringPict.stack(self, *args)
        return s, self.baseline

    def above(self, *args):
        """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
        string, baseline = stringPict.stack(*(args + (self,)))
        baseline = len(string.splitlines()) - self.height() + self.baseline
        return string, baseline

    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()
        b = self.baseline

        # XXX this is a hack -- ascii parens are ugly!
        if ifascii_nougly and not pretty_use_unicode():
            h = 1
            b = 0

        res = self

        if left:
            lparen = stringPict(vobj(left, h), baseline=b)
            res = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res = stringPict(*res.right(rparen))

        return ('\n'.join(res.picture), res.baseline)

    def leftslash(self):
        """Precede object by a slash of the proper size.
        """
        # XXX not used anywhere ?
        height = max(
            self.baseline,
            self.height() - 1 - self.baseline)*2 + 1
        slash = '\n'.join(
            ' '*(height - i - 1) + xobj('/', 1) + ' '*i
            for i in range(height)
        )
        return self.left(stringPict(slash, height//2))

    def root(self, n=None):
        """Produce a nice root symbol.
        Produces ugly results for big n inserts.
        """
        # XXX not used anywhere
        # XXX duplicate of root drawing in pretty.py
        #put line over expression
        result = self.above('_'*self.width())
        #construct right half of root symbol
        height = self.height()
        slash = '\n'.join(
            ' ' * (height - i - 1) + '/' + ' ' * i
            for i in range(height)
        )
        slash = stringPict(slash, height - 1)
        #left half of root symbol
        if height > 2:
            downline = stringPict('\\ \n \\', 1)
        else:
            downline = stringPict('\\')
        #put n on top, as low as possible
        if n is not None and n.width() > downline.width():
            downline = downline.left(' '*(n.width() - downline.width()))
            downline = downline.above(n)
        #build root symbol
        root = downline.right(slash)
        #glue it on at the proper height
        #normally, the root symbel is as high as self
        #which is one less than result
        #this moves the root symbol one down
        #if the root became higher, the baseline has to grow too
        root.baseline = result.baseline - result.height() + root.height()
        return result.left(root)

    def render(self, * args, **kwargs):
        """Return the string form of self.

           Unless the argument line_break is set to False, it will
           break the expression in a form that can be printed
           on the terminal without being broken up.
         """
        if _GLOBAL_WRAP_LINE is not None:
            kwargs["wrap_line"] = _GLOBAL_WRAP_LINE

        if kwargs["wrap_line"] is False:
            return "\n".join(self.picture)

        if kwargs["num_columns"] is not None:
            # Read the argument num_columns if it is not None
            ncols = kwargs["num_columns"]
        else:
            # Attempt to get a terminal width
            ncols = self.terminal_width()

        if ncols <= 0:
            ncols = 80

        # If smaller than the terminal width, no need to correct
        if self.width() <= ncols:
            return type(self.picture[0])(self)

        """
        Break long-lines in a visually pleasing format.
        without overflow indicators | with overflow indicators
        |   2  2        3     |     |   2  2        3    ↪|
        |6*x *y  + 4*x*y  +   |     |6*x *y  + 4*x*y  +  ↪|
        |                     |     |                     |
        |     3    4    4     |     |↪      3    4    4   |
        |4*y*x  + x  + y      |     |↪ 4*y*x  + x  + y    |
        |a*c*e + a*c*f + a*d  |     |a*c*e + a*c*f + a*d ↪|
        |*e + a*d*f + b*c*e   |     |                     |
        |+ b*c*f + b*d*e + b  |     |↪ *e + a*d*f + b*c* ↪|
        |*d*f                 |     |                     |
        |                     |     |↪ e + b*c*f + b*d*e ↪|
        |                     |     |                     |
        |                     |     |↪ + b*d*f            |
        """

        overflow_first = ""
        if kwargs["use_unicode"] or pretty_use_unicode():
            overflow_start = "\N{RIGHTWARDS ARROW WITH HOOK} "
            overflow_end   = " \N{RIGHTWARDS ARROW WITH HOOK}"
        else:
            overflow_start = "> "
            overflow_end   = " >"

        def chunks(line):
            """Yields consecutive chunks of line_width ncols"""
            prefix = overflow_first
            width, start = line_width(prefix + overflow_end), 0
            for i, x in enumerate(line):
                wx = line_width(x)
                # Only flush the screen when the current character overflows.
                # This way, combining marks can be appended even when width == ncols.
                if width + wx > ncols:
                    yield prefix + line[start:i] + overflow_end
                    prefix = overflow_start
                    width, start = line_width(prefix + overflow_end), i
                width += wx
            yield prefix + line[start:]

        # Concurrently assemble chunks of all lines into individual screens
        pictures = zip(*map(chunks, self.picture))

        # Join lines of each screen into sub-pictures
        pictures = ["\n".join(picture) for picture in pictures]

        # Add spacers between sub-pictures
        return "\n\n".join(pictures)

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        size = shutil.get_terminal_size(fallback=(0, 0))
        return size.columns

    def __eq__(self, o):
        if isinstance(o, str):
            return '\n'.join(self.picture) == o
        elif isinstance(o, stringPict):
            return o.picture == self.picture
        return False

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return '\n'.join(self.picture)

    def __repr__(self):
        return "stringPict(%r,%d)" % ('\n'.join(self.picture), self.baseline)

    def __getitem__(self, index):
        return self.picture[index]

    def __len__(self):
        return len(self.s)


class prettyForm(stringPict):
    """
    Extension of the stringPict class that knows about basic math applications,
    optimizing double minus signs.

    "Binding" is interpreted as follows::

        ATOM this is an atom: never needs to be parenthesized
        FUNC this is a function application: parenthesize if added (?)
        DIV  this is a division: make wider division if divided
        POW  this is a power: only parenthesize if exponent
        MUL  this is a multiplication: parenthesize if powered
        ADD  this is an addition: parenthesize if multiplied or powered
        NEG  this is a negative number: optimize if added, parenthesize if
             multiplied or powered
        OPEN this is an open object: parenthesize if added, multiplied, or
             powered (example: Piecewise)
    """
    ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN = range(8)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        """Initialize from stringPict and binding power."""
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        if unicode is not None:
            sympy_deprecation_warning(
                """
                The unicode argument to prettyForm is deprecated. Only the s
                argument (the first positional argument) should be passed.
                """,
                deprecated_since_version="1.7",
                active_deprecations_target="deprecated-pretty-printing-functions")
        self._unicode = unicode or s

    @property
    def unicode(self):
        sympy_deprecation_warning(
            """
            The prettyForm.unicode attribute is deprecated. Use the
            prettyForm.s attribute instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-pretty-printing-functions")
        return self._unicode

    # Note: code to handle subtraction is in _print_Add

    def __add__(self, *others):
        """Make a pretty addition.
        Addition of negative numbers is simplified.
        """
        arg = self
        if arg.binding > prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            #add parentheses for weak binders
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            #use existing minus sign if available
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(binding=prettyForm.ADD, *stringPict.next(*result))

    def __truediv__(self, den, slashed=False):
        """Make a pretty division; stacked or slashed.
        """
        if slashed:
            raise NotImplementedError("Can't do slashed fraction yet")
        num = self
        if num.binding == prettyForm.DIV:
            num = stringPict(*num.parens())
        if den.binding == prettyForm.DIV:
            den = stringPict(*den.parens())

        if num.binding==prettyForm.NEG:
            num = num.right(" ")[0]

        return prettyForm(binding=prettyForm.DIV, *stringPict.stack(
            num,
            stringPict.LINE,
            den))

    def __mul__(self, *others):
        """Make a pretty multiplication.
        Parentheses are needed around +, - and neg.
        """
        quantity = {
            'degree': "\N{DEGREE SIGN}"
        }

        if len(others) == 0:
            return self  # We aren't actually multiplying... So nothing to do here.

        # add parens on args that need them
        arg = self
        if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.picture[0] not in quantity.values():
                result.append(xsym('*'))
            #add parentheses for weak binders
            if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
                arg = stringPict(*arg.parens())
            result.append(arg)

        len_res = len(result)
        for i in range(len_res):
            if i < len_res - 1 and result[i] == '-1' and result[i + 1] == xsym('*'):
                # substitute -1 by -, like in -1*x -> -x
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        if result[0][0] == '-':
            # if there is a - sign in front of all
            # This test was failing to catch a prettyForm.__mul__(prettyForm("-1", 0, 6)) being negative
            bin = prettyForm.NEG
            if result[0] == '-':
                right = result[1]
                if right.picture[right.baseline][0] == '-':
                    result[0] = '- '
        else:
            bin = prettyForm.MUL
        return prettyForm(binding=bin, *stringPict.next(*result))

    def __repr__(self):
        return "prettyForm(%r,%d,%d)" % (
            '\n'.join(self.picture),
            self.baseline,
            self.binding)

    def __pow__(self, b):
        """Make a pretty power.
        """
        a = self
        use_inline_func_form = False
        if b.binding == prettyForm.POW:
            b = stringPict(*b.parens())
        if a.binding > prettyForm.FUNC:
            a = stringPict(*a.parens())
        elif a.binding == prettyForm.FUNC:
            # heuristic for when to use inline power
            if b.height() > 1:
                a = stringPict(*a.parens())
            else:
                use_inline_func_form = True

        if use_inline_func_form:
            #         2
            #  sin  +   + (x)
            b.baseline = a.prettyFunc.baseline + b.height()
            func = stringPict(*a.prettyFunc.right(b))
            return prettyForm(*func.right(a.prettyArgs))
        else:
            #      2    <-- top
            # (x+y)     <-- bot
            top = stringPict(*b.left(' '*a.width()))
            bot = stringPict(*a.right(' '*b.width()))

        return prettyForm(binding=prettyForm.POW, *bot.above(top))

    simpleFunctions = ["sin", "cos", "tan"]

    @staticmethod
    def apply(function, *args):
        """Functions of one or more variables.
        """
        if function in prettyForm.simpleFunctions:
            #simple function: use only space if possible
            assert len(
                args) == 1, "Simple function %s must have 1 argument" % function
            arg = args[0].__pretty__()
            if arg.binding <= prettyForm.DIV:
                #optimization: no parentheses necessary
                return prettyForm(binding=prettyForm.FUNC, *arg.left(function + ' '))
        argumentList = []
        for arg in args:
            argumentList.append(',')
            argumentList.append(arg.__pretty__())
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        argumentList = stringPict(*argumentList.parens())
        return prettyForm(binding=prettyForm.ATOM, *argumentList.left(function))
<|endoftext|>import itertools

from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
    xsym, pretty_symbol, pretty_atom, pretty_use_unicode, greek_unicode, U, \
    pretty_try_use_unicode, annotated, is_subscriptable_in_unicode, center_pad,  root as nth_root

# rename for usage from outside
pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode


class PrettyPrinter(Printer):
    """Printer, which converts an expression into 2D ASCII-art figure."""
    printmethod = "_pretty"

    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "use_unicode": None,
        "wrap_line": True,
        "num_columns": None,
        "use_unicode_sqrt_char": True,
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "perm_cyclic": True
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if not isinstance(self._settings['imaginary_unit'], str):
            raise TypeError("'imaginary_unit' must a string, not {}".format(self._settings['imaginary_unit']))
        elif self._settings['imaginary_unit'] not in ("i", "j"):
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '{}'".format(self._settings['imaginary_unit']))

    def emptyPrinter(self, expr):
        return prettyForm(str(expr))

    @property
    def _use_unicode(self):
        if self._settings['use_unicode']:
            return True
        else:
            return pretty_use_unicode()

    def doprint(self, expr):
        return self._print(expr).render(**self._settings)

    # empty op so _print(stringPict) returns the same
    def _print_stringPict(self, e):
        return e

    def _print_basestring(self, e):
        return prettyForm(e)

    def _print_atan2(self, e):
        pform = prettyForm(*self._print_seq(e.args).parens())
        pform = prettyForm(*pform.left('atan2'))
        return pform

    def _print_Symbol(self, e, bold_name=False):
        symb = pretty_symbol(e.name, bold_name)
        return prettyForm(symb)
    _print_RandomSymbol = _print_Symbol
    def _print_MatrixSymbol(self, e):
        return self._print_Symbol(e, self._settings['mat_symbol_style'] == "bold")

    def _print_Float(self, e):
        # we will use StrPrinter's Float printer, but we need to handle the
        # full_prec ourselves, according to the self._print_level
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        return prettyForm(sstr(e, full_prec=full_prec))

    def _print_Cross(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform

    def _print_Curl(self, e):
        vec = e._expr
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Divergence(self, e):
        vec = e._expr
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Dot(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform

    def _print_Gradient(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Laplacian(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('INCREMENT'))))
        return pform

    def _print_Atom(self, e):
        try:
            # print atoms like Exp1 or Pi
            return prettyForm(pretty_atom(e.__class__.__name__, printer=self))
        except KeyError:
            return self.emptyPrinter(e)

    # Infinity inherits from Number, so we have to override _print_XXX order
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Rationals = _print_Atom
    _print_Complexes = _print_Atom

    _print_EmptySequence = _print_Atom

    def _print_Reals(self, e):
        if self._use_unicode:
            return self._print_Atom(e)
        else:
            inf_list = ['-oo', 'oo']
            return self._print_seq(inf_list, '(', ')')

    def _print_subfactorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('!'))
        return pform

    def _print_factorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!'))
        return pform

    def _print_factorial2(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!!'))
        return pform

    def _print_binomial(self, e):
        n, k = e.args

        n_pform = self._print(n)
        k_pform = self._print(k)

        bar = ' '*max(n_pform.width(), k_pform.width())

        pform = prettyForm(*k_pform.above(bar))
        pform = prettyForm(*pform.above(n_pform))
        pform = prettyForm(*pform.parens('(', ')'))

        pform.baseline = (pform.baseline + 1)//2

        return pform

    def _print_Relational(self, e):
        op = prettyForm(' ' + xsym(e.rel_op) + ' ')

        l = self._print(e.lhs)
        r = self._print(e.rhs)
        pform = prettyForm(*stringPict.next(l, op, r), binding=prettyForm.OPEN)
        return pform

    def _print_Not(self, e):
        from sympy.logic.boolalg import (Equivalent, Implies)
        if self._use_unicode:
            arg = e.args[0]
            pform = self._print(arg)
            if isinstance(arg, Equivalent):
                return self._print_Equivalent(arg, altchar=pretty_atom('NotEquiv'))
            if isinstance(arg, Implies):
                return self._print_Implies(arg, altchar=pretty_atom('NotArrow'))

            if arg.is_Boolean and not arg.is_Not:
                pform = prettyForm(*pform.parens())

            return prettyForm(*pform.left(pretty_atom('Not')))
        else:
            return self._print_Function(e)

    def __print_Boolean(self, e, char, sort=True):
        args = e.args
        if sort:
            args = sorted(e.args, key=default_sort_key)
        arg = args[0]
        pform = self._print(arg)

        if arg.is_Boolean and not arg.is_Not:
            pform = prettyForm(*pform.parens())

        for arg in args[1:]:
            pform_arg = self._print(arg)

            if arg.is_Boolean and not arg.is_Not:
                pform_arg = prettyForm(*pform_arg.parens())

            pform = prettyForm(*pform.right(' %s ' % char))
            pform = prettyForm(*pform.right(pform_arg))

        return pform

    def _print_And(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('And'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Or(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Or'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Xor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom("Xor"))
        else:
            return self._print_Function(e, sort=True)

    def _print_Nand(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Nand'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Nor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Nor'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Implies(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or pretty_atom('Arrow'), sort=False)
        else:
            return self._print_Function(e)

    def _print_Equivalent(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or pretty_atom('Equiv'))
        else:
            return self._print_Function(e, sort=True)

    def _print_conjugate(self, e):
        pform = self._print(e.args[0])
        return prettyForm( *pform.above( hobj('_', pform.width())) )

    def _print_Abs(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens('|', '|'))
        return pform

    def _print_floor(self, e):
        if self._use_unicode:
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lfloor', 'rfloor'))
            return pform
        else:
            return self._print_Function(e)

    def _print_ceiling(self, e):
        if self._use_unicode:
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lceil', 'rceil'))
            return pform
        else:
            return self._print_Function(e)

    def _print_Derivative(self, deriv):
        if requires_partial(deriv.expr) and self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None
        count_total_deriv = 0

        for sym, num in reversed(deriv.variable_count):
            s = self._print(sym)
            ds = prettyForm(*s.left(deriv_symbol))
            count_total_deriv += num

            if (not num.is_Integer) or (num > 1):
                ds = ds**prettyForm(str(num))

            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        pform = prettyForm(deriv_symbol)

        if (count_total_deriv > 1) != False:
            pform = pform**prettyForm(str(count_total_deriv))

        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        return pform

    def _print_Cycle(self, dc):
        from sympy.combinatorics.permutations import Permutation, Cycle
        # for Empty Cycle
        if dc == Cycle():
            cyc = stringPict('')
            return prettyForm(*cyc.parens())

        dc_list = Permutation(dc.list()).cyclic_form
        # for Identity Cycle
        if dc_list == []:
            cyc = self._print(dc.size - 1)
            return prettyForm(*cyc.parens())

        cyc = stringPict('')
        for i in dc_list:
            l = self._print(str(tuple(i)).replace(',', ''))
            cyc = prettyForm(*cyc.right(l))
        return cyc

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            return self._print_Cycle(Cycle(expr))

        lower = expr.array_form
        upper = list(range(len(lower)))

        result = stringPict('')
        first = True
        for u, l in zip(upper, lower):
            s1 = self._print(u)
            s2 = self._print(l)
            col = prettyForm(*s1.below(s2))
            if first:
                first = False
            else:
                col = prettyForm(*col.left(" "))
            result = prettyForm(*result.right(col))
        return prettyForm(*result.parens())


    def _print_Integral(self, integral):
        f = integral.function

        # Add parentheses if arg involves addition of terms and
        # create a pretty form for the argument
        prettyF = self._print(f)
        # XXX generalize parens
        if f.is_Add:
            prettyF = prettyForm(*prettyF.parens())

        # dx dy dz ...
        arg = prettyF
        for x in integral.limits:
            prettyArg = self._print(x[0])
            # XXX qparens (parens if needs-parens)
            if prettyArg.width() > 1:
                prettyArg = prettyForm(*prettyArg.parens())

            arg = prettyForm(*arg.right(' d', prettyArg))

        # \int \int \int ...
        firstterm = True
        s = None
        for lim in integral.limits:
            # Create bar based on the height of the argument
            h = arg.height()
            H = h + 2

            # XXX hack!
            ascii_mode = not self._use_unicode
            if ascii_mode:
                H += 2

            vint = vobj('int', H)

            # Construct the pretty form with the integral sign and the argument
            pform = prettyForm(vint)
            pform.baseline = arg.baseline + (
                H - h)//2    # covering the whole argument

            if len(lim) > 1:
                # Create pretty forms for endpoints, if definite integral.
                # Do not print empty endpoints.
                if len(lim) == 2:
                    prettyA = prettyForm("")
                    prettyB = self._print(lim[1])
                if len(lim) == 3:
                    prettyA = self._print(lim[1])
                    prettyB = self._print(lim[2])

                if ascii_mode:  # XXX hack
                    # Add spacing so that endpoint can more easily be
                    # identified with the correct integral sign
                    spc = max(1, 3 - prettyB.width())
                    prettyB = prettyForm(*prettyB.left(' ' * spc))

                    spc = max(1, 4 - prettyA.width())
                    prettyA = prettyForm(*prettyA.right(' ' * spc))

                pform = prettyForm(*pform.above(prettyB))
                pform = prettyForm(*pform.below(prettyA))

            if not ascii_mode:  # XXX hack
                pform = prettyForm(*pform.right(' '))

            if firstterm:
                s = pform   # first term
                firstterm = False
            else:
                s = prettyForm(*s.left(pform))

        pform = prettyForm(*arg.left(s))
        pform.binding = prettyForm.MUL
        return pform

    def _print_Product(self, expr):
        func = expr.term
        pretty_func = self._print(func)

        horizontal_chr = xobj('_', 1)
        corner_chr = xobj('_', 1)
        vertical_chr = xobj('|', 1)

        if self._use_unicode:
            # use unicode corners
            horizontal_chr = xobj('-', 1)
            corner_chr = xobj('UpTack', 1)

        func_height = pretty_func.height()

        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            pretty_lower, pretty_upper = self.__print_SumProduct_Limits(lim)

            width = (func_height + 2) * 5 // 3 - 2
            sign_lines = [horizontal_chr + corner_chr + (horizontal_chr * (width-2)) + corner_chr + horizontal_chr]
            for _ in range(func_height + 1):
                sign_lines.append(' ' + vertical_chr + (' ' * (width-2)) + vertical_chr + ' ')

            pretty_sign = stringPict('')
            pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))


            max_upper = max(max_upper, pretty_upper.height())

            if first:
                sign_height = pretty_sign.height()

            pretty_sign = prettyForm(*pretty_sign.above(pretty_upper))
            pretty_sign = prettyForm(*pretty_sign.below(pretty_lower))

            if first:
                pretty_func.baseline = 0
                first = False

            height = pretty_sign.height()
            padding = stringPict('')
            padding = prettyForm(*padding.stack(*[' ']*(height - 1)))
            pretty_sign = prettyForm(*pretty_sign.right(padding))

            pretty_func = prettyForm(*pretty_sign.right(pretty_func))

        pretty_func.baseline = max_upper + sign_height//2
        pretty_func.binding = prettyForm.MUL
        return pretty_func

    def __print_SumProduct_Limits(self, lim):
        def print_start(lhs, rhs):
            op = prettyForm(' ' + xsym("==") + ' ')
            l = self._print(lhs)
            r = self._print(rhs)
            pform = prettyForm(*stringPict.next(l, op, r))
            return pform

        prettyUpper = self._print(lim[2])
        prettyLower = print_start(lim[0], lim[1])
        return prettyLower, prettyUpper

    def _print_Sum(self, expr):
        ascii_mode = not self._use_unicode

        def asum(hrequired, lower, upper, use_ascii):
            def adjust(s, wid=None, how='<^>'):
                if not wid or len(s) > wid:
                    return s
                need = wid - len(s)
                if how in ('<^>', "<") or how not in list('<^>'):
                    return s + ' '*need
                half = need//2
                lead = ' '*half
                if how == ">":
                    return " "*need + s
                return lead + s + ' '*(need - len(lead))

            h = max(hrequired, 2)
            d = h//2
            w = d + 1
            more = hrequired % 2

            lines = []
            if use_ascii:
                lines.append("_"*(w) + ' ')
                lines.append(r"\%s`" % (' '*(w - 1)))
                for i in range(1, d):
                    lines.append('%s\\%s' % (' '*i, ' '*(w - i)))
                if more:
                    lines.append('%s)%s' % (' '*(d), ' '*(w - d)))
                for i in reversed(range(1, d)):
                    lines.append('%s/%s' % (' '*i, ' '*(w - i)))
                lines.append("/" + "_"*(w - 1) + ',')
                return d, h + more, lines, more
            else:
                w = w + more
                d = d + more
                vsum = vobj('sum', 4)
                lines.append("_"*(w))
                for i in range(0, d):
                    lines.append('%s%s%s' % (' '*i, vsum[2], ' '*(w - i - 1)))
                for i in reversed(range(0, d)):
                    lines.append('%s%s%s' % (' '*i, vsum[4], ' '*(w - i - 1)))
                lines.append(vsum[8]*(w))
                return d, h + 2*more, lines, more

        f = expr.function

        prettyF = self._print(f)

        if f.is_Add:  # add parens
            prettyF = prettyForm(*prettyF.parens())

        H = prettyF.height() + 2

        # \sum \sum \sum ...
        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            prettyLower, prettyUpper = self.__print_SumProduct_Limits(lim)

            max_upper = max(max_upper, prettyUpper.height())

            # Create sum sign based on the height of the argument
            d, h, slines, adjustment = asum(
                H, prettyLower.width(), prettyUpper.width(), ascii_mode)
            prettySign = stringPict('')
            prettySign = prettyForm(*prettySign.stack(*slines))

            if first:
                sign_height = prettySign.height()

            prettySign = prettyForm(*prettySign.above(prettyUpper))
            prettySign = prettyForm(*prettySign.below(prettyLower))

            if first:
                # change F baseline so it centers on the sign
                prettyF.baseline -= d - (prettyF.height()//2 -
                                         prettyF.baseline)
                first = False

            # put padding to the right
            pad = stringPict('')
            pad = prettyForm(*pad.stack(*[' ']*h))
            prettySign = prettyForm(*prettySign.right(pad))
            # put the present prettyF to the right
            prettyF = prettyForm(*prettySign.right(prettyF))

        # adjust baseline of ascii mode sigma with an odd height so that it is
        # exactly through the center
        ascii_adjustment = ascii_mode if not adjustment else 0
        prettyF.baseline = max_upper + sign_height//2 + ascii_adjustment

        prettyF.binding = prettyForm.MUL
        return prettyF

    def _print_Limit(self, l):
        e, z, z0, dir = l.args

        E = self._print(e)
        if precedence(e) <= PRECEDENCE["Mul"]:
            E = prettyForm(*E.parens('(', ')'))
        Lim = prettyForm('lim')

        LimArg = self._print(z)
        if self._use_unicode:
            LimArg = prettyForm(*LimArg.right(f"{xobj('-', 1)}{pretty_atom('Arrow')}"))
        else:
            LimArg = prettyForm(*LimArg.right('->'))
        LimArg = prettyForm(*LimArg.right(self._print(z0)))

        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            dir = ""
        else:
            if self._use_unicode:
                dir = pretty_atom('SuperscriptPlus') if str(dir) == "+" else pretty_atom('SuperscriptMinus')

        LimArg = prettyForm(*LimArg.right(self._print(dir)))

        Lim = prettyForm(*Lim.below(LimArg))
        Lim = prettyForm(*Lim.right(E), binding=prettyForm.MUL)

        return Lim

    def _print_matrix_contents(self, e):
        """
        This method factors out what is essentially grid printing.
        """
        M = e   # matrix
        Ms = {}  # i,j -> pretty(M[i,j])
        for i in range(M.rows):
            for j in range(M.cols):
                Ms[i, j] = self._print(M[i, j])

        # h- and v- spacers
        hsep = 2
        vsep = 1

        # max width for columns
        maxw = [-1] * M.cols

        for j in range(M.cols):
            maxw[j] = max([Ms[i, j].width() for i in range(M.rows)] or [0])

        # drawing result
        D = None

        for i in range(M.rows):

            D_row = None
            for j in range(M.cols):
                s = Ms[i, j]

                # reshape s to maxw
                # XXX this should be generalized, and go to stringPict.reshape ?
                assert s.width() <= maxw[j]

                # hcenter it, +0.5 to the right                        2
                # ( it's better to align formula starts for say 0 and r )
                # XXX this is not good in all cases -- maybe introduce vbaseline?
                left, right = center_pad(s.width(), maxw[j])

                s = prettyForm(*s.right(right))
                s = prettyForm(*s.left(left))

                # we don't need vcenter cells -- this is automatically done in
                # a pretty way because when their baselines are taking into
                # account in .right()

                if D_row is None:
                    D_row = s   # first box in a row
                    continue

                D_row = prettyForm(*D_row.right(' '*hsep))  # h-spacer
                D_row = prettyForm(*D_row.right(s))

            if D is None:
                D = D_row       # first row in a picture
                continue

            # v-spacer
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            D = prettyForm(*D.below(D_row))

        if D is None:
            D = prettyForm('')  # Empty Matrix

        return D

    def _print_MatrixBase(self, e, lparens='[', rparens=']'):
        D = self._print_matrix_contents(e)
        D.baseline = D.height()//2
        D = prettyForm(*D.parens(lparens, rparens))
        return D

    def _print_Determinant(self, e):
        mat = e.arg
        if mat.is_MatrixExpr:
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            if isinstance(mat, BlockMatrix):
                return self._print_MatrixBase(mat.blocks, lparens='|', rparens='|')
            D = self._print(mat)
            D.baseline = D.height()//2
            return prettyForm(*D.parens('|', '|'))
        else:
            return self._print_MatrixBase(mat, lparens='|', rparens='|')

    def _print_TensorProduct(self, expr):
        # This should somehow share the code with _print_WedgeProduct:
        if self._use_unicode:
            circled_times = "\u2297"
        else:
            circled_times = ".*"
        return self._print_seq(expr.args, None, None, circled_times,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_WedgeProduct(self, expr):
        # This should somehow share the code with _print_TensorProduct:
        if self._use_unicode:
            wedge_symbol = "\u2227"
        else:
            wedge_symbol = '/\\'
        return self._print_seq(expr.args, None, None, wedge_symbol,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_Trace(self, e):
        D = self._print(e.arg)
        D = prettyForm(*D.parens('(',')'))
        D.baseline = D.height()//2
        D = prettyForm(*D.left('\n'*(0) + 'tr'))
        return D


    def _print_MatrixElement(self, expr):
        from sympy.matrices import MatrixSymbol
        if (isinstance(expr.parent, MatrixSymbol)
                and expr.i.is_number and expr.j.is_number):
            return self._print(
                    Symbol(expr.parent.name + '_%d%d' % (expr.i, expr.j)))
        else:
            prettyFunc = self._print(expr.parent)
            prettyFunc = prettyForm(*prettyFunc.parens())
            prettyIndices = self._print_seq((expr.i, expr.j), delimiter=', '
                    ).parens(left='[', right=']')[0]
            pform = prettyForm(binding=prettyForm.FUNC,
                    *stringPict.next(prettyFunc, prettyIndices))

            # store pform parts so it can be reassembled e.g. when powered
            pform.prettyFunc = prettyFunc
            pform.prettyArgs = prettyIndices

            return pform


    def _print_MatrixSlice(self, m):
        # XXX works only for applied functions
        from sympy.matrices import MatrixSymbol
        prettyFunc = self._print(m.parent)
        if not isinstance(m.parent, MatrixSymbol):
            prettyFunc = prettyForm(*prettyFunc.parens())
        def ppslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return prettyForm(*self._print_seq(x, delimiter=':'))
        prettyArgs = self._print_seq((ppslice(m.rowslice, m.parent.rows),
            ppslice(m.colslice, m.parent.cols)), delimiter=', ').parens(left='[', right=']')[0]

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    def _print_Transpose(self, expr):
        mat = expr.arg
        pform = self._print(mat)
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            pform = prettyForm(*pform.parens())
        pform = pform**(prettyForm('T'))
        return pform

    def _print_Adjoint(self, expr):
        mat = expr.arg
        pform = self._print(mat)
        if self._use_unicode:
            dag = prettyForm(pretty_atom('Dagger'))
        else:
            dag = prettyForm('+')
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            pform = prettyForm(*pform.parens())
        pform = pform**dag
        return pform

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            return self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_MatAdd(self, expr):
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                coeff = item.as_coeff_mmul()[0]
                if S(coeff).could_extract_minus_sign():
                    s = prettyForm(*stringPict.next(s, ' '))
                    pform = self._print(item)
                else:
                    s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))

        return s

    def _print_MatMul(self, expr):
        args = list(expr.args)
        from sympy.matrices.expressions.hadamard import HadamardProduct
        from sympy.matrices.expressions.kronecker import KroneckerProduct
        from sympy.matrices.expressions.matadd import MatAdd
        for i, a in enumerate(args):
            if (isinstance(a, (Add, MatAdd, HadamardProduct, KroneckerProduct))
                    and len(expr.args) > 1):
                args[i] = prettyForm(*self._print(a).parens())
            else:
                args[i] = self._print(a)

        return prettyForm.__mul__(*args)

    def _print_Identity(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('IdentityMatrix'))
        else:
            return prettyForm('I')

    def _print_ZeroMatrix(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('ZeroMatrix'))
        else:
            return prettyForm('0')

    def _print_OneMatrix(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom("OneMatrix"))
        else:
            return prettyForm('1')

    def _print_DotProduct(self, expr):
        args = list(expr.args)

        for i, a in enumerate(args):
            args[i] = self._print(a)
        return prettyForm.__mul__(*args)

    def _print_MatPow(self, expr):
        pform = self._print(expr.base)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.base, MatrixSymbol) and expr.base.is_MatrixExpr:
            pform = prettyForm(*pform.parens())
        pform = pform**(self._print(expr.exp))
        return pform

    def _print_HadamardProduct(self, expr):
        from sympy.matrices.expressions.hadamard import HadamardProduct
        from sympy.matrices.expressions.matadd import MatAdd
        from sympy.matrices.expressions.matmul import MatMul
        if self._use_unicode:
            delim = pretty_atom('Ring')
        else:
            delim = '.*'
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul, HadamardProduct)))

    def _print_HadamardPower(self, expr):
        # from sympy import MatAdd, MatMul
        if self._use_unicode:
            circ = pretty_atom('Ring')
        else:
            circ = self._print('.')
        pretty_base = self._print(expr.base)
        pretty_exp = self._print(expr.exp)
        if precedence(expr.exp) < PRECEDENCE["Mul"]:
            pretty_exp = prettyForm(*pretty_exp.parens())
        pretty_circ_exp = prettyForm(
            binding=prettyForm.LINE,
            *stringPict.next(circ, pretty_exp)
        )
        return pretty_base**pretty_circ_exp

    def _print_KroneckerProduct(self, expr):
        from sympy.matrices.expressions.matadd import MatAdd
        from sympy.matrices.expressions.matmul import MatMul
        if self._use_unicode:
            delim = f" {pretty_atom('TensorProduct')} "
        else:
            delim = ' x '
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    def _print_FunctionMatrix(self, X):
        D = self._print(X.lamda.expr)
        D = prettyForm(*D.parens('[', ']'))
        return D

    def _print_TransferFunction(self, expr):
        if not expr.num == 1:
            num, den = expr.num, expr.den
            res = Mul(num, Pow(den, -1, evaluate=False), evaluate=False)
            return self._print_Mul(res)
        else:
            return self._print(1)/self._print(expr.den)

    def _print_Series(self, expr):
        args = list(expr.args)
        for i, a in enumerate(expr.args):
            args[i] = prettyForm(*self._print(a).parens())
        return prettyForm.__mul__(*args)

    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)
        pretty_args = []
        for a in reversed(args):
            if (isinstance(a, MIMOParallel) and len(expr.args) > 1):
                expression = self._print(a)
                expression.baseline = expression.height()//2
                pretty_args.append(prettyForm(*expression.parens()))
            else:
                expression = self._print(a)
                expression.baseline = expression.height()//2
                pretty_args.append(expression)
        return prettyForm.__mul__(*pretty_args)

    def _print_Parallel(self, expr):
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                s = prettyForm(*stringPict.next(s))
                s.baseline = s.height()//2
                s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))
        return s

    def _print_MIMOParallel(self, expr):
        from sympy.physics.control.lti import TransferFunctionMatrix
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                s = prettyForm(*stringPict.next(s))
                s.baseline = s.height()//2
                s = prettyForm(*stringPict.next(s, ' + '))
                if isinstance(item, TransferFunctionMatrix):
                    s.baseline = s.height() - 1
                s = prettyForm(*stringPict.next(s, pform))
            # s.baseline = s.height()//2
        return s

    def _print_Feedback(self, expr):
        from sympy.physics.control import TransferFunction, Series

        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        den_arg_list = list(expr.sys2.args) if \
            isinstance(expr.sys2, Series) else [expr.sys2]

        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den = Series(*num_arg_list)
            else:
                den = Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den = Series(*den_arg_list)
            else:
                den = Series(num, *den_arg_list)
        else:
            if num == tf:
                den = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den = Series(*num_arg_list)
            else:
                den = Series(*num_arg_list, *den_arg_list)

        denom = prettyForm(*stringPict.next(self._print(tf)))
        denom.baseline = denom.height()//2
        denom = prettyForm(*stringPict.next(denom, ' + ')) if expr.sign == -1 \
            else prettyForm(*stringPict.next(denom, ' - '))
        denom = prettyForm(*stringPict.next(denom, self._print(den)))

        return self._print(num)/denom

    def _print_MIMOFeedback(self, expr):
        from sympy.physics.control import MIMOSeries, TransferFunctionMatrix

        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        plant = self._print(expr.sys1)
        _feedback = prettyForm(*stringPict.next(inv_mat))
        _feedback = prettyForm(*stringPict.right("I + ", _feedback)) if expr.sign == -1 \
            else prettyForm(*stringPict.right("I - ", _feedback))
        _feedback = prettyForm(*stringPict.parens(_feedback))
        _feedback.baseline = 0
        _feedback = prettyForm(*stringPict.right(_feedback, '-1 '))
        _feedback.baseline = _feedback.height()//2
        _feedback = prettyForm.__mul__(_feedback, prettyForm(" "))
        if isinstance(expr.sys1, TransferFunctionMatrix):
            _feedback.baseline = _feedback.height() - 1
        _feedback = prettyForm(*stringPict.next(_feedback, plant))
        return _feedback

    def _print_TransferFunctionMatrix(self, expr):
        mat = self._print(expr._expr_mat)
        mat.baseline = mat.height() - 1
        subscript = greek_unicode['tau'] if self._use_unicode else r'{t}'
        mat = prettyForm(*mat.right(subscript))
        return mat

    def _print_StateSpace(self, expr):
        from sympy.matrices.expressions.blockmatrix import BlockMatrix
        A = expr._A
        B = expr._B
        C = expr._C
        D = expr._D
        mat = BlockMatrix([[A, B], [C, D]])
        return self._print(mat.blocks)

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        if not self._use_unicode:
            raise NotImplementedError("ASCII pretty printing of BasisDependent is not implemented")

        if expr == expr.zero:
            return prettyForm(expr.zero._pretty_form)
        o1 = []
        vectstrs = []
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]
        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x: x[0].__str__())
            for k, v in inneritems:
                #if the coef of the basis vector is 1
                #we skip the 1
                if v == 1:
                    o1.append("" +
                              k._pretty_form)
                #Same for -1
                elif v == -1:
                    o1.append("(-1) " +
                              k._pretty_form)
                #For a general expr
                else:
                    #We always wrap the measure numbers in
                    #parentheses
                    arg_str = self._print(
                        v).parens()[0]

                    o1.append(arg_str + ' ' + k._pretty_form)
                vectstrs.append(k._pretty_form)

        #outstr = u("").join(o1)
        if o1[0].startswith(" + "):
            o1[0] = o1[0][3:]
        elif o1[0].startswith(" "):
            o1[0] = o1[0][1:]
        #Fixing the newlines
        lengths = []
        strs = ['']
        flag = []
        for i, partstr in enumerate(o1):
            flag.append(0)
            # XXX: What is this hack?
            if '\n' in partstr:
                tempstr = partstr
                tempstr = tempstr.replace(vectstrs[i], '')
                if xobj(')_ext', 1) in tempstr:   # If scalar is a fraction
                    for paren in range(len(tempstr)):
                        flag[i] = 1
                        if tempstr[paren] == xobj(')_ext', 1) and tempstr[paren + 1] == '\n':
                            # We want to place the vector string after all the right parentheses, because
                            # otherwise, the vector will be in the middle of the string
                            tempstr = tempstr[:paren] + xobj(')_ext', 1)\
                                         + ' '  + vectstrs[i] + tempstr[paren + 1:]
                            break
                elif xobj(')_lower_hook', 1) in tempstr:
                    # We want to place the vector string after all the right parentheses, because
                    # otherwise, the vector will be in the middle of the string. For this reason,
                    # we insert the vector string at the rightmost index.
                    index = tempstr.rfind(xobj(')_lower_hook', 1))
                    if index != -1: # then this character was found in this string
                        flag[i] = 1
                        tempstr = tempstr[:index] + xobj(')_lower_hook', 1)\
                                     + ' '  + vectstrs[i] + tempstr[index + 1:]
                o1[i] = tempstr

        o1 = [x.split('\n') for x in o1]
        n_newlines = max(len(x) for x in o1)  # Width of part in its pretty form

        if 1 in flag:                           # If there was a fractional scalar
            for i, parts in enumerate(o1):
                if len(parts) == 1:             # If part has no newline
                    parts.insert(0, ' ' * (len(parts[0])))
                    flag[i] = 1

        for i, parts in enumerate(o1):
            lengths.append(len(parts[flag[i]]))
            for j in range(n_newlines):
                if j+1 <= len(parts):
                    if j >= len(strs):
                        strs.append(' ' * (sum(lengths[:-1]) +
                                           3*(len(lengths)-1)))
                    if j == flag[i]:
                        strs[flag[i]] += parts[flag[i]] + ' + '
                    else:
                        strs[j] += parts[j] + ' '*(lengths[-1] -
                                                   len(parts[j])+
                                                   3)
                else:
                    if j >= len(strs):
                        strs.append(' ' * (sum(lengths[:-1]) +
                                           3*(len(lengths)-1)))
                    strs[j] += ' '*(lengths[-1]+3)

        return prettyForm('\n'.join([s[:-3] for s in strs]))

    def _print_NDimArray(self, expr):
        from sympy.matrices.immutable import ImmutableMatrix

        if expr.rank() == 0:
            return self._print(expr[()])

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        # leave eventual matrix elements unflattened
        mat = lambda x: ImmutableMatrix(x, evaluate=False)
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(expr[outer_i])
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(level_str[back_outer_i+1])
                else:
                    level_str[back_outer_i].append(mat(
                        level_str[back_outer_i+1]))
                    if len(level_str[back_outer_i + 1]) == 1:
                        level_str[back_outer_i][-1] = mat(
                            [[level_str[back_outer_i][-1]]])
                even = not even
                level_str[back_outer_i+1] = []

        out_expr = level_str[0][0]
        if expr.rank() % 2 == 1:
            out_expr = mat([out_expr])

        return self._print(out_expr)

    def _printer_tensor_indices(self, name, indices, index_map={}):
        center = stringPict(name)
        top = stringPict(" "*center.width())
        bot = stringPict(" "*center.width())

        last_valence = None
        prev_map = None

        for index in indices:
            indpic = self._print(index.args[0])
            if ((index in index_map) or prev_map) and last_valence == index.is_up:
                if index.is_up:
                    top = prettyForm(*stringPict.next(top, ","))
                else:
                    bot = prettyForm(*stringPict.next(bot, ","))
            if index in index_map:
                indpic = prettyForm(*stringPict.next(indpic, "="))
                indpic = prettyForm(*stringPict.next(indpic, self._print(index_map[index])))
                prev_map = True
            else:
                prev_map = False
            if index.is_up:
                top = stringPict(*top.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                bot = stringPict(*bot.right(" "*indpic.width()))
            else:
                bot = stringPict(*bot.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                top = stringPict(*top.right(" "*indpic.width()))
            last_valence = index.is_up

        pict = prettyForm(*center.above(top))
        pict = prettyForm(*pict.below(bot))
        return pict

    def _print_Tensor(self, expr):
        name = expr.args[0].name
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices)

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].name
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        sign, args = expr._get_args_for_traditional_printer()
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in args
        ]
        pform = prettyForm.__mul__(*args)
        if sign:
            return prettyForm(*pform.left(sign))
        else:
            return pform

    def _print_TensAdd(self, expr):
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in expr.args
        ]
        return prettyForm.__add__(*args)

    def _print_TensorIndex(self, expr):
        sym = expr.args[0]
        if not expr.is_up:
            sym = -sym
        return self._print(sym)

    def _print_PartialDerivative(self, deriv):
        if self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None

        for variable in reversed(deriv.variables):
            s = self._print(variable)
            ds = prettyForm(*s.left(deriv_symbol))

            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        pform = prettyForm(deriv_symbol)

        if len(deriv.variables) > 1:
            pform = pform**self._print(len(deriv.variables))

        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        return pform

    def _print_Piecewise(self, pexpr):

        P = {}
        for n, ec in enumerate(pexpr.args):
            P[n, 0] = self._print(ec.expr)
            if ec.cond == True:
                P[n, 1] = prettyForm('otherwise')
            else:
                P[n, 1] = prettyForm(
                    *prettyForm('for ').right(self._print(ec.cond)))
        hsep = 2
        vsep = 1
        len_args = len(pexpr.args)

        # max widths
        maxw = [max(P[i, j].width() for i in range(len_args))
                for j in range(2)]

        # FIXME: Refactor this code and matrix into some tabular environment.
        # drawing result
        D = None

        for i in range(len_args):
            D_row = None
            for j in range(2):
                p = P[i, j]
                assert p.width() <= maxw[j]

                wdelta = maxw[j] - p.width()
                wleft = wdelta // 2
                wright = wdelta - wleft

                p = prettyForm(*p.right(' '*wright))
                p = prettyForm(*p.left(' '*wleft))

                if D_row is None:
                    D_row = p
                    continue

                D_row = prettyForm(*D_row.right(' '*hsep))  # h-spacer
                D_row = prettyForm(*D_row.right(p))
            if D is None:
                D = D_row       # first row in a picture
                continue

            # v-spacer
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            D = prettyForm(*D.below(D_row))

        D = prettyForm(*D.parens('{', ''))
        D.baseline = D.height()//2
        D.binding = prettyForm.OPEN
        return D

    def _print_ITE(self, ite):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(ite.rewrite(Piecewise))

    def _hprint_vec(self, v):
        D = None

        for a in v:
            p = a
            if D is None:
                D = p
            else:
                D = prettyForm(*D.right(', '))
                D = prettyForm(*D.right(p))
        if D is None:
            D = stringPict(' ')

        return D

    def _hprint_vseparator(self, p1, p2, left=None, right=None, delimiter='', ifascii_nougly=False):
        if ifascii_nougly and not self._use_unicode:
            return self._print_seq((p1, '|', p2), left=left, right=right,
                                   delimiter=delimiter, ifascii_nougly=True)
        tmp = self._print_seq((p1, p2,), left=left, right=right, delimiter=delimiter)
        sep = stringPict(vobj('|', tmp.height()), baseline=tmp.baseline)
        return self._print_seq((p1, sep, p2), left=left, right=right,
                               delimiter=delimiter)

    def _print_hyper(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment
        ap = [self._print(a) for a in e.ap]
        bq = [self._print(b) for b in e.bq]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        # Drawing result - first create the ap, bq vectors
        D = None
        for v in [ap, bq]:
            D_row = self._hprint_vec(v)
            if D is None:
                D = D_row       # first row in a picture
            else:
                D = prettyForm(*D.below(' '))
                D = prettyForm(*D.below(D_row))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the F symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('F')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)
        add = (sz + 1)//2

        F = prettyForm(*F.left(self._print(len(e.ap))))
        F = prettyForm(*F.right(self._print(len(e.bq))))
        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D

    def _print_meijerg(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment

        v = {}
        v[(0, 0)] = [self._print(a) for a in e.an]
        v[(0, 1)] = [self._print(a) for a in e.aother]
        v[(1, 0)] = [self._print(b) for b in e.bm]
        v[(1, 1)] = [self._print(b) for b in e.bother]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        vp = {}
        for idx in v:
            vp[idx] = self._hprint_vec(v[idx])

        for i in range(2):
            maxw = max(vp[(0, i)].width(), vp[(1, i)].width())
            for j in range(2):
                s = vp[(j, i)]
                left = (maxw - s.width()) // 2
                right = maxw - left - s.width()
                s = prettyForm(*s.left(' ' * left))
                s = prettyForm(*s.right(' ' * right))
                vp[(j, i)] = s

        D1 = prettyForm(*vp[(0, 0)].right('  ', vp[(0, 1)]))
        D1 = prettyForm(*D1.below(' '))
        D2 = prettyForm(*vp[(1, 0)].right('  ', vp[(1, 1)]))
        D = prettyForm(*D1.below(D2))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the G symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('G')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)

        pp = self._print(len(e.ap))
        pq = self._print(len(e.bq))
        pm = self._print(len(e.bm))
        pn = self._print(len(e.an))

        def adjust(p1, p2):
            diff = p1.width() - p2.width()
            if diff == 0:
                return p1, p2
            elif diff > 0:
                return p1, prettyForm(*p2.left(' '*diff))
            else:
                return prettyForm(*p1.left(' '*-diff)), p2
        pp, pm = adjust(pp, pm)
        pq, pn = adjust(pq, pn)
        pu = prettyForm(*pm.right(', ', pn))
        pl = prettyForm(*pp.right(', ', pq))

        ht = F.baseline - above - 2
        if ht > 0:
            pu = prettyForm(*pu.below('\n'*ht))
        p = prettyForm(*pu.below(pl))

        F.baseline = above
        F = prettyForm(*F.right(p))

        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D

    def _print_ExpBase(self, e):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        base = prettyForm(pretty_atom('Exp1', 'e'))
        return base ** self._print(e.args[0])

    def _print_Exp1(self, e):
        return prettyForm(pretty_atom('Exp1', 'e'))

    def _print_Function(self, e, sort=False, func_name=None, left='(',
                        right=')'):
        # optional argument func_name for supplying custom names
        # XXX works only for applied functions
        return self._helper_print_function(e.func, e.args, sort=sort, func_name=func_name, left=left, right=right)

    def _print_mathieuc(self, e):
        return self._print_Function(e, func_name='C')

    def _print_mathieus(self, e):
        return self._print_Function(e, func_name='S')

    def _print_mathieucprime(self, e):
        return self._print_Function(e, func_name="C'")

    def _print_mathieusprime(self, e):
        return self._print_Function(e, func_name="S'")

    def _helper_print_function(self, func, args, sort=False, func_name=None,
                               delimiter=', ', elementwise=False, left='(',
                               right=')'):
        if sort:
            args = sorted(args, key=default_sort_key)

        if not func_name and hasattr(func, "__name__"):
            func_name = func.__name__

        if func_name:
            prettyFunc = self._print(Symbol(func_name))
        else:
            prettyFunc = prettyForm(*self._print(func).parens())

        if elementwise:
            if self._use_unicode:
                circ = pretty_atom('Modifier Letter Low Ring')
            else:
                circ = '.'
            circ = self._print(circ)
            prettyFunc = prettyForm(
                binding=prettyForm.LINE,
                *stringPict.next(prettyFunc, circ)
            )

        prettyArgs = prettyForm(*self._print_seq(args, delimiter=delimiter).parens(
                                                 left=left, right=right))

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    def _print_ElementwiseApplyFunction(self, e):
        func = e.function
        arg = e.expr
        args = [arg]
        return self._helper_print_function(func, args, delimiter="", elementwise=True)

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.zeta_functions import lerchphi
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: [greek_unicode['delta'], 'delta'],
                gamma: [greek_unicode['Gamma'], 'Gamma'],
                lerchphi: [greek_unicode['Phi'], 'lerchphi'],
                lowergamma: [greek_unicode['gamma'], 'gamma'],
                beta: [greek_unicode['Beta'], 'B'],
                DiracDelta: [greek_unicode['delta'], 'delta'],
                Chi: ['Chi', 'Chi']}

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                if self._use_unicode:
                    return prettyForm(self._special_function_classes[cls][0])
                else:
                    return prettyForm(self._special_function_classes[cls][1])
        func_name = expr.__name__
        return prettyForm(pretty_symbol(func_name))

    def _print_GeometryEntity(self, expr):
        # GeometryEntity is based on Tuple but should not print like a Tuple
        return self.emptyPrinter(expr)

    def _print_polylog(self, e):
        subscript = self._print(e.args[0])
        if self._use_unicode and is_subscriptable_in_unicode(subscript):
            return self._print_Function(Function('Li_%s' % subscript)(e.args[1]))
        return self._print_Function(e)

    def _print_lerchphi(self, e):
        func_name = greek_unicode['Phi'] if self._use_unicode else 'lerchphi'
        return self._print_Function(e, func_name=func_name)

    def _print_dirichlet_eta(self, e):
        func_name = greek_unicode['eta'] if self._use_unicode else 'dirichlet_eta'
        return self._print_Function(e, func_name=func_name)

    def _print_Heaviside(self, e):
        func_name = greek_unicode['theta'] if self._use_unicode else 'Heaviside'
        if e.args[1] is S.Half:
            pform = prettyForm(*self._print(e.args[0]).parens())
            pform = prettyForm(*pform.left(func_name))
            return pform
        else:
            return self._print_Function(e, func_name=func_name)

    def _print_fresnels(self, e):
        return self._print_Function(e, func_name="S")

    def _print_fresnelc(self, e):
        return self._print_Function(e, func_name="C")

    def _print_airyai(self, e):
        return self._print_Function(e, func_name="Ai")

    def _print_airybi(self, e):
        return self._print_Function(e, func_name="Bi")

    def _print_airyaiprime(self, e):
        return self._print_Function(e, func_name="Ai'")

    def _print_airybiprime(self, e):
        return self._print_Function(e, func_name="Bi'")

    def _print_LambertW(self, e):
        return self._print_Function(e, func_name="W")

    def _print_Covariance(self, e):
        return self._print_Function(e, func_name="Cov")

    def _print_Variance(self, e):
        return self._print_Function(e, func_name="Var")

    def _print_Probability(self, e):
        return self._print_Function(e, func_name="P")

    def _print_Expectation(self, e):
        return self._print_Function(e, func_name="E", left='[', right=']')

    def _print_Lambda(self, e):
        expr = e.expr
        sig = e.signature
        if self._use_unicode:
            arrow = f" {pretty_atom('ArrowFromBar')} "
        else:
            arrow = " -> "
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        var_form = self._print(sig)

        return prettyForm(*stringPict.next(var_form, arrow, self._print(expr)), binding=8)

    def _print_Order(self, expr):
        pform = self._print(expr.expr)
        if (expr.point and any(p != S.Zero for p in expr.point)) or \
           len(expr.variables) > 1:
            pform = prettyForm(*pform.right("; "))
            if len(expr.variables) > 1:
                pform = prettyForm(*pform.right(self._print(expr.variables)))
            elif len(expr.variables):
                pform = prettyForm(*pform.right(self._print(expr.variables[0])))
            if self._use_unicode:
                pform = prettyForm(*pform.right(f" {pretty_atom('Arrow')} "))
            else:
                pform = prettyForm(*pform.right(" -> "))
            if len(expr.point) > 1:
                pform = prettyForm(*pform.right(self._print(expr.point)))
            else:
                pform = prettyForm(*pform.right(self._print(expr.point[0])))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left("O"))
        return pform

    def _print_SingularityFunction(self, e):
        if self._use_unicode:
            shift = self._print(e.args[0]-e.args[1])
            n = self._print(e.args[2])
            base = prettyForm("<")
            base = prettyForm(*base.right(shift))
            base = prettyForm(*base.right(">"))
            pform = base**n
            return pform
        else:
            n = self._print(e.args[2])
            shift = self._print(e.args[0]-e.args[1])
            base = self._print_seq(shift, "<", ">", ' ')
            return base**n

    def _print_beta(self, e):
        func_name = greek_unicode['Beta'] if self._use_unicode else 'B'
        return self._print_Function(e, func_name=func_name)

    def _print_betainc(self, e):
        func_name = "B'"
        return self._print_Function(e, func_name=func_name)

    def _print_betainc_regularized(self, e):
        func_name = 'I'
        return self._print_Function(e, func_name=func_name)

    def _print_gamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        return self._print_Function(e, func_name=func_name)

    def _print_uppergamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        return self._print_Function(e, func_name=func_name)

    def _print_lowergamma(self, e):
        func_name = greek_unicode['gamma'] if self._use_unicode else 'lowergamma'
        return self._print_Function(e, func_name=func_name)

    def _print_DiracDelta(self, e):
        if self._use_unicode:
            if len(e.args) == 2:
                a = prettyForm(greek_unicode['delta'])
                b = self._print(e.args[1])
                b = prettyForm(*b.parens())
                c = self._print(e.args[0])
                c = prettyForm(*c.parens())
                pform = a**b
                pform = prettyForm(*pform.right(' '))
                pform = prettyForm(*pform.right(c))
                return pform
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens())
            pform = prettyForm(*pform.left(greek_unicode['delta']))
            return pform
        else:
            return self._print_Function(e)

    def _print_expint(self, e):
        subscript = self._print(e.args[0])
        if self._use_unicode and is_subscriptable_in_unicode(subscript):
            return self._print_Function(Function('E_%s' % subscript)(e.args[1]))
        return self._print_Function(e)

    def _print_Chi(self, e):
        # This needs a special case since otherwise it comes out as greek
        # letter chi...
        prettyFunc = prettyForm("Chi")
        prettyArgs = prettyForm(*self._print_seq(e.args).parens())

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    def _print_elliptic_e(self, e):
        pforma0 = self._print(e.args[0])
        if len(e.args) == 1:
            pform = pforma0
        else:
            pforma1 = self._print(e.args[1])
            pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('E'))
        return pform

    def _print_elliptic_k(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('K'))
        return pform

    def _print_elliptic_f(self, e):
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('F'))
        return pform

    def _print_elliptic_pi(self, e):
        name = greek_unicode['Pi'] if self._use_unicode else 'Pi'
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        if len(e.args) == 2:
            pform = self._hprint_vseparator(pforma0, pforma1)
        else:
            pforma2 = self._print(e.args[2])
            pforma = self._hprint_vseparator(pforma1, pforma2, ifascii_nougly=False)
            pforma = prettyForm(*pforma.left('; '))
            pform = prettyForm(*pforma.left(pforma0))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left(name))
        return pform

    def _print_GoldenRatio(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_symbol('phi'))
        return self._print(Symbol("GoldenRatio"))

    def _print_EulerGamma(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_symbol('gamma'))
        return self._print(Symbol("EulerGamma"))

    def _print_Catalan(self, expr):
        return self._print(Symbol("G"))

    def _print_Mod(self, expr):
        pform = self._print(expr.args[0])
        if pform.binding > prettyForm.MUL:
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right(' mod '))
        pform = prettyForm(*pform.right(self._print(expr.args[1])))
        pform.binding = prettyForm.OPEN
        return pform

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)
        pforms, indices = [], []

        def pretty_negative(pform, index):
            """Prepend a minus sign to a pretty form. """
            #TODO: Move this code to prettyForm
            if index == 0:
                if pform.height() > 1:
                    pform_neg = '- '
                else:
                    pform_neg = '-'
            else:
                pform_neg = ' - '

            if (pform.binding > prettyForm.NEG
                or pform.binding == prettyForm.ADD):
                p = stringPict(*pform.parens())
            else:
                p = pform
            p = stringPict.next(pform_neg, p)
            # Lower the binding to NEG, even if it was higher. Otherwise, it
            # will print as a + ( - (b)), instead of a - (b).
            return prettyForm(binding=prettyForm.NEG, *p)

        for i, term in enumerate(terms):
            if term.is_Mul and term.could_extract_minus_sign():
                coeff, other = term.as_coeff_mul(rational=False)
                if coeff == -1:
                    negterm = Mul(*other, evaluate=False)
                else:
                    negterm = Mul(-coeff, *other, evaluate=False)
                pform = self._print(negterm)
                pforms.append(pretty_negative(pform, i))
            elif term.is_Rational and term.q > 1:
                pforms.append(None)
                indices.append(i)
            elif term.is_Number and term < 0:
                pform = self._print(-term)
                pforms.append(pretty_negative(pform, i))
            elif term.is_Relational:
                pforms.append(prettyForm(*self._print(term).parens()))
            else:
                pforms.append(self._print(term))

        if indices:
            large = True

            for pform in pforms:
                if pform is not None and pform.height() > 1:
                    break
            else:
                large = False

            for i in indices:
                term, negative = terms[i], False

                if term < 0:
                    term, negative = -term, True

                if large:
                    pform = prettyForm(str(term.p))/prettyForm(str(term.q))
                else:
                    pform = self._print(term)

                if negative:
                    pform = pretty_negative(pform, i)

                pforms[i] = pform

        return prettyForm.__add__(*pforms)

    def _print_Mul(self, product):
        from sympy.physics.units import Quantity

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        args = product.args
        if args[0] is S.One or any(isinstance(arg, Number) for arg in args[1:]):
            strargs = list(map(self._print, args))
            # XXX: This is a hack to work around the fact that
            # prettyForm.__mul__ absorbs a leading -1 in the args. Probably it
            # would be better to fix this in prettyForm.__mul__ instead.
            negone = strargs[0] == '-1'
            if negone:
                strargs[0] = prettyForm('1', 0, 0)
            obj = prettyForm.__mul__(*strargs)
            if negone:
                obj = prettyForm('-' + obj.s, obj.baseline, obj.binding)
            return obj

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = product.as_ordered_factors()
        else:
            args = list(product.args)

        # If quantities are present append them at the back
        args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                     (isinstance(x, Pow) and isinstance(x.base, Quantity)))

        # Gather terms for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append( Rational(item.p) )
                if item.q != 1:
                    b.append( Rational(item.q) )
            else:
                a.append(item)

        # Convert to pretty forms. Parentheses are added by `__mul__`.
        a = [self._print(ai) for ai in a]
        b = [self._print(bi) for bi in b]

        # Construct a pretty form
        if len(b) == 0:
            return prettyForm.__mul__(*a)
        else:
            if len(a) == 0:
                a.append( self._print(S.One) )
            return prettyForm.__mul__(*a)/prettyForm.__mul__(*b)

    # A helper function for _print_Pow to print x**(1/n)
    def _print_nth_root(self, base, root):
        bpretty = self._print(base)

        # In very simple cases, use a single-char root sign
        if (self._settings['use_unicode_sqrt_char'] and self._use_unicode
            and root == 2 and bpretty.height() == 1
            and (bpretty.width() == 1
                 or (base.is_Integer and base.is_nonnegative))):
            return prettyForm(*bpretty.left(nth_root[2]))

        # Construct root sign, start with the \/ shape
        _zZ = xobj('/', 1)
        rootsign = xobj('\\', 1) + _zZ
        # Constructing the number to put on root
        rpretty = self._print(root)
        # roots look bad if they are not a single line
        if rpretty.height() != 1:
            return self._print(base)**self._print(1/root)
        # If power is half, no number should appear on top of root sign
        exp = '' if root == 2 else str(rpretty).ljust(2)
        if len(exp) > 2:
            rootsign = ' '*(len(exp) - 2) + rootsign
        # Stack the exponent
        rootsign = stringPict(exp + '\n' + rootsign)
        rootsign.baseline = 0
        # Diagonal: length is one less than height of base
        linelength = bpretty.height() - 1
        diagonal = stringPict('\n'.join(
            ' '*(linelength - i - 1) + _zZ + ' '*i
            for i in range(linelength)
        ))
        # Put baseline just below lowest line: next to exp
        diagonal.baseline = linelength - 1
        # Make the root symbol
        rootsign = prettyForm(*rootsign.right(diagonal))
        # Det the baseline to match contents to fix the height
        # but if the height of bpretty is one, the rootsign must be one higher
        rootsign.baseline = max(1, bpretty.baseline)
        #build result
        s = prettyForm(hobj('_', 2 + bpretty.width()))
        s = prettyForm(*bpretty.above(s))
        s = prettyForm(*s.left(rootsign))
        return s

    def _print_Pow(self, power):
        from sympy.simplify.simplify import fraction
        b, e = power.as_base_exp()
        if power.is_commutative:
            if e is S.NegativeOne:
                return prettyForm("1")/self._print(b)
            n, d = fraction(e)
            if n is S.One and d.is_Atom and not e.is_Integer and (e.is_Rational or d.is_Symbol) \
                    and self._settings['root_notation']:
                return self._print_nth_root(b, d)
            if e.is_Rational and e < 0:
                return prettyForm("1")/self._print(Pow(b, -e, evaluate=False))

        if b.is_Relational:
            return prettyForm(*self._print(b).parens()).__pow__(self._print(e))

        return self._print(b)**self._print(e)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def __print_numer_denom(self, p, q):
        if q == 1:
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)
            else:
                return prettyForm(str(p))
        elif abs(p) >= 10 and abs(q) >= 10:
            # If more than one digit in numer and denom, print larger fraction
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)/prettyForm(str(q))
                # Old printing method:
                #pform = prettyForm(str(-p))/prettyForm(str(q))
                #return prettyForm(binding=prettyForm.NEG, *pform.left('- '))
            else:
                return prettyForm(str(p))/prettyForm(str(q))
        else:
            return None

    def _print_Rational(self, expr):
        result = self.__print_numer_denom(expr.p, expr.q)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    def _print_Fraction(self, expr):
        result = self.__print_numer_denom(expr.numerator, expr.denominator)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    def _print_ProductSet(self, p):
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self._print(p.sets[0]) ** self._print(len(p.sets))
        else:
            prod_char = pretty_atom('Multiplication') if self._use_unicode else 'x'
            return self._print_seq(p.sets, None, None, ' %s ' % prod_char,
                                   parenthesize=lambda set: set.is_Union or
                                   set.is_Intersection or set.is_ProductSet)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_seq(items, '{', '}', ', ' )

    def _print_Range(self, s):

        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'

        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            it = iter(s)
            printset = next(it), next(it), dots
        elif len(s) > 4:
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            printset = tuple(s)

        return self._print_seq(printset, '{', '}', ', ' )

    def _print_Interval(self, i):
        if i.start == i.end:
            return self._print_seq(i.args[:1], '{', '}')

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return self._print_seq(i.args[:2], left, right)

    def _print_AccumulationBounds(self, i):
        left = '<'
        right = '>'

        return self._print_seq(i.args[:2], left, right)

    def _print_Intersection(self, u):

        delimiter = ' %s ' % pretty_atom('Intersection', 'n')

        return self._print_seq(u.args, None, None, delimiter,
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Union or set.is_Complement)

    def _print_Union(self, u):

        union_delimiter = ' %s ' % pretty_atom('Union', 'U')

        return self._print_seq(u.args, None, None, union_delimiter,
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Intersection or set.is_Complement)

    def _print_SymmetricDifference(self, u):
        if not self._use_unicode:
            raise NotImplementedError("ASCII pretty printing of SymmetricDifference is not implemented")

        sym_delimeter = ' %s ' % pretty_atom('SymmetricDifference')

        return self._print_seq(u.args, None, None, sym_delimeter)

    def _print_Complement(self, u):

        delimiter = r' \ '

        return self._print_seq(u.args, None, None, delimiter,
             parenthesize=lambda set: set.is_ProductSet or set.is_Intersection
                               or set.is_Union)

    def _print_ImageSet(self, ts):
        if self._use_unicode:
            inn = pretty_atom("SmallElementOf")
        else:
            inn = 'in'
        fun = ts.lamda
        sets = ts.base_sets
        signature = fun.signature
        expr = self._print(fun.expr)

        # TODO: the stuff to the left of the | and the stuff to the right of
        # the | should have independent baselines, that way something like
        # ImageSet(Lambda(x, 1/x**2), S.Naturals) prints the "x in N" part
        # centered on the right instead of aligned with the fraction bar on
        # the left. The same also applies to ConditionSet and ComplexRegion
        if len(signature) == 1:
            S = self._print_seq((signature[0], inn, sets[0]),
                                delimiter=' ')
            return self._hprint_vseparator(expr, S,
                                           left='{', right='}',
                                           ifascii_nougly=True, delimiter=' ')
        else:
            pargs = tuple(j for var, setv in zip(signature, sets) for j in
                          (var, ' ', inn, ' ', setv, ", "))
            S = self._print_seq(pargs[:-1], delimiter='')
            return self._hprint_vseparator(expr, S,
                                           left='{', right='}',
                                           ifascii_nougly=True, delimiter=' ')

    def _print_ConditionSet(self, ts):
        if self._use_unicode:
            inn = pretty_atom('SmallElementOf')
            # using _and because and is a keyword and it is bad practice to
            # overwrite them
            _and = pretty_atom('And')
        else:
            inn = 'in'
            _and = 'and'

        variables = self._print_seq(Tuple(ts.sym))
        as_expr = getattr(ts.condition, 'as_expr', None)
        if as_expr is not None:
            cond = self._print(ts.condition.as_expr())
        else:
            cond = self._print(ts.condition)
            if self._use_unicode:
                cond = self._print(cond)
                cond = prettyForm(*cond.parens())

        if ts.base_set is S.UniversalSet:
            return self._hprint_vseparator(variables, cond, left="{",
                                           right="}", ifascii_nougly=True,
                                           delimiter=' ')

        base = self._print(ts.base_set)
        C = self._print_seq((variables, inn, base, _and, cond),
                            delimiter=' ')
        return self._hprint_vseparator(variables, C, left="{", right="}",
                                       ifascii_nougly=True, delimiter=' ')

    def _print_ComplexRegion(self, ts):
        if self._use_unicode:
            inn = pretty_atom('SmallElementOf')
        else:
            inn = 'in'
        variables = self._print_seq(ts.variables)
        expr = self._print(ts.expr)
        prodsets = self._print(ts.sets)

        C = self._print_seq((variables, inn, prodsets),
                            delimiter=' ')
        return self._hprint_vseparator(expr, C, left="{", right="}",
                                       ifascii_nougly=True, delimiter=' ')

    def _print_Contains(self, e):
        var, set = e.args
        if self._use_unicode:
            el = f" {pretty_atom('ElementOf')} "
            return prettyForm(*stringPict.next(self._print(var),
                                               el, self._print(set)), binding=8)
        else:
            return prettyForm(sstr(e))

    def _print_FourierSeries(self, s):
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'
        return self._print_Add(s.truncate()) + self._print(dots)

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_SetExpr(self, se):
        pretty_set = prettyForm(*self._print(se.set).parens())
        pretty_name = self._print(Symbol("SetExpr"))
        return prettyForm(*pretty_name.right(pretty_set))

    def _print_SeqFormula(self, s):
        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'

        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            raise NotImplementedError("Pretty printing of sequences with symbolic bound not implemented")

        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
            printset = tuple(printset)
        else:
            printset = tuple(s)
        return self._print_list(printset)

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_seq(self, seq, left=None, right=None, delimiter=', ',
            parenthesize=lambda x: False, ifascii_nougly=True):

        pforms = []
        for item in seq:
            pform = self._print(item)
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if pforms:
                pforms.append(delimiter)
            pforms.append(pform)

        if not pforms:
            s = stringPict('')
        else:
            s = prettyForm(*stringPict.next(*pforms))

        s = prettyForm(*s.parens(left, right, ifascii_nougly=ifascii_nougly))
        return s

    def join(self, delimiter, args):
        pform = None

        for arg in args:
            if pform is None:
                pform = arg
            else:
                pform = prettyForm(*pform.right(delimiter))
                pform = prettyForm(*pform.right(arg))

        if pform is None:
            return prettyForm("")
        else:
            return pform

    def _print_list(self, l):
        return self._print_seq(l, '[', ']')

    def _print_tuple(self, t):
        if len(t) == 1:
            ptuple = prettyForm(*stringPict.next(self._print(t[0]), ','))
            return prettyForm(*ptuple.parens('(', ')', ifascii_nougly=True))
        else:
            return self._print_seq(t, '(', ')')

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for k in keys:
            K = self._print(k)
            V = self._print(d[k])
            s = prettyForm(*stringPict.next(K, ': ', V))

            items.append(s)

        return self._print_seq(items, '{', '}')

    def _print_Dict(self, d):
        return self._print_dict(d)

    def _print_set(self, s):
        if not s:
            return prettyForm('set()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        return pretty

    def _print_frozenset(self, s):
        if not s:
            return prettyForm('frozenset()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        pretty = prettyForm(*pretty.parens('(', ')', ifascii_nougly=True))
        pretty = prettyForm(*stringPict.next(type(s).__name__, pretty))
        return pretty

    def _print_UniversalSet(self, s):
        if self._use_unicode:
            return prettyForm(pretty_atom('Universe'))
        else:
            return prettyForm('UniversalSet')

    def _print_PolyRing(self, ring):
        return prettyForm(sstr(ring))

    def _print_FracField(self, field):
        return prettyForm(sstr(field))

    def _print_FreeGroupElement(self, elm):
        return prettyForm(str(elm))

    def _print_PolyElement(self, poly):
        return prettyForm(sstr(poly))

    def _print_FracElement(self, frac):
        return prettyForm(sstr(frac))

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_ComplexRootOf(self, expr):
        args = [self._print_Add(expr.expr, order='lex'), expr.index]
        pform = prettyForm(*self._print_seq(args).parens())
        pform = prettyForm(*pform.left('CRootOf'))
        return pform

    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        pform = prettyForm(*self._print_seq(args).parens())
        pform = prettyForm(*pform.left('RootSum'))

        return pform

    def _print_FiniteField(self, expr):
        if self._use_unicode:
            form = f"{pretty_atom('Integers')}_%d"
        else:
            form = 'GF(%d)'

        return prettyForm(pretty_symbol(form % expr.mod))

    def _print_IntegerRing(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('Integers'))
        else:
            return prettyForm('ZZ')

    def _print_RationalField(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('Rationals'))
        else:
            return prettyForm('QQ')

    def _print_RealField(self, domain):
        if self._use_unicode:
            prefix = pretty_atom("Reals")
        else:
            prefix = 'RR'

        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    def _print_ComplexField(self, domain):
        if self._use_unicode:
            prefix = pretty_atom('Complexes')
        else:
            prefix = 'CC'

        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    def _print_PolynomialRing(self, expr):
        args = list(expr.symbols)

        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        pform = self._print_seq(args, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_FractionField(self, expr):
        args = list(expr.symbols)

        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        pform = self._print_seq(args, '(', ')')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_PolynomialRingBase(self, expr):
        g = expr.symbols
        if str(expr.order) != str(expr.default_order):
            g = g + ("order=" + str(expr.order),)
        pform = self._print_seq(g, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_GroebnerBasis(self, basis):
        exprs = [ self._print_Add(arg, order=basis.order)
                  for arg in basis.exprs ]
        exprs = prettyForm(*self.join(", ", exprs).parens(left="[", right="]"))

        gens = [ self._print(gen) for gen in basis.gens ]

        domain = prettyForm(
            *prettyForm("domain=").right(self._print(basis.domain)))
        order = prettyForm(
            *prettyForm("order=").right(self._print(basis.order)))

        pform = self.join(", ", [exprs] + gens + [domain, order])

        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left(basis.__class__.__name__))

        return pform

    def _print_Subs(self, e):
        pform = self._print(e.expr)
        pform = prettyForm(*pform.parens())

        h = pform.height() if pform.height() > 1 else 2
        rvert = stringPict(vobj('|', h), baseline=pform.baseline)
        pform = prettyForm(*pform.right(rvert))

        b = pform.baseline
        pform.baseline = pform.height() - 1
        pform = prettyForm(*pform.right(self._print_seq([
            self._print_seq((self._print(v[0]), xsym('=='), self._print(v[1])),
                delimiter='') for v in zip(e.variables, e.point) ])))

        pform.baseline = b
        return pform

    def _print_number_function(self, e, name):
        # Print name_arg[0] for one argument or name_arg[0](arg[1])
        # for more than one argument
        pform = prettyForm(name)
        arg = self._print(e.args[0])
        pform_arg = prettyForm(" "*arg.width())
        pform_arg = prettyForm(*pform_arg.below(arg))
        pform = prettyForm(*pform.right(pform_arg))
        if len(e.args) == 1:
            return pform
        m, x = e.args
        # TODO: copy-pasted from _print_Function: can we do better?
        prettyFunc = pform
        prettyArgs = prettyForm(*self._print_seq([x]).parens())
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs
        return pform

    def _print_euler(self, e):
        return self._print_number_function(e, "E")

    def _print_catalan(self, e):
        return self._print_number_function(e, "C")

    def _print_bernoulli(self, e):
        return self._print_number_function(e, "B")

    _print_bell = _print_bernoulli

    def _print_lucas(self, e):
        return self._print_number_function(e, "L")

    def _print_fibonacci(self, e):
        return self._print_number_function(e, "F")

    def _print_tribonacci(self, e):
        return self._print_number_function(e, "T")

    def _print_stieltjes(self, e):
        if self._use_unicode:
            return self._print_number_function(e, greek_unicode['gamma'])
        else:
            return self._print_number_function(e, "stieltjes")

    def _print_KroneckerDelta(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(self._print(e.args[1])))
        if self._use_unicode:
            a = stringPict(pretty_symbol('delta'))
        else:
            a = stringPict('d')
        b = pform
        top = stringPict(*b.left(' '*a.width()))
        bot = stringPict(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            pform = self._print('Domain: ')
            pform = prettyForm(*pform.right(self._print(d.as_boolean())))
            return pform
        elif hasattr(d, 'set'):
            pform = self._print('Domain: ')
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            pform = prettyForm(*pform.right(self._print(' in ')))
            pform = prettyForm(*pform.right(self._print(d.set)))
            return pform
        elif hasattr(d, 'symbols'):
            pform = self._print('Domain on ')
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            return pform
        else:
            return self._print(None)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(pretty_symbol(object.name))

    def _print_Morphism(self, morphism):
        arrow = xsym("-->")

        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        tail = domain.right(arrow, codomain)[0]

        return prettyForm(tail)

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(pretty_symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(":", pretty_morphism)[0])

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(
            NamedMorphism(morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):

        circle = xsym(".")

        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [pretty_symbol(component.name) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = circle.join(component_names_list) + ":"

        pretty_name = self._print(component_names)
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(pretty_morphism)[0])

    def _print_Category(self, category):
        return self._print(pretty_symbol(category.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        pretty_result = self._print(diagram.premises)
        if diagram.conclusions:
            results_arrow = " %s " % xsym("==>")

            pretty_conclusions = self._print(diagram.conclusions)[0]
            pretty_result = pretty_result.right(
                results_arrow, pretty_conclusions)

        return prettyForm(pretty_result[0])

    def _print_DiagramGrid(self, grid):
        from sympy.matrices import Matrix
        matrix = Matrix([[grid[i, j] if grid[i, j] else Symbol(" ")
                          for j in range(grid.width)]
                         for i in range(grid.height)])
        return self._print_matrix_contents(matrix)

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return self._print_seq(m, '[', ']')

    def _print_SubModule(self, M):
        gens = [[M.ring.to_sympy(g) for g in gen] for gen in M.gens]
        return self._print_seq(gens, '<', '>')

    def _print_FreeModule(self, M):
        return self._print(M.ring)**self._print(M.rank)

    def _print_ModuleImplementedIdeal(self, M):
        sym = M.ring.to_sympy
        return self._print_seq([sym(x) for [x] in M._module.gens], '<', '>')

    def _print_QuotientRing(self, R):
        return self._print(R.ring) / self._print(R.base_ideal)

    def _print_QuotientRingElement(self, R):
        return self._print(R.ring.to_sympy(R)) + self._print(R.ring.base_ideal)

    def _print_QuotientModuleElement(self, m):
        return self._print(m.data) + self._print(m.module.killed_module)

    def _print_QuotientModule(self, M):
        return self._print(M.base) / self._print(M.killed_module)

    def _print_MatrixHomomorphism(self, h):
        matrix = self._print(h._sympy_matrix())
        matrix.baseline = matrix.height() // 2
        pform = prettyForm(*matrix.right(' : ', self._print(h.domain),
            ' %s> ' % hobj('-', 2), self._print(h.codomain)))
        return pform

    def _print_Manifold(self, manifold):
        return self._print(manifold.name)

    def _print_Patch(self, patch):
        return self._print(patch.name)

    def _print_CoordSystem(self, coords):
        return self._print(coords.name)

    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return self._print(pretty_symbol(string))

    def _print_BaseVectorField(self, field):
        s = U('PARTIAL DIFFERENTIAL') + '_' + field._coord_sys.symbols[field._index].name
        return self._print(pretty_symbol(s))

    def _print_Differential(self, diff):
        if self._use_unicode:
            d = pretty_atom('Differential')
        else:
            d = 'd'
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return self._print(d + ' ' + pretty_symbol(string))
        else:
            pform = self._print(field)
            pform = prettyForm(*pform.parens())
            return prettyForm(*pform.left(d))

    def _print_Tr(self, p):
        #TODO: Handle indices
        pform = self._print(p.args[0])
        pform = prettyForm(*pform.left('%s(' % (p.__class__.__name__)))
        pform = prettyForm(*pform.right(')'))
        return pform

    def _print_primenu(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['nu']))
        else:
            pform = prettyForm(*pform.left('nu'))
        return pform

    def _print_primeomega(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['Omega']))
        else:
            pform = prettyForm(*pform.left('Omega'))
        return pform

    def _print_Quantity(self, e):
        if e.name.name == 'degree':
            if self._use_unicode:
                pform = self._print(pretty_atom('Degree'))
            else:
                pform = self._print(chr(176))
            return pform
        else:
            return self.emptyPrinter(e)

    def _print_AssignmentBase(self, e):

        op = prettyForm(' ' + xsym(e.op) + ' ')

        l = self._print(e.lhs)
        r = self._print(e.rhs)
        pform = prettyForm(*stringPict.next(l, op, r))
