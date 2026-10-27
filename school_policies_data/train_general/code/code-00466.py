        if exp is not None:
            return r"\gamma%s^{%s}" % (tex, exp)
        return r"\gamma%s" % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        if exp is None:
            return r"\Phi%s" % tex
        return r"\Phi^{%s}%s" % (exp, tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, exp, tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def __print_mathieu_functions(self, character, args, prime=False, exp=None):
        a, q, z = map(self._print, args)
        sup = r"^{\prime}" if prime else ""
        exp = "" if not exp else "^{%s}" % exp
        return r"%s%s\left(%s, %s, %s\right)%s" % (character, sup, a, q, z, exp)

    def _print_mathieuc(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, exp=exp)

    def _print_mathieus(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, exp=exp)

    def _print_mathieucprime(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, prime=True, exp=exp)

    def _print_mathieusprime(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, prime=True, exp=exp)

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            if self._settings['fold_short_frac']:
                return r"%s%d / %d" % (sign, p, expr.q)
            return r"%s\frac{%d}{%d}" % (sign, p, expr.q)
        else:
            return self._print(expr.p)

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif expr.variables:
                s += self._print(expr.variables[0])
            s += r'\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return r"O\left(%s\right)" % s

    def _print_Symbol(self, expr: Symbol, style='plain'):
        name: str = self._settings['symbol_names'].get(expr)
        if name is not None:
            return name

        return self._deal_with_super_sub(expr.name, style=style)

    _print_RandomSymbol = _print_Symbol

    def _split_super_sub(self, name: str) -> tuple[str, list[str], list[str]]:
        if name is None or '{' in name:
            return (name, [], [])
        elif self._settings["disable_split_super_sub"]:
            name, supers, subs = (name.replace('_', '\\_').replace('^', '\\^'), [], [])
        else:
            name, supers, subs = split_super_sub(name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]
        return (name, supers, subs)

    def _deal_with_super_sub(self, string: str, style='plain') -> str:
        name, supers, subs = self._split_super_sub(string)

        # apply the style only to the name
        if style == 'bold':
            name = "\\mathbf{{{}}}".format(name)

        # glue all items together:
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    def _print_Relational(self, expr):
        if self._settings['itex']:
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
        }

        return "%s %s %s" % (self._print(expr.lhs),
                             charmap[expr.rel_op], self._print(expr.rhs))

    def _print_Piecewise(self, expr):
        ecpairs = [r"%s & \text{for}\: %s" % (self._print(e), self._print(c))
                   for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            ecpairs.append(r"%s & \text{otherwise}" %
                           self._print(expr.args[-1].expr))
        else:
            ecpairs.append(r"%s & \text{for}\: %s" %
                           (self._print(expr.args[-1].expr),
                            self._print(expr.args[-1].cond)))
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)

    def _print_matrix_contents(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" & ".join([self._print(i) for i in expr[line, :]]))

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.cols <= 10) is True:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        return out_str % r"\\".join(lines)

    def _print_MatrixBase(self, expr):
        out_str = self._print_matrix_contents(expr)
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        return out_str

    def _print_MatrixElement(self, expr):
        matrix_part = self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True)
        index_part = f"{self._print(expr.i)},{self._print(expr.j)}"
        return f"{{{matrix_part}}}_{{{index_part}}}"

    def _print_MatrixSlice(self, expr):
        def latexslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = None
            if x[1] == dim:
                x[1] = None
            return ':'.join(self._print(xi) if xi is not None else '' for xi in x)
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + r'\left[' +
                latexslice(expr.rowslice, expr.parent.rows) + ', ' +
                latexslice(expr.colslice, expr.parent.cols) + r'\right]')

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            return r"\left(%s\right)^{T}" % self._print(mat)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{T}" % s
            else:
                return "%s^{T}" % s

    def _print_Trace(self, expr):
        mat = expr.arg
        return r"\operatorname{tr}\left(%s \right)" % self._print(mat)

    def _print_Adjoint(self, expr):
        style_to_latex = {
            "dagger"   : r"\dagger",
            "star"     : r"\ast",
            "hermitian": r"\mathsf{H}"
        }
        adjoint_style = style_to_latex.get(self._settings["adjoint_style"], r"\dagger")
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            return r"\left(%s\right)^{%s}" % (self._print(mat), adjoint_style)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{%s}" % (s, adjoint_style)
            else:
                return r"%s^{%s}" % (s, adjoint_style)

    def _print_MatMul(self, expr):
        from sympy import MatMul

        # Parenthesize nested MatMul but not other types of Mul objects:
        parens = lambda x: self._print(x) if isinstance(x, Mul) and not isinstance(x, MatMul) else \
            self.parenthesize(x, precedence_traditional(expr), False)

        args = list(expr.args)
        if expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            return '- ' + ' '.join(map(parens, args))
        else:
            return ' '.join(map(parens, args))

    def _print_DotProduct(self, expr):
        level = precedence_traditional(expr)
        left, right = expr.args
        return rf"{self.parenthesize(left, level)} \cdot {self.parenthesize(right, level)}"

    def _print_Determinant(self, expr):
        mat = expr.arg
        if mat.is_MatrixExpr:
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            if isinstance(mat, BlockMatrix):
                return r"\left|{%s}\right|" % self._print_matrix_contents(mat.blocks)
            return r"\left|{%s}\right|" % self._print(mat)
        return r"\left|{%s}\right|" % self._print_matrix_contents(mat)


    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s \bmod %s\right)^{%s}' % \
                (self.parenthesize(expr.args[0], PRECEDENCE['Mul'],
                                   strict=True),
                 self.parenthesize(expr.args[1], PRECEDENCE['Mul'],
                                   strict=True),
                 exp)
        return r'%s \bmod %s' % (self.parenthesize(expr.args[0],
                                                   PRECEDENCE['Mul'],
                                                   strict=True),
                                 self.parenthesize(expr.args[1],
                                                   PRECEDENCE['Mul'],
                                                   strict=True))

    def _print_HadamardProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \circ '.join(
            (parens(arg, prec, strict=True) for arg in args))

    def _print_HadamardPower(self, expr):
        if precedence_traditional(expr.exp) < PRECEDENCE["Mul"]:
            template = r"%s^{\circ \left({%s}\right)}"
        else:
            template = r"%s^{\circ {%s}}"
        return self._helper_print_standard_power(expr, template)

    def _print_KroneckerProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \otimes '.join(
            (parens(arg, prec, strict=True) for arg in args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol) and base.is_MatrixExpr:
            return "\\left(%s\\right)^{%s}" % (self._print(base),
                                              self._print(exp))
        else:
            base_str = self._print(base)
            if '^' in base_str:
                return r"\left(%s\right)^{%s}" % (base_str, self._print(exp))
            else:
                return "%s^{%s}" % (base_str, self._print(exp))

    def _print_MatrixSymbol(self, expr):
        return self._print_Symbol(expr, style=self._settings[
            'mat_symbol_style'])

    def _print_ZeroMatrix(self, Z):
        return "0" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{0}"

    def _print_OneMatrix(self, O):
        return "1" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{1}"

    def _print_Identity(self, I):
        return r"\mathbb{I}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{I}"

    def _print_PermutationMatrix(self, P):
        perm_str = self._print(P.args[0])
        return "P_{%s}" % perm_str

    def _print_NDimArray(self, expr: NDimArray):

        if expr.rank() == 0:
            return self._print(expr[()])

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            block_str = block_str.replace('%s', '{' + 'c'*expr.shape[0] + '}%s')

        if self._settings['mat_delim']:
            left_delim: str = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                        r'\right' + right_delim

        if expr.rank() == 0:
            return block_str % ""

        level_str: list[list[str]] = [[] for i in range(expr.rank() + 1)]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(
                        r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(
                        block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + \
                            level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str

    def _printer_tensor_indices(self, name, indices, index_map: dict):
        out_str = self._print(name)
        last_valence = None
        prev_map = None
        for index in indices:
            new_valence = index.is_up
            if ((index in index_map) or prev_map) and \
                    last_valence == new_valence:
                out_str += ","
            if last_valence != new_valence:
                if last_valence is not None:
                    out_str += "}"
                if index.is_up:
                    out_str += "{}^{"
                else:
                    out_str += "{}_{"
            out_str += self._print(index.args[0])
            if index in index_map:
                out_str += "="
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            last_valence = new_valence
        if last_valence is not None:
            out_str += "}"
        return out_str

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices, {})

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )

    def _print_PartialDerivative(self, expr):
        if len(expr.variables) == 1:
            return r"\frac{\partial}{\partial {%s}}{%s}" % (
                self._print(expr.variables[0]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )
        else:
            return r"\frac{\partial^{%s}}{%s}{%s}" % (
                len(expr.variables),
                " ".join([r"\partial {%s}" % self._print(i) for i in expr.variables]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return "{{%s}_{%s}}" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True),
            ", ".join([f"{self._print(i)}" for i in expr.indices]))

    def _print_UniversalSet(self, expr):
        return r"\mathbb{U}"

    def _print_frac(self, expr, exp=None):
        if exp is None:
            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
        else:
            return r"\operatorname{frac}{\left(%s\right)}^{%s}" % (
                    self._print(expr.args[0]), exp)

    def _print_tuple(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            sep = ";"
        elif self._settings['decimal_separator'] == 'period':
            sep = ","
        else:
            raise ValueError('Unknown Decimal Separator')

        if len(expr) == 1:
            # 1-tuple needs a trailing separator
            return self._add_parens_lspace(self._print(expr[0]) + sep)
        else:
            return self._add_parens_lspace(
                (sep + r" \  ").join([self._print(i) for i in expr]))

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            return r"\left[ %s\right]" % \
                r"; \  ".join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] == 'period':
            return r"\left[ %s\right]" % \
                r", \  ".join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')


    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left\{ %s\right\}" % r", \  ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_SingularityFunction(self, expr, exp=None):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\left\langle %s \right\rangle}^{%s}" % (shift, power)
        if exp is not None:
            tex = r"{\left({\langle %s \rangle}^{%s}\right)}^{%s}" % (shift, power, exp)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        pargs = ', '.join(self._print(arg) for arg in expr.pargs)
        tex = r"\theta\left(%s\right)" % pargs
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        if exp is not None:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return '\\text{Domain: }' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('\\text{Domain: }' + self._print(d.symbols) + ' \\in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return '\\text{Domain on }' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        if self._settings['decimal_separator'] == 'comma':
            items = "; ".join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ", ".join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        return r"\left\{%s\right\}" % items


    _print_frozenset = _print_set

    def _print_Range(self, s):
        def _print_symbolic_range():
            # Symbolic Range that cannot be resolved
            if s.args[0] == 0:
                if s.args[2] == 1:
                    cont = self._print(s.args[1])
                else:
                    cont = ", ".join(self._print(arg) for arg in s.args)
            else:
                if s.args[2] == 1:
                    cont = ", ".join(self._print(arg) for arg in s.args[:2])
                else:
                    cont = ", ".join(self._print(arg) for arg in s.args)

            return(f"\\text{{Range}}\\left({cont}\\right)")

        dots = object()

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
        elif s.is_empty is not None:
            if (s.size < 4) == True:
                printset = tuple(s)
            elif s.is_iterable:
                it = iter(s)
                printset = next(it), next(it), dots, s[-1]
            else:
                return _print_symbolic_range()
        else:
            return _print_symbolic_range()
        return (r"\left\{" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right\}")

    def __print_number_polynomial(self, expr, letter, exp=None):
        if len(expr.args) == 2:
            if exp is not None:
                return r"%s_{%s}^{%s}\left(%s\right)" % (letter,
                            self._print(expr.args[0]), exp,
                            self._print(expr.args[1]))
            return r"%s_{%s}\left(%s\right)" % (letter,
                        self._print(expr.args[0]), self._print(expr.args[1]))

        tex = r"%s_{%s}" % (letter, self._print(expr.args[0]))
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_bernoulli(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_genocchi(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "G", exp)

    def _print_bell(self, expr, exp=None):
        if len(expr.args) == 3:
            tex1 = r"B_{%s, %s}" % (self._print(expr.args[0]),
                                self._print(expr.args[1]))
            tex2 = r"\left(%s\right)" % r", ".join(self._print(el) for
                                               el in expr.args[2])
            if exp is not None:
                tex = r"%s^{%s}%s" % (tex1, exp, tex2)
            else:
                tex = tex1 + tex2
            return tex
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_fibonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "F", exp)

    def _print_lucas(self, expr, exp=None):
        tex = r"L_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_tribonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "T", exp)

    def _print_mobius(self, expr, exp=None):
        if exp is None:
            return r'\mu\left(%s\right)' % self._print(expr.args[0])
        return r'\mu^{%s}\left(%s\right)' % (exp, self._print(expr.args[0]))

    def _print_SeqFormula(self, s):
        dots = object()
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            return r"\left\{%s\right\}_{%s=%s}^{%s}" % (
                self._print(s.formula),
                self._print(s.variables[0]),
                self._print(s.start),
                self._print(s.stop)
            )
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                        s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
        else:
            printset = tuple(s)

        return (r"\left[" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right]")

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return r"\left%s%s, %s\right%s" % \
                   (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        return r"\left\langle %s, %s\right\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cup ".join(args_str)

    def _print_Complement(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \setminus ".join(args_str)

    def _print_Intersection(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cap ".join(args_str)

    def _print_SymmetricDifference(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \triangle ".join(args_str)

    def _print_ProductSet(self, p):
        prec = precedence_traditional(p)
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self.parenthesize(p.sets[0], prec) + "^{%d}" % len(p.sets)
        return r" \times ".join(
            self.parenthesize(set, prec) for set in p.sets)

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Rationals(self, i):
        return r"\mathbb{Q}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"

    def _print_ImageSet(self, s):
        expr = s.lamda.expr
        sig = s.lamda.signature
        xys = ((self._print(x), self._print(y)) for x, y in zip(sig, s.base_sets))
        xinys = r", ".join(r"%s \in %s" % xy for xy in xys)
        return r"\left\{%s\; \middle|\; %s\right\}" % (self._print(expr), xinys)

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        if s.base_set is S.UniversalSet:
            return r"\left\{%s\; \middle|\; %s \right\}" % \
                (vars_print, self._print(s.condition))

        return r"\left\{%s\; \middle|\; %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition))

    def _print_PowerSet(self, expr):
        arg_print = self._print(expr.args[0])
        return r"\mathcal{{P}}\left({}\right)".format(arg_print)

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return r"\left\{%s\; \middle|\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        return self._print_Add(s.truncate()) + r' + \ldots'

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left[%s\right]" % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left(%s\right)" % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        inv = ""
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = r"\left(%s\right)" % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = "domain=%s" % self._print(poly.get_domain())

        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = r"\%s {\left(%s \right)}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        return tex

    def _print_ComplexRootOf(self, root):
        cls = root.__class__.__name__
        if cls == "ComplexRootOf":
            cls = "CRootOf"
        expr = self._print(root.expr)
        index = root.index
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s, %d\right)}" % (cls, expr, index)
        else:
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr,
                                                                 index)

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls,
                                                             ", ".join(args))

    def _print_OrdinalOmega(self, expr):
        return r"\omega"

    def _print_OmegaPower(self, expr):
        exp, mul = expr.args
        if mul != 1:
            if exp != 1:
                return r"{} \omega^{{{}}}".format(mul, exp)
            else:
                return r"{} \omega".format(mul)
        else:
            if exp != 1:
                return r"\omega^{{{}}}".format(exp)
            else:
                return r"\omega"

    def _print_Ordinal(self, expr):
        return " + ".join([self._print(arg) for arg in expr.args])

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_UnifiedTransform(self, expr, s, inverse=False):
        return r"\mathcal{{{}}}{}_{{{}}}\left[{}\right]\left({}\right)".format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')

    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

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
        return self._print(Symbol(object.name))

    def _print_LambertW(self, expr, exp=None):
        arg0 = self._print(expr.args[0])
        exp = r"^{%s}" % (exp,) if exp is not None else ""
        if len(expr.args) == 1:
            result = r"W%s\left(%s\right)" % (exp, arg0)
        else:
            arg1 = self._print(expr.args[1])
            result = "W{0}_{{{1}}}\\left({2}\\right)".format(exp, arg1, arg0)
        return result

    def _print_Expectation(self, expr):
        return r"\operatorname{{E}}\left[{}\right]".format(self._print(expr.args[0]))

    def _print_Variance(self, expr):
        return r"\operatorname{{Var}}\left({}\right)".format(self._print(expr.args[0]))

    def _print_Covariance(self, expr):
        return r"\operatorname{{Cov}}\left({}\right)".format(", ".join(self._print(arg) for arg in expr.args))

    def _print_Probability(self, expr):
        return r"\operatorname{{P}}\left({}\right)".format(self._print(expr.args[0]))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    def _print_TransferFunction(self, expr):
        num, den = self._print(expr.num), self._print(expr.den)
        return r"\frac{%s}{%s}" % (num, den)

    def _print_Series(self, expr):
        args = list(expr.args)
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                            False)
        return ' '.join(map(parens, args))

    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)[::-1]
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                             False) if isinstance(x, MIMOParallel) else self._print(x)
        return r"\cdot".join(map(parens, args))

    def _print_Parallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_MIMOParallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_Feedback(self, expr):
        from sympy.physics.control import TransferFunction, Series

        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        den_arg_list = list(expr.sys2.args) if \
            isinstance(expr.sys2, Series) else [expr.sys2]
        den_term_1 = tf

        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den_term_2 = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = tf, Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            else:
                den_term_2 = Series(num, *den_arg_list)
        else:
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = Series(*num_arg_list, *den_arg_list)

        numer = self._print(num)
        denom_1 = self._print(den_term_1)
        denom_2 = self._print(den_term_2)
        _sign = "+" if expr.sign == -1 else "-"

        return r"\frac{%s}{%s %s %s}" % (numer, denom_1, _sign, denom_2)

    def _print_MIMOFeedback(self, expr):
        from sympy.physics.control import MIMOSeries
        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        sys1 = self._print(expr.sys1)
        _sign = "+" if expr.sign == -1 else "-"
        return r"\left(I_{\tau} %s %s\right)^{-1} \cdot %s" % (_sign, inv_mat, sys1)

    def _print_TransferFunctionMatrix(self, expr):
        mat = self._print(expr._expr_mat)
        return r"%s_\tau" % mat

    def _print_DFT(self, expr):
        return r"\text{{{}}}_{{{}}}".format(expr.__class__.__name__, expr.n)
    _print_IDFT = _print_DFT

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return "%s:%s" % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):
        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return r"\mathbf{{{}}}".format(self._print(Symbol(morphism.name)))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        return latex_result

    def _print_DiagramGrid(self, grid):
        latex_result = "\\begin{array}{%s}\n" % ("c" * grid.width)

        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    latex_result += latex(grid[i, j])
                latex_result += " "
                if j != grid.width - 1:
                    latex_result += "& "

            if i != grid.height - 1:
                latex_result += "\\\\"
            latex_result += "\n"

        latex_result += "\\end{array}\n"
        return latex_result

    def _print_FreeModule(self, M):
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in m))

    def _print_SubModule(self, m):
        gens = [[self._print(m.ring.to_sympy(x)) for x in g] for g in m.gens]
        curly = lambda o: r"{" + o + r"}"
        square = lambda o: r"\left[ " + o + r" \right]"
        gens_latex = ",".join(curly(square(",".join(curly(x) for x in g))) for g in gens)
        return r"\left\langle {} \right\rangle".format(gens_latex)

    def _print_SubQuotientModule(self, m):
        gens_latex = ",".join(["{" + self._print(g) + "}" for g in m.gens])
        return r"\left\langle {} \right\rangle".format(gens_latex)

    def _print_ModuleImplementedIdeal(self, m):
        gens = [m.ring.to_sympy(x) for [x] in m._module.gens]
        gens_latex = ",".join('{' + self._print(x) + '}' for x in gens)
        return r"\left\langle {} \right\rangle".format(gens_latex)

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True)
             for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(R.ring),
                 self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        x_latex = self._print(x.ring.to_sympy(x))
        return r"{{{}}} + {{{}}}".format(x_latex,
                 self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        data = [m.module.ring.to_sympy(x) for x in m.data]
        data_latex = r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in data))
        return r"{{{}}} + {{{}}}".format(data_latex,
                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(M.base),
                 self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{{{}}} : {{{}}} \to {{{}}}".format(self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_Manifold(self, manifold):
        name, supers, subs = self._split_super_sub(manifold.name.name)

        name = r'\text{%s}' % name
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    def _print_Patch(self, patch):
        return r'\text{%s}_{%s}' % (self._print(patch.name), self._print(patch.manifold))

    def _print_CoordSystem(self, coordsys):
        return r'\text{%s}^{\text{%s}}_{%s}' % (
            self._print(coordsys.name), self._print(coordsys.patch.name), self._print(coordsys.manifold)
        )

    def _print_CovarDerivativeOp(self, cvd):
        return r'\mathbb{\nabla}_{%s}' % self._print(cvd._wrt)

    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return r'\mathbf{{{}}}'.format(self._print(Symbol(string)))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return r'\partial_{{{}}}'.format(self._print(Symbol(string)))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return r'\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            string = self._print(field)
            return r'\operatorname{{d}}\left({}\right)'.format(string)

    def _print_Tr(self, p):
        # TODO: Handle indices
        contents = self._print(p.args[0])
        return r'\operatorname{{tr}}\left({}\right)'.format(contents)

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^{%s}%s" % (exp, tex)
        return r"\sigma%s" % tex

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (exp, tex)
        return r"\sigma^*%s" % tex

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])

    def _print_Str(self, s):
        return str(s.name)

    def _print_float(self, expr):
        return self._print(Float(expr))

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_mpq(self, expr):
        return str(expr)

    def _print_fmpz(self, expr):
        return str(expr)

    def _print_fmpq(self, expr):
        return str(expr)

    def _print_Predicate(self, expr):
        return r"\operatorname{{Q}}_{{\text{{{}}}}}".format(latex_escape(str(expr.name)))

    def _print_AppliedPredicate(self, expr):
        pred = expr.function
        args = expr.arguments
        pred_latex = self._print(pred)
        args_latex = ', '.join([self._print(a) for a in args])
        return '%s(%s)' % (pred_latex, args_latex)

    def emptyPrinter(self, expr):
        # default to just printing as monospace, like would normally be shown
        s = super().emptyPrinter(expr)

        return r"\mathtt{\text{%s}}" % latex_escape(s)


def translate(s: str) -> str:
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # Process the rest
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    elif s in other_symbols:
        return "\\" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=len, reverse=True):
            if s.lower().endswith(key) and len(s) > len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s



@print_function(LatexPrinter)
def latex(expr, **settings):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    full_prec: boolean, optional
        If set to True, a floating point number is printed with full precision.
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``'abbreviated'``, ``'full'``, or ``'power'``. Defaults to
        ``'abbreviated'``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``'['``, ``'('``,
        or the empty string ``''``. Defaults to ``'['``.
    mat_str : string, optional
        Which matrix environment string to emit. ``'smallmatrix'``,
        ``'matrix'``, ``'array'``, etc. Defaults to ``'smallmatrix'`` for
        inline mode, ``'matrix'`` for matrices of no more than 10 columns, and
        ``'array'`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``'plain'``, ``'inline'``, ``'equation'`` or ``'equation*'``.  If
        ``mode`` is set to ``'plain'``, then the resulting code will not be
        delimited at all (this is the default). If ``mode`` is set to
        ``'inline'`` then inline LaTeX ``$...$`` will be used. If ``mode`` is
        set to ``'equation'`` or ``'equation*'``, the resulting code will be
        enclosed in the ``equation`` or ``equation*`` environment (remember to
        import ``amsmath`` for ``equation*``), unless the ``itex`` option is
        set. In the latter case, the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``,
        ``'ldot'``, ``'dot'``, or ``'times'``.
    order: string, optional
        Any of the supported monomial orderings (currently ``'lex'``,
        ``'grlex'``, or ``'grevlex'``), ``'old'``, and ``'none'``. This
        parameter does nothing for `~.Mul` objects. Setting order to ``'old'``
        uses the compatibility ordering for ``~.Add`` defined in Printer. For
        very large expressions, set the ``order`` keyword to ``'none'`` if
        speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.
    mat_symbol_style : string, optional
        Can be either ``'plain'`` (default) or ``'bold'``. If set to
        ``'bold'``, a `~.MatrixSymbol` A will be printed as ``\mathbf{A}``,
        otherwise as ``A``.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are ``'i'``
        (default) and ``'j'``. Adding ``r`` or ``t`` in front gives ``\mathrm``
        or ``\text``, so ``'ri'`` leads to ``\mathrm{i}`` which gives
        `\mathrm{i}`.
    gothic_re_im : boolean, optional
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
        The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
    decimal_separator : string, optional
        Specifies what separator to use to separate the whole and fractional parts of a
        floating point number as in `2.5` for the default, ``period`` or `2{,}5`
        when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
        separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
        ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
    parenthesize_super : boolean, optional
        If set to ``False``, superscripted expressions will not be parenthesized when
        powered. Default is ``True``, which parenthesizes the expression when powered.
    min: Integer or None, optional
        Sets the lower bound for the exponent to print floating point numbers in
        fixed-point format.
    max: Integer or None, optional
        Sets the upper bound for the exponent to print floating point numbers in
        fixed-point format.
    diff_operator: string, optional
        String to use for differential operator. Default is ``'d'``, to print in italic
        form. ``'rd'``, ``'td'`` are shortcuts for ``\mathrm{d}`` and ``\text{d}``.
    adjoint_style: string, optional
        String to use for the adjoint symbol. Defined options are ``'dagger'``
        (default),``'star'``, and ``'hermitian'``.

    Notes
    =====

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> from sympy import latex, Rational
    >>> from sympy.abc import tau
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Examples
    ========

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    Basic usage:

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    ``mode`` and ``itex`` options:

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    Fraction options:

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left(\frac{7}{2} \right)}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}
    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    Multiplication options:

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left(\frac{7}{2} \right)}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left(\frac{7}{2} \right)}

    Matrix options:

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    Custom printing of symbols:

    >>> print(latex(x**2, symbol_names={x: 'x_i'}))
    x_i^{2}

    Logarithms:

    >>> print(latex(log(10)))
    \log{\left(10 \right)}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left(10 \right)}

    ``latex()`` also supports the builtin container types :class:`list`,
    :class:`tuple`, and :class:`dict`:

    >>> print(latex([2/x, y], mode='inline'))
    $\left[ 2 / x, \  y\right]$

    Unsupported types are rendered as monospaced plaintext:

    >>> print(latex(int))
    \mathtt{\text{<class 'int'>}}
    >>> print(latex("plain % text"))
    \mathtt{\text{plain \% text}}

    See :ref:`printer_method_example` for an example of how to override
    this behavior for your own types by implementing ``_latex``.

    .. versionchanged:: 1.7.0
        Unsupported types no longer have their ``str`` representation treated as valid latex.

    """
    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""

    print(latex(expr, **settings))


def multiline_latex(lhs, rhs, terms_per_line=1, environment="align*", use_dots=False, **settings):
    r"""
    This function generates a LaTeX equation with a multiline right-hand side
    in an ``align*``, ``eqnarray`` or ``IEEEeqnarray`` environment.

    Parameters
    ==========

    lhs : Expr
        Left-hand side of equation

    rhs : Expr
        Right-hand side of equation

    terms_per_line : integer, optional
        Number of terms per line to print. Default is 1.

    environment : "string", optional
        Which LaTeX wnvironment to use for the output. Options are "align*"
        (default), "eqnarray", and "IEEEeqnarray".

    use_dots : boolean, optional
        If ``True``, ``\\dots`` is added to the end of each line. Default is ``False``.

    Examples
    ========

    >>> from sympy import multiline_latex, symbols, sin, cos, exp, log, I
    >>> x, y, alpha = symbols('x y alpha')
    >>> expr = sin(alpha*y) + exp(I*alpha) - cos(log(y))
    >>> print(multiline_latex(x, expr))
    \begin{align*}
    x = & e^{i \alpha} \\
    & + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using at most two terms per line:
    >>> print(multiline_latex(x, expr, 2))
    \begin{align*}
    x = & e^{i \alpha} + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using ``eqnarray`` and dots:
    >>> print(multiline_latex(x, expr, terms_per_line=2, environment="eqnarray", use_dots=True))
    \begin{eqnarray}
    x & = & e^{i \alpha} + \sin{\left(\alpha y \right)} \dots\nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{eqnarray}

    Using ``IEEEeqnarray``:
    >>> print(multiline_latex(x, expr, environment="IEEEeqnarray"))
    \begin{IEEEeqnarray}{rCl}
    x & = & e^{i \alpha} \nonumber\\
    & & + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{IEEEeqnarray}

    Notes
    =====

    All optional parameters from ``latex`` can also be used.

    """

    # Based on code from https://github.com/sympy/sympy/issues/3001
    l = LatexPrinter(**settings)
    if environment == "eqnarray":
        result = r'\begin{eqnarray}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{eqnarray}'
        doubleet = True
    elif environment == "IEEEeqnarray":
        result = r'\begin{IEEEeqnarray}{rCl}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{IEEEeqnarray}'
        doubleet = True
    elif environment == "align*":
        result = r'\begin{align*}' + '\n'
        first_term = '= &'
        nonumber = ''
        end_term =  '\n\\end{align*}'
        doubleet = False
    else:
        raise ValueError("Unknown environment: {}".format(environment))
    dots = ''
    if use_dots:
        dots=r'\dots'
    terms = rhs.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = ''
        term_end = ''
        sign = '+'
        if term_count > terms_per_line:
            if doubleet:
                term_start = '& & '
            else:
                term_start = '& '
            term_count = 1
        if term_count == terms_per_line:
            # End of line
            if i < n_terms-1:
                # There are terms remaining
                term_end = dots + nonumber + r'\\' + '\n'
            else:
                term_end = ''

        if term.as_ordered_factors()[0] == -1:
            term = -1*term
            sign = r'-'
        if i == 0: # beginning
            if sign == '+':
                sign = ''
            result += r'{:s} {:s}{:s} {:s} {:s}'.format(l.doprint(lhs),
                        first_term, sign, l.doprint(term), term_end)
        else:
            result += r'{:s}{:s} {:s} {:s}'.format(term_start, sign,
                        l.doprint(term), term_end)
        term_count += 1
    result += end_term
    return result
<|endoftext|>"""
Octave (and Matlab) code printer

The `OctaveCodePrinter` converts SymPy expressions into Octave expressions.
It uses a subset of the Octave language for Matlab compatibility.

A complete code generator, which uses `octave_code` extensively, can be found
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
# SymPy and Octave.   This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "acot", "atan", "atan2", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "csch", "sech",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "erfc", "erfi", "erf", "erfinv", "erfcinv",
                   "besseli", "besselj", "besselk", "bessely",
                   "bernoulli", "beta", "euler", "exp", "factorial", "floor",
                   "fresnelc", "fresnels", "gamma", "harmonic", "log",
                   "polylog", "sign", "zeta", "legendre"]

# These functions have different names ("SymPy": "Octave"), more
# generally a mapping to (argument_conditions, octave_function).
known_fcns_src2 = {
    "Abs": "abs",
    "arg": "angle",  # arg/angle ok in Octave but only angle in Matlab
    "binomial": "bincoeff",
    "ceiling": "ceil",
    "chebyshevu": "chebyshevU",
    "chebyshevt": "chebyshevT",
    "Chi": "coshint",
    "Ci": "cosint",
    "conjugate": "conj",
    "DiracDelta": "dirac",
    "Heaviside": "heaviside",
    "im": "imag",
    "laguerre": "laguerreL",
    "LambertW": "lambertw",
    "li": "logint",
    "loggamma": "gammaln",
    "Max": "max",
    "Min": "min",
    "Mod": "mod",
    "polygamma": "psi",
    "re": "real",
    "RisingFactorial": "pochhammer",
    "Shi": "sinhint",
    "Si": "sinint",
}


class OctaveCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Octave/Matlab code.
    """
    printmethod = "_octave"
    language = "Octave"

    _operators = {
        'and': '&',
        'or': '|',
        'not': '~',
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'inline': True,
    })
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Octave.


    def __init__(self, settings={}):
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s;" % codestring


    def _get_comment(self, text):
        return "% {}".format(text)


    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Octave uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Octave arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines


    def _print_Mul(self, expr):
        # print complex numbers nicely in Octave
        if (expr.is_number and expr.is_imaginary and
                (S.ImaginaryUnit*expr).is_Integer):
            return "%si" % self._print(-S.ImaginaryUnit*expr)

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
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
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
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))

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
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_MatrixSolve(self, expr):
        PREC = precedence(expr)
        return "%s \\ %s" % (self.parenthesize(expr.matrix, PREC),
                             self.parenthesize(expr.vector, PREC))

    def _print_Pi(self, expr):
        return 'pi'


    def _print_ImaginaryUnit(self, expr):
        return "1i"


    def _print_Exp1(self, expr):
        return "exp(1)"


    def _print_GoldenRatio(self, expr):
        # FIXME: how to do better, e.g., for octave_code(2*GoldenRatio)?
        #return self._print((1+sqrt(S(5)))/2)
        return "(1+sqrt(5))/2"


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
        return 'inf'


    def _print_NegativeInfinity(self, expr):
        return '-inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return '{' + ', '.join(self._print(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list
    _print_List = _print_list


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
        if (A.rows, A.cols) == (0, 0):
            return '[]'
        elif S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            # Octave does not distinguish between scalars and 1x1 matrices
            return self._print(A[0, 0])
        return "[%s]" % "; ".join(" ".join([self._print(a) for a in A[r, :]])
                                  for r in range(A.rows))


    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()
        # make row vectors of the indices and entries
        I = Matrix([[k[0] + 1 for k in L]])
        J = Matrix([[k[1] + 1 for k in L]])
        AIJ = Matrix([[k[2] for k in L]])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '(%s, %s)' % (expr.i + 1, expr.j + 1)


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
        return (self._print(expr.parent) + '(' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ', ' +
                strslice(expr.colslice, expr.parent.shape[1]) + ')')


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


    def _print_KroneckerDelta(self, expr):
        prec = PRECEDENCE["Pow"]
        return "double(%s == %s)" % tuple(self.parenthesize(x, prec)
                                          for x in expr.args)

    def _print_HadamardProduct(self, expr):
        return '.*'.join([self.parenthesize(arg, precedence(expr))
                          for arg in expr.args])

    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([
            self.parenthesize(expr.base, PREC),
            self.parenthesize(expr.exp, PREC)
            ])

    def _print_Identity(self, expr):
        shape = expr.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            shape = [shape[0]]
        s = ", ".join(self._print(n) for n in shape)
        return "eye(" + s + ")"

    def _print_lowergamma(self, expr):
        # Octave implements regularized incomplete gamma function
        return "(gammainc({1}, {0}).*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))


    def _print_uppergamma(self, expr):
        return "(gammainc({1}, {0}, 'upper').*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))


    def _print_sinc(self, expr):
        #Note: Divide by pi because Octave implements normalized sinc function.
        return "sinc(%s)" % self._print(expr.args[0]/S.Pi)


    def _print_hankel1(self, expr):
        return "besselh(%s, 1, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    def _print_hankel2(self, expr):
        return "besselh(%s, 2, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    # Note: as of 2015, Octave doesn't have spherical Bessel functions
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


    def _print_airyai(self, expr):
        return "airy(0, %s)" % self._print(expr.args[0])


    def _print_airyaiprime(self, expr):
        return "airy(1, %s)" % self._print(expr.args[0])


    def _print_airybi(self, expr):
        return "airy(2, %s)" % self._print(expr.args[0])


    def _print_airybiprime(self, expr):
        return "airy(3, %s)" % self._print(expr.args[0])


    def _print_expint(self, expr):
        mu, x = expr.args
        if mu != 1:
            return self._print_not_supported(expr)
        return "expint(%s)" % self._print(x)


    def _one_or_two_reversed_args(self, expr):
        assert len(expr.args) <= 2
        return '{name}({args})'.format(
            name=self.known_functions[expr.__class__.__name__],
            args=", ".join([self._print(x) for x in reversed(expr.args)])
        )


    _print_DiracDelta = _print_LambertW = _one_or_two_reversed_args


    def _nested_binary_math_func(self, expr):
        return '{name}({arg1}, {arg2})'.format(
            name=self.known_functions[expr.__class__.__name__],
            arg1=self._print(expr.args[0]),
            arg2=self._print(expr.func(*expr.args[1:]))
            )

    _print_Max = _print_Min = _nested_binary_math_func


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
            ecpairs = ["({0}).*({1}) + (~({0})).*(".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = "%s" % self._print(expr.args[-1].expr)
            pw = " ...\n".join(ecpairs) + elast + ")"*len(ecpairs)
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


    def _print_zeta(self, expr):
        if len(expr.args) == 1:
            return "zeta(%s)" % self._print(expr.args[0])
        else:
            # Matlab two argument zeta is not equivalent to SymPy's
            return self._print_not_supported(expr)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "  "
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


def octave_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Octave (or Matlab) code.

    The string uses a subset of the Octave language for Matlab compatibility.

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

    >>> from sympy import octave_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> octave_code(sin(x).series(x).removeO())
    'x.^5/120 - x.^3/6 + x'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> octave_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau.^(7/2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its very common in Octave to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> octave_code(sin(pi*x*y), assign_to="s")
    's = sin(pi*x.*y);'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> octave_code(3*pi*A**3)
    '(3*pi)*A^3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> octave_code(x**2*y*A**3)
    '(x.^2.*y)*A^3'

    Matrices are supported using Octave inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 sin(x) ceil(x)];'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> octave_code(pw, assign_to=tau)
    'tau = ((x > 0).*(x + 1) + (~(x > 0)).*(x));'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 ((x > 0).*(x + 1) + (~(x > 0)).*(x)) sin(x)];'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Octave function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_octave_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> octave_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_octave_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

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
    >>> octave_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy(i) = (y(i + 1) - y(i))./(t(i + 1) - t(i));'
    """
    return OctaveCodePrinter(settings).doprint(expr, assign_to)


def print_octave_code(expr, **settings):
    """Prints the Octave (or Matlab) representation of the given expression.

    See `octave_code` for the meaning of the optional arguments.
    """
    print(octave_code(expr, **settings))
<|endoftext|>from sympy.printing.mathml import mathml
from sympy.utilities.mathml import c2p
import tempfile
import subprocess


def print_gtk(x, start_viewer=True):
    """Print to Gtkmathview, a gtk widget capable of rendering MathML.

    Needs libgtkmathview-bin"""
    with tempfile.NamedTemporaryFile('w') as file:
        file.write(c2p(mathml(x), simple=True))
        file.flush()

        if start_viewer:
            subprocess.check_call(('mathmlviewer', file.name))
<|endoftext|>
from sympy.printing.pycode import AbstractPythonCodePrinter, ArrayPrinter
from sympy.matrices.expressions import MatrixExpr
from sympy.core.mul import Mul
from sympy.printing.precedence import PRECEDENCE
from sympy.external import import_module
from sympy.codegen.cfunctions import Sqrt
from sympy import S
from sympy import Integer

import sympy

torch = import_module('torch')


class TorchPrinter(ArrayPrinter, AbstractPythonCodePrinter):

    printmethod = "_torchcode"

    mapping = {
        sympy.Abs: "torch.abs",
        sympy.sign: "torch.sign",

        # XXX May raise error for ints.
        sympy.ceiling: "torch.ceil",
        sympy.floor: "torch.floor",
        sympy.log: "torch.log",
        sympy.exp: "torch.exp",
        Sqrt: "torch.sqrt",
        sympy.cos: "torch.cos",
        sympy.acos: "torch.acos",
        sympy.sin: "torch.sin",
        sympy.asin: "torch.asin",
        sympy.tan: "torch.tan",
        sympy.atan: "torch.atan",
        sympy.atan2: "torch.atan2",
        # XXX Also may give NaN for complex results.
        sympy.cosh: "torch.cosh",
        sympy.acosh: "torch.acosh",
        sympy.sinh: "torch.sinh",
        sympy.asinh: "torch.asinh",
        sympy.tanh: "torch.tanh",
        sympy.atanh: "torch.atanh",
        sympy.Pow: "torch.pow",

        sympy.re: "torch.real",
        sympy.im: "torch.imag",
        sympy.arg: "torch.angle",

        # XXX May raise error for ints and complexes
        sympy.erf: "torch.erf",
        sympy.loggamma: "torch.lgamma",

        sympy.Eq: "torch.eq",
        sympy.Ne: "torch.ne",
        sympy.StrictGreaterThan: "torch.gt",
        sympy.StrictLessThan: "torch.lt",
        sympy.LessThan: "torch.le",
        sympy.GreaterThan: "torch.ge",

        sympy.And: "torch.logical_and",
        sympy.Or: "torch.logical_or",
        sympy.Not: "torch.logical_not",
        sympy.Max: "torch.max",
        sympy.Min: "torch.min",

        # Matrices
        sympy.MatAdd: "torch.add",
        sympy.HadamardProduct: "torch.mul",
        sympy.Trace: "torch.trace",

        # XXX May raise error for integer matrices.
        sympy.Determinant: "torch.det",
    }

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        torch_version=None,
        requires_grad=False,
        dtype="torch.float64",
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        version = self._settings['torch_version']
        self.requires_grad = self._settings['requires_grad']
        self.dtype = self._settings['dtype']
        if version is None and torch:
            version = torch.__version__
        self.torch_version = version

    def _print_Function(self, expr):

        op = self.mapping.get(type(expr), None)
        if op is None:
            return super()._print_Basic(expr)
        children = [self._print(arg) for arg in expr.args]
        if len(children) == 1:
            return "%s(%s)" % (
                self._module_format(op),
                children[0]
            )
        else:
            return self._expand_fold_binary_op(op, children)

    # mirrors the tensorflow version
    _print_Expr = _print_Function
    _print_Application = _print_Function
    _print_MatrixExpr = _print_Function
    _print_Relational = _print_Function
    _print_Not = _print_Function
    _print_And = _print_Function
    _print_Or = _print_Function
    _print_HadamardProduct = _print_Function
    _print_Trace = _print_Function
    _print_Determinant = _print_Function

    def _print_Inverse(self, expr):
        return '{}({})'.format(self._module_format("torch.linalg.inv"),
                               self._print(expr.args[0]))

    def _print_Transpose(self, expr):
        if expr.arg.is_Matrix and expr.arg.shape[0] == expr.arg.shape[1]:
            # For square matrices, we can use the .t() method
            return "{}({}).t()".format("torch.transpose", self._print(expr.arg))
        else:
            # For non-square matrices or more general cases
            # transpose first and second dimensions (typical matrix transpose)
            return "{}.permute({})".format(
                self._print(expr.arg),
                ", ".join([str(i) for i in range(len(expr.arg.shape))])[::-1]
            )

    def _print_PermuteDims(self, expr):
        return "%s.permute(%s)" % (
            self._print(expr.expr),
            ", ".join(str(i) for i in expr.permutation.array_form)
        )

    def _print_Derivative(self, expr):
        # this version handles multi-variable and mixed partial derivatives. The tensorflow version does not.
        variables = expr.variables
        expr_arg = expr.expr

        # Handle multi-variable or repeated derivatives
        if len(variables) > 1 or (
            len(variables) == 1 and not isinstance(variables[0], tuple) and variables.count(variables[0]) > 1):
            result = self._print(expr_arg)
            var_groups = {}

            # Group variables by base symbol
            for var in variables:
                if isinstance(var, tuple):
                    base_var, order = var
                    var_groups[base_var] = var_groups.get(base_var, 0) + order
                else:
                    var_groups[var] = var_groups.get(var, 0) + 1

            # Apply gradients in sequence
            for var, order in var_groups.items():
                for _ in range(order):
                    result = "torch.autograd.grad({}, {}, create_graph=True)[0]".format(result, self._print(var))
            return result

        # Handle single variable case
        if len(variables) == 1:
            variable = variables[0]
            if isinstance(variable, tuple) and len(variable) == 2:
                base_var, order = variable
                if not isinstance(order, Integer): raise NotImplementedError("Only integer orders are supported")
                result = self._print(expr_arg)
                for _ in range(order):
                    result = "torch.autograd.grad({}, {}, create_graph=True)[0]".format(result, self._print(base_var))
                return result
            return "torch.autograd.grad({}, {})[0]".format(self._print(expr_arg), self._print(variable))

        return self._print(expr_arg)  # Empty variables case

    def _print_Piecewise(self, expr):
        from sympy import Piecewise
        e, cond = expr.args[0].args
        if len(expr.args) == 1:
            return '{}({}, {}, {})'.format(
                self._module_format("torch.where"),
                self._print(cond),
                self._print(e),
                0)

        return '{}({}, {}, {})'.format(
            self._module_format("torch.where"),
            self._print(cond),
            self._print(e),
            self._print(Piecewise(*expr.args[1:])))

    def _print_Pow(self, expr):
        # XXX May raise error for
        # int**float or int**complex or float**complex
        base, exp = expr.args
        if expr.exp == S.Half:
            return "{}({})".format(
                self._module_format("torch.sqrt"), self._print(base))
        return "{}({}, {})".format(
            self._module_format("torch.pow"),
            self._print(base), self._print(exp))

    def _print_MatMul(self, expr):
        # Separate matrix and scalar arguments
        mat_args = [arg for arg in expr.args if isinstance(arg, MatrixExpr)]
        args = [arg for arg in expr.args if arg not in mat_args]
        # Handle scalar multipliers if present
        if args:
            return "%s*%s" % (
                self.parenthesize(Mul.fromiter(args), PRECEDENCE["Mul"]),
                self._expand_fold_binary_op("torch.matmul", mat_args)
            )
        else:
            return self._expand_fold_binary_op("torch.matmul", mat_args)

    def _print_MatPow(self, expr):
        return self._expand_fold_binary_op("torch.mm", [expr.base]*expr.exp)

    def _print_MatrixBase(self, expr):
        data = "[" + ", ".join(["[" + ", ".join([self._print(j) for j in i]) + "]" for i in expr.tolist()]) + "]"
        params = [str(data)]
        params.append(f"dtype={self.dtype}")
        if self.requires_grad:
            params.append("requires_grad=True")

        return "{}({})".format(
            self._module_format("torch.tensor"),
            ", ".join(params)
        )

    def _print_isnan(self, expr):
        return f'torch.isnan({self._print(expr.args[0])})'

    def _print_isinf(self, expr):
        return f'torch.isinf({self._print(expr.args[0])})'

    def _print_Identity(self, expr):
        if all(dim.is_Integer for dim in expr.shape):
            return "{}({})".format(
                self._module_format("torch.eye"),
                self._print(expr.shape[0])
            )
        else:
            # For symbolic dimensions, fall back to a more general approach
            return "{}({}, {})".format(
                self._module_format("torch.eye"),
                self._print(expr.shape[0]),
                self._print(expr.shape[1])
            )

    def _print_ZeroMatrix(self, expr):
        return "{}({})".format(
            self._module_format("torch.zeros"),
            self._print(expr.shape)
        )

    def _print_OneMatrix(self, expr):
        return "{}({})".format(
            self._module_format("torch.ones"),
            self._print(expr.shape)
        )

    def _print_conjugate(self, expr):
        return f"{self._module_format('torch.conj')}({self._print(expr.args[0])})"

    def _print_ImaginaryUnit(self, expr):
        return "1j"  # uses the Python built-in 1j notation for the imaginary unit

    def _print_Heaviside(self, expr):
        args = [self._print(expr.args[0]), "0.5"]
        if len(expr.args) > 1:
            args[1] = self._print(expr.args[1])
        return f"{self._module_format('torch.heaviside')}({args[0]}, {args[1]})"

    def _print_gamma(self, expr):
        return f"{self._module_format('torch.special.gamma')}({self._print(expr.args[0])})"

    def _print_polygamma(self, expr):
        if expr.args[0] == S.Zero:
            return f"{self._module_format('torch.special.digamma')}({self._print(expr.args[1])})"
        else:
            raise NotImplementedError("PyTorch only supports digamma (0th order polygamma)")

    _module = "torch"
    _einsum = "einsum"
    _add = "add"
    _transpose = "t"
    _ones = "ones"
    _zeros = "zeros"

def torch_code(expr, requires_grad=False, dtype="torch.float64", **settings):
    printer = TorchPrinter(settings={'requires_grad': requires_grad, 'dtype': dtype})
    return printer.doprint(expr, **settings)
<|endoftext|>"""
Mathematica code printer
"""

from __future__ import annotations
from typing import Any

from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key

from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence

# Used in MCodePrinter._print_Function(self)
known_functions = {
    "exp": [(lambda x: True, "Exp")],
    "log": [(lambda x: True, "Log")],
    "sin": [(lambda x: True, "Sin")],
    "cos": [(lambda x: True, "Cos")],
    "tan": [(lambda x: True, "Tan")],
    "cot": [(lambda x: True, "Cot")],
    "sec": [(lambda x: True, "Sec")],
    "csc": [(lambda x: True, "Csc")],
    "asin": [(lambda x: True, "ArcSin")],
    "acos": [(lambda x: True, "ArcCos")],
    "atan": [(lambda x: True, "ArcTan")],
    "acot": [(lambda x: True, "ArcCot")],
    "asec": [(lambda x: True, "ArcSec")],
    "acsc": [(lambda x: True, "ArcCsc")],
    "sinh": [(lambda x: True, "Sinh")],
    "cosh": [(lambda x: True, "Cosh")],
    "tanh": [(lambda x: True, "Tanh")],
    "coth": [(lambda x: True, "Coth")],
    "sech": [(lambda x: True, "Sech")],
    "csch": [(lambda x: True, "Csch")],
    "asinh": [(lambda x: True, "ArcSinh")],
    "acosh": [(lambda x: True, "ArcCosh")],
    "atanh": [(lambda x: True, "ArcTanh")],
    "acoth": [(lambda x: True, "ArcCoth")],
    "asech": [(lambda x: True, "ArcSech")],
    "acsch": [(lambda x: True, "ArcCsch")],
    "sinc": [(lambda x: True, "Sinc")],
    "conjugate": [(lambda x: True, "Conjugate")],
    "Max": [(lambda *x: True, "Max")],
    "Min": [(lambda *x: True, "Min")],
    "erf": [(lambda x: True, "Erf")],
    "erf2": [(lambda *x: True, "Erf")],
    "erfc": [(lambda x: True, "Erfc")],
    "erfi": [(lambda x: True, "Erfi")],
    "erfinv": [(lambda x: True, "InverseErf")],
    "erfcinv": [(lambda x: True, "InverseErfc")],
    "erf2inv": [(lambda *x: True, "InverseErf")],
    "expint": [(lambda *x: True, "ExpIntegralE")],
    "Ei": [(lambda x: True, "ExpIntegralEi")],
    "fresnelc": [(lambda x: True, "FresnelC")],
    "fresnels": [(lambda x: True, "FresnelS")],
    "gamma": [(lambda x: True, "Gamma")],
    "uppergamma": [(lambda *x: True, "Gamma")],
    "polygamma": [(lambda *x: True, "PolyGamma")],
    "loggamma": [(lambda x: True, "LogGamma")],
    "beta": [(lambda *x: True, "Beta")],
    "Ci": [(lambda x: True, "CosIntegral")],
    "Si": [(lambda x: True, "SinIntegral")],
    "Chi": [(lambda x: True, "CoshIntegral")],
    "Shi": [(lambda x: True, "SinhIntegral")],
    "li": [(lambda x: True, "LogIntegral")],
    "factorial": [(lambda x: True, "Factorial")],
    "factorial2": [(lambda x: True, "Factorial2")],
    "subfactorial": [(lambda x: True, "Subfactorial")],
    "catalan": [(lambda x: True, "CatalanNumber")],
    "harmonic": [(lambda *x: True, "HarmonicNumber")],
    "lucas": [(lambda x: True, "LucasL")],
    "RisingFactorial": [(lambda *x: True, "Pochhammer")],
    "FallingFactorial": [(lambda *x: True, "FactorialPower")],
    "laguerre": [(lambda *x: True, "LaguerreL")],
    "assoc_laguerre": [(lambda *x: True, "LaguerreL")],
    "hermite": [(lambda *x: True, "HermiteH")],
    "jacobi": [(lambda *x: True, "JacobiP")],
    "gegenbauer": [(lambda *x: True, "GegenbauerC")],
    "chebyshevt": [(lambda *x: True, "ChebyshevT")],
    "chebyshevu": [(lambda *x: True, "ChebyshevU")],
    "legendre": [(lambda *x: True, "LegendreP")],
    "assoc_legendre": [(lambda *x: True, "LegendreP")],
    "mathieuc": [(lambda *x: True, "MathieuC")],
    "mathieus": [(lambda *x: True, "MathieuS")],
    "mathieucprime": [(lambda *x: True, "MathieuCPrime")],
    "mathieusprime": [(lambda *x: True, "MathieuSPrime")],
    "stieltjes": [(lambda x: True, "StieltjesGamma")],
    "elliptic_e": [(lambda *x: True, "EllipticE")],
    "elliptic_f": [(lambda *x: True, "EllipticE")],
    "elliptic_k": [(lambda x: True, "EllipticK")],
    "elliptic_pi": [(lambda *x: True, "EllipticPi")],
    "zeta": [(lambda *x: True, "Zeta")],
    "dirichlet_eta": [(lambda x: True, "DirichletEta")],
    "riemann_xi": [(lambda x: True, "RiemannXi")],
    "besseli": [(lambda *x: True, "BesselI")],
    "besselj": [(lambda *x: True, "BesselJ")],
    "besselk": [(lambda *x: True, "BesselK")],
    "bessely": [(lambda *x: True, "BesselY")],
    "hankel1": [(lambda *x: True, "HankelH1")],
    "hankel2": [(lambda *x: True, "HankelH2")],
    "airyai": [(lambda x: True, "AiryAi")],
    "airybi": [(lambda x: True, "AiryBi")],
    "airyaiprime": [(lambda x: True, "AiryAiPrime")],
    "airybiprime": [(lambda x: True, "AiryBiPrime")],
    "polylog": [(lambda *x: True, "PolyLog")],
    "lerchphi": [(lambda *x: True, "LerchPhi")],
    "gcd": [(lambda *x: True, "GCD")],
    "lcm": [(lambda *x: True, "LCM")],
    "jn": [(lambda *x: True, "SphericalBesselJ")],
    "yn": [(lambda *x: True, "SphericalBesselY")],
    "hyper": [(lambda *x: True, "HypergeometricPFQ")],
    "meijerg": [(lambda *x: True, "MeijerG")],
    "appellf1": [(lambda *x: True, "AppellF1")],
    "DiracDelta": [(lambda x: True, "DiracDelta")],
    "Heaviside": [(lambda x: True, "HeavisideTheta")],
    "KroneckerDelta": [(lambda *x: True, "KroneckerDelta")],
    "sqrt": [(lambda x: True, "Sqrt")],  # For automatic rewrites
}


class MCodePrinter(CodePrinter):
    """A printer to convert Python expressions to
    strings of the Wolfram's Mathematica code
    """
    printmethod = "_mcode"
    language = "Wolfram Language"

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
    })

    _number_symbols: set[tuple[Expr, Float]] = set()
    _not_supported: set[Basic] = set()

    def __init__(self, settings={}):
        """Register function mappings supplied by user"""
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {}).copy()
        for k, v in userfuncs.items():
            if not isinstance(v, list):
                userfuncs[k] = [(lambda *x: True, v)]
        self.known_functions.update(userfuncs)

    def _format_code(self, lines):
        return lines

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_Mul(self, expr):
        PREC = precedence(expr)
        c, nc = expr.args_cnc()
        res = super()._print_Mul(expr.func(*c))
        if nc:
            res += '*'
            res += '**'.join(self.parenthesize(a, PREC) for a in nc)
        return res

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # Primitive numbers
    def _print_Zero(self, expr):
        return '0'

    def _print_One(self, expr):
        return '1'

    def _print_NegativeOne(self, expr):
        return '-1'

    def _print_Half(self, expr):
        return '1/2'

    def _print_ImaginaryUnit(self, expr):
        return 'I'


    # Infinity and invalid numbers
    def _print_Infinity(self, expr):
        return 'Infinity'

    def _print_NegativeInfinity(self, expr):
        return '-Infinity'

    def _print_ComplexInfinity(self, expr):
        return 'ComplexInfinity'

    def _print_NaN(self, expr):
        return 'Indeterminate'


    # Mathematical constants
    def _print_Exp1(self, expr):
        return 'E'

    def _print_Pi(self, expr):
        return 'Pi'

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        expanded = expr.expand(func=True)
        PREC = precedence(expr)
        return self.parenthesize(expanded, PREC)

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Catalan(self, expr):
        return 'Catalan'


    def _print_list(self, expr):
        return '{' + ', '.join(self.doprint(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _print_ImmutableDenseMatrix(self, expr):
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseMatrix(self, expr):

        def print_rule(pos, val):
            return '{} -> {}'.format(
            self.doprint((pos[0]+1, pos[1]+1)), self.doprint(val))

        def print_data():
            items = sorted(expr.todok().items(), key=default_sort_key)
            return '{' + \
                ', '.join(print_rule(k, v) for k, v in items) + \
                '}'

        def print_dims():
            return self.doprint(expr.shape)

        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_ImmutableDenseNDimArray(self, expr):
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseNDimArray(self, expr):
        def print_string_list(string_list):
            return '{' + ', '.join(a for a in string_list) + '}'

        def to_mathematica_index(*args):
            """Helper function to change Python style indexing to
            Pathematica indexing.

            Python indexing (0, 1 ... n-1)
            -> Mathematica indexing (1, 2 ... n)
            """
            return tuple(i + 1 for i in args)

        def print_rule(pos, val):
            """Helper function to print a rule of Mathematica"""
            return '{} -> {}'.format(self.doprint(pos), self.doprint(val))

        def print_data():
            """Helper function to print data part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html

            ``data`` must be formatted with rule.
            """
            return print_string_list(
                [print_rule(
                    to_mathematica_index(*(expr._get_tuple_index(key))),
                    value)
                for key, value in sorted(expr._sparse_array.items())]
            )

        def print_dims():
            """Helper function to print dimensions part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html
            """
            return self.doprint(expr.shape)

        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_mfunc = self.known_functions[expr.func.__name__]
            for cond, mfunc in cond_mfunc:
                if cond(*expr.args):
                    return "%s[%s]" % (mfunc, self.stringify(expr.args, ", "))
        elif expr.func.__name__ in self._rewriteable_functions:
            # Simple rewrite to supported function possible
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return self._print(expr.rewrite(target_f))
        return expr.func.__name__ + "[%s]" % self.stringify(expr.args, ", ")

    _print_MinMaxBase = _print_Function

    def _print_LambertW(self, expr):
        if len(expr.args) == 1:
            return "ProductLog[{}]".format(self._print(expr.args[0]))
        return "ProductLog[{}, {}]".format(
            self._print(expr.args[1]), self._print(expr.args[0]))

    def _print_atan2(self, expr):
        return "ArcTan[{}, {}]".format(
            self._print(expr.args[1]), self._print(expr.args[0]))

    def _print_Integral(self, expr):
        if len(expr.variables) == 1 and not expr.limits[0][1:]:
            args = [expr.args[0], expr.variables[0]]
        else:
            args = expr.args
        return "Hold[Integrate[" + ', '.join(self.doprint(a) for a in args) + "]]"

    def _print_Sum(self, expr):
        return "Hold[Sum[" + ', '.join(self.doprint(a) for a in expr.args) + "]]"

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return "Hold[D[" + ', '.join(self.doprint(a) for a in [dexpr] + dvars) + "]]"


    def _get_comment(self, text):
        return "(* {} *)".format(text)


def mathematica_code(expr, **settings):
    r"""Converts an expr to a string of the Wolfram Mathematica code

    Examples
    ========

    >>> from sympy import mathematica_code as mcode, symbols, sin
    >>> x = symbols('x')
    >>> mcode(sin(x).series(x).removeO())
    '(1/120)*x^5 - 1/6*x^3 + x'
    """
    return MCodePrinter(settings).doprint(expr)
<|endoftext|>"""
R code printer

The RCodePrinter converts single SymPy expressions into single R expressions,
using the functions defined in math.h where possible.



"""

from __future__ import annotations
from typing import Any

from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# dictionary mapping SymPy function to (argument_conditions, C_function).
# Used in RCodePrinter._print_Function(self)
known_functions = {
    #"Abs": [(lambda x: not x.is_integer, "fabs")],
    "Abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "erf": "erf",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "floor": "floor",
    "ceiling": "ceiling",
    "sign": "sign",
    "Max": "max",
    "Min": "min",
    "factorial": "factorial",
    "gamma": "gamma",
    "digamma": "digamma",
    "trigamma": "trigamma",
    "beta": "beta",
    "sqrt": "sqrt",  # To enable automatic rewrite
}

# These are the core reserved words in the R language. Taken from:
# https://cran.r-project.org/doc/manuals/r-release/R-lang.html#Reserved-words

reserved_words = ['if',
                  'else',
                  'repeat',
                  'while',
                  'function',
                  'for',
                  'in',
                  'next',
                  'break',
                  'TRUE',
                  'FALSE',
                  'NULL',
                  'Inf',
                  'NaN',
                  'NA',
                  'NA_integer_',
                  'NA_real_',
                  'NA_complex_',
                  'NA_character_',
                  'volatile']


class RCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of R code"""
    printmethod = "_rcode"
    language = "R"

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })
    _operators = {
       'and': '&',
        'or': '|',
       'not': '!',
    }

    _relationals: dict[str, str] = {}

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {}".format(text)

    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists of codelines
        """
        open_lines = []
        close_lines = []
        loopstart = "for (%(var)s in %(start)s:%(end)s){"
        for i in indices:
            # R arrays start at 1 and end at dimension
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower+1),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return '%s^%s' % (self.parenthesize(expr.base, PREC),
                                 self.parenthesize(expr.exp, PREC))


    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0/%d.0' % (p, q)

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s[%s]" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Exp1(self, expr):
        return "exp(1)"

    def _print_Pi(self, expr):
        return 'pi'

    def _print_Infinity(self, expr):
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        return '-Inf'

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment

        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        #if isinstance(expr.rhs, Piecewise):
        #    from sympy.functions.elementary.piecewise import Piecewise
        #    # Here we modify Piecewise so each expression is now
        #    # an Assignment, and then continue on the print.
        #    expressions = []
        #    conditions = []
        #    for (e, c) in rhs.args:
        #        expressions.append(Assignment(lhs, e))
        #        conditions.append(c)
        #    temp = Piecewise(*zip(expressions, conditions))
        #    return self._print(temp)
        #elif isinstance(lhs, MatrixSymbol):
        if isinstance(lhs, MatrixSymbol):
            # Here we form an Assignment for each element in the array,
            # printing each one.
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        elif self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_Piecewise(self, expr):
        # This method is called only for inline if constructs
        # Top level piecewise is handled in doprint()
        if expr.args[-1].cond == True:
            last_line = "%s" % self._print(expr.args[-1].expr)
        else:
            last_line = "ifelse(%s,%s,NA)" % (self._print(expr.args[-1].cond), self._print(expr.args[-1].expr))
        code=last_line
        for e, c in reversed(expr.args[:-1]):
            code= "ifelse(%s,%s," % (self._print(c), self._print(e))+code+")"
        return(code)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_MatrixElement(self, expr):
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return "{} {} {};".format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(target=target, start=start,
                stop=stop-1, step=step, body=body)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
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


def rcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of r code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired R string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        rfunction_string)] or [(argument_test, rfunction_formater)]. See below
        for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import rcode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rcode((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau^(7.0/2.0)'
    >>> rcode(sin(x), assign_to="s")
    's = sin(x);'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rcode(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the R-function takes a subset of the original arguments:

    >>> rcode(2**x + 3**x, user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rcode(expr, assign_to=tau))
    tau = ifelse(x > 0,x + 1,x);

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
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> rcode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rcode(mat, A))
    A[0] = x^2;
    A[1] = ifelse(x > 0,x + 1,x);
    A[2] = sin(x);

    """

    return RCodePrinter(settings).doprint(expr, assign_to)


def print_rcode(expr, **settings):
    """Prints R representation of the given expression."""
    print(rcode(expr, **settings))
<|endoftext|>import sympy.codegen
import sympy.codegen.cfunctions
from sympy.external.importtools import version_tuple
from collections.abc import Iterable

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.codegen.cfunctions import Sqrt
from sympy.external import import_module
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pycode import AbstractPythonCodePrinter, ArrayPrinter
import sympy

tensorflow = import_module('tensorflow')

class TensorflowPrinter(ArrayPrinter, AbstractPythonCodePrinter):
    """
    Tensorflow printer which handles vectorized piecewise functions,
    logical operators, max/min, and relational operators.
    """
    printmethod = "_tensorflowcode"

    mapping = {
        sympy.Abs: "tensorflow.math.abs",
        sympy.sign: "tensorflow.math.sign",

        # XXX May raise error for ints.
        sympy.ceiling: "tensorflow.math.ceil",
        sympy.floor: "tensorflow.math.floor",
        sympy.log: "tensorflow.math.log",
        sympy.exp: "tensorflow.math.exp",
        Sqrt: "tensorflow.math.sqrt",
        sympy.cos: "tensorflow.math.cos",
        sympy.acos: "tensorflow.math.acos",
        sympy.sin: "tensorflow.math.sin",
        sympy.asin: "tensorflow.math.asin",
        sympy.tan: "tensorflow.math.tan",
        sympy.atan: "tensorflow.math.atan",
        sympy.atan2: "tensorflow.math.atan2",
        # XXX Also may give NaN for complex results.
        sympy.cosh: "tensorflow.math.cosh",
        sympy.acosh: "tensorflow.math.acosh",
        sympy.sinh: "tensorflow.math.sinh",
        sympy.asinh: "tensorflow.math.asinh",
        sympy.tanh: "tensorflow.math.tanh",
        sympy.atanh: "tensorflow.math.atanh",

        sympy.re: "tensorflow.math.real",
        sympy.im: "tensorflow.math.imag",
        sympy.arg: "tensorflow.math.angle",

        # XXX May raise error for ints and complexes
        sympy.erf: "tensorflow.math.erf",
        sympy.loggamma: "tensorflow.math.lgamma",

        sympy.Eq: "tensorflow.math.equal",
        sympy.Ne: "tensorflow.math.not_equal",
        sympy.StrictGreaterThan: "tensorflow.math.greater",
        sympy.StrictLessThan: "tensorflow.math.less",
        sympy.LessThan: "tensorflow.math.less_equal",
        sympy.GreaterThan: "tensorflow.math.greater_equal",

        sympy.And: "tensorflow.math.logical_and",
        sympy.Or: "tensorflow.math.logical_or",
        sympy.Not: "tensorflow.math.logical_not",
        sympy.Max: "tensorflow.math.maximum",
        sympy.Min: "tensorflow.math.minimum",

        # Matrices
        sympy.MatAdd: "tensorflow.math.add",
        sympy.HadamardProduct: "tensorflow.math.multiply",
        sympy.Trace: "tensorflow.linalg.trace",

        # XXX May raise error for integer matrices.
        sympy.Determinant : "tensorflow.linalg.det",
    }

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        tensorflow_version=None
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        version = self._settings['tensorflow_version']
        if version is None and tensorflow:
            version = tensorflow.__version__
        self.tensorflow_version = version

    def _print_Function(self, expr):
        op = self.mapping.get(type(expr), None)
        if op is None:
            return super()._print_Basic(expr)
        children = [self._print(arg) for arg in expr.args]
        if len(children) == 1:
            return "%s(%s)" % (
                self._module_format(op),
                children[0]
            )
        else:
            return self._expand_fold_binary_op(op, children)

    _print_Expr = _print_Function
    _print_Application = _print_Function
    _print_MatrixExpr = _print_Function
    # TODO: a better class structure would avoid this mess:
    _print_Relational = _print_Function
    _print_Not = _print_Function
    _print_And = _print_Function
    _print_Or = _print_Function
    _print_HadamardProduct = _print_Function
    _print_Trace = _print_Function
    _print_Determinant = _print_Function

    def _print_Inverse(self, expr):
        op = self._module_format('tensorflow.linalg.inv')
        return "{}({})".format(op, self._print(expr.arg))

    def _print_Transpose(self, expr):
        version = self.tensorflow_version
        if version and version_tuple(version) < version_tuple('1.14'):
            op = self._module_format('tensorflow.matrix_transpose')
        else:
            op = self._module_format('tensorflow.linalg.matrix_transpose')
        return "{}({})".format(op, self._print(expr.arg))

    def _print_Derivative(self, expr):
        variables = expr.variables
        if any(isinstance(i, Iterable) for i in variables):
            raise NotImplementedError("derivation by multiple variables is not supported")
        def unfold(expr, args):
            if not args:
                return self._print(expr)
            return "%s(%s, %s)[0]" % (
                    self._module_format("tensorflow.gradients"),
                    unfold(expr, args[:-1]),
                    self._print(args[-1]),
                )
        return unfold(expr.expr, variables)

    def _print_Piecewise(self, expr):
        version = self.tensorflow_version
        if version and version_tuple(version) < version_tuple('1.0'):
            tensorflow_piecewise = "tensorflow.select"
        else:
            tensorflow_piecewise = "tensorflow.where"

        from sympy.functions.elementary.piecewise import Piecewise
        e, cond = expr.args[0].args
        if len(expr.args) == 1:
            return '{}({}, {}, {})'.format(
                self._module_format(tensorflow_piecewise),
                self._print(cond),
                self._print(e),
                0)

        return '{}({}, {}, {})'.format(
            self._module_format(tensorflow_piecewise),
            self._print(cond),
            self._print(e),
            self._print(Piecewise(*expr.args[1:])))

    def _print_Pow(self, expr):
        # XXX May raise error for
        # int**float or int**complex or float**complex
        base, exp = expr.args
        if expr.exp == S.Half:
            return "{}({})".format(
                self._module_format("tensorflow.math.sqrt"), self._print(base))
        return "{}({}, {})".format(
            self._module_format("tensorflow.math.pow"),
            self._print(base), self._print(exp))

    def _print_MatrixBase(self, expr):
        tensorflow_f = "tensorflow.Variable" if expr.free_symbols else "tensorflow.constant"
        data = "["+", ".join(["["+", ".join([self._print(j) for j in i])+"]" for i in expr.tolist()])+"]"
        return "%s(%s)" % (
            self._module_format(tensorflow_f),
            data,
        )

    def _print_MatMul(self, expr):
        from sympy.matrices.expressions import MatrixExpr
        mat_args = [arg for arg in expr.args if isinstance(arg, MatrixExpr)]
        args = [arg for arg in expr.args if arg not in mat_args]
        if args:
            return "%s*%s" % (
                self.parenthesize(Mul.fromiter(args), PRECEDENCE["Mul"]),
                self._expand_fold_binary_op(
                    "tensorflow.linalg.matmul", mat_args)
            )
        else:
            return self._expand_fold_binary_op(
                "tensorflow.linalg.matmul", mat_args)

    def _print_MatPow(self, expr):
        return self._expand_fold_binary_op(
            "tensorflow.linalg.matmul", [expr.base]*expr.exp)

    def _print_CodeBlock(self, expr):
        # TODO: is this necessary?
        ret = []
        for subexpr in expr.args:
            ret.append(self._print(subexpr))
        return "\n".join(ret)

    def _print_isnan(self, exp):
        return f'tensorflow.math.is_nan({self._print(*exp.args)})'

    def _print_isinf(self, exp):
        return f'tensorflow.math.is_inf({self._print(*exp.args)})'

    _module = "tensorflow"
    _einsum = "linalg.einsum"
    _add = "math.add"
    _transpose = "transpose"
    _ones = "ones"
    _zeros = "zeros"


def tensorflow_code(expr, **settings):
    printer = TensorflowPrinter(settings)
    return printer.doprint(expr)
<|endoftext|>"""
A Printer for generating executable code.

The most important function here is srepr that returns a string so that the
relation eval(srepr(expr))=expr holds in an appropriate environment.
"""

from __future__ import annotations
from typing import Any

from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str

from .printer import Printer, print_function


class ReprPrinter(Printer):
    printmethod = "_sympyrepr"

    _default_settings: dict[str, Any] = {
        "order": None,
        "perm_cyclic" : True,
    }

    def reprify(self, args, sep):
        """
        Prints each item in `args` and joins them with `sep`.
        """
        return sep.join([self.doprint(item) for item in args])

    def emptyPrinter(self, expr):
        """
        The fallback printer.
        """
        if isinstance(expr, str):
            return expr
        elif hasattr(expr, "__srepr__"):
            return expr.__srepr__()
        elif hasattr(expr, "args") and hasattr(expr.args, "__iter__"):
            l = []
            for o in expr.args:
                l.append(self._print(o))
            return expr.__class__.__name__ + '(%s)' % ', '.join(l)
        elif hasattr(expr, "__module__") and hasattr(expr, "__name__"):
            return "<'%s.%s'>" % (expr.__module__, expr.__name__)
        else:
            return str(expr)

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    def _print_Cycle(self, expr):
        return expr.__repr__()

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
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
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            if not expr.size:
                return 'Permutation()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            return 'Permutation%s' %s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % str(expr.array_form)
                return 'Permutation([], size=%s)' % expr.size
            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
            use = full = str(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    def _print_Function(self, expr):
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.args])
        return r

    def _print_Heaviside(self, expr):
        # Same as _print_Function but uses pargs to suppress default value for
        # 2nd arg.
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.pargs])
        return r

    def _print_FunctionClass(self, expr):
        if issubclass(expr, AppliedUndef):
            return 'Function(%r)' % (expr.__name__)
        else:
            return expr.__name__

    def _print_Half(self, expr):
        return 'Rational(1, 2)'

    def _print_RationalConstant(self, expr):
        return str(expr)

    def _print_AtomicExpr(self, expr):
        return str(expr)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Integer(self, expr):
        return 'Integer(%i)' % expr.p

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_UniversalSet(self, expr):
        return 'UniversalSet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_list(self, expr):
        return "[%s]" % self.reprify(expr, ", ")

    def _print_dict(self, expr):
        sep = ", "
        dict_kvs = ["%s: %s" % (self.doprint(key), self.doprint(value)) for key, value in expr.items()]
        return "{%s}" % sep.join(dict_kvs)

    def _print_set(self, expr):
        if not expr:
            return "set()"
        return "{%s}" % self.reprify(expr, ", ")

    def _print_MatrixBase(self, expr):
        # special case for some empty matrices
        if (expr.rows == 0) ^ (expr.cols == 0):
            return '%s(%s, %s, %s)' % (expr.__class__.__name__,
                                       self._print(expr.rows),
                                       self._print(expr.cols),
                                       self._print([]))
        l = []
        for i in range(expr.rows):
            l.append([])
            for j in range(expr.cols):
                l[-1].append(expr[i, j])
        return '%s(%s)' % (expr.__class__.__name__, self._print(l))

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_NaN(self, expr):
        return "nan"

    def _print_Mul(self, expr, order=None):
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    def _print_Rational(self, expr):
        return 'Rational(%s, %s)' % (self._print(expr.p), self._print(expr.q))

    def _print_PythonRational(self, expr):
        return "%s(%d, %d)" % (expr.__class__.__name__, expr.p, expr.q)

    def _print_Fraction(self, expr):
        return 'Fraction(%s, %s)' % (self._print(expr.numerator), self._print(expr.denominator))

    def _print_Float(self, expr):
        r = mlib_to_str(expr._mpf_, repr_dps(expr._prec))
        return "%s('%s', precision=%i)" % (expr.__class__.__name__, r, expr._prec)

    def _print_Sum2(self, expr):
        return "Sum2(%s, (%s, %s, %s))" % (self._print(expr.f), self._print(expr.i),
                                           self._print(expr.a), self._print(expr.b))

    def _print_Str(self, s):
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))

    def _print_Symbol(self, expr):
        d = expr._assumptions_orig
        # print the dummy_index like it was an assumption
        if expr.is_Dummy:
            d = d.copy()
            d['dummy_index'] = expr.dummy_index

        if d == {}:
            return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s)" % (expr.__class__.__name__,
                                   self._print(expr.name), ', '.join(attr))

    def _print_CoordinateSymbol(self, expr):
        d = expr._assumptions.generator

        if d == {}:
            return "%s(%s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index)
            )
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index),
                ', '.join(attr)
            )

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_AppliedPredicate(self, expr):
        # will be changed to just expr.args when args overriding is removed
        args = expr._args
        return "%s(%s)" % (expr.__class__.__name__, self.reprify(args, ", "))

    def _print_str(self, expr):
        return repr(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.reprify(expr, ", ")

    def _print_WildFunction(self, expr):
        return "%s('%s')" % (expr.__class__.__name__, expr.name)

    def _print_AlgebraicNumber(self, expr):
        return "%s(%s, %s)" % (expr.__class__.__name__,
            self._print(expr.root), self._print(expr.coeffs()))

    def _print_PolyRing(self, ring):
        return "%s(%s, %s, %s)" % (ring.__class__.__name__,
            self._print(ring.symbols), self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "%s(%s, %s, %s)" % (field.__class__.__name__,
            self._print(field.symbols), self._print(field.domain), self._print(field.order))

    def _print_PolyElement(self, poly):
        terms = list(poly.terms())
        terms.sort(key=poly.ring.order, reverse=True)
        return "%s(%s, %s)" % (poly.__class__.__name__, self._print(poly.ring), self._print(terms))

    def _print_FracElement(self, frac):
        numer_terms = list(frac.numer.terms())
        numer_terms.sort(key=frac.field.order, reverse=True)
        denom_terms = list(frac.denom.terms())
        denom_terms.sort(key=frac.field.order, reverse=True)
        numer = self._print(numer_terms)
        denom = self._print(denom_terms)
        return "%s(%s, %s, %s)" % (frac.__class__.__name__, self._print(frac.field), numer, denom)

    def _print_FractionField(self, domain):
        cls = domain.__class__.__name__
        field = self._print(domain.field)
        return "%s(%s)" % (cls, field)

    def _print_PolynomialRingBase(self, ring):
        cls = ring.__class__.__name__
        dom = self._print(ring.domain)
        gens = ', '.join(map(self._print, ring.gens))
        order = str(ring.order)
        if order != ring.default_order:
            orderstr = ", order=" + order
        else:
            orderstr = ""
        return "%s(%s, %s%s)" % (cls, dom, gens, orderstr)

    def _print_DMP(self, p):
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)
        return "%s(%s, %s)" % (cls, rep, dom)

    def _print_MonogenicFiniteExtension(self, ext):
        # The expanded tree shown by srepr(ext.modulus)
        # is not practical.
        return "FiniteExtension(%s)" % str(ext.modulus)

    def _print_ExtensionElement(self, f):
        rep = self._print(f.rep)
        ext = self._print(f.ext)
        return "ExtElem(%s, %s)" % (rep, ext)

@print_function(ReprPrinter)
def srepr(expr, **settings):
    """return expr in repr form"""
    return ReprPrinter(settings).doprint(expr)
<|endoftext|>"""
A few practical conventions common to all printers.
"""

import re

from collections.abc import Iterable
from sympy.core.function import Derivative

_name_with_digits_p = re.compile(r'^([^\W\d_]+)(\d+)$', re.UNICODE)


def split_super_sub(text):
    """Split a symbol name into a name, superscripts and subscripts

    The first part of the symbol name is considered to be its actual
    'name', followed by super- and subscripts. Each superscript is
    preceded with a "^" character or by "__". Each subscript is preceded
    by a "_" character.  The three return values are the actual name, a
    list with superscripts and a list with subscripts.

    Examples
    ========

    >>> from sympy.printing.conventions import split_super_sub
    >>> split_super_sub('a_x^1')
    ('a', ['1'], ['x'])
    >>> split_super_sub('var_sub1__sup_sub2')
    ('var', ['sup'], ['sub1', 'sub2'])

    """
    if not text:
        return text, [], []

    pos = 0
    name = None
    supers = []
    subs = []
    while pos < len(text):
        start = pos + 1
        if text[pos:pos + 2] == "__":
            start += 1
        pos_hat = text.find("^", start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find("_", start)
        if pos_usc < 0:
            pos_usc = len(text)
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        if name is None:
            name = part
        elif part.startswith("^"):
            supers.append(part[1:])
        elif part.startswith("__"):
            supers.append(part[2:])
        elif part.startswith("_"):
            subs.append(part[1:])
        else:
            raise RuntimeError("This should never happen.")

    # Make a little exception when a name ends with digits, i.e. treat them
    # as a subscript too.
    m = _name_with_digits_p.match(name)
    if m:
        name, sub = m.groups()
        subs.insert(0, sub)

    return name, supers, subs


def requires_partial(expr):
    """Return whether a partial derivative symbol is required for printing

    This requires checking how many free variables there are,
    filtering out the ones that are integers. Some expressions do not have
    free variables. In that case, check its variable list explicitly to
    get the context of the expression.
    """

    if isinstance(expr, Derivative):
        return requires_partial(expr.expr)

    if not isinstance(expr.free_symbols, Iterable):
        return len(set(expr.variables)) > 1

    return sum(not s.is_integer for s in expr.free_symbols) > 1
<|endoftext|>"""
A Printer for generating readable representation of most SymPy classes.
"""

from __future__ import annotations
from typing import Any

from sympy.core import S, Rational, Pow, Basic, Mul, Number
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import Integer
from sympy.core.relational import Relational
from sympy.core.sorting import default_sort_key
from sympy.utilities.iterables import sift
from .precedence import precedence, PRECEDENCE
from .printer import Printer, print_function

from mpmath.libmp import prec_to_dps, to_str as mlib_to_str


class StrPrinter(Printer):
    printmethod = "_sympystr"
    _default_settings: dict[str, Any] = {
        "order": None,
        "full_prec": "auto",
        "sympy_integers": False,
        "abbrev": False,
        "perm_cyclic": True,
        "min": None,
        "max": None,
        "dps" : None
    }

    _relationals: dict[str, str] = {}

    def parenthesize(self, item, level, strict=False):
        if (precedence(item) < level) or ((not strict) and precedence(item) <= level):
            return "(%s)" % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        prec = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-') and not term.is_Add:
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < prec or term.is_Add:
                l.extend([sign, "(%s)" % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_Not(self, expr):
        return '~%s' %(self.parenthesize(expr.args[0],PRECEDENCE["Not"]))

    def _print_And(self, expr):
        args = list(expr.args)
        for j, i in enumerate(args):
            if isinstance(i, Relational) and (
                    i.canonical.rhs is S.NegativeInfinity):
                args.insert(0, args.pop(j))
        return self.stringify(args, " & ", PRECEDENCE["BitwiseAnd"])

    def _print_Or(self, expr):
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    def _print_Xor(self, expr):
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])

    def _print_AppliedPredicate(self, expr):
        return '%s(%s)' % (
            self._print(expr.function), self.stringify(expr.arguments, ", "))

    def _print_Basic(self, expr):
        l = [self._print(o) for o in expr.args]
        return expr.__class__.__name__ + "(%s)" % ", ".join(l)

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_Catalan(self, expr):
        return 'Catalan'

    def _print_ComplexInfinity(self, expr):
        return 'zoo'

    def _print_ConditionSet(self, s):
        args = tuple([self._print(i) for i in (s.sym, s.condition)])
        if s.base_set is S.UniversalSet:
            return 'ConditionSet(%s, %s)' % args
        args += (self._print(s.base_set),)
        return 'ConditionSet(%s, %s, %s)' % args

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return 'Derivative(%s)' % ", ".join((self._print(arg) for arg in [dexpr] + dvars))

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            item = "%s: %s" % (self._print(key), self._print(d[key]))
            items.append(item)

        return "{%s}" % ", ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('Domain: ' + self._print(d.symbols) + ' in ' +
                    self._print(d.set))
        else:
            return 'Domain on ' + self._print(d.symbols)

    def _print_Dummy(self, expr):
        return '_' + expr.name

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Exp1(self, expr):
        return 'E'

    def _print_ExprCondPair(self, expr):
        return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))

    def _print_Function(self, expr):
        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_Heaviside(self, expr):
        # Same as _print_Function but uses pargs to suppress default 1/2 for
        # 2nd args
        return expr.func.__name__ + "(%s)" % self.stringify(expr.pargs, ", ")

    def _print_TribonacciConstant(self, expr):
        return 'TribonacciConstant'

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    def _print_Infinity(self, expr):
        return 'oo'

    def _print_Integral(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Integral(%s, %s)' % (self._print(expr.function), L)

    def _print_Interval(self, i):
        fin =  'Interval{m}({a}, {b})'
        a, b, l, r = i.args
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and not r:
            m = ''
        elif b.is_infinite and not l:
            m = ''
        elif not l and not r:
            m = ''
        elif l and r:
            m = '.open'
        elif l:
            m = '.Lopen'
        else:
            m = '.Ropen'
        return fin.format(**{'a': a, 'b': b, 'm': m})

    def _print_AccumulationBounds(self, i):
        return "AccumBounds(%s, %s)" % (self._print(i.min),
                                        self._print(i.max))

    def _print_Inverse(self, I):
        return "%s**(-1)" % self.parenthesize(I.arg, PRECEDENCE["Pow"])

    def _print_Lambda(self, obj):
        expr = obj.expr
        sig = obj.signature
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        return "Lambda(%s, %s)" % (self._print(sig), self._print(expr))

    def _print_LatticeOp(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + "(%s)" % ", ".join(self._print(arg) for arg in args)

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print, (e, z, z0, dir)))


    def _print_list(self, expr):
        return "[%s]" % self.stringify(expr, ", ")

    def _print_List(self, expr):
        return self._print_list(expr)

    def _print_MatrixBase(self, expr):
        return expr._format_str(self)

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))

    def _print_MatrixSlice(self, expr):
        def strslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join((self._print(arg) for arg in x))
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + '[' +
                strslice(expr.rowslice, expr.parent.rows) + ', ' +
                strslice(expr.colslice, expr.parent.cols) + ']')

    def _print_DeferredVector(self, expr):
        return expr.name

    def _print_Mul(self, expr):

        prec = precedence(expr)

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        args = expr.args
        if args[0] is S.One or any(
                isinstance(a, Number) or
                a.is_Pow and all(ai.is_Integer for ai in a.args)
                for a in args[1:]):
            d, n = sift(args, lambda x:
                isinstance(x, Pow) and bool(x.exp.as_coeff_Mul()[0] < 0),
                binary=True)
            for i, di in enumerate(d):
                if di.exp.is_Number:
                    e = -di.exp
                else:
                    dargs = list(di.exp.args)
                    dargs[0] = -dargs[0]
                    e = Mul._from_args(dargs)
                d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base

            pre = []
            # don't parenthesize first factor if negative
            if n and not n[0].is_Add and n[0].could_extract_minus_sign():
                pre = [self._print(n.pop(0))]

            nfactors = pre + [self.parenthesize(a, prec, strict=False)
                for a in n]
            if not nfactors:
                nfactors = ['1']

            # don't parenthesize first of denominator unless singleton
            if len(d) > 1 and d[0].could_extract_minus_sign():
                pre = [self._print(d.pop(0))]
            else:
                pre = []
            dfactors = pre + [self.parenthesize(a, prec, strict=False)
                for a in d]

            n = '*'.join(nfactors)
            d = '*'.join(dfactors)
            if len(dfactors) > 1:
                return '%s/(%s)' % (n, d)
            elif dfactors:
                return '%s/%s' % (n, d)
            return n

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
        def apow(i):
            b, e = i.as_base_exp()
            eargs = list(Mul.make_args(e))
            if eargs[0] is S.NegativeOne:
                eargs = eargs[1:]
            else:
                eargs[0] = -eargs[0]
            e = Mul._from_args(eargs)
            if isinstance(i, Pow):
                return i.func(b, e, evaluate=False)
            return i.func(e, evaluate=False)
        for item in args:
            if (item.is_commutative and
                    isinstance(item, Pow) and
                    bool(item.exp.as_coeff_Mul()[0] < 0)):
                if item.exp is not S.NegativeOne:
                    b.append(apow(item))
                else:
                    if (len(item.args[0].args) != 1 and
                            isinstance(item.base, (Mul, Pow))):
                        # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(item.base)
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

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

        return sign + '*'.join(
            [self.parenthesize(arg, precedence(expr)) for arg in expr.args]
        )

    def _print_ElementwiseApplyFunction(self, expr):
        return "{}.({})".format(
            expr.function,
            self._print(expr.expr),
        )

    def _print_NaN(self, expr):
        return 'nan'

    def _print_NegativeInfinity(self, expr):
        return '-oo'

    def _print_Order(self, expr):
        if not expr.variables or all(p is S.Zero for p in expr.point):
            if len(expr.variables) <= 1:
                return 'O(%s)' % self._print(expr.expr)
            else:
                return 'O(%s)' % self.stringify((expr.expr,) + expr.variables, ', ', 0)
        else:
            return 'O(%s)' % self.stringify(expr.args, ', ', 0)

    def _print_Ordinal(self, expr):
        return expr.__str__()

    def _print_Cycle(self, expr):
        return expr.__str__()

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
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
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            if not expr.size:
                return '()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            s = s.replace(',', '')
            return s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % self._print(expr.array_form)
                return 'Permutation([], size=%s)' % self._print(expr.size)
            trim = self._print(expr.array_form[:s[-1] + 1]) + ', size=%s' % self._print(expr.size)
            use = full = self._print(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    def _print_Subs(self, obj):
        expr, old, new = obj.args
        if len(obj.point) == 1:
            old = old[0]
            new = new[0]
        return "Subs(%s, %s, %s)" % (
            self._print(expr), self._print(old), self._print(new))

    def _print_TensorIndex(self, expr):
        return expr._print()

    def _print_TensorHead(self, expr):
        return expr._print()

    def _print_Tensor(self, expr):
        return expr._print()

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "*".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        return expr._print()

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return "%s[%s]" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True), ", ".join([self._print(i) for i in expr.indices]))

    def _print_PermutationGroup(self, expr):
        p = ['    %s' % self._print(a) for a in expr.args]
        return 'PermutationGroup([\n%s])' % ',\n'.join(p)

    def _print_Pi(self, expr):
        return 'pi'

    def _print_PolyRing(self, ring):
        return "Polynomial ring in %s over %s with %s order" % \
            (", ".join((self._print(rs) for rs in ring.symbols)),
            self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "Rational function field in %s over %s with %s order" % \
            (", ".join((self._print(fs) for fs in field.symbols)),
            self._print(field.domain), self._print(field.order))

    def _print_FreeGroupElement(self, elm):
        return elm.__str__()

    def _print_GaussianElement(self, poly):
        return "(%s + %s*I)" % (poly.x, poly.y)

    def _print_PolyElement(self, poly):
        return poly.str(self, PRECEDENCE, "%s**%s", "*")

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self.parenthesize(frac.numer, PRECEDENCE["Mul"], strict=True)
            denom = self.parenthesize(frac.denom, PRECEDENCE["Atom"], strict=True)
            return numer + "/" + denom

    def _print_Poly(self, expr):
        ATOM_PREC = PRECEDENCE["Atom"] - 1
        terms, gens = [], [ self.parenthesize(s, ATOM_PREC) for s in expr.gens ]

        for monom, coeff in expr.terms():
            s_monom = []

            for i, e in enumerate(monom):
                if e > 0:
                    if e == 1:
                        s_monom.append(gens[i])
                    else:
                        s_monom.append(gens[i] + "**%d" % e)

            s_monom = "*".join(s_monom)

            if coeff.is_Add:
                if s_monom:
                    s_coeff = "(" + self._print(coeff) + ")"
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + "*" + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        format = expr.__class__.__name__ + "(%s, %s"

        from sympy.polys.polyerrors import PolynomialError

        try:
            format += ", modulus=%s" % expr.get_modulus()
        except PolynomialError:
            format += ", domain='%s'" % expr.get_domain()

        format += ")"

        for index, item in enumerate(gens):
            if len(item) > 2 and (item[:1] == "(" and item[len(item) - 1:] == ")"):
                gens[index] = item[1:len(item) - 1]

        return format % (' '.join(terms), ', '.join(gens))

    def _print_UniversalSet(self, p):
        return 'UniversalSet'

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_Pow(self, expr, rational=False):
        """Printing helper function for ``Pow``

        Parameters
        ==========

        rational : bool, optional
            If ``True``, it will not attempt printing ``sqrt(x)`` or
            ``x**S.Half`` as ``sqrt``, and will use ``x**(1/2)``
            instead.

            See examples for additional details

        Examples
        ========

        >>> from sympy import sqrt, StrPrinter
        >>> from sympy.abc import x

        How ``rational`` keyword works with ``sqrt``:

        >>> printer = StrPrinter()
        >>> printer._print_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._print_Pow(sqrt(x), rational=False)
        'sqrt(x)'
        >>> printer._print_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._print_Pow(1/sqrt(x), rational=False)
        '1/sqrt(x)'

        Notes
        =====

        ``sqrt(x)`` is canonicalized as ``Pow(x, S.Half)`` in SymPy,
        so there is no need of defining a separate printer for ``sqrt``.
        Instead, it should be handled here as well.
        """
        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that will
                # match -0.5, which we don't want.
                return "%s/sqrt(%s)" % tuple((self._print(arg) for arg in (S.One, expr.base)))
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '%s/%s' % (self._print(S.One),
                                  self.parenthesize(expr.base, PREC, strict=False))

        e = self.parenthesize(expr.exp, PREC, strict=False)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False),
                         self.parenthesize(expr.exp, PREC, strict=False))

    def _print_Integer(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(%s)" % (expr)
        return str(expr.p)

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            if self._settings.get("sympy_integers", False):
                return "S(%s)/%s" % (expr.p, expr.q)
            return "%s/%s" % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        else:
            return "%d/%d" % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return "%s/%s" % (expr.numerator, expr.denominator)

    def _print_Float(self, expr):
        prec = expr._prec
        dps = self._settings.get('dps', None)
        if dps is None:
            dps = 0 if prec < 5 else prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        rv = rv.removeprefix('+') # e.g., +inf -> inf
        return rv

    def _print_Relational(self, expr):

        charmap = {
            "==": "Eq",
            "!=": "Ne",
            ":=": "Assignment",
            '+=': "AddAugmentedAssignment",
            "-=": "SubAugmentedAssignment",
            "*=": "MulAugmentedAssignment",
            "/=": "DivAugmentedAssignment",
            "%=": "ModAugmentedAssignment",
        }

        if expr.rel_op in charmap:
            return '%s(%s, %s)' % (charmap[expr.rel_op], self._print(expr.lhs),
                                   self._print(expr.rhs))

        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)),
                           self._relationals.get(expr.rel_op) or expr.rel_op,
                           self.parenthesize(expr.rhs, precedence(expr)))

    def _print_ComplexRootOf(self, expr):
        return "CRootOf(%s, %d)" % (self._print_Add(expr.expr,  order='lex'),
                                    expr.index)

    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        return "RootSum(%s)" % ", ".join(args)

    def _print_GroebnerBasis(self, basis):
        cls = basis.__class__.__name__

        exprs = [self._print_Add(arg, order=basis.order) for arg in basis.exprs]
        exprs = "[%s]" % ", ".join(exprs)

        gens = [ self._print(gen) for gen in basis.gens ]
        domain = "domain='%s'" % self._print(basis.domain)
        order = "order='%s'" % self._print(basis.order)

        args = [exprs] + gens + [domain, order]

        return "%s(%s)" % (cls, ", ".join(args))

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if not args:
            return "set()"
        return '{%s}' % args

    def _print_FiniteSet(self, s):
        from sympy.sets.sets import FiniteSet
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(item) for item in items)
        if any(item.has(FiniteSet) for item in items):
            return 'FiniteSet({})'.format(args)
        return '{{{}}}'.format(args)

    def _print_Partition(self, s):
        items = sorted(s, key=default_sort_key)

        args = ', '.join(self._print(arg) for arg in items)
        return 'Partition({})'.format(args)

    def _print_frozenset(self, s):
        if not s:
            return "frozenset()"
        return "frozenset(%s)" % self._print_set(s)

    def _print_Sum(self, expr):
        def _xab_tostr(xab):
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Sum(%s, %s)' % (self._print(expr.function), L)

    def _print_Symbol(self, expr):
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Identity(self, expr):
        return "I"

    def _print_ZeroMatrix(self, expr):
        return "0"

    def _print_OneMatrix(self, expr):
        return "1"

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_str(self, expr):
        return str(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.stringify(expr, ", ")

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_Transpose(self, T):
        return "%s.T" % self.parenthesize(T.arg, PRECEDENCE["Pow"])

    def _print_Uniform(self, expr):
        return "Uniform(%s, %s)" % (self._print(expr.a), self._print(expr.b))

    def _print_Quantity(self, expr):
        if self._settings.get("abbrev", False):
            return "%s" % expr.abbrev
        return "%s" % expr.name

    def _print_Quaternion(self, expr):
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True) for i in expr.args]
        a = [s[0]] + [i+"*"+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_Dimension(self, expr):
        return str(expr)

    def _print_Wild(self, expr):
        return expr.name + '_'

    def _print_WildFunction(self, expr):
        return expr.name + '_'

    def _print_WildDot(self, expr):
        return expr.name

    def _print_WildPlus(self, expr):
        return expr.name

    def _print_WildStar(self, expr):
        return expr.name

    def _print_Zero(self, expr):
        if self._settings.get("sympy_integers", False):
            return "S(0)"
        return self._print_Integer(Integer(0))

    def _print_DMP(self, p):
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)

        return "%s(%s, %s)" % (cls, rep, dom)

    def _print_DMF(self, expr):
        cls = expr.__class__.__name__
        num = self._print(expr.num)
        den = self._print(expr.den)
        dom = self._print(expr.dom)

        return "%s(%s, %s, %s)" % (cls, num, den, dom)

    def _print_Object(self, obj):
        return 'Object("%s")' % obj.name

    def _print_IdentityMorphism(self, morphism):
        return 'IdentityMorphism(%s)' % morphism.domain

    def _print_NamedMorphism(self, morphism):
        return 'NamedMorphism(%s, %s, "%s")' % \
               (morphism.domain, morphism.codomain, morphism.name)

    def _print_Category(self, category):
        return 'Category("%s")' % category.name

    def _print_Manifold(self, manifold):
        return manifold.name.name

    def _print_Patch(self, patch):
        return patch.name.name

    def _print_CoordSystem(self, coords):
        return coords.name.name

    def _print_BaseScalarField(self, field):
        return field._coord_sys.symbols[field._index].name

    def _print_BaseVectorField(self, field):
        return 'e_%s' % field._coord_sys.symbols[field._index].name

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            return 'd%s' % field._coord_sys.symbols[field._index].name
        else:
            return 'd(%s)' % self._print(field)

    def _print_Tr(self, expr):
        #TODO : Handle indices
        return "%s(%s)" % ("Tr", self._print(expr.args[0]))

    def _print_Str(self, s):
        return self._print(s.name)

    def _print_AppliedBinaryRelation(self, expr):
        rel = expr.function
        return '%s(%s, %s)' % (self._print(rel),
                               self._print(expr.lhs),
                               self._print(expr.rhs))


@print_function(StrPrinter)
def sstr(expr, **settings):
    """Returns the expression as a string.

    For large expressions where speed is a concern, use the setting
    order='none'. If abbrev=True setting is used then units are printed in
    abbreviated form.

    Examples
    ========

    >>> from sympy import symbols, Eq, sstr
    >>> a, b = symbols('a b')
    >>> sstr(Eq(a + b, 0))
    'Eq(a + b, 0)'
    """

    p = StrPrinter(settings)
    s = p.doprint(expr)

    return s


class StrReprPrinter(StrPrinter):
    """(internal) -- see sstrrepr"""

    def _print_str(self, s):
        return repr(s)

    def _print_Str(self, s):
        # Str does not to be printed same as str here
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))

@print_function(StrReprPrinter)
def sstrrepr(expr, **settings):
    """return expr in mixed str/repr form

       i.e. strings are returned in repr form with quotes, and everything else
       is returned in str form.

       This function could be useful for hooking into sys.displayhook
    """

    p = StrReprPrinter(settings)
    s = p.doprint(expr)

    return s
<|endoftext|>from __future__ import annotations
from typing import Any

from functools import wraps

from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import Expr, UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE


class requires:
    """ Decorator for registering requirements on print methods. """
    def __init__(self, **kwargs):
        self._req = kwargs

    def __call__(self, method):
        def _method_wrapper(self_, *args, **kwargs):
            for k, v in self._req.items():
                getattr(self_, k).update(v)
            return method(self_, *args, **kwargs)
        return wraps(method)(_method_wrapper)


class AssignmentError(Exception):
    """
    Raised if an assignment variable for a loop is missing.
    """
    pass

class PrintMethodNotImplementedError(NotImplementedError):
    """
    Raised if a _print_* method is missing in the Printer.
    """
    pass

def _convert_python_lists(arg):
    if isinstance(arg, list):
        from sympy.codegen.abstract_nodes import List
        return List(*(_convert_python_lists(e) for e in arg))
    elif isinstance(arg, tuple):
        return tuple(_convert_python_lists(e) for e in arg)
    else:
        return arg


class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """

    _operators = {
        'and': '&&',
        'or': '||',
        'not': '!',
    }

    _default_settings: dict[str, Any] = {
        'order': None,
        'full_prec': 'auto',
        'error_on_reserved': False,
        'reserved_word_suffix': '_',
        'human': True,
        'inline': False,
        'allow_unknown_functions': False,
        'strict': None  # True or False; None => True if human == True
    }

    # Functions which are "simple" to rewrite to other functions that
    # may be supported
    # function_to_rewrite : (function_to_rewrite_to, iterable_with_other_functions_required)
    _rewriteable_functions = {
            'cot': ('tan', []),
            'csc': ('sin', []),
            'sec': ('cos', []),
            'acot': ('atan', []),
            'acsc': ('asin', []),
            'asec': ('acos', []),
            'coth': ('exp', []),
            'csch': ('exp', []),
            'sech': ('exp', []),
            'acoth': ('log', []),
            'acsch': ('log', []),
            'asech': ('log', []),
            'catalan': ('gamma', []),
            'fibonacci': ('sqrt', []),
            'lucas': ('sqrt', []),
            'beta': ('gamma', []),
            'sinc': ('sin', ['Piecewise']),
            'Mod': ('floor', []),
            'factorial': ('gamma', []),
            'factorial2': ('gamma', ['Piecewise']),
            'subfactorial': ('uppergamma', []),
            'RisingFactorial': ('gamma', ['Piecewise']),
            'FallingFactorial': ('gamma', ['Piecewise']),
            'binomial': ('gamma', []),
            'frac': ('floor', []),
            'Max': ('Piecewise', []),
            'Min': ('Piecewise', []),
            'Heaviside': ('Piecewise', []),
            'erf2': ('erf', []),
            'erfc': ('erf', []),
            'Li': ('li', []),
            'Ei': ('li', []),
            'dirichlet_eta': ('zeta', []),
            'riemann_xi': ('zeta', ['gamma']),
            'SingularityFunction': ('Piecewise', []),
    }

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        if self._settings.get('strict', True) == None:
            # for backwards compatibility, human=False need not to throw:
            self._settings['strict'] = self._settings.get('human', True) == True
        if not hasattr(self, 'reserved_words'):
            self.reserved_words = set()

    def _handle_UnevaluatedExpr(self, expr):
        return expr.replace(re, lambda arg: arg if isinstance(
            arg, UnevaluatedExpr) and arg.args[0].is_real else re(arg))

    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)
            If provided, the printed code will set the expression to a variable or multiple variables
            with the name or names given in ``assign_to``.
        """
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.codegen.ast import CodeBlock, Assignment

        def _handle_assign_to(expr, assign_to):
            if assign_to is None:
                return sympify(expr)
            if isinstance(assign_to, (list, tuple)):
                if len(expr) != len(assign_to):
                    raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
                return CodeBlock(*[_handle_assign_to(lhs, rhs) for lhs, rhs in zip(expr, assign_to)])
            if isinstance(assign_to, str):
                if expr.is_Matrix:
                    assign_to = MatrixSymbol(assign_to, *expr.shape)
                else:
                    assign_to = Symbol(assign_to)
            elif not isinstance(assign_to, Basic):
                raise TypeError("{} cannot assign to object of type {}".format(
                        type(self).__name__, type(assign_to)))
            return Assignment(assign_to, expr)

        expr = _convert_python_lists(expr)
        expr = _handle_assign_to(expr, assign_to)

        # Remove re(...) nodes due to UnevaluatedExpr.is_real always is None:
        expr = self._handle_UnevaluatedExpr(expr)

        # keep a set of expressions that are not strictly translatable to Code
        # and number constants that must be declared and initialized
        self._not_supported = set()
        self._number_symbols = set()

        lines = self._print(expr).splitlines()

        # format the output
        if self._settings["human"]:
            frontlines = []
            if self._not_supported:
                frontlines.append(self._get_comment(
                        "Not supported in {}:".format(self.language)))
                for expr in sorted(self._not_supported, key=str):
                    frontlines.append(self._get_comment(type(expr).__name__))
            for name, value in sorted(self._number_symbols, key=str):
                frontlines.append(self._declare_number_const(name, value))
            lines = frontlines + lines
            lines = self._format_code(lines)
            result = "\n".join(lines)
        else:
            lines = self._format_code(lines)
            num_syms = {(k, self._print(v)) for k, v in self._number_symbols}
            result = (num_syms, self._not_supported, "\n".join(lines))
        self._not_supported = set()
        self._number_symbols = set()
        return result

    def _doprint_loops(self, expr, assign_to=None):
        # Here we print an expression that contains Indexed objects, they
        # correspond to arrays in the generated code.  The low-level implementation
        # involves looping over array elements and possibly storing results in temporary
        # variables or accumulate it in the assign_to object.

        if self._settings.get('contract', True):
            from sympy.tensor import get_contraction_structure
            # Setup loops over non-dummy indices  --  all terms need these
            indices = self._get_expression_indices(expr, assign_to)
            # Setup loops over dummy indices  --  each term needs separate treatment
            dummies = get_contraction_structure(expr)
        else:
            indices = []
            dummies = {None: (expr,)}
        openloop, closeloop = self._get_loop_opening_ending(indices)

        # terms with no summations first
        if None in dummies:
            text = StrPrinter.doprint(self, Add(*dummies[None]))
        else:
            # If all terms have summations we must initialize array to Zero
            text = StrPrinter.doprint(self, 0)

        # skip redundant assignments (where lhs == rhs)
        lhs_printed = self._print(assign_to)
        lines = []
        if text != lhs_printed:
            lines.extend(openloop)
            if assign_to is not None:
                text = self._get_statement("%s = %s" % (lhs_printed, text))
            lines.append(text)
            lines.extend(closeloop)

        # then terms with summations
        for d in dummies:
            if isinstance(d, tuple):
                indices = self._sort_optimized(d, expr)
                openloop_d, closeloop_d = self._get_loop_opening_ending(
                    indices)

                for term in dummies[d]:
                    if term in dummies and not ([list(f.keys()) for f in dummies[term]]
                            == [[None] for f in dummies[term]]):
                        # If one factor in the term has it's own internal
                        # contractions, those must be computed first.
                        # (temporary variables?)
                        raise NotImplementedError(
                            "FIXME: no support for contractions in factor yet")
                    else:

                        # We need the lhs expression as an accumulator for
                        # the loops, i.e
                        #
                        # for (int d=0; d < dim; d++){
                        #    lhs[] = lhs[] + term[][d]
                        # }           ^.................. the accumulator
                        #
                        # We check if the expression already contains the
                        # lhs, and raise an exception if it does, as that
                        # syntax is currently undefined.  FIXME: What would be
                        # a good interpretation?
                        if assign_to is None:
                            raise AssignmentError(
                                "need assignment variable for loops")
                        if term.has(assign_to):
                            raise ValueError("FIXME: lhs present in rhs,\
                                this is undefined in CodePrinter")

                        lines.extend(openloop)
                        lines.extend(openloop_d)
                        text = "%s = %s" % (lhs_printed, StrPrinter.doprint(
                            self, assign_to + term))
                        lines.append(self._get_statement(text))
                        lines.extend(closeloop_d)
                        lines.extend(closeloop)

        return "\n".join(lines)

    def _get_expression_indices(self, expr, assign_to):
        from sympy.tensor import get_indices
        rinds, junk = get_indices(expr)
        linds, junk = get_indices(assign_to)

        # support broadcast of scalar
        if linds and not rinds:
            rinds = linds
        if rinds != linds:
            raise ValueError("lhs indices must match non-dummy"
                    " rhs indices in %s" % expr)

        return self._sort_optimized(rinds, assign_to)

    def _sort_optimized(self, indices, expr):

        from sympy.tensor.indexed import Indexed

        if not indices:
            return []

        # determine optimized loop order by giving a score to each index
        # the index with the highest score are put in the innermost loop.
        score_table = {}
        for i in indices:
            score_table[i] = 0

        arrays = expr.atoms(Indexed)
        for arr in arrays:
            for p, ind in enumerate(arr.indices):
                try:
                    score_table[ind] += self._rate_index_position(p)
                except KeyError:
                    pass

        return sorted(indices, key=lambda x: score_table[x])

    def _rate_index_position(self, p):
