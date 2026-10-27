        $c_3 = 33.93669377311689$

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.

        """
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(l_T_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, l_T_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_T_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_T_tilde = l_T_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0*exp(c3*(l_T_tilde - c1)) - c2

        return c0*exp(c3*UnevaluatedExpr(l_T_tilde - c1)) - c2

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_T_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 2:
            return exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 3:
            return -c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 4:
            return Integer(-1)
        elif argindex == 5:
            return c0*(l_T_tilde - c1)*exp(c3*UnevaluatedExpr(l_T_tilde - c1))

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return TendonForceLengthInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_T_tilde = self.args[0]
        _l_T_tilde = printer._print(l_T_tilde)
        return r'\operatorname{fl}^T \left( %s \right)' % _l_T_tilde


class TendonForceLengthInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon length that produces a specific normalized
    tendon force.

    The function is defined by the equation:

    ${fl^T}^{-1} = frac{\log{\frac{fl^T + c_2}{c_0}}}{c_3} + c_1$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$. This function is the exact analytical inverse
    of the related tendon force-length curve ``TendonForceLengthDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthInverseDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon force-length, which is
    equal to the tendon force. We'll create a :class:`~.Symbol` called ``fl_T`` to
    represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    >>> fl_T = Symbol('fl_T')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016.with_defaults(fl_T)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_T_tilde.doit(evaluate=False)
    c1 + log((c2 + fl_T)/c0)/c3

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> l_T_tilde.diff(fl_T)
    1/(c3*(c2 + fl_T))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_T):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse tendon force-length function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.

        """
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(fl_T, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fl_T, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fl_T, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_T = fl_T.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return log((fl_T + c2)/c0)/c3 + c1

        return log(UnevaluatedExpr((fl_T + c2)/c0))/c3 + c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fl_T, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return 1/(c3*(fl_T + c2))
        elif argindex == 2:
            return -1/(c0*c3)
        elif argindex == 3:
            return Integer(1)
        elif argindex == 4:
            return 1/(c3*(fl_T + c2))
        elif argindex == 5:
            return -log(UnevaluatedExpr((fl_T + c2)/c0))/c3**2

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return TendonForceLengthDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fl_T = self.args[0]
        _fl_T = printer._print(fl_T)
        return r'\left( \operatorname{fl}^T \right)^{-1} \left( %s \right)' % _fl_T


class FiberForceLengthPassiveDeGroote2016(CharacteristicCurveFunction):
    r"""Passive muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl^M_{pas} = \frac{\frac{\exp{c_1 \left(\tilde{l^M} - 1\right)}}{c_0} - 1}{\exp{c_1} - 1}$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthPassiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M/l_M_opt, 0.6, 4.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.0186573603637741*(-1 + exp(6.66666666666667*(l_M/l_M_opt - 1)))

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    0.12438240242516*exp(6.66666666666667*(l_M/l_M_opt - 1))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.

        """
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(l_M_tilde, c0, c1)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return (exp((c1*(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

        return (exp((c1*UnevaluatedExpr(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_M_tilde, c0, c1 = self.args
        if argindex == 1:
            return c1*exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)/(c0*(exp(c1) - 1))
        elif argindex == 2:
            return (
                -c1*exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)
                *UnevaluatedExpr(l_M_tilde - 1)/(c0**2*(exp(c1) - 1))
            )
        elif argindex == 3:
            return (
                -exp(c1)*(-1 + exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0))/(exp(c1) - 1)**2
                + exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)*(l_M_tilde - 1)/(c0*(exp(c1) - 1))
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceLengthPassiveInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_M_tilde = self.args[0]
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{pas} \left( %s \right)' % _l_M_tilde


class FiberForceLengthPassiveInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse passive muscle fiber force-length curve based on De Groote et
    al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber length that produces a specific normalized
    passive muscle fiber force.

    The function is defined by the equation:

    ${fl^M_{pas}}^{-1} = \frac{c_0 \log{\left(\exp{c_1} - 1\right)fl^M_pas + 1}}{c_1} + 1$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$. This function is the
    exact analytical inverse of the related tendon force-length curve
    ``FiberForceLengthPassiveDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate
    :class:`FiberForceLengthPassiveInverseDeGroote2016` is using the
    :meth:`~.with_defaults` constructor because this will automatically populate the
    constants within the characteristic curve equation with the floating point
    values from the original publication. This constructor takes a single
    argument corresponding to the normalized passive muscle fiber length-force
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fl_M_pas`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    >>> fl_M_pas = Symbol('fl_M_pas')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(fl_M_pas)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_M_tilde.doit(evaluate=False)
    c0*log(1 + fl_M_pas*(exp(c1) - 1))/c1 + 1

    The function can also be differentiated. We'll differentiate with respect
    to fl_M_pas using the ``diff`` method on an instance with the single positional
    argument ``fl_M_pas``.

    >>> l_M_tilde.diff(fl_M_pas)
    c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_M_pas):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(fl_M_pas, c0, c1)

    @classmethod
    def eval(cls, fl_M_pas, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fl_M_pas, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_M_pas = fl_M_pas.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1 + 1

        return c0*log(UnevaluatedExpr(fl_M_pas*(exp(c1) - 1)) + 1)/c1 + 1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fl_M_pas, c0, c1 = self.args
        if argindex == 1:
            return c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
        elif argindex == 2:
            return log(fl_M_pas*(exp(c1) - 1) + 1)/c1
        elif argindex == 3:
            return (
                c0*fl_M_pas*exp(c1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
                - c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1**2
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceLengthPassiveDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fl_M_pas = self.args[0]
        _fl_M_pas = printer._print(fl_M_pas)
        return r'\left( \operatorname{fl}^M_{pas} \right)^{-1} \left( %s \right)' % _fl_M_pas


class FiberForceLengthActiveDeGroote2016(CharacteristicCurveFunction):
    r"""Active muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl_{\text{act}}^M = c_0 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_1}{c_2 + c_3 \tilde{l}^M}\right)^2\right)
    + c_4 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_5}{c_6 + c_7 \tilde{l}^M}\right)^2\right)
    + c_8 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_9}{c_{10} + c_{11} \tilde{l}^M}\right)^2\right)$

    with constant values of $c0 = 0.814$, $c1 = 1.06$, $c2 = 0.162$,
    $c3 = 0.0633$, $c4 = 0.433$, $c5 = 0.717$, $c6 = -0.0299$, $c7 = 0.2$,
    $c8 = 0.1$, $c9 = 1.0$, $c10 = 0.354$, and $c11 = 0.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    active fiber force of 1 at a normalized fiber length of 1, and an active
    fiber force of 0 at normalized fiber lengths of 0 and 2.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthActiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = symbols('c0:12')
    >>> fl_M = FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3,
    ...     c4, c5, c6, c7, c8, c9, c10, c11)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6,
    c7, c8, c9, c10, c11)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M/l_M_opt, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.814*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2))
    + 0.433*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + 0.1*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    ((-0.79798269973507*l_M/l_M_opt
    + 0.79798269973507)*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)
    + (0.433*(-l_M/l_M_opt + 0.717)/(0.2*l_M/l_M_opt - 0.0299)**2
    + 0.0866*(l_M/l_M_opt - 0.717)**2/(0.2*l_M/l_M_opt
    - 0.0299)**3)*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + (0.814*(-l_M/l_M_opt + 1.06)/(0.0633*l_M/l_M_opt
    + 0.162)**2 + 0.0515262*(l_M/l_M_opt
    - 1.06)**2/(0.0633*l_M/l_M_opt
    + 0.162)**3)*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2)))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber act force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c0 = 0.814$
        $c1 = 1.06$
        $c2 = 0.162$
        $c3 = 0.0633$
        $c4 = 0.433$
        $c5 = 0.717$
        $c6 = -0.0299$
        $c7 = 0.2$
        $c8 = 0.1$
        $c9 = 1.0$
        $c10 = 0.354$
        $c11 = 0.0$

        Parameters
        ==========

        fl_M_act : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        c0 = Float('0.814')
        c1 = Float('1.06')
        c2 = Float('0.162')
        c3 = Float('0.0633')
        c4 = Float('0.433')
        c5 = Float('0.717')
        c6 = Float('-0.0299')
        c7 = Float('0.2')
        c8 = Float('0.1')
        c9 = Float('1.0')
        c10 = Float('0.354')
        c11 = Float('0.0')
        return cls(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.814``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``1.06``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.162``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.0633``.
        c4 : Any (sympifiable)
            The fifth constant in the characteristic equation. The published
            value is ``0.433``.
        c5 : Any (sympifiable)
            The sixth constant in the characteristic equation. The published
            value is ``0.717``.
        c6 : Any (sympifiable)
            The seventh constant in the characteristic equation. The published
            value is ``-0.0299``.
        c7 : Any (sympifiable)
            The eighth constant in the characteristic equation. The published
            value is ``0.2``.
        c8 : Any (sympifiable)
            The ninth constant in the characteristic equation. The published
            value is ``0.1``.
        c9 : Any (sympifiable)
            The tenth constant in the characteristic equation. The published
            value is ``1.0``.
        c10 : Any (sympifiable)
            The eleventh constant in the characteristic equation. The published
            value is ``0.354``.
        c11 : Any (sympifiable)
            The tweflth constant in the characteristic equation. The published
            value is ``0.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            constants = [c.doit(deep=deep, **hints) for c in constants]
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = constants

        if evaluate:
            return (
                c0*exp(-(((l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
                + c4*exp(-(((l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
                + c8*exp(-(((l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
            )

        return (
            c0*exp(-((UnevaluatedExpr(l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
            + c4*exp(-((UnevaluatedExpr(l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
            + c8*exp(-((UnevaluatedExpr(l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
        )

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = self.args
        if argindex == 1:
            return (
                c0*(
                    c3*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                    + (c1 - l_M_tilde)/((c2 + c3*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
                + c4*(
                    c7*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                    + (c5 - l_M_tilde)/((c6 + c7*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
                + c8*(
                    c11*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                    + (c9 - l_M_tilde)/((c10 + c11*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 2:
            return exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
        elif argindex == 3:
            return (
                c0*(l_M_tilde - c1)/(c2 + c3*l_M_tilde)**2
                *exp(-(l_M_tilde - c1)**2 /(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 4:
            return (
                c0*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                *exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 5:
            return (
                c0*l_M_tilde*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                *exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 6:
            return exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
        elif argindex == 7:
            return (
                c4*(l_M_tilde - c5)/(c6 + c7*l_M_tilde)**2
                *exp(-(l_M_tilde - c5)**2 /(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 8:
            return (
                c4*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                *exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 9:
            return (
                c4*l_M_tilde*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                *exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 10:
            return exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
        elif argindex == 11:
            return (
                c8*(l_M_tilde - c9)/(c10 + c11*l_M_tilde)**2
                *exp(-(l_M_tilde - c9)**2 /(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 12:
            return (
                c8*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                *exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 13:
            return (
                c8*l_M_tilde*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                *exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )

        raise ArgumentIndexError(self, argindex)

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_M_tilde = self.args[0]
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{act} \left( %s \right)' % _l_M_tilde


class FiberForceVelocityDeGroote2016(CharacteristicCurveFunction):
    r"""Muscle fiber force-velocity curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber force produced as a function of
    normalized tendon velocity.

    The function is defined by the equation:

    $fv^M = c_0 \log{\left(c_1 \tilde{v}_m + c_2\right) + \sqrt{\left(c_1 \tilde{v}_m + c_2\right)^2 + 1}} + c_3$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically populate
    the constants within the characteristic curve equation with the floating
    point values from the original publication. This constructor takes a single
    argument corresponding to normalized muscle fiber extension velocity. We'll
    create a :class:`~.Symbol` called ``v_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    >>> v_M_tilde = Symbol('v_M_tilde')
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fv_M = FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``v_M`` and
    ``v_M_max``, representing muscle fiber extension velocity and maximum
    muscle fiber extension velocity respectively. We can then represent
    ``v_M_tilde`` as an expression, the ratio of these.

    >>> v_M, v_M_max = symbols('v_M v_M_max')
    >>> v_M_tilde = v_M/v_M_max
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M/v_M_max, -0.318, -8.149, -0.374, 0.886)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fv_M.doit(evaluate=False)
    0.886 - 0.318*log(-8.149*v_M/v_M_max - 0.374 + sqrt(1 + (-8.149*v_M/v_M_max
    - 0.374)**2))

    The function can also be differentiated. We'll differentiate with respect
    to v_M using the ``diff`` method on an instance with the single positional
    argument ``v_M``.

    >>> fv_M.diff(v_M)
    2.591382*(1 + (-8.149*v_M/v_M_max - 0.374)**2)**(-1/2)/v_M_max

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, v_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber force-velocity function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(v_M_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, v_M_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``v_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        v_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            v_M_tilde = v_M_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0*log(c1*v_M_tilde + c2 + sqrt((c1*v_M_tilde + c2)**2 + 1)) + c3

        return c0*log(c1*v_M_tilde + c2 + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)) + c3

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        v_M_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0*c1/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 2:
            return log(
                c1*v_M_tilde + c2
                + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
            )
        elif argindex == 3:
            return c0*v_M_tilde/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 4:
            return c0/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 5:
            return Integer(1)

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceVelocityInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        v_M_tilde = self.args[0]
        _v_M_tilde = printer._print(v_M_tilde)
        return r'\operatorname{fv}^M \left( %s \right)' % _v_M_tilde


class FiberForceVelocityInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse muscle fiber force-velocity curve based on De Groote et al.,
    2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber velocity that produces a specific
    normalized muscle fiber force.

    The function is defined by the equation:

    ${fv^M}^{-1} = \frac{\sinh{\frac{fv^M - c_3}{c_0}} - c_2}{c_1}$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$. This function is the exact analytical inverse of the related
    muscle fiber force-velocity curve ``FiberForceVelocityDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityInverseDeGroote2016`
    is using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber force-velocity
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fv_M`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    >>> fv_M = Symbol('fv_M')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016.with_defaults(fv_M)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> v_M_tilde.doit(evaluate=False)
    (-c2 + sinh((-c3 + fv_M)/c0))/c1

    The function can also be differentiated. We'll differentiate with respect
    to fv_M using the ``diff`` method on an instance with the single positional
    argument ``fv_M``.

    >>> v_M_tilde.diff(fv_M)
    cosh((-c3 + fv_M)/c0)/(c0*c1)

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fv_M):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber force-velocity
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(fv_M, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fv_M, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber force as a function of muscle fiber
            extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``fv_M`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fv_M, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fv_M = fv_M.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return (sinh((fv_M - c3)/c0) - c2)/c1

        return (sinh(UnevaluatedExpr(fv_M - c3)/c0) - c2)/c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fv_M, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return cosh((fv_M - c3)/c0)/(c0*c1)
        elif argindex == 2:
            return (c3 - fv_M)*cosh((fv_M - c3)/c0)/(c0**2*c1)
        elif argindex == 3:
            return (c2 - sinh((fv_M - c3)/c0))/c1**2
        elif argindex == 4:
            return -1/c1
        elif argindex == 5:
            return -cosh((fv_M - c3)/c0)/(c0*c1)

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceVelocityDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fv_M = self.args[0]
        _fv_M = printer._print(fv_M)
        return r'\left( \operatorname{fv}^M \right)^{-1} \left( %s \right)' % _fv_M


@dataclass(frozen=True)
class CharacteristicCurveCollection:
    """Simple data container to group together related characteristic curves."""
    tendon_force_length: CharacteristicCurveFunction
    tendon_force_length_inverse: CharacteristicCurveFunction
    fiber_force_length_passive: CharacteristicCurveFunction
    fiber_force_length_passive_inverse: CharacteristicCurveFunction
    fiber_force_length_active: CharacteristicCurveFunction
    fiber_force_velocity: CharacteristicCurveFunction
    fiber_force_velocity_inverse: CharacteristicCurveFunction

    def __iter__(self):
        """Iterator support for ``CharacteristicCurveCollection``."""
        yield self.tendon_force_length
        yield self.tendon_force_length_inverse
        yield self.fiber_force_length_passive
        yield self.fiber_force_length_passive_inverse
        yield self.fiber_force_length_active
        yield self.fiber_force_velocity
        yield self.fiber_force_velocity_inverse
<|endoftext|>"""Mixin classes for sharing functionality between unrelated classes.

This module is named with a leading underscore to signify to users that it's
"private" and only intended for internal use by the biomechanics module.

"""


__all__ = ['_NamedMixin']


class _NamedMixin:
    """Mixin class for adding `name` properties.

    Valid names, as will typically be used by subclasses as a suffix when
    naming automatically-instantiated symbol attributes, must be nonzero length
    strings.

    Attributes
    ==========

    name : str
        The name identifier associated with the instance. Must be a string of
        length at least 1.

    """

    @property
    def name(self) -> str:
        """The name associated with the class instance."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if hasattr(self, '_name'):
            msg = (
                f'Can\'t set attribute `name` to {repr(name)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(name, str):
            msg = (
                f'Name {repr(name)} passed to `name` was of type '
                f'{type(name)}, must be {str}.'
            )
            raise TypeError(msg)
        if name in {''}:
            msg = (
                f'Name {repr(name)} is invalid, must be a nonzero length '
                f'{type(str)}.'
            )
            raise ValueError(msg)
        self._name = name
<|endoftext|>from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import (exp, log)
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.plotting.plot import plot_parametric
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex
from sympy.geometry.polygon import deg

__all__ = ['pole_zero_numerical_data', 'pole_zero_plot',
    'step_response_numerical_data', 'step_response_plot',
    'impulse_response_numerical_data', 'impulse_response_plot',
    'ramp_response_numerical_data', 'ramp_response_plot',
    'bode_magnitude_numerical_data', 'bode_phase_numerical_data',
    'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot',
    'nyquist_plot_expr', 'nyquist_plot', 'nichols_plot_expr',
    'nichols_plot']


matplotlib = import_module(
        'matplotlib', import_kwargs={'fromlist': ['pyplot']},
        catch=(RuntimeError,))

if matplotlib:
    plt = matplotlib.pyplot


def _check_system(system):
    """Function to check whether the dynamical system passed for plots is
    compatible or not."""
    if not isinstance(system, SISOLinearTimeInvariant):
        raise NotImplementedError("Only SISO LTI systems are currently supported.")
    sys = system.to_expr()
    len_free_symbols = len(sys.free_symbols)
    if len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")
    if sys.has(exp):
        # Should test that exp is not part of a constant, in which case
        # no exception is required, compare exp(s) with s*exp(1)
        raise NotImplementedError("Time delay terms are not supported.")


def _poly_roots(poly):
    """Function to get the roots of a polynomial."""
    def _eval(l):
        return [float(i) if i.is_real else complex(i) for i in l]
    if poly.domain in (QQ, ZZ):
        return _eval(poly.all_roots())
    # XXX: Use all_roots() for irrational coefficients when possible
    # See https://github.com/sympy/sympy/issues/22943
    return _eval(poly.nroots())


def pole_zero_numerical_data(system):
    """
    Returns the numerical data of poles and zeros of the system.
    It is internally used by ``pole_zero_plot`` to get the data
    for plotting poles and zeros. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the pole-zero data is to be computed.

    Returns
    =======

    tuple : (zeros, poles)
        zeros = Zeros of the system as a list of Python float/complex.
        poles = Poles of the system as a list of Python float/complex.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import pole_zero_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> pole_zero_numerical_data(tf1)
    ([-1j, 1j], [-2.0, -1.0, (-0.5-0.8660254037844386j), (-0.5+0.8660254037844386j)])

    See Also
    ========

    pole_zero_plot

    """
    _check_system(system)
    system = system.doit()  # Get the equivalent TransferFunction object.

    num_poly = Poly(system.num, system.var)
    den_poly = Poly(system.den, system.var)

    return _poly_roots(num_poly), _poly_roots(den_poly)


def pole_zero_plot(system, pole_color='blue', pole_markersize=10,
    zero_color='orange', zero_markersize=7, grid=True, show_axes=True,
    show=True, **kwargs):
    r"""
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
    pole_color : str, tuple, optional
        The color of the pole points on the plot. Default color
        is blue. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    pole_markersize : Number, optional
        The size of the markers used to mark the poles in the plot.
        Default pole markersize is 10.
    zero_color : str, tuple, optional
        The color of the zero points on the plot. Default color
        is orange. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    zero_markersize : Number, optional
        The size of the markers used to mark the zeros in the plot.
        Default zero markersize is 7.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import pole_zero_plot
        >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
        >>> pole_zero_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    pole_zero_numerical_data

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot

    """
    zeros, poles = pole_zero_numerical_data(system)

    zero_real = [i.real for i in zeros]
    zero_imag = [i.imag for i in zeros]

    pole_real = [i.real for i in poles]
    pole_imag = [i.imag for i in poles]

    plt.plot(pole_real, pole_imag, 'x', mfc='none',
        markersize=pole_markersize, color=pole_color)
    plt.plot(zero_real, zero_imag, 'o', markersize=zero_markersize,
        color=zero_color)
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title(f'Poles and Zeros of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def step_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the step response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the unit step response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the step response. NumPy array.
        y = Amplitude-axis values of the points in the step response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import step_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> step_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.025413462339411542, 0.0484508722725343, ... , 9.670250533855183, 9.844291913708725, 10.0],
    [0.0, 0.023844582399907256, 0.042894276802320226, ..., 6.828770759094287e-12, 6.456457160755703e-12])

    See Also
    ========

    step_response_plot

    """
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = system.to_expr()/(system.var)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def step_response_plot(system, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the unit step response of a continuous-time system. It is
    the response of the system when the input signal is a step function.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Step Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import step_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> step_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    impulse_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/lti.step.html

    """
    x, y = step_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Unit Step Response of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def impulse_response_numerical_data(system, prec=8, lower_limit=0,
    upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the impulse response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the impulse response data is to be computed.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the impulse response. NumPy array.
        y = Amplitude-axis values of the points in the impulse response. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import impulse_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> impulse_response_numerical_data(tf1)   # doctest: +SKIP
    ([0.0, 0.06616480200395854,... , 9.854500743565858, 10.0],
    [0.9999999799999999, 0.7042848373025861,...,7.170748906965121e-13, -5.1901263495547205e-12])

    See Also
    ========

    impulse_response_plot

    """
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = system.to_expr()
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def impulse_response_plot(system, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the unit impulse response (Input is the Dirac-Delta Function) of a
    continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Impulse Response is to be computed.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import impulse_response_plot
        >>> tf1 = TransferFunction(8*s**2 + 18*s + 32, s**3 + 6*s**2 + 14*s + 24, s)
        >>> impulse_response_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://www.mathworks.com/help/control/ref/dynamicsystem.impulse.html

    """
    x, y = impulse_response_numerical_data(system, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Impulse Response of ${latex(system)}$', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def ramp_response_numerical_data(system, slope=1, prec=8,
    lower_limit=0, upper_limit=10, **kwargs):
    """
    Returns the numerical values of the points in the ramp response plot
    of a SISO continuous-time system. By default, adaptive sampling
    is used. If the user wants to instead get an uniformly
    sampled response, then ``adaptive`` kwarg should be passed ``False``
    and ``n`` must be passed as additional kwargs.
    Refer to the parameters of class :class:`sympy.plotting.series.LineOver1DRangeSeries`
    for more details.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the ramp response data is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    kwargs :
        Additional keyword arguments are passed to the underlying
        :class:`sympy.plotting.series.LineOver1DRangeSeries` class.

    Returns
    =======

    tuple : (x, y)
        x = Time-axis values of the points in the ramp response plot. NumPy array.
        y = Amplitude-axis values of the points in the ramp response plot. NumPy array.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When ``lower_limit`` parameter is less than 0.

        When ``slope`` is negative.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import ramp_response_numerical_data
    >>> tf1 = TransferFunction(s, s**2 + 5*s + 8, s)
    >>> ramp_response_numerical_data(tf1)   # doctest: +SKIP
    (([0.0, 0.12166980856813935,..., 9.861246379582118, 10.0],
    [1.4504508011325967e-09, 0.006046440489058766,..., 0.12499999999568202, 0.12499999999661349]))

    See Also
    ========

    ramp_response_plot

    """
    if slope < 0:
        raise ValueError("Slope must be greater than or equal"
            " to zero.")
    if lower_limit < 0:
        raise ValueError("Lower limit of time must be greater "
            "than or equal to zero.")
    _check_system(system)
    _x = Dummy("x")
    expr = (slope*system.to_expr())/((system.var)**2)
    expr = apart(expr, system.var, full=True)
    _y = _fast_inverse_laplace(expr, system.var, _x).evalf(prec)
    return LineOver1DRangeSeries(_y, (_x, lower_limit, upper_limit),
        **kwargs).get_points()


def ramp_response_plot(system, slope=1, color='b', prec=8, lower_limit=0,
    upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    r"""
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line
    passing through origin ($f(x) = mx$). The slope of
    the ramp function can be varied by the user and
    the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import ramp_response_plot
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, impulse_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    x, y = ramp_response_numerical_data(system, slope=slope, prec=prec,
        lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Ramp Response of ${latex(system)}$ [Slope = {slope}]', pad=20)

    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_magnitude_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', **kwargs):
    """
    Returns the numerical data of the Bode magnitude plot of the system.
    It is internally used by ``bode_magnitude_plot`` to get the data
    for plotting Bode magnitude plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode magnitude plot.
        y = y-axis values of the Bode magnitude plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_magnitude_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_magnitude_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.5148378120533502e-05,..., 68437.36188804005, 100000.0],
    [-6.020599914256786, -6.0205999155219505,..., -193.4117304087953, -200.00000000260573])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)

    x, y = LineOver1DRangeSeries(mag,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    return x, y


def bode_magnitude_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', **kwargs):
    r"""
    Returns the Bode magnitude plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    x, y = bode_magnitude_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')


    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Bode Plot (Magnitude) of ${latex(system)}$', pad=20)

    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_phase_numerical_data(system, initial_exp=-5, final_exp=5, freq_unit='rad/sec', phase_unit='rad', phase_unwrap = True, **kwargs):
    """
    Returns the numerical data of the Bode phase plot of the system.
    It is internally used by ``bode_phase_plot`` to get the data
    for plotting Bode phase plot. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the Bode phase plot data is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and '``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.
    phase_unwrap : bool, optional
        Set to ``True`` by default.

    Returns
    =======

    tuple : (x, y)
        x = x-axis values of the Bode phase plot.
        y = y-axis values of the Bode phase plot.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

        When incorrect frequency or phase units are given as input.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import bode_phase_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> bode_phase_numerical_data(tf1)   # doctest: +SKIP
    ([1e-05, 1.4472354033813751e-05, 2.035581932165858e-05,..., 47577.3248186011, 67884.09326036123, 100000.0],
    [-2.5000000000291665e-05, -3.6180885085e-05, -5.08895483066e-05,...,-3.1415085799262523, -3.14155265358979])

    See Also
    ========

    bode_magnitude_plot, bode_phase_numerical_data

    """
    _check_system(system)
    expr = system.to_expr()
    freq_units = ('rad/sec', 'Hz')
    phase_units = ('rad', 'deg')
    if freq_unit not in freq_units:
        raise ValueError('Only "rad/sec" and "Hz" are accepted frequency units.')
    if phase_unit not in phase_units:
        raise ValueError('Only "rad" and "deg" are accepted phase units.')

    _w = Dummy("w", real=True)
    if freq_unit == 'Hz':
        repl = I*_w*2*pi
    else:
        repl = I*_w
    w_expr = expr.subs({system.var: repl})

    if phase_unit == 'deg':
        phase = arg(w_expr)*180/pi
    else:
        phase = arg(w_expr)

    x, y = LineOver1DRangeSeries(phase,
        (_w, 10**initial_exp, 10**final_exp), xscale='log', **kwargs).get_points()

    half = None
    if phase_unwrap:
        if(phase_unit == 'rad'):
            half = pi
        elif(phase_unit == 'deg'):
            half = 180
    if half:
        unit = 2*half
        for i in range(1, len(y)):
            diff = y[i] - y[i - 1]
            if diff > half:      # Jump from -half to half
                y[i] = (y[i] - unit)
            elif diff < -half:   # Jump from half to -half
                y[i] = (y[i] + unit)

    return x, y


def bode_phase_plot(system, initial_exp=-5, final_exp=5,
    color='b', show_axes=False, grid=True, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase plot of a continuous-time system.

    See ``bode_plot`` for all the parameters.
    """
    x, y = bode_phase_numerical_data(system, initial_exp=initial_exp,
        final_exp=final_exp, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap)
    plt.plot(x, y, color=color, **kwargs)
    plt.xscale('log')

    plt.xlabel('Frequency (%s) [Log Scale]' % freq_unit)
    plt.ylabel('Phase (%s)' % phase_unit)
    plt.title(f'Bode Plot (Phase) of ${latex(system)}$', pad=20)

    if grid:
        plt.grid(True)
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return

    return plt


def bode_plot(system, initial_exp=-5, final_exp=5,
    grid=True, show_axes=False, show=True, freq_unit='rad/sec', phase_unit='rad', phase_unwrap=True, **kwargs):
    r"""
    Returns the Bode phase and magnitude plots of a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Bode Plot is to be computed.
    initial_exp : Number, optional
        The initial exponent of 10 of the semilog plot. Defaults to -5.
    final_exp : Number, optional
        The final exponent of 10 of the semilog plot. Defaults to 5.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    freq_unit : string, optional
        User can choose between ``'rad/sec'`` (radians/second) and ``'Hz'`` (Hertz) as frequency units.
    phase_unit : string, optional
        User can choose between ``'rad'`` (radians) and ``'deg'`` (degree) as phase units.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import bode_plot
        >>> tf1 = TransferFunction(1*s**2 + 0.1*s + 7.5, 1*s**4 + 0.12*s**3 + 9*s**2, s)
        >>> bode_plot(tf1, initial_exp=0.2, final_exp=0.7)   # doctest: +SKIP

    See Also
    ========

    bode_magnitude_plot, bode_phase_plot

    """
    plt.subplot(211)
    mag = bode_magnitude_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes,
        freq_unit=freq_unit, **kwargs)
    mag.title(f'Bode Plot of ${latex(system)}$', pad=20)
    mag.xlabel(None)
    plt.subplot(212)
    bode_phase_plot(system, initial_exp=initial_exp, final_exp=final_exp,
        show=False, grid=grid, show_axes=show_axes, freq_unit=freq_unit, phase_unit=phase_unit, phase_unwrap=phase_unwrap, **kwargs).title(None)

    if show:
        plt.show()
        return

    return plt


def nyquist_plot_expr(system):
    """Function to get the expression for Nyquist plot."""
    s = system.var
    w = Dummy('w', real=True)
    repl = I * w
    expr = system.to_expr()
    w_expr = expr.subs({s: repl})
    w_expr = w_expr.as_real_imag()
    real_expr = w_expr[0]
    imag_expr = w_expr[1]
    return real_expr, imag_expr, w


def nichols_plot_expr(system):
    """Function to get the expression for Nichols plot."""
    s = system.var
    w = Dummy('w', real=True)
    sys_expr = system.to_expr()
    H_jw = sys_expr.subs(s, I*w)
    mag_expr = Abs(H_jw)
    mag_dB_expr = 20*log(mag_expr, 10)
    phase_expr = arg(H_jw)
    phase_deg_expr = deg(phase_expr)
    return mag_dB_expr, phase_deg_expr, w


def nyquist_plot(system, initial_omega=0.01, final_omega=100, show=True,
                color='b', **kwargs):
    r"""
    Generates the Nyquist plot for a continuous-time system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The LTI SISO system for which the Nyquist plot is to be generated.
    initial_omega : float, optional
        The starting frequency value. Defaults to 0.01.
    final_omega : float, optional
        The ending frequency value. Defaults to 100.
    show : bool, optional
        If True, the plot is displayed. Default is True.
    color : str, optional
        The color of the Nyquist plot. Default is 'b' (blue).
    grid : bool, optional
        If True, grid lines are displayed. Default is False.
    **kwargs
        Additional keyword arguments for customization.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import nyquist_plot
        >>> tf1 = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
        >>> nyquist_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    nichols_plot, bode_plot

    """
    _check_system(system)
    real_expr, imag_expr, w = nyquist_plot_expr(system)
    w_values = [(w, initial_omega, final_omega)]
    p = plot_parametric(
        (real_expr, imag_expr),   # The curve
        (real_expr, -imag_expr),  # Its mirror image
        *w_values,
        show=False,
        line_color=color,
        adaptive=True,
        title=f'Nyquist Plot of ${latex(system)}$',
        xlabel='Real Axis',
        ylabel='Imaginary Axis',
        size=(6, 5),
        kwargs=kwargs)
    if show:
        p.show()
        return
    return p


def nichols_plot(system, initial_omega=0.01, final_omega=100, show=True, color='b', **kwargs):
    r"""
    Generates the Nichols plot for a LTI system.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The LTI SISO system for which the Nyquist plot is to be generated.
    initial_omega : float, optional
        The starting frequency value. Defaults to 0.01.
    final_omega : float, optional
        The ending frequency value. Defaults to 100.
    show : bool, optional
        If True, the plot is displayed. Default is True.
    color : str, optional
        The color of the Nyquist plot. Default is 'b' (blue).
    grid : bool, optional
        If True, grid lines are displayed. Default is False.
    **kwargs
        Additional keyword arguments for customization.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import nichols_plot
        >>> tf1 = TransferFunction(1.5, s**2+14*s+40.02, s)
        >>> nichols_plot(tf1)   # doctest: +SKIP

    See Also
    ========

    nyquist_plot, bode_plot

    """
    _check_system(system)
    magnitude_dB_expr, phase_deg_expr, w = nichols_plot_expr(system)
    w_values = [(w, initial_omega, final_omega)]
    p = plot_parametric(
        (phase_deg_expr, magnitude_dB_expr),
        *w_values,
        show=False,
        line_color=color,
        title=f'Nichols Plot of ${latex(system)}$',
        xlabel='Phase [deg]',
        ylabel='Magnitude [dB]',
        size=(6,5),
        kwargs=kwargs)
    if show:
        p.show()
        return
    return p
<|endoftext|>"""
Module defining unit prefixe class and some constants.

Constant dict for SI and binary prefixes are defined as PREFIXES and
BIN_PREFIXES.
"""
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.singleton import S

class Prefix(Expr):
    """
    This class represent prefixes, with their name, symbol and factor.

    Prefixes are used to create derived units from a given unit. They should
    always be encapsulated into units.

    The factor is constructed from a base (default is 10) to some power, and
    it gives the total multiple or fraction. For example the kilometer km
    is constructed from the meter (factor 1) and the kilo (10 to the power 3,
    i.e. 1000). The base can be changed to allow e.g. binary prefixes.

    A prefix multiplied by something will always return the product of this
    other object times the factor, except if the other object:

    - is a prefix and they can be combined into a new prefix;
    - defines multiplication with prefixes (which is the case for the Unit
      class).
    """
    _op_priority = 13.0
    is_commutative = True

    def __new__(cls, name, abbrev, exponent, base=sympify(10), latex_repr=None):

        name = sympify(name)
        abbrev = sympify(abbrev)
        exponent = sympify(exponent)
        base = sympify(base)

        obj = Expr.__new__(cls, name, abbrev, exponent, base)
        obj._name = name
        obj._abbrev = abbrev
        obj._scale_factor = base**exponent
        obj._exponent = exponent
        obj._base = base
        obj._latex_repr = latex_repr
        return obj

    @property
    def name(self):
        return self._name

    @property
    def abbrev(self):
        return self._abbrev

    @property
    def scale_factor(self):
        return self._scale_factor

    def _latex(self, printer):
        if self._latex_repr is None:
            return r'\text{%s}' % self._abbrev
        return self._latex_repr

    @property
    def base(self):
        return self._base

    def __str__(self):
        return str(self._abbrev)

    def __repr__(self):
        if self.base == 10:
            return "Prefix(%r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent)
        else:
            return "Prefix(%r, %r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent, self.base)

    def __mul__(self, other):
        from sympy.physics.units import Quantity
        if not isinstance(other, (Quantity, Prefix)):
            return super().__mul__(other)

        fact = self.scale_factor * other.scale_factor

        if isinstance(other, Prefix):
            if fact == 1:
                return S.One
            # simplify prefix
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        return self.scale_factor * other

    def __truediv__(self, other):
        if not hasattr(other, "scale_factor"):
            return super().__truediv__(other)

        fact = self.scale_factor / other.scale_factor

        if fact == 1:
            return S.One
        elif isinstance(other, Prefix):
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        return self.scale_factor / other

    def __rtruediv__(self, other):
        if other == 1:
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == 1 / self.scale_factor:
                    return PREFIXES[p]
        return other / self.scale_factor


def prefix_unit(unit, prefixes):
    """
    Return a list of all units formed by unit and the given prefixes.

    You can use the predefined PREFIXES or BIN_PREFIXES, but you can also
    pass as argument a subdict of them if you do not want all prefixed units.

        >>> from sympy.physics.units.prefixes import (PREFIXES,
        ...                                                 prefix_unit)
        >>> from sympy.physics.units import m
        >>> pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}
        >>> prefix_unit(m, pref)  # doctest: +SKIP
        [millimeter, centimeter, decimeter]
    """

    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units import UnitSystem

    prefixed_units = []

    for prefix in prefixes.values():
        quantity = Quantity(
                "%s%s" % (prefix.name, unit.name),
                abbrev=("%s%s" % (prefix.abbrev, unit.abbrev)),
                is_prefixed=True,
           )
        UnitSystem._quantity_dimensional_equivalence_map_global[quantity] = unit
        UnitSystem._quantity_scale_factors_global[quantity] = (prefix.scale_factor, unit)
        prefixed_units.append(quantity)

    return prefixed_units


yotta = Prefix('yotta', 'Y', 24)
zetta = Prefix('zetta', 'Z', 21)
exa = Prefix('exa', 'E', 18)
peta = Prefix('peta', 'P', 15)
tera = Prefix('tera', 'T', 12)
giga = Prefix('giga', 'G', 9)
mega = Prefix('mega', 'M', 6)
kilo = Prefix('kilo', 'k', 3)
hecto = Prefix('hecto', 'h', 2)
deca = Prefix('deca', 'da', 1)
deci = Prefix('deci', 'd', -1)
centi = Prefix('centi', 'c', -2)
milli = Prefix('milli', 'm', -3)
micro = Prefix('micro', 'mu', -6, latex_repr=r"\mu")
nano = Prefix('nano', 'n', -9)
pico = Prefix('pico', 'p', -12)
femto = Prefix('femto', 'f', -15)
atto = Prefix('atto', 'a', -18)
zepto = Prefix('zepto', 'z', -21)
yocto = Prefix('yocto', 'y', -24)


# https://physics.nist.gov/cuu/Units/prefixes.html
PREFIXES = {
    'Y': yotta,
    'Z': zetta,
    'E': exa,
    'P': peta,
    'T': tera,
    'G': giga,
    'M': mega,
    'k': kilo,
    'h': hecto,
    'da': deca,
    'd': deci,
    'c': centi,
    'm': milli,
    'mu': micro,
    'n': nano,
    'p': pico,
    'f': femto,
    'a': atto,
    'z': zepto,
    'y': yocto,
}


kibi = Prefix('kibi', 'Y', 10, 2)
mebi = Prefix('mebi', 'Y', 20, 2)
gibi = Prefix('gibi', 'Y', 30, 2)
tebi = Prefix('tebi', 'Y', 40, 2)
pebi = Prefix('pebi', 'Y', 50, 2)
exbi = Prefix('exbi', 'Y', 60, 2)


# https://physics.nist.gov/cuu/Units/binary.html
BIN_PREFIXES = {
    'Ki': kibi,
    'Mi': mebi,
    'Gi': gibi,
    'Ti': tebi,
    'Pi': pebi,
    'Ei': exbi,
}
<|endoftext|>"""
Physical quantities.
"""

from sympy.core.expr import AtomicExpr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.prefixes import Prefix


class Quantity(AtomicExpr):
    """
    Physical quantity: can be a unit of measure, a constant or a generic quantity.
    """

    is_commutative = True
    is_real = True
    is_number = False
    is_nonzero = True
    is_physical_constant = False
    _diff_wrt = True

    def __new__(cls, name, abbrev=None,
                latex_repr=None, pretty_unicode_repr=None,
                pretty_ascii_repr=None, mathml_presentation_repr=None,
                is_prefixed=False,
                **assumptions):

        if not isinstance(name, Symbol):
            name = Symbol(name)

        if abbrev is None:
            abbrev = name
        elif isinstance(abbrev, str):
            abbrev = Symbol(abbrev)

        # HACK: These are here purely for type checking. They actually get assigned below.
        cls._is_prefixed = is_prefixed

        obj = AtomicExpr.__new__(cls, name, abbrev)
        obj._name = name
        obj._abbrev = abbrev
        obj._latex_repr = latex_repr
        obj._unicode_repr = pretty_unicode_repr
        obj._ascii_repr = pretty_ascii_repr
        obj._mathml_repr = mathml_presentation_repr
        obj._is_prefixed = is_prefixed
        return obj

    def set_global_dimension(self, dimension):
        _QuantityMapper._quantity_dimension_global[self] = dimension

    def set_global_relative_scale_factor(self, scale_factor, reference_quantity):
        """
        Setting a scale factor that is valid across all unit system.
        """
        from sympy.physics.units import UnitSystem
        scale_factor = sympify(scale_factor)
        if isinstance(scale_factor, Prefix):
            self._is_prefixed = True
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        scale_factor = sympify(scale_factor)
        UnitSystem._quantity_scale_factors_global[self] = (scale_factor, reference_quantity)
        UnitSystem._quantity_dimensional_equivalence_map_global[self] = reference_quantity

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_dimension(self)

    @property
    def abbrev(self):
        """
        Symbol representing the unit name.

        Prepend the abbreviation with the prefix symbol if it is defines.
        """
        return self._abbrev

    @property
    def scale_factor(self):
        """
        Overall magnitude of the quantity as compared to the canonical units.
        """
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_scale_factor(self)

    def _eval_is_positive(self):
        return True

    def _eval_is_constant(self):
        return True

    def _eval_Abs(self):
        return self

    def _eval_subs(self, old, new):
        if isinstance(new, Quantity) and self != old:
            return self

    def _latex(self, printer):
        if self._latex_repr:
            return self._latex_repr
        else:
            return r'\text{{{}}}'.format(self.args[1] \
                          if len(self.args) >= 2 else self.args[0])

    def convert_to(self, other, unit_system="SI"):
        """
        Convert the quantity to another quantity of same dimensions.

        Examples
        ========

        >>> from sympy.physics.units import speed_of_light, meter, second
        >>> speed_of_light
        speed_of_light
        >>> speed_of_light.convert_to(meter/second)
        299792458*meter/second

        >>> from sympy.physics.units import liter
        >>> liter.convert_to(meter**3)
        meter**3/1000
        """
        from .util import convert_to
        return convert_to(self, other, unit_system)

    @property
    def free_symbols(self):
        """Return free symbols from quantity."""
        return set()

    @property
    def is_prefixed(self):
        """Whether or not the quantity is prefixed. Eg. `kilogram` is prefixed, but `gram` is not."""
        return self._is_prefixed

class PhysicalConstant(Quantity):
    """Represents a physical constant, eg. `speed_of_light` or `avogadro_constant`."""

    is_physical_constant = True
<|endoftext|>"""
Unit system for physical quantities; include definition of constants.
"""
from __future__ import annotations

from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.quantities import Quantity

from .dimensions import Dimension


class UnitSystem(_QuantityMapper):
    """
    UnitSystem represents a coherent set of units.

    A unit system is basically a dimension system with notions of scales. Many
    of the methods are defined in the same way.

    It is much better if all base units have a symbol.
    """

    _unit_systems: dict[str, UnitSystem] = {}

    def __init__(self, base_units, units=(), name="", descr="", dimension_system=None, derived_units: dict[Dimension, Quantity]={}):

        UnitSystem._unit_systems[name] = self

        self.name = name
        self.descr = descr

        self._base_units = base_units
        self._dimension_system = dimension_system
        self._units = tuple(set(base_units) | set(units))
        self._base_units = tuple(base_units)
        self._derived_units = derived_units

        super().__init__()

    def __str__(self):
        """
        Return the name of the system.

        If it does not exist, then it makes a list of symbols (or names) of
        the base dimensions.
        """

        if self.name != "":
            return self.name
        else:
            return "UnitSystem((%s))" % ", ".join(
                str(d) for d in self._base_units)

    def __repr__(self):
        return '<UnitSystem: %s>' % repr(self._base_units)

    def extend(self, base, units=(), name="", description="", dimension_system=None, derived_units: dict[Dimension, Quantity]={}):
        """Extend the current system into a new one.

        Take the base and normal units of the current system to merge
        them to the base and normal units given in argument.
        If not provided, name and description are overridden by empty strings.
        """

        base = self._base_units + tuple(base)
        units = self._units + tuple(units)

        return UnitSystem(base, units, name, description, dimension_system, {**self._derived_units, **derived_units})

    def get_dimension_system(self):
        return self._dimension_system

    def get_quantity_dimension(self, unit):
        qdm = self.get_dimension_system()._quantity_dimension_map
        if unit in qdm:
            return qdm[unit]
        return super().get_quantity_dimension(unit)

    def get_quantity_scale_factor(self, unit):
        qsfm = self.get_dimension_system()._quantity_scale_factors
        if unit in qsfm:
            return qsfm[unit]
        return super().get_quantity_scale_factor(unit)

    @staticmethod
    def get_unit_system(unit_system):
        if isinstance(unit_system, UnitSystem):
            return unit_system

        if unit_system not in UnitSystem._unit_systems:
            raise ValueError(
                "Unit system is not supported. Currently"
                "supported unit systems are {}".format(
                    ", ".join(sorted(UnitSystem._unit_systems))
                )
            )

        return UnitSystem._unit_systems[unit_system]

    @staticmethod
    def get_default_unit_system():
        return UnitSystem._unit_systems["SI"]

    @property
    def dim(self):
        """
        Give the dimension of the system.

        That is return the number of units forming the basis.
        """
        return len(self._base_units)

    @property
    def is_consistent(self):
        """
        Check if the underlying dimension system is consistent.
        """
        # test is performed in DimensionSystem
        return self.get_dimension_system().is_consistent

    @property
    def derived_units(self) -> dict[Dimension, Quantity]:
        return self._derived_units

    def get_dimensional_expr(self, expr):
        from sympy.physics.units import Quantity
        if isinstance(expr, Mul):
            return Mul(*[self.get_dimensional_expr(i) for i in expr.args])
        elif isinstance(expr, Pow):
            return self.get_dimensional_expr(expr.base) ** expr.exp
        elif isinstance(expr, Add):
            return self.get_dimensional_expr(expr.args[0])
        elif isinstance(expr, Derivative):
            dim = self.get_dimensional_expr(expr.expr)
            for independent, count in expr.variable_count:
                dim /= self.get_dimensional_expr(independent)**count
            return dim
        elif isinstance(expr, Function):
            args = [self.get_dimensional_expr(arg) for arg in expr.args]
            if all(i == 1 for i in args):
                return S.One
            return expr.func(*args)
        elif isinstance(expr, Quantity):
            return self.get_quantity_dimension(expr).name
        return S.One

    def _collect_factor_and_dimension(self, expr):
        """
        Return tuple with scale factor expression and dimension expression.
        """
        from sympy.physics.units import Quantity
        if isinstance(expr, Quantity):
            return expr.scale_factor, expr.dimension
        elif isinstance(expr, Mul):
            factor = 1
            dimension = Dimension(1)
            for arg in expr.args:
                arg_factor, arg_dim = self._collect_factor_and_dimension(arg)
                factor *= arg_factor
                dimension *= arg_dim
            return factor, dimension
        elif isinstance(expr, Pow):
            factor, dim = self._collect_factor_and_dimension(expr.base)
            exp_factor, exp_dim = self._collect_factor_and_dimension(expr.exp)
            if self.get_dimension_system().is_dimensionless(exp_dim):
                exp_dim = 1
            return factor ** exp_factor, dim ** (exp_factor * exp_dim)
        elif isinstance(expr, Add):
            factor, dim = self._collect_factor_and_dimension(expr.args[0])
            for addend in expr.args[1:]:
                addend_factor, addend_dim = \
                    self._collect_factor_and_dimension(addend)
                if not self.get_dimension_system().equivalent_dims(dim, addend_dim):
                    raise ValueError(
                        'Dimension of "{}" is {}, '
                        'but it should be {}'.format(
                            addend, addend_dim, dim))
                factor += addend_factor
            return factor, dim
        elif isinstance(expr, Derivative):
            factor, dim = self._collect_factor_and_dimension(expr.args[0])
            for independent, count in expr.variable_count:
                ifactor, idim = self._collect_factor_and_dimension(independent)
                factor /= ifactor**count
                dim /= idim**count
            return factor, dim
        elif isinstance(expr, Function):
            fds = [self._collect_factor_and_dimension(arg) for arg in expr.args]
            dims = [Dimension(1) if self.get_dimension_system().is_dimensionless(d[1]) else d[1] for d in fds]
            return (expr.func(*(f[0] for f in fds)), *dims)
        elif isinstance(expr, Dimension):
            return S.One, expr
        else:
            return expr, Dimension(1)

    def get_units_non_prefixed(self) -> set[Quantity]:
        """
        Return the units of the system that do not have a prefix.
        """
        return set(filter(lambda u: not u.is_prefixed and not u.is_physical_constant, self._units))
<|endoftext|>"""
Several methods to simplify expressions involving unit objects.
"""
from functools import reduce
from collections.abc import Iterable
from typing import Optional

from sympy import default_sort_key
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.core.function import Function
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.physics.units.dimensions import Dimension, DimensionSystem
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.utilities.iterables import sift


def _get_conversion_matrix_for_expr(expr, target_units, unit_system):
    from sympy.matrices.dense import Matrix

    dimension_system = unit_system.get_dimension_system()

    expr_dim = Dimension(unit_system.get_dimensional_expr(expr))
    dim_dependencies = dimension_system.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)
    target_dims = [Dimension(unit_system.get_dimensional_expr(x)) for x in target_units]
    canon_dim_units = [i for x in target_dims for i in dimension_system.get_dimensional_dependencies(x, mark_dimensionless=True)]
    canon_expr_units = set(dim_dependencies)

    if not canon_expr_units.issubset(set(canon_dim_units)):
        return None

    seen = set()
    canon_dim_units = [i for i in canon_dim_units if not (i in seen or seen.add(i))]

    camat = Matrix([[dimension_system.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims] for j in canon_dim_units])
    exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])

    try:
        res_exponents = camat.solve(exprmat)
    except NonInvertibleMatrixError:
        return None

    return res_exponents


def convert_to(expr, target_units, unit_system="SI"):
    """
    Convert ``expr`` to the same expression with all of its units and quantities
    represented as factors of ``target_units``, whenever the dimension is compatible.

    ``target_units`` may be a single unit/quantity, or a collection of
    units/quantities.

    Examples
    ========

    >>> from sympy.physics.units import speed_of_light, meter, gram, second, day
    >>> from sympy.physics.units import mile, newton, kilogram, atomic_mass_constant
    >>> from sympy.physics.units import kilometer, centimeter
    >>> from sympy.physics.units import gravitational_constant, hbar
    >>> from sympy.physics.units import convert_to
    >>> convert_to(mile, kilometer)
    25146*kilometer/15625
    >>> convert_to(mile, kilometer).n()
    1.609344*kilometer
    >>> convert_to(speed_of_light, meter/second)
    299792458*meter/second
    >>> convert_to(day, second)
    86400*second
    >>> 3*newton
    3*newton
    >>> convert_to(3*newton, kilogram*meter/second**2)
    3*kilogram*meter/second**2
    >>> convert_to(atomic_mass_constant, gram)
    1.660539060e-24*gram

    Conversion to multiple units:

    >>> convert_to(speed_of_light, [meter, second])
    299792458*meter/second
    >>> convert_to(3*newton, [centimeter, gram, second])
    300000*centimeter*gram/second**2

    Conversion to Planck units:

    >>> convert_to(atomic_mass_constant, [gravitational_constant, speed_of_light, hbar]).n()
    7.62963087839509e-20*hbar**0.5*speed_of_light**0.5/gravitational_constant**0.5

    """
    from sympy.physics.units import UnitSystem
    unit_system = UnitSystem.get_unit_system(unit_system)

    if not isinstance(target_units, (Iterable, Tuple)):
        target_units = [target_units]

    def handle_Adds(expr):
        return Add.fromiter(convert_to(i, target_units, unit_system)
            for i in expr.args)

    if isinstance(expr, Add):
        return handle_Adds(expr)
    elif isinstance(expr, Pow) and isinstance(expr.base, Add):
        return handle_Adds(expr.base) ** expr.exp

    expr = sympify(expr)
    target_units = sympify(target_units)

    if isinstance(expr, Function):
        expr = expr.together()

    if not isinstance(expr, Quantity) and expr.has(Quantity):
        expr = expr.replace(lambda x: isinstance(x, Quantity),
            lambda x: x.convert_to(target_units, unit_system))

    def get_total_scale_factor(expr):
        if isinstance(expr, Mul):
            return reduce(lambda x, y: x * y,
                [get_total_scale_factor(i) for i in expr.args])
        elif isinstance(expr, Pow):
            return get_total_scale_factor(expr.base) ** expr.exp
        elif isinstance(expr, Quantity):
            return unit_system.get_quantity_scale_factor(expr)
        return expr

    depmat = _get_conversion_matrix_for_expr(expr, target_units, unit_system)
    if depmat is None:
        return expr

    expr_scale_factor = get_total_scale_factor(expr)
    return expr_scale_factor * Mul.fromiter(
        (1/get_total_scale_factor(u)*u)**p for u, p in
        zip(target_units, depmat))


def quantity_simplify(expr, across_dimensions: bool=False, unit_system=None):
    """Return an equivalent expression in which prefixes are replaced
    with numerical values and all units of a given dimension are the
    unified in a canonical manner by default. `across_dimensions` allows
    for units of different dimensions to be simplified together.

    `unit_system` must be specified if `across_dimensions` is True.

    Examples
    ========

    >>> from sympy.physics.units.util import quantity_simplify
    >>> from sympy.physics.units.prefixes import kilo
    >>> from sympy.physics.units import foot, inch, joule, coulomb
    >>> quantity_simplify(kilo*foot*inch)
    250*foot**2/3
    >>> quantity_simplify(foot - 6*inch)
    foot/2
    >>> quantity_simplify(5*joule/coulomb, across_dimensions=True, unit_system="SI")
    5*volt
    """

    if expr.is_Atom or not expr.has(Prefix, Quantity):
        return expr

    # replace all prefixes with numerical values
    p = expr.atoms(Prefix)
    expr = expr.xreplace({p: p.scale_factor for p in p})

    # replace all quantities of given dimension with a canonical
    # quantity, chosen from those in the expression
    d = sift(expr.atoms(Quantity), lambda i: i.dimension)
    for k in d:
        if len(d[k]) == 1:
            continue
        v = list(ordered(d[k]))
        ref = v[0]/v[0].scale_factor
        expr = expr.xreplace({vi: ref*vi.scale_factor for vi in v[1:]})

    if across_dimensions:
        # combine quantities of different dimensions into a single
        # quantity that is equivalent to the original expression

        if unit_system is None:
            raise ValueError("unit_system must be specified if across_dimensions is True")

        unit_system = UnitSystem.get_unit_system(unit_system)
        dimension_system: DimensionSystem = unit_system.get_dimension_system()
        dim_expr = unit_system.get_dimensional_expr(expr)
        dim_deps = dimension_system.get_dimensional_dependencies(dim_expr, mark_dimensionless=True)

        target_dimension: Optional[Dimension] = None
        for ds_dim, ds_dim_deps in dimension_system.dimensional_dependencies.items():
            if ds_dim_deps == dim_deps:
                target_dimension = ds_dim
                break

        if target_dimension is None:
            # if we can't find a target dimension, we can't do anything. unsure how to handle this case.
            return expr

        target_unit = unit_system.derived_units.get(target_dimension)
        if target_unit:
            expr = convert_to(expr, target_unit, unit_system)

    return expr


def check_dimensions(expr, unit_system="SI"):
    """Return expr if units in addends have the same
    base dimensions, else raise a ValueError."""
    # the case of adding a number to a dimensional quantity
    # is ignored for the sake of SymPy core routines, so this
    # function will raise an error now if such an addend is
    # found.
    # Also, when doing substitutions, multiplicative constants
    # might be introduced, so remove those now

    from sympy.physics.units import UnitSystem
    unit_system = UnitSystem.get_unit_system(unit_system)

    def addDict(dict1, dict2):
        """Merge dictionaries by adding values of common keys and
        removing keys with value of 0."""
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
            if key in dict1 and key in dict2:
                dict3[key] = value + dict1[key]
        return {key:val for key, val in dict3.items() if val != 0}

    adds = expr.atoms(Add)
    DIM_OF = unit_system.get_dimension_system().get_dimensional_dependencies
    for a in adds:
        deset = set()
        for ai in a.args:
            if ai.is_number:
                deset.add(())
                continue
            dims = []
            skip = False
            dimdict = {}
            for i in Mul.make_args(ai):
                if i.has(Quantity):
                    i = Dimension(unit_system.get_dimensional_expr(i))
                if i.has(Dimension):
                    dimdict = addDict(dimdict, DIM_OF(i))
                elif i.free_symbols:
                    skip = True
                    break
            dims.extend(dimdict.items())
            if not skip:
                deset.add(tuple(sorted(dims, key=default_sort_key)))
                if len(deset) > 1:
                    raise ValueError(
                        "addends have incompatible dimensions: {}".format(deset))

    # clear multiplicative constants on Dimensions which may be
    # left after substitution
    reps = {}
    for m in expr.atoms(Mul):
        if any(isinstance(i, Dimension) for i in m.args):
            reps[m] = m.func(*[
                i for i in m.args if not i.is_number])

    return expr.xreplace(reps)
<|endoftext|>"""
Definition of physical dimensions.

Unit systems will be constructed on top of these dimensions.

Most of the examples in the doc use MKS system and are presented from the
computer point of view: from a human point, adding length to time is not legal
in MKS but it is in natural system; for a computer in natural system there is
no time dimension (but a velocity dimension instead) - in the basis - so the
question of adding time to length has no meaning.
"""

from __future__ import annotations

import collections
from functools import reduce

from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow


class _QuantityMapper:

    _quantity_scale_factors_global: dict[Expr, Expr] = {}
    _quantity_dimensional_equivalence_map_global: dict[Expr, Expr] = {}
    _quantity_dimension_global: dict[Expr, Expr] = {}

    def __init__(self, *args, **kwargs):
        self._quantity_dimension_map = {}
        self._quantity_scale_factors = {}

    def set_quantity_dimension(self, quantity, dimension):
        """
        Set the dimension for the quantity in a unit system.

        If this relation is valid in every unit system, use
        ``quantity.set_global_dimension(dimension)`` instead.
        """
        from sympy.physics.units import Quantity
        dimension = sympify(dimension)
        if not isinstance(dimension, Dimension):
            if dimension == 1:
                dimension = Dimension(1)
            else:
                raise ValueError("expected dimension or 1")
        elif isinstance(dimension, Quantity):
            dimension = self.get_quantity_dimension(dimension)
        self._quantity_dimension_map[quantity] = dimension

    def set_quantity_scale_factor(self, quantity, scale_factor):
        """
        Set the scale factor of a quantity relative to another quantity.

        It should be used only once per quantity to just one other quantity,
        the algorithm will then be able to compute the scale factors to all
        other quantities.

        In case the scale factor is valid in every unit system, please use
        ``quantity.set_global_relative_scale_factor(scale_factor)`` instead.
        """
        from sympy.physics.units import Quantity
        from sympy.physics.units.prefixes import Prefix
        scale_factor = sympify(scale_factor)
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        # replace all quantities by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Quantity),
            lambda x: self.get_quantity_scale_factor(x)
        )
        self._quantity_scale_factors[quantity] = scale_factor

    def get_quantity_dimension(self, unit):
        from sympy.physics.units import Quantity
        # First look-up the local dimension map, then the global one:
        if unit in self._quantity_dimension_map:
            return self._quantity_dimension_map[unit]
        if unit in self._quantity_dimension_global:
            return self._quantity_dimension_global[unit]
        if unit in self._quantity_dimensional_equivalence_map_global:
            dep_unit = self._quantity_dimensional_equivalence_map_global[unit]
            if isinstance(dep_unit, Quantity):
                return self.get_quantity_dimension(dep_unit)
            else:
                return Dimension(self.get_dimensional_expr(dep_unit))
        if isinstance(unit, Quantity):
            return Dimension(unit.name)
        else:
            return Dimension(1)

    def get_quantity_scale_factor(self, unit):
        if unit in self._quantity_scale_factors:
            return self._quantity_scale_factors[unit]
        if unit in self._quantity_scale_factors_global:
            mul_factor, other_unit = self._quantity_scale_factors_global[unit]
            return mul_factor*self.get_quantity_scale_factor(other_unit)
        return S.One


class Dimension(Expr):
    """
    This class represent the dimension of a physical quantities.

    The ``Dimension`` constructor takes as parameters a name and an optional
    symbol.

    For example, in classical mechanics we know that time is different from
    temperature and dimensions make this difference (but they do not provide
    any measure of these quantities.

        >>> from sympy.physics.units import Dimension
        >>> length = Dimension('length')
        >>> length
        Dimension(length)
        >>> time = Dimension('time')
        >>> time
        Dimension(time)

    Dimensions can be composed using multiplication, division and
    exponentiation (by a number) to give new dimensions. Addition and
    subtraction is defined only when the two objects are the same dimension.

        >>> velocity = length / time
        >>> velocity
        Dimension(length/time)

    It is possible to use a dimension system object to get the dimensionsal
    dependencies of a dimension, for example the dimension system used by the
    SI units convention can be used:

        >>> from sympy.physics.units.systems.si import dimsys_SI
        >>> dimsys_SI.get_dimensional_dependencies(velocity)
        {Dimension(length, L): 1, Dimension(time, T): -1}
        >>> length + length
        Dimension(length)
        >>> l2 = length**2
        >>> l2
        Dimension(length**2)
        >>> dimsys_SI.get_dimensional_dependencies(l2)
        {Dimension(length, L): 2}

    """

    _op_priority = 13.0

    # XXX: This doesn't seem to be used anywhere...
    _dimensional_dependencies = {}  # type: ignore

    is_commutative = True
    is_number = False
    # make sqrt(M**2) --> M
    is_positive = True
    is_real = True

    def __new__(cls, name, symbol=None):

        if isinstance(name, str):
            name = Symbol(name)
        else:
            name = sympify(name)

        if not isinstance(name, Expr):
            raise TypeError("Dimension name needs to be a valid math expression")

        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        elif symbol is not None:
            assert isinstance(symbol, Symbol)

        obj = Expr.__new__(cls, name)

        obj._name = name
        obj._symbol = symbol
        return obj

    @property
    def name(self):
        return self._name

    @property
    def symbol(self):
        return self._symbol

    def __str__(self):
        """
        Display the string representation of the dimension.
        """
        if self.symbol is None:
            return "Dimension(%s)" % (self.name)
        else:
            return "Dimension(%s, %s)" % (self.name, self.symbol)

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return self

    def __add__(self, other):
        from sympy.physics.units.quantities import Quantity
        other = sympify(other)
        if isinstance(other, Basic):
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            if isinstance(other, Dimension) and self == other:
                return self
            return super().__add__(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # there is no notion of ordering (or magnitude) among dimension,
        # subtraction is equivalent to addition when the operation is legal
        return self + other

    def __rsub__(self, other):
        # there is no notion of ordering (or magnitude) among dimension,
        # subtraction is equivalent to addition when the operation is legal
        return self + other

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        other = sympify(other)
        return Dimension(self.name**other)

    def __mul__(self, other):
        from sympy.physics.units.quantities import Quantity
        if isinstance(other, Basic):
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            if isinstance(other, Dimension):
                return Dimension(self.name*other.name)
            if not other.free_symbols:  # other.is_number cannot be used
                return self
            return super().__mul__(other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self*Pow(other, -1)

    def __rtruediv__(self, other):
        return other * pow(self, -1)

    @classmethod
    def _from_dimensional_dependencies(cls, dependencies):
        return reduce(lambda x, y: x * y, (
            d**e for d, e in dependencies.items()
        ), 1)

    def has_integer_powers(self, dim_sys):
        """
        Check if the dimension object has only integer powers.

        All the dimension powers should be integers, but rational powers may
        appear in intermediate steps. This method may be used to check that the
        final result is well-defined.
        """

        return all(dpow.is_Integer for dpow in dim_sys.get_dimensional_dependencies(self).values())


# Create dimensions according to the base units in MKSA.
# For other unit systems, they can be derived by transforming the base
# dimensional dependency dictionary.


class DimensionSystem(Basic, _QuantityMapper):
    r"""
    DimensionSystem represents a coherent set of dimensions.

    The constructor takes three parameters:

    - base dimensions;
    - derived dimensions: these are defined in terms of the base dimensions
      (for example velocity is defined from the division of length by time);
    - dependency of dimensions: how the derived dimensions depend
      on the base dimensions.

    Optionally either the ``derived_dims`` or the ``dimensional_dependencies``
    may be omitted.
    """

    def __new__(cls, base_dims, derived_dims=(), dimensional_dependencies={}):
        dimensional_dependencies = dict(dimensional_dependencies)

        def parse_dim(dim):
            if isinstance(dim, str):
                dim = Dimension(Symbol(dim))
            elif isinstance(dim, Dimension):
                pass
            elif isinstance(dim, Symbol):
                dim = Dimension(dim)
            else:
                raise TypeError("%s wrong type" % dim)
            return dim

        base_dims = [parse_dim(i) for i in base_dims]
        derived_dims = [parse_dim(i) for i in derived_dims]

        for dim in base_dims:
            if (dim in dimensional_dependencies
                and (len(dimensional_dependencies[dim]) != 1 or
                dimensional_dependencies[dim].get(dim, None) != 1)):
                raise IndexError("Repeated value in base dimensions")
            dimensional_dependencies[dim] = Dict({dim: 1})

        def parse_dim_name(dim):
            if isinstance(dim, Dimension):
                return dim
            elif isinstance(dim, str):
                return Dimension(Symbol(dim))
            elif isinstance(dim, Symbol):
                return Dimension(dim)
            else:
                raise TypeError("unrecognized type %s for %s" % (type(dim), dim))

        for dim in dimensional_dependencies.keys():
            dim = parse_dim(dim)
            if (dim not in derived_dims) and (dim not in base_dims):
                derived_dims.append(dim)

        def parse_dict(d):
            return Dict({parse_dim_name(i): j for i, j in d.items()})

        # Make sure everything is a SymPy type:
        dimensional_dependencies = {parse_dim_name(i): parse_dict(j) for i, j in
                                    dimensional_dependencies.items()}

        for dim in derived_dims:
            if dim in base_dims:
                raise ValueError("Dimension %s both in base and derived" % dim)
            if dim not in dimensional_dependencies:
                # TODO: should this raise a warning?
                dimensional_dependencies[dim] = Dict({dim: 1})

        base_dims.sort(key=default_sort_key)
        derived_dims.sort(key=default_sort_key)

        base_dims = Tuple(*base_dims)
        derived_dims = Tuple(*derived_dims)
        dimensional_dependencies = Dict({i: Dict(j) for i, j in dimensional_dependencies.items()})
        obj = Basic.__new__(cls, base_dims, derived_dims, dimensional_dependencies)
        return obj

    @property
    def base_dims(self):
        return self.args[0]

    @property
    def derived_dims(self):
        return self.args[1]

    @property
    def dimensional_dependencies(self):
        return self.args[2]

    def _get_dimensional_dependencies_for_name(self, dimension):
        if isinstance(dimension, str):
            dimension = Dimension(Symbol(dimension))
        elif not isinstance(dimension, Dimension):
            dimension = Dimension(dimension)

        if dimension.name.is_Symbol:
            # Dimensions not included in the dependencies are considered
            # as base dimensions:
            return dict(self.dimensional_dependencies.get(dimension, {dimension: 1}))

        if dimension.name.is_number or dimension.name.is_NumberSymbol:
            return {}

        get_for_name = self._get_dimensional_dependencies_for_name

        if dimension.name.is_Mul:
            ret = collections.defaultdict(int)
            dicts = [get_for_name(i) for i in dimension.name.args]
            for d in dicts:
                for k, v in d.items():
                    ret[k] += v
            return {k: v for (k, v) in ret.items() if v != 0}

        if dimension.name.is_Add:
            dicts = [get_for_name(i) for i in dimension.name.args]
            if all(d == dicts[0] for d in dicts[1:]):
                return dicts[0]
            raise TypeError("Only equivalent dimensions can be added or subtracted.")

        if dimension.name.is_Pow:
            dim_base = get_for_name(dimension.name.base)
            dim_exp = get_for_name(dimension.name.exp)
            if dim_exp == {} or dimension.name.exp.is_Symbol:
                return {k: v * dimension.name.exp for (k, v) in dim_base.items()}
            else:
                raise TypeError("The exponent for the power operator must be a Symbol or dimensionless.")

        if dimension.name.is_Function:
            args = (Dimension._from_dimensional_dependencies(
                get_for_name(arg)) for arg in dimension.name.args)
            result = dimension.name.func(*args)

            dicts = [get_for_name(i) for i in dimension.name.args]

            if isinstance(result, Dimension):
                return self.get_dimensional_dependencies(result)
            elif result.func == dimension.name.func:
                if isinstance(dimension.name, TrigonometricFunction):
                    if dicts[0] in ({}, {Dimension('angle'): 1}):
                        return {}
                    else:
                        raise TypeError("The input argument for the function {} must be dimensionless or have dimensions of angle.".format(dimension.func))
                else:
                    if all(item == {} for item in dicts):
                        return {}
                    else:
                        raise TypeError("The input arguments for the function {} must be dimensionless.".format(dimension.func))
            else:
                return get_for_name(result)

        raise TypeError("Type {} not implemented for get_dimensional_dependencies".format(type(dimension.name)))

    def get_dimensional_dependencies(self, name, mark_dimensionless=False):
        dimdep = self._get_dimensional_dependencies_for_name(name)
        if mark_dimensionless and dimdep == {}:
            return {Dimension(1): 1}
        return dict(dimdep.items())

    def equivalent_dims(self, dim1, dim2):
        deps1 = self.get_dimensional_dependencies(dim1)
        deps2 = self.get_dimensional_dependencies(dim2)
        return deps1 == deps2

    def extend(self, new_base_dims, new_derived_dims=(), new_dim_deps=None):
        deps = dict(self.dimensional_dependencies)
        if new_dim_deps:
            deps.update(new_dim_deps)

        new_dim_sys = DimensionSystem(
            tuple(self.base_dims) + tuple(new_base_dims),
            tuple(self.derived_dims) + tuple(new_derived_dims),
            deps
        )
        new_dim_sys._quantity_dimension_map.update(self._quantity_dimension_map)
        new_dim_sys._quantity_scale_factors.update(self._quantity_scale_factors)
        return new_dim_sys

    def is_dimensionless(self, dimension):
        """
        Check if the dimension object really has a dimension.

        A dimension should have at least one component with non-zero power.
        """
        if dimension.name == 1:
            return True
        return self.get_dimensional_dependencies(dimension) == {}

    @property
    def list_can_dims(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        List all canonical dimension names.
        """
        dimset = set()
        for i in self.base_dims:
            dimset.update(set(self.get_dimensional_dependencies(i).keys()))
        return tuple(sorted(dimset, key=str))

    @property
    def inv_can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Compute the inverse transformation matrix from the base to the
        canonical dimension basis.

        It corresponds to the matrix where columns are the vector of base
        dimensions in canonical basis.

        This matrix will almost never be used because dimensions are always
        defined with respect to the canonical basis, so no work has to be done
        to get them in this basis. Nonetheless if this matrix is not square
        (or not invertible) it means that we have chosen a bad basis.
        """
        matrix = reduce(lambda x, y: x.row_join(y),
                        [self.dim_can_vector(d) for d in self.base_dims])
        return matrix

    @property
    def can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Return the canonical transformation matrix from the canonical to the
        base dimension basis.

        It is the inverse of the matrix computed with inv_can_transf_matrix().
        """

        #TODO: the inversion will fail if the system is inconsistent, for
        #      example if the matrix is not a square
        return reduce(lambda x, y: x.row_join(y),
                      [self.dim_can_vector(d) for d in sorted(self.base_dims, key=str)]
                      ).inv()

    def dim_can_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Dimensional representation in terms of the canonical base dimensions.
        """

        vec = []
        for d in self.list_can_dims:
            vec.append(self.get_dimensional_dependencies(dim).get(d, 0))
        return Matrix(vec)

    def dim_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.


        Vector representation in terms of the base dimensions.
        """
        return self.can_transf_matrix * Matrix(self.dim_can_vector(dim))

    def print_dim_base(self, dim):
        """
        Give the string expression of a dimension in term of the basis symbols.
        """
        dims = self.dim_vector(dim)
        symbols = [i.symbol if i.symbol is not None else i.name for i in self.base_dims]
        res = S.One
        for (s, p) in zip(symbols, dims):
            res *= s**p
        return res

    @property
    def dim(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Give the dimension of the system.

        That is return the number of dimensions forming the basis.
        """
        return len(self.base_dims)

    @property
    def is_consistent(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Check if the system is well defined.
        """

        # not enough or too many base dimensions compared to independent
        # dimensions
        # in vector language: the set of vectors do not form a basis
        return self.inv_can_transf_matrix.is_square
<|endoftext|># isort:skip_file
"""
Dimensional analysis and unit systems.

This module defines dimension/unit systems and physical quantities. It is
based on a group-theoretical construction where dimensions are represented as
vectors (coefficients being the exponents), and units are defined as a dimension
to which we added a scale.

Quantities are built from a factor and a unit, and are the basic objects that
one will use when doing computations.

All objects except systems and prefixes can be used in SymPy expressions.
Note that as part of a CAS, various objects do not combine automatically
under operations.

Details about the implementation can be found in the documentation, and we
will not repeat all the explanations we gave there concerning our approach.
Ideas about future developments can be found on the `Github wiki
<https://github.com/sympy/sympy/wiki/Unit-systems>`_, and you should consult
this page if you are willing to help.

Useful functions:

- ``find_unit``: easily lookup pre-defined units.
- ``convert_to(expr, newunit)``: converts an expression into the same
    expression expressed in another unit.

"""

from .dimensions import Dimension, DimensionSystem
from .unitsystem import UnitSystem
from .util import convert_to
from .quantities import Quantity

from .definitions.dimension_definitions import (
    amount_of_substance, acceleration, action, area,
    capacitance, charge, conductance, current, energy,
    force, frequency, impedance, inductance, length,
    luminous_intensity, magnetic_density,
    magnetic_flux, mass, momentum, power, pressure, temperature, time,
    velocity, voltage, volume
)

Unit = Quantity

speed = velocity
luminosity = luminous_intensity
magnetic_flux_density = magnetic_density
amount = amount_of_substance

from .prefixes import (
    # 10-power based:
    yotta,
    zetta,
    exa,
    peta,
    tera,
    giga,
    mega,
    kilo,
    hecto,
    deca,
    deci,
    centi,
    milli,
    micro,
    nano,
    pico,
    femto,
    atto,
    zepto,
    yocto,
    # 2-power based:
    kibi,
    mebi,
    gibi,
    tebi,
    pebi,
    exbi,
)

from .definitions import (
    percent, percents,
    permille,
    rad, radian, radians,
    deg, degree, degrees,
    sr, steradian, steradians,
    mil, angular_mil, angular_mils,
    m, meter, meters,
    kg, kilogram, kilograms,
    s, second, seconds,
    A, ampere, amperes,
    K, kelvin, kelvins,
    mol, mole, moles,
    cd, candela, candelas,
    g, gram, grams,
    mg, milligram, milligrams,
    ug, microgram, micrograms,
    t, tonne, metric_ton,
    newton, newtons, N,
    joule, joules, J,
    watt, watts, W,
    pascal, pascals, Pa, pa,
    hertz, hz, Hz,
    coulomb, coulombs, C,
    volt, volts, v, V,
    ohm, ohms,
    siemens, S, mho, mhos,
    farad, farads, F,
    henry, henrys, H,
    tesla, teslas, T,
    weber, webers, Wb, wb,
    optical_power, dioptre, D,
    lux, lx,
    katal, kat,
    gray, Gy,
    becquerel, Bq,
    km, kilometer, kilometers,
    dm, decimeter, decimeters,
    cm, centimeter, centimeters,
    mm, millimeter, millimeters,
    um, micrometer, micrometers, micron, microns,
    nm, nanometer, nanometers,
    pm, picometer, picometers,
    ft, foot, feet,
    inch, inches,
    yd, yard, yards,
    mi, mile, miles,
    nmi, nautical_mile, nautical_miles,
    angstrom, angstroms,
    ha, hectare,
    l, L, liter, liters,
    dl, dL, deciliter, deciliters,
    cl, cL, centiliter, centiliters,
    ml, mL, milliliter, milliliters,
    ms, millisecond, milliseconds,
    us, microsecond, microseconds,
    ns, nanosecond, nanoseconds,
    ps, picosecond, picoseconds,
    minute, minutes,
    h, hour, hours,
    day, days,
    anomalistic_year, anomalistic_years,
    sidereal_year, sidereal_years,
    tropical_year, tropical_years,
    common_year, common_years,
    julian_year, julian_years,
    draconic_year, draconic_years,
    gaussian_year, gaussian_years,
    full_moon_cycle, full_moon_cycles,
    year, years,
    G, gravitational_constant,
    c, speed_of_light,
    elementary_charge,
    hbar,
    planck,
    eV, electronvolt, electronvolts,
    avogadro_number,
    avogadro, avogadro_constant,
    boltzmann, boltzmann_constant,
    stefan, stefan_boltzmann_constant,
    R, molar_gas_constant,
    faraday_constant,
    josephson_constant,
    von_klitzing_constant,
    Da, dalton, amu, amus, atomic_mass_unit, atomic_mass_constant,
    me, electron_rest_mass,
    gee, gees, acceleration_due_to_gravity,
    u0, magnetic_constant, vacuum_permeability,
    e0, electric_constant, vacuum_permittivity,
    Z0, vacuum_impedance,
    coulomb_constant, electric_force_constant,
    atmosphere, atmospheres, atm,
    kPa,
    bar, bars,
    pound, pounds,
    psi,
    dHg0,
    mmHg, torr,
    mmu, mmus, milli_mass_unit,
    quart, quarts,
    ly, lightyear, lightyears,
    au, astronomical_unit, astronomical_units,
    planck_mass,
    planck_time,
    planck_temperature,
    planck_length,
    planck_charge,
    planck_area,
    planck_volume,
    planck_momentum,
    planck_energy,
    planck_force,
    planck_power,
    planck_density,
    planck_energy_density,
    planck_intensity,
    planck_angular_frequency,
    planck_pressure,
    planck_current,
    planck_voltage,
    planck_impedance,
    planck_acceleration,
    bit, bits,
    byte,
    kibibyte, kibibytes,
    mebibyte, mebibytes,
    gibibyte, gibibytes,
    tebibyte, tebibytes,
    pebibyte, pebibytes,
    exbibyte, exbibytes,
)

from .systems import (
    mks, mksa, si
)


def find_unit(quantity, unit_system="SI"):
    """
    Return a list of matching units or dimension names.

    - If ``quantity`` is a string -- units/dimensions containing the string
    `quantity`.
    - If ``quantity`` is a unit or dimension -- units having matching base
    units or dimensions.

    Examples
    ========

    >>> from sympy.physics import units as u
    >>> u.find_unit('charge')
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit(u.charge)
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit("ampere")
    ['ampere', 'amperes']
    >>> u.find_unit('angstrom')
    ['angstrom', 'angstroms']
    >>> u.find_unit('volt')
    ['volt', 'volts', 'electronvolt', 'electronvolts', 'planck_voltage']
    >>> u.find_unit(u.inch**3)[:9]
    ['L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter']
    """
    unit_system = UnitSystem.get_unit_system(unit_system)

    import sympy.physics.units as u
    rv = []
    if isinstance(quantity, str):
        rv = [i for i in dir(u) if quantity in i and isinstance(getattr(u, i), Quantity)]
        dim = getattr(u, quantity)
        if isinstance(dim, Dimension):
            rv.extend(find_unit(dim))
    else:
        for i in sorted(dir(u)):
            other = getattr(u, i)
            if not isinstance(other, Quantity):
                continue
            if isinstance(quantity, Quantity):
                if quantity.dimension == other.dimension:
                    rv.append(str(i))
            elif isinstance(quantity, Dimension):
                if other.dimension == quantity:
                    rv.append(str(i))
            elif other.dimension == Dimension(unit_system.get_dimensional_expr(quantity)):
                rv.append(str(i))
    return sorted(set(rv), key=lambda x: (len(x), x))

# NOTE: the old units module had additional variables:
# 'density', 'illuminance', 'resistance'.
# They were not dimensions, but units (old Unit class).

__all__ = [
    'Dimension', 'DimensionSystem',
    'UnitSystem',
    'convert_to',
    'Quantity',

    'amount_of_substance', 'acceleration', 'action', 'area',
    'capacitance', 'charge', 'conductance', 'current', 'energy',
    'force', 'frequency', 'impedance', 'inductance', 'length',
    'luminous_intensity', 'magnetic_density',
    'magnetic_flux', 'mass', 'momentum', 'power', 'pressure', 'temperature', 'time',
    'velocity', 'voltage', 'volume',

    'Unit',

    'speed',
    'luminosity',
    'magnetic_flux_density',
    'amount',

    'yotta',
    'zetta',
    'exa',
    'peta',
    'tera',
    'giga',
    'mega',
    'kilo',
    'hecto',
    'deca',
    'deci',
    'centi',
    'milli',
    'micro',
    'nano',
    'pico',
    'femto',
    'atto',
    'zepto',
    'yocto',

    'kibi',
    'mebi',
    'gibi',
    'tebi',
    'pebi',
    'exbi',

    'percent', 'percents',
    'permille',
    'rad', 'radian', 'radians',
    'deg', 'degree', 'degrees',
    'sr', 'steradian', 'steradians',
    'mil', 'angular_mil', 'angular_mils',
    'm', 'meter', 'meters',
    'kg', 'kilogram', 'kilograms',
    's', 'second', 'seconds',
    'A', 'ampere', 'amperes',
    'K', 'kelvin', 'kelvins',
    'mol', 'mole', 'moles',
    'cd', 'candela', 'candelas',
    'g', 'gram', 'grams',
    'mg', 'milligram', 'milligrams',
    'ug', 'microgram', 'micrograms',
    't', 'tonne', 'metric_ton',
    'newton', 'newtons', 'N',
    'joule', 'joules', 'J',
    'watt', 'watts', 'W',
    'pascal', 'pascals', 'Pa', 'pa',
    'hertz', 'hz', 'Hz',
    'coulomb', 'coulombs', 'C',
    'volt', 'volts', 'v', 'V',
    'ohm', 'ohms',
    'siemens', 'S', 'mho', 'mhos',
    'farad', 'farads', 'F',
    'henry', 'henrys', 'H',
    'tesla', 'teslas', 'T',
    'weber', 'webers', 'Wb', 'wb',
    'optical_power', 'dioptre', 'D',
    'lux', 'lx',
    'katal', 'kat',
    'gray', 'Gy',
    'becquerel', 'Bq',
    'km', 'kilometer', 'kilometers',
    'dm', 'decimeter', 'decimeters',
    'cm', 'centimeter', 'centimeters',
    'mm', 'millimeter', 'millimeters',
    'um', 'micrometer', 'micrometers', 'micron', 'microns',
    'nm', 'nanometer', 'nanometers',
    'pm', 'picometer', 'picometers',
    'ft', 'foot', 'feet',
    'inch', 'inches',
    'yd', 'yard', 'yards',
    'mi', 'mile', 'miles',
    'nmi', 'nautical_mile', 'nautical_miles',
    'angstrom', 'angstroms',
    'ha', 'hectare',
    'l', 'L', 'liter', 'liters',
    'dl', 'dL', 'deciliter', 'deciliters',
    'cl', 'cL', 'centiliter', 'centiliters',
    'ml', 'mL', 'milliliter', 'milliliters',
    'ms', 'millisecond', 'milliseconds',
    'us', 'microsecond', 'microseconds',
    'ns', 'nanosecond', 'nanoseconds',
    'ps', 'picosecond', 'picoseconds',
    'minute', 'minutes',
    'h', 'hour', 'hours',
    'day', 'days',
    'anomalistic_year', 'anomalistic_years',
    'sidereal_year', 'sidereal_years',
    'tropical_year', 'tropical_years',
    'common_year', 'common_years',
    'julian_year', 'julian_years',
    'draconic_year', 'draconic_years',
    'gaussian_year', 'gaussian_years',
    'full_moon_cycle', 'full_moon_cycles',
    'year', 'years',
    'G', 'gravitational_constant',
    'c', 'speed_of_light',
    'elementary_charge',
    'hbar',
    'planck',
    'eV', 'electronvolt', 'electronvolts',
    'avogadro_number',
    'avogadro', 'avogadro_constant',
    'boltzmann', 'boltzmann_constant',
    'stefan', 'stefan_boltzmann_constant',
    'R', 'molar_gas_constant',
    'faraday_constant',
    'josephson_constant',
    'von_klitzing_constant',
    'Da', 'dalton', 'amu', 'amus', 'atomic_mass_unit', 'atomic_mass_constant',
    'me', 'electron_rest_mass',
    'gee', 'gees', 'acceleration_due_to_gravity',
    'u0', 'magnetic_constant', 'vacuum_permeability',
    'e0', 'electric_constant', 'vacuum_permittivity',
    'Z0', 'vacuum_impedance',
    'coulomb_constant', 'electric_force_constant',
    'atmosphere', 'atmospheres', 'atm',
    'kPa',
    'bar', 'bars',
    'pound', 'pounds',
    'psi',
    'dHg0',
    'mmHg', 'torr',
    'mmu', 'mmus', 'milli_mass_unit',
    'quart', 'quarts',
    'ly', 'lightyear', 'lightyears',
    'au', 'astronomical_unit', 'astronomical_units',
    'planck_mass',
    'planck_time',
    'planck_temperature',
    'planck_length',
    'planck_charge',
    'planck_area',
    'planck_volume',
    'planck_momentum',
    'planck_energy',
    'planck_force',
    'planck_power',
    'planck_density',
    'planck_energy_density',
    'planck_intensity',
    'planck_angular_frequency',
    'planck_pressure',
    'planck_current',
    'planck_voltage',
    'planck_impedance',
    'planck_acceleration',
    'bit', 'bits',
    'byte',
    'kibibyte', 'kibibytes',
    'mebibyte', 'mebibytes',
    'gibibyte', 'gibibytes',
    'tebibyte', 'tebibytes',
    'pebibyte', 'pebibytes',
    'exbibyte', 'exbibytes',

    'mks', 'mksa', 'si',
]
<|endoftext|>"""
Naturalunit system.

The natural system comes from "setting c = 1, hbar = 1". From the computer
point of view it means that we use velocity and action instead of length and
time. Moreover instead of mass we use energy.
"""

from sympy.physics.units import DimensionSystem
from sympy.physics.units.definitions import c, eV, hbar
from sympy.physics.units.definitions.dimension_definitions import (
    action, energy, force, frequency, length, mass, momentum,
    power, time, velocity)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.unitsystem import UnitSystem


# dimension system
_natural_dim = DimensionSystem(
    base_dims=(action, energy, velocity),
    derived_dims=(length, mass, time, momentum, force, power, frequency)
)

units = prefix_unit(eV, PREFIXES)

# unit system
natural = UnitSystem(base_units=(hbar, eV, c), units=units, name="Natural system")
<|endoftext|>from sympy.core.singleton import S

from sympy.core.numbers import pi

from sympy.physics.units import DimensionSystem, hertz, kilogram
from sympy.physics.units.definitions import (
    G, Hz, J, N, Pa, W, c, g, kg, m, s, meter, gram, second, newton,
    joule, watt, pascal)
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, energy, force, frequency, momentum,
    power, pressure, velocity, length, mass, time)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.prefixes import (
    kibi, mebi, gibi, tebi, pebi, exbi
)
from sympy.physics.units.definitions import (
    cd, K, coulomb, volt, ohm, siemens, farad, henry, tesla, weber, dioptre,
    lux, katal, gray, becquerel, inch, hectare, liter, julian_year,
    gravitational_constant, speed_of_light, elementary_charge, planck, hbar,
    electronvolt, avogadro_number, avogadro_constant, boltzmann_constant,
    stefan_boltzmann_constant, atomic_mass_constant, molar_gas_constant,
    faraday_constant, josephson_constant, von_klitzing_constant,
    acceleration_due_to_gravity, magnetic_constant, vacuum_permittivity,
    vacuum_impedance, coulomb_constant, atmosphere, bar, pound, psi, mmHg,
    milli_mass_unit, quart, lightyear, astronomical_unit, planck_mass,
    planck_time, planck_temperature, planck_length, planck_charge,
    planck_area, planck_volume, planck_momentum, planck_energy, planck_force,
    planck_power, planck_density, planck_energy_density, planck_intensity,
    planck_angular_frequency, planck_pressure, planck_current, planck_voltage,
    planck_impedance, planck_acceleration, bit, byte, kibibyte, mebibyte,
    gibibyte, tebibyte, pebibyte, exbibyte, curie, rutherford, radian, degree,
    steradian, angular_mil, atomic_mass_unit, gee, kPa, ampere, u0, kelvin,
    mol, mole, candela, electric_constant, boltzmann, angstrom
)


dimsys_length_weight_time = DimensionSystem([
    # Dimensional dependencies for MKS base dimensions
    length,
    mass,
    time,
], dimensional_dependencies={
    # Dimensional dependencies for derived dimensions
    "velocity": {"length": 1, "time": -1},
    "acceleration": {"length": 1, "time": -2},
    "momentum": {"mass": 1, "length": 1, "time": -1},
    "force": {"mass": 1, "length": 1, "time": -2},
    "energy": {"mass": 1, "length": 2, "time": -2},
    "power": {"length": 2, "mass": 1, "time": -3},
    "pressure": {"mass": 1, "length": -1, "time": -2},
    "frequency": {"time": -1},
    "action": {"length": 2, "mass": 1, "time": -1},
    "area": {"length": 2},
    "volume": {"length": 3},
})


One = S.One


# Base units:
dimsys_length_weight_time.set_quantity_dimension(meter, length)
dimsys_length_weight_time.set_quantity_scale_factor(meter, One)

# gram; used to define its prefixed units
dimsys_length_weight_time.set_quantity_dimension(gram, mass)
dimsys_length_weight_time.set_quantity_scale_factor(gram, One)

dimsys_length_weight_time.set_quantity_dimension(second, time)
dimsys_length_weight_time.set_quantity_scale_factor(second, One)

# derived units

dimsys_length_weight_time.set_quantity_dimension(newton, force)
dimsys_length_weight_time.set_quantity_scale_factor(newton, kilogram*meter/second**2)

dimsys_length_weight_time.set_quantity_dimension(joule, energy)
dimsys_length_weight_time.set_quantity_scale_factor(joule, newton*meter)

dimsys_length_weight_time.set_quantity_dimension(watt, power)
dimsys_length_weight_time.set_quantity_scale_factor(watt, joule/second)

dimsys_length_weight_time.set_quantity_dimension(pascal, pressure)
dimsys_length_weight_time.set_quantity_scale_factor(pascal, newton/meter**2)

dimsys_length_weight_time.set_quantity_dimension(hertz, frequency)
dimsys_length_weight_time.set_quantity_scale_factor(hertz, One)

# Other derived units:

dimsys_length_weight_time.set_quantity_dimension(dioptre, 1 / length)
dimsys_length_weight_time.set_quantity_scale_factor(dioptre, 1/meter)

# Common volume and area units

dimsys_length_weight_time.set_quantity_dimension(hectare, length**2)
dimsys_length_weight_time.set_quantity_scale_factor(hectare, (meter**2)*(10000))

dimsys_length_weight_time.set_quantity_dimension(liter, length**3)
dimsys_length_weight_time.set_quantity_scale_factor(liter, meter**3/1000)


# Newton constant
# REF: NIST SP 959 (June 2019)

dimsys_length_weight_time.set_quantity_dimension(gravitational_constant, length ** 3 * mass ** -1 * time ** -2)
dimsys_length_weight_time.set_quantity_scale_factor(gravitational_constant, 6.67430e-11*m**3/(kg*s**2))

# speed of light

dimsys_length_weight_time.set_quantity_dimension(speed_of_light, velocity)
dimsys_length_weight_time.set_quantity_scale_factor(speed_of_light, 299792458*meter/second)


# Planck constant
# REF: NIST SP 959 (June 2019)

dimsys_length_weight_time.set_quantity_dimension(planck, action)
dimsys_length_weight_time.set_quantity_scale_factor(planck, 6.62607015e-34*joule*second)

# Reduced Planck constant
# REF: NIST SP 959 (June 2019)

dimsys_length_weight_time.set_quantity_dimension(hbar, action)
dimsys_length_weight_time.set_quantity_scale_factor(hbar, planck / (2 * pi))


__all__ = [
    'mmHg', 'atmosphere', 'newton', 'meter', 'vacuum_permittivity', 'pascal',
    'magnetic_constant', 'angular_mil', 'julian_year', 'weber', 'exbibyte',
    'liter', 'molar_gas_constant', 'faraday_constant', 'avogadro_constant',
    'planck_momentum', 'planck_density', 'gee', 'mol', 'bit', 'gray', 'kibi',
    'bar', 'curie', 'prefix_unit', 'PREFIXES', 'planck_time', 'gram',
    'candela', 'force', 'planck_intensity', 'energy', 'becquerel',
    'planck_acceleration', 'speed_of_light', 'dioptre', 'second', 'frequency',
    'Hz', 'power', 'lux', 'planck_current', 'momentum', 'tebibyte',
    'planck_power', 'degree', 'mebi', 'K', 'planck_volume',
    'quart', 'pressure', 'W', 'joule', 'boltzmann_constant', 'c', 'g',
    'planck_force', 'exbi', 's', 'watt', 'action', 'hbar', 'gibibyte',
    'DimensionSystem', 'cd', 'volt', 'planck_charge', 'angstrom',
    'dimsys_length_weight_time', 'pebi', 'vacuum_impedance', 'planck',
    'farad', 'gravitational_constant', 'u0', 'hertz', 'tesla', 'steradian',
    'josephson_constant', 'planck_area', 'stefan_boltzmann_constant',
    'astronomical_unit', 'J', 'N', 'planck_voltage', 'planck_energy',
    'atomic_mass_constant', 'rutherford', 'elementary_charge', 'Pa',
    'planck_mass', 'henry', 'planck_angular_frequency', 'ohm', 'pound',
    'planck_pressure', 'G', 'avogadro_number', 'psi', 'von_klitzing_constant',
    'planck_length', 'radian', 'mole', 'acceleration',
    'planck_energy_density', 'mebibyte', 'length',
    'acceleration_due_to_gravity', 'planck_temperature', 'tebi', 'inch',
    'electronvolt', 'coulomb_constant', 'kelvin', 'kPa', 'boltzmann',
    'milli_mass_unit', 'gibi', 'planck_impedance', 'electric_constant', 'kg',
    'coulomb', 'siemens', 'byte', 'atomic_mass_unit', 'm', 'kibibyte',
    'kilogram', 'lightyear', 'mass', 'time', 'pebibyte', 'velocity',
    'ampere', 'katal',
]
<|endoftext|>"""
MKS unit system.

MKS stands for "meter, kilogram, second".
"""

from sympy.physics.units import UnitSystem
from sympy.physics.units.definitions import gravitational_constant, hertz, joule, newton, pascal, watt, speed_of_light, gram, kilogram, meter, second
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, energy, force, frequency, momentum,
    power, pressure, velocity, length, mass, time)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.length_weight_time import dimsys_length_weight_time

dims = (velocity, acceleration, momentum, force, energy, power, pressure,
        frequency, action)

units = [meter, gram, second, joule, newton, watt, pascal, hertz]
all_units = []

# Prefixes of units like gram, joule, newton etc get added using `prefix_unit`
# in the for loop, but the actual units have to be added manually.
all_units.extend([gram, joule, newton, watt, pascal, hertz])

for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))
all_units.extend([gravitational_constant, speed_of_light])

# unit system
MKS = UnitSystem(base_units=(meter, kilogram, second), units=all_units, name="MKS", dimension_system=dimsys_length_weight_time, derived_units={
    power: watt,
    time: second,
    pressure: pascal,
    length: meter,
    frequency: hertz,
    mass: kilogram,
    force: newton,
    energy: joule,
    velocity: meter/second,
    acceleration: meter/(second**2),
})


__all__ = [
    'MKS', 'units', 'all_units', 'dims',
]
<|endoftext|>from sympy.physics.units.systems.mks import MKS
from sympy.physics.units.systems.mksa import MKSA
from sympy.physics.units.systems.natural import natural
from sympy.physics.units.systems.si import SI

__all__ = ['MKS', 'MKSA', 'natural', 'SI']
<|endoftext|>"""
MKS unit system.

MKS stands for "meter, kilogram, second, ampere".
"""

from __future__ import annotations

from sympy.physics.units.definitions import Z0, ampere, coulomb, farad, henry, siemens, tesla, volt, weber, ohm
from sympy.physics.units.definitions.dimension_definitions import (
    capacitance, charge, conductance, current, impedance, inductance,
    magnetic_density, magnetic_flux, voltage)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.mks import MKS, dimsys_length_weight_time
from sympy.physics.units.quantities import Quantity

dims = (voltage, impedance, conductance, current, capacitance, inductance, charge,
        magnetic_density, magnetic_flux)

units = [ampere, volt, ohm, siemens, farad, henry, coulomb, tesla, weber]

all_units: list[Quantity] = []
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))
all_units.extend(units)

all_units.append(Z0)

dimsys_MKSA = dimsys_length_weight_time.extend([
    # Dimensional dependencies for base dimensions (MKSA not in MKS)
    current,
], new_dim_deps={
    # Dimensional dependencies for derived dimensions
    "voltage": {"mass": 1, "length": 2, "current": -1, "time": -3},
    "impedance": {"mass": 1, "length": 2, "current": -2, "time": -3},
    "conductance": {"mass": -1, "length": -2, "current": 2, "time": 3},
    "capacitance": {"mass": -1, "length": -2, "current": 2, "time": 4},
    "inductance": {"mass": 1, "length": 2, "current": -2, "time": -2},
    "charge": {"current": 1, "time": 1},
    "magnetic_density": {"mass": 1, "current": -1, "time": -2},
    "magnetic_flux": {"length": 2, "mass": 1, "current": -1, "time": -2},
})

MKSA = MKS.extend(base=(ampere,), units=all_units, name='MKSA', dimension_system=dimsys_MKSA, derived_units={
    magnetic_flux: weber,
    impedance: ohm,
    current: ampere,
    voltage: volt,
    inductance: henry,
    conductance: siemens,
    magnetic_density: tesla,
    charge: coulomb,
    capacitance: farad,
})
<|endoftext|>"""
SI unit system.
Based on MKSA, which stands for "meter, kilogram, second, ampere".
Added kelvin, candela and mole.

"""

from __future__ import annotations

from sympy.physics.units import DimensionSystem, Dimension, dHg0

from sympy.physics.units.quantities import Quantity

from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, current, impedance, length, mass, time, velocity,
    amount_of_substance, temperature, information, frequency, force, pressure,
    energy, power, charge, voltage, capacitance, conductance, magnetic_flux,
    magnetic_density, inductance, luminous_intensity
)
from sympy.physics.units.definitions import (
    kilogram, newton, second, meter, gram, cd, K, joule, watt, pascal, hertz,
    coulomb, volt, ohm, siemens, farad, henry, tesla, weber, dioptre, lux,
    katal, gray, becquerel, inch, liter, julian_year, gravitational_constant,
    speed_of_light, elementary_charge, planck, hbar, electronvolt,
    avogadro_number, avogadro_constant, boltzmann_constant, electron_rest_mass,
    stefan_boltzmann_constant, Da, atomic_mass_constant, molar_gas_constant,
    faraday_constant, josephson_constant, von_klitzing_constant,
    acceleration_due_to_gravity, magnetic_constant, vacuum_permittivity,
    vacuum_impedance, coulomb_constant, atmosphere, bar, pound, psi, mmHg,
    milli_mass_unit, quart, lightyear, astronomical_unit, planck_mass,
    planck_time, planck_temperature, planck_length, planck_charge, planck_area,
    planck_volume, planck_momentum, planck_energy, planck_force, planck_power,
    planck_density, planck_energy_density, planck_intensity,
    planck_angular_frequency, planck_pressure, planck_current, planck_voltage,
    planck_impedance, planck_acceleration, bit, byte, kibibyte, mebibyte,
    gibibyte, tebibyte, pebibyte, exbibyte, curie, rutherford, radian, degree,
    steradian, angular_mil, atomic_mass_unit, gee, kPa, ampere, u0, c, kelvin,
    mol, mole, candela, m, kg, s, electric_constant, G, boltzmann
)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.mksa import MKSA, dimsys_MKSA

derived_dims = (frequency, force, pressure, energy, power, charge, voltage,
                capacitance, conductance, magnetic_flux,
                magnetic_density, inductance, luminous_intensity)
base_dims = (amount_of_substance, luminous_intensity, temperature)

units = [mol, cd, K, lux, hertz, newton, pascal, joule, watt, coulomb, volt,
        farad, ohm, siemens, weber, tesla, henry, candela, lux, becquerel,
        gray, katal]

all_units: list[Quantity] = []
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))

all_units.extend(units)
all_units.extend([mol, cd, K, lux])


dimsys_SI = dimsys_MKSA.extend(
    [
        # Dimensional dependencies for other base dimensions:
        temperature,
        amount_of_substance,
        luminous_intensity,
    ])

dimsys_default = dimsys_SI.extend(
    [information],
)

SI = MKSA.extend(base=(mol, cd, K), units=all_units, name='SI', dimension_system=dimsys_SI, derived_units={
    power: watt,
    magnetic_flux: weber,
    time: second,
    impedance: ohm,
    pressure: pascal,
    current: ampere,
    voltage: volt,
    length: meter,
    frequency: hertz,
    inductance: henry,
    temperature: kelvin,
    amount_of_substance: mole,
    luminous_intensity: candela,
    conductance: siemens,
    mass: kilogram,
    magnetic_density: tesla,
    charge: coulomb,
    force: newton,
    capacitance: farad,
    energy: joule,
    velocity: meter/second,
})

One = S.One

SI.set_quantity_dimension(radian, One)

SI.set_quantity_scale_factor(ampere, One)

SI.set_quantity_scale_factor(kelvin, One)

SI.set_quantity_scale_factor(mole, One)

SI.set_quantity_scale_factor(candela, One)

# MKSA extension to MKS: derived units

SI.set_quantity_scale_factor(coulomb, One)

SI.set_quantity_scale_factor(volt, joule/coulomb)

SI.set_quantity_scale_factor(ohm, volt/ampere)

SI.set_quantity_scale_factor(siemens, ampere/volt)

SI.set_quantity_scale_factor(farad, coulomb/volt)

SI.set_quantity_scale_factor(henry, volt*second/ampere)

SI.set_quantity_scale_factor(tesla, volt*second/meter**2)

SI.set_quantity_scale_factor(weber, joule/ampere)


SI.set_quantity_dimension(lux, luminous_intensity / length ** 2)
SI.set_quantity_scale_factor(lux, steradian*candela/meter**2)

# katal is the SI unit of catalytic activity

SI.set_quantity_dimension(katal, amount_of_substance / time)
SI.set_quantity_scale_factor(katal, mol/second)

# gray is the SI unit of absorbed dose

SI.set_quantity_dimension(gray, energy / mass)
SI.set_quantity_scale_factor(gray, meter**2/second**2)

# becquerel is the SI unit of radioactivity

SI.set_quantity_dimension(becquerel, 1 / time)
SI.set_quantity_scale_factor(becquerel, 1/second)

#### CONSTANTS ####

# elementary charge
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(elementary_charge, charge)
SI.set_quantity_scale_factor(elementary_charge, 1.602176634e-19*coulomb)

# Electronvolt
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(electronvolt, energy)
SI.set_quantity_scale_factor(electronvolt, 1.602176634e-19*joule)

# Avogadro number
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(avogadro_number, One)
SI.set_quantity_scale_factor(avogadro_number, 6.02214076e23)

# Avogadro constant

SI.set_quantity_dimension(avogadro_constant, amount_of_substance ** -1)
SI.set_quantity_scale_factor(avogadro_constant, avogadro_number / mol)

# Boltzmann constant
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(boltzmann_constant, energy / temperature)
SI.set_quantity_scale_factor(boltzmann_constant, 1.380649e-23*joule/kelvin)

# Stefan-Boltzmann constant
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(stefan_boltzmann_constant, energy * time ** -1 * length ** -2 * temperature ** -4)
SI.set_quantity_scale_factor(stefan_boltzmann_constant, pi**2 * boltzmann_constant**4 / (60 * hbar**3 * speed_of_light ** 2))

# Atomic mass
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(atomic_mass_constant, mass)
SI.set_quantity_scale_factor(atomic_mass_constant, 1.66053906660e-24*gram)

# Molar gas constant
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(molar_gas_constant, energy / (temperature * amount_of_substance))
SI.set_quantity_scale_factor(molar_gas_constant, boltzmann_constant * avogadro_constant)

# Faraday constant

SI.set_quantity_dimension(faraday_constant, charge / amount_of_substance)
SI.set_quantity_scale_factor(faraday_constant, elementary_charge * avogadro_constant)

# Josephson constant

SI.set_quantity_dimension(josephson_constant, frequency / voltage)
SI.set_quantity_scale_factor(josephson_constant, 0.5 * planck / elementary_charge)

# Von Klitzing constant

SI.set_quantity_dimension(von_klitzing_constant, voltage / current)
SI.set_quantity_scale_factor(von_klitzing_constant, hbar / elementary_charge ** 2)

# Acceleration due to gravity (on the Earth surface)

SI.set_quantity_dimension(acceleration_due_to_gravity, acceleration)
SI.set_quantity_scale_factor(acceleration_due_to_gravity, 9.80665*meter/second**2)

# magnetic constant:

SI.set_quantity_dimension(magnetic_constant, force / current ** 2)
SI.set_quantity_scale_factor(magnetic_constant, 4*pi/10**7 * newton/ampere**2)

# electric constant:

SI.set_quantity_dimension(vacuum_permittivity, capacitance / length)
SI.set_quantity_scale_factor(vacuum_permittivity, 1/(u0 * c**2))

# vacuum impedance:

SI.set_quantity_dimension(vacuum_impedance, impedance)
SI.set_quantity_scale_factor(vacuum_impedance, u0 * c)

# Electron rest mass
SI.set_quantity_dimension(electron_rest_mass, mass)
SI.set_quantity_scale_factor(electron_rest_mass, 9.1093837015e-31*kilogram)

# Coulomb's constant:
SI.set_quantity_dimension(coulomb_constant, force * length ** 2 / charge ** 2)
SI.set_quantity_scale_factor(coulomb_constant, 1/(4*pi*vacuum_permittivity))

SI.set_quantity_dimension(psi, pressure)
SI.set_quantity_scale_factor(psi, pound * gee / inch ** 2)

SI.set_quantity_dimension(mmHg, pressure)
SI.set_quantity_scale_factor(mmHg, dHg0 * acceleration_due_to_gravity * kilogram / meter**2)

SI.set_quantity_dimension(milli_mass_unit, mass)
SI.set_quantity_scale_factor(milli_mass_unit, atomic_mass_unit/1000)

SI.set_quantity_dimension(quart, length ** 3)
SI.set_quantity_scale_factor(quart, Rational(231, 4) * inch**3)

# Other convenient units and magnitudes

SI.set_quantity_dimension(lightyear, length)
SI.set_quantity_scale_factor(lightyear, speed_of_light*julian_year)

SI.set_quantity_dimension(astronomical_unit, length)
SI.set_quantity_scale_factor(astronomical_unit, 149597870691*meter)

# Fundamental Planck units:

SI.set_quantity_dimension(planck_mass, mass)
SI.set_quantity_scale_factor(planck_mass, sqrt(hbar*speed_of_light/G))

SI.set_quantity_dimension(planck_time, time)
SI.set_quantity_scale_factor(planck_time, sqrt(hbar*G/speed_of_light**5))

SI.set_quantity_dimension(planck_temperature, temperature)
SI.set_quantity_scale_factor(planck_temperature, sqrt(hbar*speed_of_light**5/G/boltzmann**2))

SI.set_quantity_dimension(planck_length, length)
SI.set_quantity_scale_factor(planck_length, sqrt(hbar*G/speed_of_light**3))

SI.set_quantity_dimension(planck_charge, charge)
SI.set_quantity_scale_factor(planck_charge, sqrt(4*pi*electric_constant*hbar*speed_of_light))

# Derived Planck units:

SI.set_quantity_dimension(planck_area, length ** 2)
SI.set_quantity_scale_factor(planck_area, planck_length**2)

SI.set_quantity_dimension(planck_volume, length ** 3)
SI.set_quantity_scale_factor(planck_volume, planck_length**3)

SI.set_quantity_dimension(planck_momentum, mass * velocity)
SI.set_quantity_scale_factor(planck_momentum, planck_mass * speed_of_light)

SI.set_quantity_dimension(planck_energy, energy)
SI.set_quantity_scale_factor(planck_energy, planck_mass * speed_of_light**2)

SI.set_quantity_dimension(planck_force, force)
SI.set_quantity_scale_factor(planck_force, planck_energy / planck_length)

SI.set_quantity_dimension(planck_power, power)
SI.set_quantity_scale_factor(planck_power, planck_energy / planck_time)

SI.set_quantity_dimension(planck_density, mass / length ** 3)
SI.set_quantity_scale_factor(planck_density, planck_mass / planck_length**3)

SI.set_quantity_dimension(planck_energy_density, energy / length ** 3)
SI.set_quantity_scale_factor(planck_energy_density, planck_energy / planck_length**3)

SI.set_quantity_dimension(planck_intensity, mass * time ** (-3))
SI.set_quantity_scale_factor(planck_intensity, planck_energy_density * speed_of_light)

SI.set_quantity_dimension(planck_angular_frequency, 1 / time)
SI.set_quantity_scale_factor(planck_angular_frequency, 1 / planck_time)

SI.set_quantity_dimension(planck_pressure, pressure)
SI.set_quantity_scale_factor(planck_pressure, planck_force / planck_length**2)

SI.set_quantity_dimension(planck_current, current)
SI.set_quantity_scale_factor(planck_current, planck_charge / planck_time)

SI.set_quantity_dimension(planck_voltage, voltage)
SI.set_quantity_scale_factor(planck_voltage, planck_energy / planck_charge)

SI.set_quantity_dimension(planck_impedance, impedance)
SI.set_quantity_scale_factor(planck_impedance, planck_voltage / planck_current)

SI.set_quantity_dimension(planck_acceleration, acceleration)
SI.set_quantity_scale_factor(planck_acceleration, speed_of_light / planck_time)

# Older units for radioactivity

SI.set_quantity_dimension(curie, 1 / time)
SI.set_quantity_scale_factor(curie, 37000000000*becquerel)

SI.set_quantity_dimension(rutherford, 1 / time)
SI.set_quantity_scale_factor(rutherford, 1000000*becquerel)


# check that scale factors are the right SI dimensions:
for _scale_factor, _dimension in zip(
    SI._quantity_scale_factors.values(),
    SI._quantity_dimension_map.values()
):
    dimex = SI.get_dimensional_expr(_scale_factor)
    if dimex != 1:
        # XXX: equivalent_dims is an instance method taking two arguments in
        # addition to self so this can not work:
        if not DimensionSystem.equivalent_dims(_dimension, Dimension(dimex)):  # type: ignore
            raise ValueError("quantity value and dimension mismatch")
del _scale_factor, _dimension

__all__ = [
    'mmHg', 'atmosphere', 'inductance', 'newton', 'meter',
    'vacuum_permittivity', 'pascal', 'magnetic_constant', 'voltage',
    'angular_mil', 'luminous_intensity', 'all_units',
    'julian_year', 'weber', 'exbibyte', 'liter',
    'molar_gas_constant', 'faraday_constant', 'avogadro_constant',
    'lightyear', 'planck_density', 'gee', 'mol', 'bit', 'gray',
    'planck_momentum', 'bar', 'magnetic_density', 'prefix_unit', 'PREFIXES',
    'planck_time', 'dimex', 'gram', 'candela', 'force', 'planck_intensity',
    'energy', 'becquerel', 'planck_acceleration', 'speed_of_light',
    'conductance', 'frequency', 'coulomb_constant', 'degree', 'lux', 'planck',
    'current', 'planck_current', 'tebibyte', 'planck_power', 'MKSA', 'power',
    'K', 'planck_volume', 'quart', 'pressure', 'amount_of_substance',
    'joule', 'boltzmann_constant', 'Dimension', 'c', 'planck_force', 'length',
    'watt', 'action', 'hbar', 'gibibyte', 'DimensionSystem', 'cd', 'volt',
    'planck_charge', 'dioptre', 'vacuum_impedance', 'dimsys_default', 'farad',
    'charge', 'gravitational_constant', 'temperature', 'u0', 'hertz',
    'capacitance', 'tesla', 'steradian', 'planck_mass', 'josephson_constant',
    'planck_area', 'stefan_boltzmann_constant', 'base_dims',
    'astronomical_unit', 'radian', 'planck_voltage', 'impedance',
    'planck_energy', 'Da', 'atomic_mass_constant', 'rutherford', 'second', 'inch',
    'elementary_charge', 'SI', 'electronvolt', 'dimsys_SI', 'henry',
    'planck_angular_frequency', 'ohm', 'pound', 'planck_pressure', 'G', 'psi',
    'dHg0', 'von_klitzing_constant', 'planck_length', 'avogadro_number',
    'mole', 'acceleration', 'information', 'planck_energy_density',
    'mebibyte', 's', 'acceleration_due_to_gravity', 'electron_rest_mass',
    'planck_temperature', 'units', 'mass', 'dimsys_MKSA', 'kelvin', 'kPa',
    'boltzmann', 'milli_mass_unit', 'planck_impedance', 'electric_constant',
    'derived_dims', 'kg', 'coulomb', 'siemens', 'byte', 'magnetic_flux',
    'atomic_mass_unit', 'm', 'kibibyte', 'kilogram', 'One', 'curie', 'u',
    'time', 'pebibyte', 'velocity', 'ampere', 'katal',
]
<|endoftext|>from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import UnitSystem, centimeter, gram, second, coulomb, charge, speed_of_light, current, mass, \
    length, voltage, magnetic_density, magnetic_flux
from sympy.physics.units.definitions import coulombs_constant
from sympy.physics.units.definitions.unit_definitions import statcoulomb, statampere, statvolt, volt, tesla, gauss, \
    weber, maxwell, debye, oersted, ohm, farad, henry, erg, ampere, coulomb_constant
from sympy.physics.units.systems.mks import dimsys_length_weight_time

One = S.One

dimsys_cgs = dimsys_length_weight_time.extend(
    [],
    new_dim_deps={
        # Dimensional dependencies for derived dimensions
        "impedance": {"time": 1, "length": -1},
        "conductance": {"time": -1, "length": 1},
        "capacitance": {"length": 1},
        "inductance": {"time": 2, "length": -1},
        "charge": {"mass": S.Half, "length": S(3)/2, "time": -1},
        "current": {"mass": One/2, "length": 3*One/2, "time": -2},
        "voltage": {"length": -One/2, "mass": One/2, "time": -1},
        "magnetic_density": {"length": -One/2, "mass": One/2, "time": -1},
        "magnetic_flux": {"length": 3*One/2, "mass": One/2, "time": -1},
    }
)

cgs_gauss = UnitSystem(
    base_units=[centimeter, gram, second],
    units=[],
    name="cgs_gauss",
    dimension_system=dimsys_cgs)


cgs_gauss.set_quantity_scale_factor(coulombs_constant, 1)

cgs_gauss.set_quantity_dimension(statcoulomb, charge)
cgs_gauss.set_quantity_scale_factor(statcoulomb, centimeter**(S(3)/2)*gram**(S.Half)/second)

cgs_gauss.set_quantity_dimension(coulomb, charge)

cgs_gauss.set_quantity_dimension(statampere, current)
cgs_gauss.set_quantity_scale_factor(statampere, statcoulomb/second)

cgs_gauss.set_quantity_dimension(statvolt, voltage)
cgs_gauss.set_quantity_scale_factor(statvolt, erg/statcoulomb)

cgs_gauss.set_quantity_dimension(volt, voltage)

cgs_gauss.set_quantity_dimension(gauss, magnetic_density)
cgs_gauss.set_quantity_scale_factor(gauss, sqrt(gram/centimeter)/second)

cgs_gauss.set_quantity_dimension(tesla, magnetic_density)

cgs_gauss.set_quantity_dimension(maxwell, magnetic_flux)
cgs_gauss.set_quantity_scale_factor(maxwell, sqrt(centimeter**3*gram)/second)

# SI units expressed in CGS-gaussian units:
cgs_gauss.set_quantity_scale_factor(coulomb, 10*speed_of_light*statcoulomb)
cgs_gauss.set_quantity_scale_factor(ampere, 10*speed_of_light*statcoulomb/second)
cgs_gauss.set_quantity_scale_factor(volt, 10**6/speed_of_light*statvolt)
cgs_gauss.set_quantity_scale_factor(weber, 10**8*maxwell)
cgs_gauss.set_quantity_scale_factor(tesla, 10**4*gauss)
cgs_gauss.set_quantity_scale_factor(debye, One/10**18*statcoulomb*centimeter)
cgs_gauss.set_quantity_scale_factor(oersted, sqrt(gram/centimeter)/second)
cgs_gauss.set_quantity_scale_factor(ohm, 10**5/speed_of_light**2*second/centimeter)
cgs_gauss.set_quantity_scale_factor(farad, One/10**5*speed_of_light**2*centimeter)
cgs_gauss.set_quantity_scale_factor(henry, 10**5/speed_of_light**2/centimeter*second**2)

# Coulomb's constant:
cgs_gauss.set_quantity_dimension(coulomb_constant, 1)
cgs_gauss.set_quantity_scale_factor(coulomb_constant, 1)

__all__ = [
    'ohm', 'tesla', 'maxwell', 'speed_of_light', 'volt', 'second', 'voltage',
    'debye', 'dimsys_length_weight_time', 'centimeter', 'coulomb_constant',
    'farad', 'sqrt', 'UnitSystem', 'current', 'charge', 'weber', 'gram',
    'statcoulomb', 'gauss', 'S', 'statvolt', 'oersted', 'statampere',
    'dimsys_cgs', 'coulomb', 'magnetic_density', 'magnetic_flux', 'One',
    'length', 'erg', 'mass', 'coulombs_constant', 'henry', 'ampere',
    'cgs_gauss',
]
<|endoftext|>from functools import wraps

from sympy.core.basic import Basic
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices.dense import Matrix, eye, zeros
from sympy.core.containers import OrderedSet
from sympy.physics.mechanics.actuator import ActuatorBase
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.functions import (
    Lagrangian, _validate_coordinates, find_dynamicsymbols)
from sympy.physics.mechanics.joint import Joint
from sympy.physics.mechanics.kane import KanesMethod
from sympy.physics.mechanics.lagrange import LagrangesMethod
from sympy.physics.mechanics.loads import _parse_load, gravity
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent

__all__ = ['SymbolicSystem', 'System']


def _reset_eom_method(method):
    """Decorator to reset the eom_method if a property is changed."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._eom_method = None
        return method(self, *args, **kwargs)

    return wrapper


class System(_Methods):
    """Class to define a multibody system and form its equations of motion.

    Explanation
    ===========

    A ``System`` instance stores the different objects associated with a model,
    including bodies, joints, constraints, and other relevant information. With
    all the relationships between components defined, the ``System`` can be used
    to form the equations of motion using a backend, such as ``KanesMethod``.
    The ``System`` has been designed to be compatible with third-party
    libraries for greater flexibility and integration with other tools.

    Attributes
    ==========

    frame : ReferenceFrame
        Inertial reference frame of the system.
    fixed_point : Point
        A fixed point in the inertial reference frame.
    x : Vector
        Unit vector fixed in the inertial reference frame.
    y : Vector
        Unit vector fixed in the inertial reference frame.
    z : Vector
        Unit vector fixed in the inertial reference frame.
    q : ImmutableMatrix
        Matrix of all the generalized coordinates, i.e. the independent
        generalized coordinates stacked upon the dependent.
    u : ImmutableMatrix
        Matrix of all the generalized speeds, i.e. the independent generealized
        speeds stacked upon the dependent.
    q_ind : ImmutableMatrix
        Matrix of the independent generalized coordinates.
    q_dep : ImmutableMatrix
        Matrix of the dependent generalized coordinates.
    u_ind : ImmutableMatrix
        Matrix of the independent generalized speeds.
    u_dep : ImmutableMatrix
        Matrix of the dependent generalized speeds.
    u_aux : ImmutableMatrix
        Matrix of auxiliary generalized speeds.
    kdes : ImmutableMatrix
        Matrix of the kinematical differential equations as expressions equated
        to the zero matrix.
    bodies : tuple of BodyBase subclasses
        Tuple of all bodies that make up the system.
    joints : tuple of Joint
        Tuple of all joints that connect bodies in the system.
    loads : tuple of LoadBase subclasses
        Tuple of all loads that have been applied to the system.
    actuators : tuple of ActuatorBase subclasses
        Tuple of all actuators present in the system.
    holonomic_constraints : ImmutableMatrix
        Matrix with the holonomic constraints as expressions equated to the zero
        matrix.
    nonholonomic_constraints : ImmutableMatrix
        Matrix with the nonholonomic constraints as expressions equated to the
        zero matrix.
    velocity_constraints : ImmutableMatrix
        Matrix with the velocity constraints as expressions equated to the zero
        matrix. These are by default derived as the time derivatives of the
        holonomic constraints extended with the nonholonomic constraints.
    eom_method : subclass of KanesMethod or LagrangesMethod
        Backend for forming the equations of motion.

    Examples
    ========

    In the example below a cart with a pendulum is created. The cart moves along
    the x axis of the rail and the pendulum rotates about the z axis. The length
    of the pendulum is ``l`` with the pendulum represented as a particle. To
    move the cart a time dependent force ``F`` is applied to the cart.

    We first need to import some functions and create some of our variables.

    >>> from sympy import symbols, simplify
    >>> from sympy.physics.mechanics import (
    ...     mechanics_printing, dynamicsymbols, RigidBody, Particle,
    ...     ReferenceFrame, PrismaticJoint, PinJoint, System)
    >>> mechanics_printing(pretty_print=False)
    >>> g, l = symbols('g l')
    >>> F = dynamicsymbols('F')

    The next step is to create bodies. It is also useful to create a frame for
    locating the particle with respect to the pin joint later on, as a particle
    does not have a body-fixed frame.

    >>> rail = RigidBody('rail')
    >>> cart = RigidBody('cart')
    >>> bob = Particle('bob')
    >>> bob_frame = ReferenceFrame('bob_frame')

    Initialize the system, with the rail as the Newtonian reference. The body is
    also automatically added to the system.

    >>> system = System.from_newtonian(rail)
    >>> print(system.bodies[0])
    rail

    Create the joints, while immediately also adding them to the system.

    >>> system.add_joints(
    ...     PrismaticJoint('slider', rail, cart, joint_axis=rail.x),
    ...     PinJoint('pin', cart, bob, joint_axis=cart.z,
    ...              child_interframe=bob_frame,
    ...              child_point=l * bob_frame.y)
    ... )
    >>> system.joints
    (PrismaticJoint: slider  parent: rail  child: cart,
    PinJoint: pin  parent: cart  child: bob)

    While adding the joints, the associated generalized coordinates, generalized
    speeds, kinematic differential equations and bodies are also added to the
    system.

    >>> system.q
    Matrix([
    [q_slider],
    [   q_pin]])
    >>> system.u
    Matrix([
    [u_slider],
    [   u_pin]])
    >>> system.kdes
    Matrix([
    [u_slider - q_slider'],
    [      u_pin - q_pin']])
    >>> [body.name for body in system.bodies]
    ['rail', 'cart', 'bob']

    With the kinematics established, we can now apply gravity and the cart force
    ``F``.

    >>> system.apply_uniform_gravity(-g * system.y)
    >>> system.add_loads((cart.masscenter, F * rail.x))
    >>> system.loads
    ((rail_masscenter, - g*rail_mass*rail_frame.y),
     (cart_masscenter, - cart_mass*g*rail_frame.y),
     (bob_masscenter, - bob_mass*g*rail_frame.y),
     (cart_masscenter, F*rail_frame.x))

    With the entire system defined, we can now form the equations of motion.
    Before forming the equations of motion, one can also run some checks that
    will try to identify some common errors.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) - bob_mass*l*cos(q_pin)*u_pin'
     - (bob_mass + cart_mass)*u_slider' + F],
    [                   -bob_mass*g*l*sin(q_pin) - bob_mass*l**2*u_pin'
     - bob_mass*l*cos(q_pin)*u_slider']])
    >>> simplify(system.mass_matrix)
    Matrix([
    [ bob_mass + cart_mass, bob_mass*l*cos(q_pin)],
    [bob_mass*l*cos(q_pin),         bob_mass*l**2]])
    >>> system.forcing
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) + F],
    [          -bob_mass*g*l*sin(q_pin)]])

    The complexity of the above example can be increased if we add a constraint
    to prevent the particle from moving in the horizontal (x) direction. This
    can be done by adding a holonomic constraint. After which we should also
    redefine what our (in)dependent generalized coordinates and speeds are.

    >>> system.add_holonomic_constraints(
    ...     bob.masscenter.pos_from(rail.masscenter).dot(system.x)
    ... )
    >>> system.q_ind = system.get_joint('pin').coordinates
    >>> system.q_dep = system.get_joint('slider').coordinates
    >>> system.u_ind = system.get_joint('pin').speeds
    >>> system.u_dep = system.get_joint('slider').speeds

    With the updated system the equations of motion can be formed again.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([[-bob_mass*g*l*sin(q_pin)
             - bob_mass*l**2*u_pin'
             - bob_mass*l*cos(q_pin)*u_slider'
             - l*(bob_mass*l*u_pin**2*sin(q_pin)
             - bob_mass*l*cos(q_pin)*u_pin'
             - (bob_mass + cart_mass)*u_slider')*cos(q_pin)
             - l*F*cos(q_pin)]])
    >>> simplify(system.mass_matrix)
    Matrix([
    [bob_mass*l**2*sin(q_pin)**2, -cart_mass*l*cos(q_pin)],
    [               l*cos(q_pin),                       1]])
    >>> simplify(system.forcing)
    Matrix([
    [-l*(bob_mass*g*sin(q_pin) + bob_mass*l*u_pin**2*sin(2*q_pin)/2
     + F*cos(q_pin))],
    [
    l*u_pin**2*sin(q_pin)]])

    """

    def __init__(self, frame=None, fixed_point=None):
        """Initialize the system.

        Parameters
        ==========

        frame : ReferenceFrame, optional
            The inertial frame of the system. If none is supplied, a new frame
            will be created.
        fixed_point : Point, optional
            A fixed point in the inertial reference frame. If none is supplied,
            a new fixed_point will be created.

        """
        if frame is None:
            frame = ReferenceFrame('inertial_frame')
        elif not isinstance(frame, ReferenceFrame):
            raise TypeError('Frame must be an instance of ReferenceFrame.')
        self._frame = frame
        if fixed_point is None:
            fixed_point = Point('inertial_point')
        elif not isinstance(fixed_point, Point):
            raise TypeError('Fixed point must be an instance of Point.')
        self._fixed_point = fixed_point
        self._fixed_point.set_vel(self._frame, 0)
        self._q_ind = ImmutableMatrix(1, 0, []).T
        self._q_dep = ImmutableMatrix(1, 0, []).T
        self._u_ind = ImmutableMatrix(1, 0, []).T
        self._u_dep = ImmutableMatrix(1, 0, []).T
        self._u_aux = ImmutableMatrix(1, 0, []).T
        self._kdes = ImmutableMatrix(1, 0, []).T
        self._hol_coneqs = ImmutableMatrix(1, 0, []).T
        self._nonhol_coneqs = ImmutableMatrix(1, 0, []).T
        self._vel_constrs = None
        self._bodies = []
        self._joints = []
        self._loads = []
        self._actuators = []
        self._eom_method = None

    @classmethod
    def from_newtonian(cls, newtonian):
        """Constructs the system with respect to a Newtonian body."""
        if isinstance(newtonian, Particle):
            raise TypeError('A Particle has no frame so cannot act as '
                            'the Newtonian.')
        system = cls(frame=newtonian.frame, fixed_point=newtonian.masscenter)
        system.add_bodies(newtonian)
        return system

    @property
    def fixed_point(self):
        """Fixed point in the inertial reference frame."""
        return self._fixed_point

    @property
    def frame(self):
        """Inertial reference frame of the system."""
        return self._frame

    @property
    def x(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.x

    @property
    def y(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.y

    @property
    def z(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.z

    @property
    def bodies(self):
        """Tuple of all bodies that have been added to the system."""
        return tuple(self._bodies)

    @bodies.setter
    @_reset_eom_method
    def bodies(self, bodies):
        bodies = self._objects_to_list(bodies)
        self._check_objects(bodies, [], BodyBase, 'Bodies', 'bodies')
        self._bodies = bodies

    @property
    def joints(self):
        """Tuple of all joints that have been added to the system."""
        return tuple(self._joints)

    @joints.setter
    @_reset_eom_method
    def joints(self, joints):
        joints = self._objects_to_list(joints)
