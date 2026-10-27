        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        """Interpolate a function at the Chebyshev points of the first kind.

        Returns the series that interpolates `func` at the Chebyshev points of
        the first kind scaled and shifted to the `domain`. The resulting series
        tends to a minmax approximation of `func` when the function is
        continuous in the domain.

        Parameters
        ----------
        func : function
            The function to be interpolated. It must be a function of a single
            variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
            extra arguments passed in the `args` parameter.
        deg : int
            Degree of the interpolating polynomial.
        domain : {None, [beg, end]}, optional
            Domain over which `func` is interpolated. The default is None, in
            which case the domain is [-1, 1].
        args : tuple, optional
            Extra arguments to be used in the function call. Default is no
            extra arguments.

        Returns
        -------
        polynomial : Chebyshev instance
            Interpolating Chebyshev instance.

        Notes
        -----
        See `numpy.polynomial.chebinterpolate` for more details.

        """
        if domain is None:
            domain = cls.domain
        xfunc = lambda x: func(pu.mapdomain(x, cls.window, domain), *args)
        coef = chebinterpolate(xfunc, deg)
        return cls(coef, domain=domain)

    # Virtual properties
    domain = np.array(chebdomain)
    window = np.array(chebdomain)
    basis_name = 'T'
<|endoftext|>"""
Abstract base class for the various polynomial Classes.

The ABCPolyBase class provides the methods needed to implement the common API
for the various polynomial classes. It operates as a mixin, but uses the
abc module from the stdlib, hence it is only available for Python >= 2.6.

"""
import os
import abc
import numbers
from typing import Callable

import numpy as np
from . import polyutils as pu

__all__ = ['ABCPolyBase']

class ABCPolyBase(abc.ABC):
    """An abstract base class for immutable series classes.

    ABCPolyBase provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the
    methods listed below.

    Parameters
    ----------
    coef : array_like
        Series coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where
        ``P_i`` is the basis polynomials of degree ``i``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is the derived class domain.
    window : (2,) array_like, optional
        Window, see domain for its use. The default value is the
        derived class window.
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    Attributes
    ----------
    coef : (N,) ndarray
        Series coefficients in order of increasing degree.
    domain : (2,) ndarray
        Domain that is mapped to window.
    window : (2,) ndarray
        Window that domain is mapped to.
    symbol : str
        Symbol representing the independent variable.

    Class Attributes
    ----------------
    maxpower : int
        Maximum power allowed, i.e., the largest number ``n`` such that
        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
    domain : (2,) ndarray
        Default domain of the class.
    window : (2,) ndarray
        Default window of the class.

    """

    # Not hashable
    __hash__ = None

    # Opt out of numpy ufuncs and Python ops with ndarray subclasses.
    __array_ufunc__ = None

    # Limit runaway size. T_n^m has degree n*m
    maxpower = 100

    # Unicode character mappings for improved __str__
    _superscript_mapping = str.maketrans({
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    })
    _subscript_mapping = str.maketrans({
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉"
    })
    # Some fonts don't support full unicode character ranges necessary for
    # the full set of superscripts and subscripts, including common/default
    # fonts in Windows shells/terminals. Therefore, default to ascii-only
    # printing on windows.
    _use_unicode = not os.name == 'nt'

    @property
    def symbol(self):
        return self._symbol

    @property
    @abc.abstractmethod
    def domain(self):
        pass

    @property
    @abc.abstractmethod
    def window(self):
        pass

    @property
    @abc.abstractmethod
    def basis_name(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _add(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(c, pow, maxpower=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def _val(x, c):
        pass

    @staticmethod
    @abc.abstractmethod
    def _int(c, m, k, lbnd, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _der(c, m, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _fit(x, y, deg, rcond, full):
        pass

    @staticmethod
    @abc.abstractmethod
    def _line(off, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _roots(c):
        pass

    @staticmethod
    @abc.abstractmethod
    def _fromroots(r):
        pass

    def has_samecoef(self, other):
        """Check if coefficients match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``coef`` attribute.

        Returns
        -------
        bool : boolean
            True if the coefficients are the same, False otherwise.

        """
        if len(self.coef) != len(other.coef):
            return False
        elif not np.all(self.coef == other.coef):
            return False
        else:
            return True

    def has_samedomain(self, other):
        """Check if domains match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``domain`` attribute.

        Returns
        -------
        bool : boolean
            True if the domains are the same, False otherwise.

        """
        return np.all(self.domain == other.domain)

    def has_samewindow(self, other):
        """Check if windows match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``window`` attribute.

        Returns
        -------
        bool : boolean
            True if the windows are the same, False otherwise.

        """
        return np.all(self.window == other.window)

    def has_sametype(self, other):
        """Check if types match.

        Parameters
        ----------
        other : object
            Class instance.

        Returns
        -------
        bool : boolean
            True if other is same class as self

        """
        return isinstance(other, self.__class__)

    def _get_coefficients(self, other):
        """Interpret other as polynomial coefficients.

        The `other` argument is checked to see if it is of the same
        class as self with identical domain and window. If so,
        return its coefficients, otherwise return `other`.

        Parameters
        ----------
        other : anything
            Object to be checked.

        Returns
        -------
        coef
            The coefficients of`other` if it is a compatible instance,
            of ABCPolyBase, otherwise `other`.

        Raises
        ------
        TypeError
            When `other` is an incompatible instance of ABCPolyBase.

        """
        if isinstance(other, ABCPolyBase):
            if not isinstance(other, self.__class__):
                raise TypeError("Polynomial types differ")
            elif not np.all(self.domain == other.domain):
                raise TypeError("Domains differ")
            elif not np.all(self.window == other.window):
                raise TypeError("Windows differ")
            elif self.symbol != other.symbol:
                raise ValueError("Polynomial symbols differ")
            return other.coef
        return other

    def __init__(self, coef, domain=None, window=None, symbol='x'):
        [coef] = pu.as_series([coef], trim=False)
        self.coef = coef

        if domain is not None:
            [domain] = pu.as_series([domain], trim=False)
            if len(domain) != 2:
                raise ValueError("Domain has wrong number of elements.")
            self.domain = domain

        if window is not None:
            [window] = pu.as_series([window], trim=False)
            if len(window) != 2:
                raise ValueError("Window has wrong number of elements.")
            self.window = window

        # Validation for symbol
        try:
            if not symbol.isidentifier():
                raise ValueError(
                    "Symbol string must be a valid Python identifier"
                )
        # If a user passes in something other than a string, the above
        # results in an AttributeError. Catch this and raise a more
        # informative exception
        except AttributeError:
            raise TypeError("Symbol must be a non-empty string")

        self._symbol = symbol

    def __repr__(self):
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        window = repr(self.window)[6:-1]
        name = self.__class__.__name__
        return (f"{name}({coef}, domain={domain}, window={window}, "
                f"symbol='{self.symbol}')")

    def __format__(self, fmt_str):
        if fmt_str == '':
            return self.__str__()
        if fmt_str not in ('ascii', 'unicode'):
            raise ValueError(
                f"Unsupported format string '{fmt_str}' passed to "
                f"{self.__class__}.__format__. Valid options are "
                f"'ascii' and 'unicode'"
            )
        if fmt_str == 'ascii':
            return self._generate_string(self._str_term_ascii)
        return self._generate_string(self._str_term_unicode)

    def __str__(self):
        if self._use_unicode:
            return self._generate_string(self._str_term_unicode)
        return self._generate_string(self._str_term_ascii)

    def _generate_string(self, term_method):
        """
        Generate the full string representation of the polynomial, using
        ``term_method`` to generate each polynomial term.
        """
        # Get configuration for line breaks
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        out = pu.format_float(self.coef[0])

        off, scale = self.mapparms()

        scaled_symbol, needs_parens = self._format_term(pu.format_float,
                                                        off, scale)
        if needs_parens:
            scaled_symbol = '(' + scaled_symbol + ')'

        for i, coef in enumerate(self.coef[1:]):
            out += " "
            power = str(i + 1)
            # Polynomial coefficient
            # The coefficient array can be an object array with elements that
            # will raise a TypeError with >= 0 (e.g. strings or Python
            # complex). In this case, represent the coefficient as-is.
            try:
                if coef >= 0:
                    next_term = "+ " + pu.format_float(coef, parens=True)
                else:
                    next_term = "- " + pu.format_float(-coef, parens=True)
            except TypeError:
                next_term = f"+ {coef}"
            # Polynomial term
            next_term += term_method(power, scaled_symbol)
            # Length of the current line with next term added
            line_len = len(out.split('\n')[-1]) + len(next_term)
            # If not the last term in the polynomial, it will be two
            # characters longer due to the +/- with the next term
            if i < len(self.coef[1:]) - 1:
                line_len += 2
            # Handle linebreaking
            if line_len >= linewidth:
                next_term = next_term.replace(" ", "\n", 1)
            out += next_term
        return out

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        """
        String representation of single polynomial term using unicode
        characters for superscripts and subscripts.
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_unicode(cls, i, arg_str)"
            )
        return (f"·{cls.basis_name}{i.translate(cls._subscript_mapping)}"
                f"({arg_str})")

    @classmethod
    def _str_term_ascii(cls, i, arg_str):
        """
        String representation of a single polynomial term using ** and _ to
        represent superscripts and subscripts, respectively.
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_ascii(cls, i, arg_str)"
            )
        return f" {cls.basis_name}_{i}({arg_str})"

    @classmethod
    def _repr_latex_term(cls, i, arg_str, needs_parens):
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis name, or override "
                "_repr_latex_term(i, arg_str, needs_parens)")
        # since we always add parens, we don't care if the expression needs them
        return f"{{{cls.basis_name}}}_{{{i}}}({arg_str})"

    @staticmethod
    def _repr_latex_scalar(x, parens=False):
        # TODO: we're stuck with disabling math formatting until we handle
        # exponents in this function
        return r'\text{{{}}}'.format(pu.format_float(x, parens=parens))

    def _format_term(self, scalar_format: Callable, off: float, scale: float):
        """ Format a single term in the expansion """
        if off == 0 and scale == 1:
            term = self.symbol
            needs_parens = False
        elif scale == 1:
            term = f"{scalar_format(off)} + {self.symbol}"
            needs_parens = True
        elif off == 0:
            term = f"{scalar_format(scale)}{self.symbol}"
            needs_parens = True
        else:
            term = (
                f"{scalar_format(off)} + "
                f"{scalar_format(scale)}{self.symbol}"
            )
            needs_parens = True
        return term, needs_parens

    def _repr_latex_(self):
        # get the scaled argument string to the basis functions
        off, scale = self.mapparms()
        term, needs_parens = self._format_term(self._repr_latex_scalar,
                                               off, scale)

        mute = r"\color{{LightGray}}{{{}}}".format

        parts = []
        for i, c in enumerate(self.coef):
            # prevent duplication of + and - signs
            if i == 0:
                coef_str = f"{self._repr_latex_scalar(c)}"
            elif not isinstance(c, numbers.Real):
                coef_str = f" + ({self._repr_latex_scalar(c)})"
            elif c >= 0:
                coef_str = f" + {self._repr_latex_scalar(c, parens=True)}"
            else:
                coef_str = f" - {self._repr_latex_scalar(-c, parens=True)}"

            # produce the string for the term
            term_str = self._repr_latex_term(i, term, needs_parens)
            if term_str == '1':
                part = coef_str
            else:
                part = rf"{coef_str}\,{term_str}"

            if c == 0:
                part = mute(part)

            parts.append(part)

        if parts:
            body = ''.join(parts)
        else:
            # in case somehow there are no coefficients at all
            body = '0'

        return rf"${self.symbol} \mapsto {body}$"



    # Pickle and copy

    def __getstate__(self):
        ret = self.__dict__.copy()
        ret['coef'] = self.coef.copy()
        ret['domain'] = self.domain.copy()
        ret['window'] = self.window.copy()
        ret['symbol'] = self.symbol
        return ret

    def __setstate__(self, dict):
        self.__dict__ = dict

    # Call

    def __call__(self, arg):
        arg = pu.mapdomain(arg, self.domain, self.window)
        return self._val(arg, self.coef)

    def __iter__(self):
        return iter(self.coef)

    def __len__(self):
        return len(self.coef)

    # Numeric properties.

    def __neg__(self):
        return self.__class__(
            -self.coef, self.domain, self.window, self.symbol
        )

    def __pos__(self):
        return self

    def __add__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._add(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __sub__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._sub(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __mul__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._mul(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __truediv__(self, other):
        # there is no true divide if the rhs is not a Number, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        if not isinstance(other, numbers.Number) or isinstance(other, bool):
            raise TypeError(
                f"unsupported types for true division: "
                f"'{type(self)}', '{type(other)}'"
            )
        return self.__floordiv__(other)

    def __floordiv__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __mod__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __divmod__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            quo, rem = self._div(self.coef, othercoef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __pow__(self, other):
        coef = self._pow(self.coef, other, maxpower=self.maxpower)
        res = self.__class__(coef, self.domain, self.window, self.symbol)
        return res

    def __radd__(self, other):
        try:
            coef = self._add(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rsub__(self, other):
        try:
            coef = self._sub(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rmul__(self, other):
        try:
            coef = self._mul(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rdiv__(self, other):
        # set to __floordiv__ /.
        return self.__rfloordiv__(other)

    def __rtruediv__(self, other):
        # An instance of ABCPolyBase is not considered a
        # Number.
        return NotImplemented

    def __rfloordiv__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __rmod__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __rdivmod__(self, other):
        try:
            quo, rem = self._div(other, self.coef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __eq__(self, other):
        res = (isinstance(other, self.__class__) and
               np.all(self.domain == other.domain) and
               np.all(self.window == other.window) and
               (self.coef.shape == other.coef.shape) and
               np.all(self.coef == other.coef) and
               (self.symbol == other.symbol))
        return res

    def __ne__(self, other):
        return not self.__eq__(other)

    #
    # Extra methods.
    #

    def copy(self):
        """Return a copy.

        Returns
        -------
        new_series : series
            Copy of self.

        """
        return self.__class__(self.coef, self.domain, self.window, self.symbol)

    def degree(self):
        """The degree of the series.

        Returns
        -------
        degree : int
            Degree of the series, one less than the number of coefficients.

        Examples
        --------

        Create a polynomial object for ``1 + 7*x + 4*x**2``:

        >>> poly = np.polynomial.Polynomial([1, 7, 4])
        >>> print(poly)
        1.0 + 7.0·x + 4.0·x²
        >>> poly.degree()
        2

        Note that this method does not check for non-zero coefficients.
        You must trim the polynomial to remove any trailing zeroes:

        >>> poly = np.polynomial.Polynomial([1, 7, 0])
        >>> print(poly)
        1.0 + 7.0·x + 0.0·x²
        >>> poly.degree()
        2
        >>> poly.trim().degree()
        1

        """
        return len(self) - 1

    def cutdeg(self, deg):
        """Truncate series to the given degree.

        Reduce the degree of the series to `deg` by discarding the
        high order terms. If `deg` is greater than the current degree a
        copy of the current series is returned. This can be useful in least
        squares where the coefficients of the high degree terms may be very
        small.

        Parameters
        ----------
        deg : non-negative int
            The series is reduced to degree `deg` by discarding the high
            order terms. The value of `deg` must be a non-negative integer.

        Returns
        -------
        new_series : series
            New instance of series with reduced degree.

        """
        return self.truncate(deg + 1)

    def trim(self, tol=0):
        """Remove trailing coefficients

        Remove trailing coefficients until a coefficient is reached whose
        absolute value greater than `tol` or the beginning of the series is
        reached. If all the coefficients would be removed the series is set
        to ``[0]``. A new series instance is returned with the new
        coefficients.  The current instance remains unchanged.

        Parameters
        ----------
        tol : non-negative number.
            All trailing coefficients less than `tol` will be removed.

        Returns
        -------
        new_series : series
            New instance of series with trimmed coefficients.

        """
        coef = pu.trimcoef(self.coef, tol)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def truncate(self, size):
        """Truncate series to length `size`.

        Reduce the series to length `size` by discarding the high
        degree terms. The value of `size` must be a positive integer. This
        can be useful in least squares where the coefficients of the
        high degree terms may be very small.

        Parameters
        ----------
        size : positive int
            The series is reduced to length `size` by discarding the high
            degree terms. The value of `size` must be a positive integer.

        Returns
        -------
        new_series : series
            New instance of series with truncated coefficients.

        """
        isize = int(size)
        if isize != size or isize < 1:
            raise ValueError("size must be a positive integer")
        if isize >= len(self.coef):
            coef = self.coef
        else:
            coef = self.coef[:isize]
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def convert(self, domain=None, kind=None, window=None):
        """Convert series to a different kind and/or domain and/or window.

        Parameters
        ----------
        domain : array_like, optional
            The domain of the converted series. If the value is None,
            the default domain of `kind` is used.
        kind : class, optional
            The polynomial series type class to which the current instance
            should be converted. If kind is None, then the class of the
            current instance is used.
        window : array_like, optional
            The window of the converted series. If the value is None,
            the default window of `kind` is used.

        Returns
        -------
        new_series : series
            The returned class can be of different type than the current
            instance and/or have a different domain and/or different
            window.

        Notes
        -----
        Conversion between domains and class types can result in
        numerically ill defined series.

        """
        if kind is None:
            kind = self.__class__
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        return self(kind.identity(domain, window=window, symbol=self.symbol))

    def mapparms(self):
        """Return the mapping parameters.

        The returned values define a linear map ``off + scl*x`` that is
        applied to the input arguments before the series is evaluated. The
        map depends on the ``domain`` and ``window``; if the current
        ``domain`` is equal to the ``window`` the resulting map is the
        identity.  If the coefficients of the series instance are to be
        used by themselves outside this class, then the linear function
        must be substituted for the ``x`` in the standard representation of
        the base polynomials.

        Returns
        -------
        off, scl : float or complex
            The mapping function is defined by ``off + scl*x``.

        Notes
        -----
        If the current domain is the interval ``[l1, r1]`` and the window
        is ``[l2, r2]``, then the linear mapping function ``L`` is
        defined by the equations::

            L(l1) = l2
            L(r1) = r2

        """
        return pu.mapparms(self.domain, self.window)

    def integ(self, m=1, k=[], lbnd=None):
        """Integrate.

        Return a series instance that is the definite integral of the
        current series.

        Parameters
        ----------
        m : non-negative int
            The number of integrations to perform.
        k : array_like
            Integration constants. The first constant is applied to the
            first integration, the second to the second, and so on. The
            list of values must less than or equal to `m` in length and any
            missing values are set to zero.
        lbnd : Scalar
            The lower bound of the definite integral.

        Returns
        -------
        new_series : series
            A new series representing the integral. The domain is the same
            as the domain of the integrated series.

        """
        off, scl = self.mapparms()
        if lbnd is None:
            lbnd = 0
        else:
            lbnd = off + scl*lbnd
        coef = self._int(self.coef, m, k, lbnd, 1./scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def deriv(self, m=1):
        """Differentiate.

        Return a series instance of that is the derivative of the current
        series.

        Parameters
        ----------
        m : non-negative int
            Find the derivative of order `m`.

        Returns
        -------
        new_series : series
            A new series representing the derivative. The domain is the same
            as the domain of the differentiated series.

        """
        off, scl = self.mapparms()
        coef = self._der(self.coef, m, scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def roots(self):
        """Return the roots of the series polynomial.

        Compute the roots for the series. Note that the accuracy of the
        roots decreases the further outside the `domain` they lie.

        Returns
        -------
        roots : ndarray
            Array containing the roots of the series.

        """
        roots = self._roots(self.coef)
        return pu.mapdomain(roots, self.window, self.domain)

    def linspace(self, n=100, domain=None):
        """Return x, y values at equally spaced points in domain.

        Returns the x, y values at `n` linearly spaced points across the
        domain.  Here y is the value of the polynomial at the points x. By
        default the domain is the same as that of the series instance.
        This method is intended mostly as a plotting aid.

        Parameters
        ----------
        n : int, optional
            Number of point pairs to return. The default value is 100.
        domain : {None, array_like}, optional
            If not None, the specified domain is used instead of that of
            the calling instance. It should be of the form ``[beg,end]``.
            The default is None which case the class domain is used.

        Returns
        -------
        x, y : ndarray
            x is equal to linspace(self.domain[0], self.domain[1], n) and
            y is the series evaluated at element of x.

        """
        if domain is None:
            domain = self.domain
        x = np.linspace(domain[0], domain[1], n)
        y = self(x)
        return x, y

    @classmethod
    def fit(cls, x, y, deg, domain=None, rcond=None, full=False, w=None,
        window=None, symbol='x'):
        """Least squares fit to data.

        Return a series instance that is the least squares fit to the data
        `y` sampled at `x`. The domain of the returned instance can be
        specified and this will often result in a superior fit with less
        chance of ill conditioning.

        Parameters
        ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points ``(x[i], y[i])``.
        y : array_like, shape (M,)
            y-coordinates of the M sample points ``(x[i], y[i])``.
        deg : int or 1-D array_like
            Degree(s) of the fitting polynomials. If `deg` is a single integer
            all terms up to and including the `deg`'th term are included in the
            fit. For NumPy versions >= 1.11.0 a list of integers specifying the
            degrees of the terms to include may be used instead.
        domain : {None, [beg, end], []}, optional
            Domain to use for the returned series. If ``None``,
            then a minimal domain that covers the points `x` is chosen.  If
            ``[]`` the class domain is used. The default value was the
            class domain in NumPy 1.4 and ``None`` in later versions.
            The ``[]`` option was added in numpy 1.5.0.
        rcond : float, optional
            Relative condition number of the fit. Singular values smaller
            than this relative to the largest singular value will be
            ignored. The default value is ``len(x)*eps``, where eps is the
            relative precision of the float type, about 2e-16 in most
            cases.
        full : bool, optional
            Switch determining nature of return value. When it is False
            (the default) just the coefficients are returned, when True
            diagnostic information from the singular value decomposition is
            also returned.
        w : array_like, shape (M,), optional
            Weights. If not None, the weight ``w[i]`` applies to the unsquared
            residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
            chosen so that the errors of the products ``w[i]*y[i]`` all have
            the same variance.  When using inverse-variance weighting, use
            ``w[i] = 1/sigma(y[i])``.  The default value is None.
        window : {[beg, end]}, optional
            Window to use for the returned series. The default
            value is the default class domain
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            A series that represents the least squares fit to the data and
            has the domain and window specified in the call. If the
            coefficients for the unscaled and unshifted basis polynomials are
            of interest, do ``new_series.convert().coef``.

        [resid, rank, sv, rcond] : list
            These values are only returned if ``full == True``

            - resid -- sum of squared residuals of the least squares fit
            - rank -- the numerical rank of the scaled Vandermonde matrix
            - sv -- singular values of the scaled Vandermonde matrix
            - rcond -- value of `rcond`.

            For more details, see `linalg.lstsq`.

        """
        if domain is None:
            domain = pu.getdomain(x)
            if domain[0] == domain[1]:
                domain[0] -= 1
                domain[1] += 1
        elif type(domain) is list and len(domain) == 0:
            domain = cls.domain

        if window is None:
            window = cls.window

        xnew = pu.mapdomain(x, domain, window)
        res = cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)
        if full:
            [coef, status] = res
            return (
                cls(coef, domain=domain, window=window, symbol=symbol), status
            )
        else:
            coef = res
            return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def fromroots(cls, roots, domain=[], window=None, symbol='x'):
        """Return series instance that has the specified roots.

        Returns a series representing the product
        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a
        list of roots.

        Parameters
        ----------
        roots : array_like
            List of roots.
        domain : {[], None, array_like}, optional
            Domain for the resulting series. If None the domain is the
            interval from the smallest root to the largest. If [] the
            domain is the class domain. The default is [].
        window : {None, array_like}, optional
            Window for the returned series. If None the class window is
            used. The default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            Series with the specified roots.

        """
        [roots] = pu.as_series([roots], trim=False)
        if domain is None:
            domain = pu.getdomain(roots)
        elif type(domain) is list and len(domain) == 0:
            domain = cls.domain

        if window is None:
            window = cls.window

        deg = len(roots)
        off, scl = pu.mapparms(domain, window)
        rnew = off + scl*roots
        coef = cls._fromroots(rnew) / scl**deg
        return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def identity(cls, domain=None, window=None, symbol='x'):
        """Identity function.

        If ``p`` is the returned series, then ``p(x) == x`` for all
        values of x.

        Parameters
        ----------
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
             Series of representing the identity.

        """
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        off, scl = pu.mapparms(window, domain)
        coef = cls._line(off, scl)
        return cls(coef, domain, window, symbol)

    @classmethod
    def basis(cls, deg, domain=None, window=None, symbol='x'):
        """Series basis polynomial of degree `deg`.

        Returns the series representing the basis polynomial of degree `deg`.

        Parameters
        ----------
        deg : int
            Degree of the basis polynomial for the series. Must be >= 0.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            A series with the coefficient of the `deg` term set to one and
            all others zero.

        """
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        ideg = int(deg)

        if ideg != deg or ideg < 0:
            raise ValueError("deg must be non-negative integer")
        return cls([0]*ideg + [1], domain, window, symbol)

    @classmethod
    def cast(cls, series, domain=None, window=None):
        """Convert series to series of this class.

        The `series` is expected to be an instance of some polynomial
        series of one of the types supported by by the numpy.polynomial
        module, but could be some other class that supports the convert
        method.

        Parameters
        ----------
        series : series
            The series instance to be converted.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.

        Returns
        -------
        new_series : series
            A series of the same kind as the calling class and equal to
            `series` when evaluated.

        See Also
        --------
        convert : similar instance method

        """
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        return series.convert(domain, cls, window)
<|endoftext|>"""
A sub-package for efficiently dealing with polynomials.

Within the documentation for this sub-package, a "finite power series,"
i.e., a polynomial (also referred to simply as a "series") is represented
by a 1-D numpy array of the polynomial's coefficients, ordered from lowest
order term to highest.  For example, array([1,2,3]) represents
``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
applicable to the specific module in question, e.g., `polynomial` (which
"wraps" the "standard" basis) or `chebyshev`.  For optimal performance,
all operations on polynomials, including evaluation at an argument, are
implemented as operations on the coefficients.  Additional (module-specific)
information can be found in the docstring for the module of interest.

This package provides *convenience classes* for each of six different kinds
of polynomials:

========================    ================
**Name**                    **Provides**
========================    ================
`~polynomial.Polynomial`    Power series
`~chebyshev.Chebyshev`      Chebyshev series
`~legendre.Legendre`        Legendre series
`~laguerre.Laguerre`        Laguerre series
`~hermite.Hermite`          Hermite series
`~hermite_e.HermiteE`       HermiteE series
========================    ================

These *convenience classes* provide a consistent interface for creating,
manipulating, and fitting data with polynomials of different bases.
The convenience classes are the preferred interface for the `~numpy.polynomial`
package, and are available from the ``numpy.polynomial`` namespace.
This eliminates the need to navigate to the corresponding submodules, e.g.
``np.polynomial.Polynomial`` or ``np.polynomial.Chebyshev`` instead of
``np.polynomial.polynomial.Polynomial`` or
``np.polynomial.chebyshev.Chebyshev``, respectively.
The classes provide a more consistent and concise interface than the
type-specific functions defined in the submodules for each type of polynomial.
For example, to fit a Chebyshev polynomial with degree ``1`` to data given
by arrays ``xdata`` and ``ydata``, the
`~chebyshev.Chebyshev.fit` class method::

    >>> from numpy.polynomial import Chebyshev
    >>> xdata = [1, 2, 3, 4]
    >>> ydata = [1, 4, 9, 16]
    >>> c = Chebyshev.fit(xdata, ydata, deg=1)

is preferred over the `chebyshev.chebfit` function from the
``np.polynomial.chebyshev`` module::

    >>> from numpy.polynomial.chebyshev import chebfit
    >>> c = chebfit(xdata, ydata, deg=1)

See :doc:`routines.polynomials.classes` for more details.

Convenience Classes
===================

The following lists the various constants and methods common to all of
the classes representing the various kinds of polynomials. In the following,
the term ``Poly`` represents any one of the convenience classes (e.g.
`~polynomial.Polynomial`, `~chebyshev.Chebyshev`, `~hermite.Hermite`, etc.)
while the lowercase ``p`` represents an **instance** of a polynomial class.

Constants
---------

- ``Poly.domain``     -- Default domain
- ``Poly.window``     -- Default window
- ``Poly.basis_name`` -- String used to represent the basis
- ``Poly.maxpower``   -- Maximum value ``n`` such that ``p**n`` is allowed
- ``Poly.nickname``   -- String used in printing

Creation
--------

Methods for creating polynomial instances.

- ``Poly.basis(degree)``    -- Basis polynomial of given degree
- ``Poly.identity()``       -- ``p`` where ``p(x) = x`` for all ``x``
- ``Poly.fit(x, y, deg)``   -- ``p`` of degree ``deg`` with coefficients
  determined by the least-squares fit to the data ``x``, ``y``
- ``Poly.fromroots(roots)`` -- ``p`` with specified roots
- ``p.copy()``              -- Create a copy of ``p``

Conversion
----------

Methods for converting a polynomial instance of one kind to another.

- ``p.cast(Poly)``    -- Convert ``p`` to instance of kind ``Poly``
- ``p.convert(Poly)`` -- Convert ``p`` to instance of kind ``Poly`` or map
  between ``domain`` and ``window``

Calculus
--------
- ``p.deriv()`` -- Take the derivative of ``p``
- ``p.integ()`` -- Integrate ``p``

Validation
----------
- ``Poly.has_samecoef(p1, p2)``   -- Check if coefficients match
- ``Poly.has_samedomain(p1, p2)`` -- Check if domains match
- ``Poly.has_sametype(p1, p2)``   -- Check if types match
- ``Poly.has_samewindow(p1, p2)`` -- Check if windows match

Misc
----
- ``p.linspace()`` -- Return ``x, p(x)`` at equally-spaced points in ``domain``
- ``p.mapparms()`` -- Return the parameters for the linear mapping between
  ``domain`` and ``window``.
- ``p.roots()``    -- Return the roots of ``p``.
- ``p.trim()``     -- Remove trailing coefficients.
- ``p.cutdeg(degree)`` -- Truncate ``p`` to given degree
- ``p.truncate(size)`` -- Truncate ``p`` to given size

"""
from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre

__all__ = [
    "set_default_printstyle",
    "polynomial", "Polynomial",
    "chebyshev", "Chebyshev",
    "legendre", "Legendre",
    "hermite", "Hermite",
    "hermite_e", "HermiteE",
    "laguerre", "Laguerre",
]


def set_default_printstyle(style):
    """
    Set the default format for the string representation of polynomials.

    Values for ``style`` must be valid inputs to ``__format__``, i.e. 'ascii'
    or 'unicode'.

    Parameters
    ----------
    style : str
        Format string for default printing style. Must be either 'ascii' or
        'unicode'.

    Notes
    -----
    The default format depends on the platform: 'unicode' is used on
    Unix-based systems and 'ascii' on Windows. This determination is based on
    default font support for the unicode superscript and subscript ranges.

    Examples
    --------
    >>> p = np.polynomial.Polynomial([1, 2, 3])
    >>> c = np.polynomial.Chebyshev([1, 2, 3])
    >>> np.polynomial.set_default_printstyle('unicode')
    >>> print(p)
    1.0 + 2.0·x + 3.0·x²
    >>> print(c)
    1.0 + 2.0·T₁(x) + 3.0·T₂(x)
    >>> np.polynomial.set_default_printstyle('ascii')
    >>> print(p)
    1.0 + 2.0 x + 3.0 x**2
    >>> print(c)
    1.0 + 2.0 T_1(x) + 3.0 T_2(x)
    >>> # Formatting supersedes all class/package-level defaults
    >>> print(f"{p:unicode}")
    1.0 + 2.0·x + 3.0·x²
    """
    if style not in ('unicode', 'ascii'):
        raise ValueError(
            f"Unsupported format string '{style}'. Valid options are 'ascii' "
            f"and 'unicode'"
        )
    _use_unicode = True
    if style == 'ascii':
        _use_unicode = False
    from ._polybase import ABCPolyBase
    ABCPolyBase._use_unicode = _use_unicode


from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""
=================================================
Power Series (:mod:`numpy.polynomial.polynomial`)
=================================================

This module provides a number of objects (mostly functions) useful for
dealing with polynomials, including a `Polynomial` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with polynomial objects is in
the docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Polynomial

Constants
---------
.. autosummary::
   :toctree: generated/

   polydomain
   polyzero
   polyone
   polyx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   polyadd
   polysub
   polymulx
   polymul
   polydiv
   polypow
   polyval
   polyval2d
   polyval3d
   polygrid2d
   polygrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   polyder
   polyint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   polyfromroots
   polyroots
   polyvalfromroots
   polyvander
   polyvander2d
   polyvander3d
   polycompanion
   polyfit
   polytrim
   polyline

See Also
--------
`numpy.polynomial`

"""
__all__ = [
    'polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd',
    'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval',
    'polyvalfromroots', 'polyder', 'polyint', 'polyfromroots', 'polyvander',
    'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d',
    'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d',
    'polycompanion']

import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

polytrim = pu.trimcoef

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Polynomial default domain.
polydomain = np.array([-1., 1.])

# Polynomial coefficients representing zero.
polyzero = np.array([0])

# Polynomial coefficients representing one.
polyone = np.array([1])

# Polynomial coefficients representing the identity x.
polyx = np.array([0, 1])

#
# Polynomial series functions
#


def polyline(off, scl):
    """
    Returns an array representing a linear polynomial.

    Parameters
    ----------
    off, scl : scalars
        The "y-intercept" and "slope" of the line, respectively.

    Returns
    -------
    y : ndarray
        This module's representation of the linear polynomial ``off +
        scl*x``.

    See Also
    --------
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyline(1, -1)
    array([ 1, -1])
    >>> P.polyval(1, P.polyline(1, -1))  # should be 0
    0.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def polyfromroots(roots):
    """
    Generate a monic polynomial with given roots.

    Return the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    where the :math:`r_n` are the roots specified in `roots`.  If a zero has
    multiplicity n, then it must appear in `roots` n times. For instance,
    if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,
    then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear
    in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * x + ... +  x^n

    The coefficient of the last term is 1 for monic polynomials in this
    form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of the polynomial's coefficients If all the roots are
        real, then `out` is also real, otherwise it is complex.  (see
        Examples below).

    See Also
    --------
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Notes
    -----
    The coefficients are determined by multiplying together linear factors
    of the form ``(x - r_i)``, i.e.

    .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)

    where ``n == len(roots) - 1``; note that this implies that ``1`` is always
    returned for :math:`a_n`.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyfromroots((-1,0,1))  # x(x - 1)(x + 1) = x^3 - x
    array([ 0., -1.,  0.,  1.])
    >>> j = complex(0,1)
    >>> P.polyfromroots((-j,j))  # complex returned, though values are real
    array([1.+0.j,  0.+0.j,  1.+0.j])

    """
    return pu._fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    """
    Add one polynomial to another.

    Returns the sum of two polynomials `c1` + `c2`.  The arguments are
    sequences of coefficients from lowest order term to highest, i.e.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        The coefficient array representing their sum.

    See Also
    --------
    polysub, polymulx, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> sum = P.polyadd(c1,c2); sum
    array([4.,  4.,  4.])
    >>> P.polyval(2, sum)  # 4 + 4(2) + 4(2**2)
    28.0

    """
    return pu._add(c1, c2)


def polysub(c1, c2):
    """
    Subtract one polynomial from another.

    Returns the difference of two polynomials `c1` - `c2`.  The arguments
    are sequences of coefficients from lowest order term to highest, i.e.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of coefficients representing their difference.

    See Also
    --------
    polyadd, polymulx, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polysub(c1,c2)
    array([-2.,  0.,  2.])
    >>> P.polysub(c2, c1)  # -P.polysub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return pu._sub(c1, c2)


def polymulx(c):
    """Multiply a polynomial by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    polyadd, polysub, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polymulx(c)
    array([0., 1., 2., 3.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1:] = c
    return prd


def polymul(c1, c2):
    """
    Multiply one polynomial by another.

    Returns the product of two polynomials `c1` * `c2`.  The arguments are
    sequences of coefficients, from lowest order term to highest, e.g.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of coefficients representing a polynomial, relative to the
        "standard" basis, and ordered from lowest order term to highest.

    Returns
    -------
    out : ndarray
        Of the coefficients of their product.

    See Also
    --------
    polyadd, polysub, polymulx, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polymul(c1, c2)
    array([  3.,   8.,  14.,   8.,   3.])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    ret = np.convolve(c1, c2)
    return pu.trimseq(ret)


def polydiv(c1, c2):
    """
    Divide one polynomial by another.

    Returns the quotient-with-remainder of two polynomials `c1` / `c2`.
    The arguments are sequences of coefficients, from lowest order term
    to highest, e.g., [1,2,3] represents ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of coefficient series representing the quotient and remainder.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polydiv(c1, c2)
    (array([3.]), array([-8., -4.]))
    >>> P.polydiv(c2, c1)
    (array([ 0.33333333]), array([ 2.66666667,  1.33333333]))  # may vary

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError  # FIXME: add message with details to exception

    # note: this is more efficient than `pu._div(polymul, c1, c2)`
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1]*0, c1
    elif lc2 == 1:
        return c1/c2[-1], c1[:1]*0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1]/scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2*c1[j]
            i -= 1
            j -= 1
        return c1[j+1:]/scl, pu.trimseq(c1[:j+1])


def polypow(c, pow, maxpower=None):
    """Raise a polynomial to a power.

    Returns the polynomial `c` raised to the power `pow`. The argument
    `c` is a sequence of coefficients ordered from low to high. i.e.,
    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c : array_like
        1-D array of array of series coefficients ordered from low to
        high degree.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Power series of power.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polydiv

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polypow([1, 2, 3], 2)
    array([ 1., 4., 10., 12., 9.])

    """
    # note: this is more efficient than `pu._pow(polymul, c1, c2)`, as it
    # avoids calling `as_series` repeatedly
    return pu._pow(np.convolve, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    """
    Differentiate a polynomial.

    Returns the polynomial coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable).  The
    argument `c` is an array of coefficients from low to high degree along
    each axis, e.g., [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``
    while [[1,2],[1,2]] represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is
    ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of polynomial coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree
        in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change
        of variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Polynomial coefficients of the derivative.

    See Also
    --------
    polyint

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3, 4)
    >>> P.polyder(c)  # (d/dx)(c)
    array([  2.,   6.,  12.])
    >>> P.polyder(c, 3)  # (d**3/dx**3)(c)
    array([24.])
    >>> P.polyder(c, scl=-1)  # (d/d(-x))(c)
    array([ -2.,  -6., -12.])
    >>> P.polyder(c, 2, -1)  # (d**2/d(-x)**2)(c)
    array([  6.,  24.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    cdt = c.dtype
    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1]*0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=cdt)
            for j in range(n, 0, -1):
                der[j - 1] = j*c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a polynomial.

    Returns the polynomial coefficients `c` integrated `m` times from
    `lbnd` along `axis`.  At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.) The argument `c` is an array of
    coefficients, from low to high degree along each axis, e.g., [1,2,3]
    represents the polynomial ``1 + 2*x + 3*x**2`` while [[1,2],[1,2]]
    represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients, ordered from low to high.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    polyder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.  Why
    is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`. Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polyint(c)  # should return array([0, 1, 1, 1])
    array([0.,  1.,  1.,  1.])
    >>> P.polyint(c, 3)  # should return array([0, 0, 0, 1/6, 1/12, 1/20])
     array([ 0.        ,  0.        ,  0.        ,  0.16666667,  0.08333333, # may vary
             0.05      ])
    >>> P.polyint(c, k=3)  # should return array([3, 1, 1, 1])
    array([3.,  1.,  1.,  1.])
    >>> P.polyint(c,lbnd=-2)  # should return array([6, 1, 1, 1])
    array([6.,  1.,  1.,  1.])
    >>> P.polyint(c,scl=-2)  # should return array([0, -2, -2, -2])
    array([ 0., -2., -2., -2.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype doesn't preserve mask attribute.
        c = c + 0.0
    cdt = c.dtype
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0]*(cnt - len(k))
    c = np.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j]/(j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def polyval(x, c, tensor=True):
    """
    Evaluate a polynomial at points x.

    If `c` is of length ``n + 1``, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyval2d, polygrid2d, polyval3d, polygrid3d

    Notes
    -----
    The evaluation uses Horner's method.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.polynomial import polyval
    >>> polyval(1, [1,2,3])
    6.0
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyval(a, [1, 2, 3])
    array([[ 1.,   6.],
           [17.,  34.]])
    >>> coef = np.arange(4).reshape(2, 2)  # multidimensional coefficients
    >>> coef
    array([[0, 1],
           [2, 3]])
    >>> polyval([1, 2], coef, tensor=True)
    array([[2.,  4.],
           [4.,  7.]])
    >>> polyval([1, 2], coef, tensor=False)
    array([2.,  7.])

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    c0 = c[-1] + x*0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0*x
    return c0


def polyvalfromroots(x, r, tensor=True):
    """
    Evaluate a polynomial specified by its roots at points x.

    If `r` is of length ``N``, this function returns the value

    .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `r`.

    If `r` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If `r`
    is multidimensional, then the shape of the result depends on the value of
    `tensor`. If `tensor` is ``True`` the shape will be r.shape[1:] + x.shape;
    that is, each polynomial is evaluated at every value of `x`. If `tensor` is
    ``False``, the shape will be r.shape[1:]; that is, each polynomial is
    evaluated only for the corresponding broadcast value of `x`. Note that
    scalars have shape (,).

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `r`.
    r : array_like
        Array of roots. If `r` is multidimensional the first index is the
        root index, while the remaining indices enumerate multiple
        polynomials. For instance, in the two dimensional case the roots
        of each polynomial may be thought of as stored in the columns of `r`.
    tensor : boolean, optional
        If True, the shape of the roots array is extended with ones on the
        right, one for each dimension of `x`. Scalars have dimension 0 for this
        action. The result is that every column of coefficients in `r` is
        evaluated for every element of `x`. If False, `x` is broadcast over the
        columns of `r` for the evaluation.  This keyword is useful when `r` is
        multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyroots, polyfromroots, polyval

    Examples
    --------
    >>> from numpy.polynomial.polynomial import polyvalfromroots
    >>> polyvalfromroots(1, [1, 2, 3])
    0.0
    >>> a = np.arange(4).reshape(2, 2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyvalfromroots(a, [-1, 0, 1])
    array([[-0.,   0.],
           [ 6.,  24.]])
    >>> r = np.arange(-2, 2).reshape(2,2)  # multidimensional coefficients
    >>> r # each column of r defines one polynomial
    array([[-2, -1],
           [ 0,  1]])
    >>> b = [-2, 1]
    >>> polyvalfromroots(b, r, tensor=True)
    array([[-0.,  3.],
           [ 3., 0.]])
    >>> polyvalfromroots(b, r, tensor=False)
    array([-0.,  0.])

    """
    r = np.array(r, ndmin=1, copy=None)
    if r.dtype.char in '?bBhHiIlLqQpP':
        r = r.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,)*x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")
    return np.prod(x - r, axis=0)


def polyval2d(x, y, c):
    """
    Evaluate a 2-D polynomial at points (x, y).

    This function returns the value

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    polyval, polygrid2d, polyval3d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polyval2d(1, 1, c)
    21.0

    """
    return pu._valnd(polyval, c, x, y)


def polygrid2d(x, y, c):
    """
    Evaluate a 2-D polynomial on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * a^i * b^j

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    polyval, polyval2d, polyval3d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polygrid2d([0, 1], [0, 1], c)
    array([[ 1.,  6.],
           [ 5., 21.]])

    """
    return pu._gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    """
    Evaluate a 3-D polynomial at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * x^i * y^j * z^k

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    polyval, polyval2d, polygrid2d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polyval3d(1, 1, 1, c)
    45.0

    """
    return pu._valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    """
    Evaluate a 3-D polynomial on the Cartesian product of x, y and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * a^i * b^j * c^k

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    polyval, polyval2d, polygrid2d, polyval3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polygrid3d([0, 1], [0, 1], [0, 1], c)
    array([[ 1., 13.],
           [ 6., 51.]])

    """
    return pu._gridnd(polyval, c, x, y, z)


def polyvander(x, deg):
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points
    `x`. The Vandermonde matrix is defined by

    .. math:: V[..., i] = x^i,

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the power of `x`.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    matrix ``V = polyvander(x, n)``, then ``np.dot(V, c)`` and
    ``polyval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of polynomials of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray.
        The Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where the last index is the power of `x`.
        The dtype will be the same as the converted `x`.

    See Also
    --------
    polyvander2d, polyvander3d

    Examples
    --------
    The Vandermonde matrix of degree ``deg = 5`` and sample points
    ``x = [-1, 2, 3]`` contains the element-wise powers of `x`
    from 0 to 5 as its columns.

    >>> from numpy.polynomial import polynomial as P
    >>> x, deg = [-1, 2, 3], 5
    >>> P.polyvander(x=x, deg=deg)
    array([[  1.,  -1.,   1.,  -1.,   1.,  -1.],
           [  1.,   2.,   4.,   8.,  16.,  32.],
           [  1.,   3.,   9.,  27.,  81., 243.]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x*0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i-1]*x
    return np.moveaxis(v, 0, -1)


def polyvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = x^i * y^j,

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the powers of
    `x` and `y`.

    If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np

    The 2-D pseudo-Vandermonde matrix of degree ``[1, 2]`` and sample
    points ``x = [-1, 2]`` and ``y = [1, 3]`` is as follows:

    >>> from numpy.polynomial import polynomial as P
    >>> x = np.array([-1, 2])
    >>> y = np.array([1, 3])
    >>> m, n = 1, 2
    >>> deg = np.array([m, n])
    >>> V = P.polyvander2d(x=x, y=y, deg=deg)
    >>> V
    array([[ 1.,  1.,  1., -1., -1., -1.],
           [ 1.,  3.,  9.,  2.,  6., 18.]])

    We can verify the columns for any ``0 <= i <= m`` and ``0 <= j <= n``:

    >>> i, j = 0, 1
    >>> V[:, (deg[1]+1)*i + j] == x**i * y**j
    array([ True,  True])

    The (1D) Vandermonde matrix of sample points ``x`` and degree ``m`` is a
    special case of the (2D) pseudo-Vandermonde matrix with ``y`` points all
    zero and degree ``[m, 0]``.

    >>> P.polyvander2d(x=x, y=0*x, deg=(m, 0)) == P.polyvander(x=x, deg=m)
    array([[ True,  True],
           [ True,  True]])

    """
    return pu._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the powers of `x`, `y`, and `z`.

    If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.asarray([-1, 2, 1])
    >>> y = np.asarray([1, -2, -3])
    >>> z = np.asarray([2, 2, 5])
    >>> l, m, n = [2, 2, 1]
    >>> deg = [l, m, n]
    >>> V = P.polyvander3d(x=x, y=y, z=z, deg=deg)
    >>> V
    array([[  1.,   2.,   1.,   2.,   1.,   2.,  -1.,  -2.,  -1.,
             -2.,  -1.,  -2.,   1.,   2.,   1.,   2.,   1.,   2.],
           [  1.,   2.,  -2.,  -4.,   4.,   8.,   2.,   4.,  -4.,
             -8.,   8.,  16.,   4.,   8.,  -8., -16.,  16.,  32.],
           [  1.,   5.,  -3., -15.,   9.,  45.,   1.,   5.,  -3.,
            -15.,   9.,  45.,   1.,   5.,  -3., -15.,   9.,  45.]])

    We can verify the columns for any ``0 <= i <= l``, ``0 <= j <= m``,
    and ``0 <= k <= n``

    >>> i, j, k = 2, 1, 0
    >>> V[:, (m+1)*(n+1)*i + (n+1)*j + k] == x**i * y**j * z**k
    array([ True,  True,  True])

    """
    return pu._vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least-squares fit of a polynomial to data.

    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit.  Singular values smaller
        than `rcond`, relative to the largest singular value, will be
        ignored.  The default value is ``len(x)*eps``, where `eps` is the
        relative precision of the platform's float type, about 2e-16 in
        most cases.
    full : bool, optional
        Switch determining the nature of the return value.  When ``False``
        (the default) just the coefficients are returned; when ``True``,
        diagnostic information from the singular value decomposition (used
        to solve the fit's matrix equation) is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Raises
    ------
    RankWarning
        Raised if the matrix in the least-squares fit is rank deficient.
        The warning is only raised if ``full == False``.  The warnings can
        be turned off by:

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    polyval : Evaluates a polynomial.
    polyvander : Vandermonde matrix for powers.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the polynomial `p` that minimizes
    the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) over-determined matrix equation:

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected (and `full` == ``False``), a `~exceptions.RankWarning` will be
    raised.  This means that the coefficient values may be poorly determined.
    Fitting to a lower order polynomial will usually get rid of the warning
    (but may not be what you want, of course; if you have independent
    reason(s) for choosing the degree which isn't working, you may have to:
    a) reconsider those reasons, and/or b) reconsider the quality of your
    data).  The `rcond` parameter can also be set to a value smaller than
    its default, but the resulting fit may be spurious and have large
    contributions from roundoff error.

    Polynomial fits using double precision tend to "fail" at about
    (polynomial) degree 20. Fits using Chebyshev or Legendre series are
    generally better conditioned, but much can still depend on the
    distribution of the sample points and the smoothness of the data.  If
    the quality of the fit is inadequate, splines may be a good
    alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.linspace(-1,1,51)  # x "data": [-1, -0.96, ..., 0.96, 1]
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(size=len(x))
    >>> y = x**3 - x + err  # x^3 - x + Gaussian noise
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] approx. -1, c[2] should be approx. 0, c[3] approx. 1
    array([ 0.23111996, -1.02785049, -0.2241444 ,  1.08405657]) # may vary
    >>> stats # note the large SSR, explaining the rather poor results
    [array([48.312088]),                                        # may vary
     4,
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    Same thing without the added noise

    >>> y = x**3 - x
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] ~= -1, c[2] should be "very close to 0", c[3] ~= 1
    array([-6.73496154e-17, -1.00000000e+00,  0.00000000e+00,  1.00000000e+00])
    >>> stats # note the minuscule SSR
    [array([8.79579319e-31]),
     np.int32(4),
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    """
    return pu._fit(polyvander, x, y, deg, rcond, full, w)


def polycompanion(c):
    """
    Return the companion matrix of c.

    The companion matrix for power series cannot be made symmetric by
    scaling the basis, so this function differs from those for the
    orthogonal polynomials.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polycompanion(c)
    array([[ 0.        , -0.33333333],
           [ 1.        , -0.66666667]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0]/c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    bot = mat.reshape(-1)[n::n+1]
    bot[...] = 1
    mat[:, -1] -= c[:-1]/c[-1]
    return mat


def polyroots(c):
    """
    Compute the roots of a polynomial.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * x^i.

    Parameters
    ----------
    c : 1-D array_like
        1-D array of polynomial coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the polynomial. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the power series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    Examples
    --------
    >>> import numpy.polynomial.polynomial as poly
    >>> poly.polyroots(poly.polyfromroots((-1,0,1)))
    array([-1.,  0.,  1.])
    >>> poly.polyroots(poly.polyfromroots((-1,0,1))).dtype
    dtype('float64')
    >>> j = complex(0,1)
    >>> poly.polyroots(poly.polyfromroots((-j,0,j)))
    array([  0.00000000e+00+0.j,   0.00000000e+00+1.j,   2.77555756e-17-1.j])  # may vary

    """  # noqa: E501
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0]/c[1]])

    # rotated companion matrix reduces error
    m = polycompanion(c)[::-1,::-1]
    r = la.eigvals(m)
    r.sort()
    return r


#
# polynomial class
#

class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)

    # Virtual properties
    domain = np.array(polydomain)
    window = np.array(polydomain)
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        if i == '1':
            return f"·{arg_str}"
        else:
            return f"·{arg_str}{i.translate(cls._superscript_mapping)}"

    @staticmethod
    def _str_term_ascii(i, arg_str):
        if i == '1':
            return f" {arg_str}"
        else:
            return f" {arg_str}**{i}"

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        if needs_parens:
            arg_str = rf"\left({arg_str}\right)"
        if i == 0:
            return '1'
        elif i == 1:
            return arg_str
        else:
            return f"{arg_str}^{{{i}}}"
<|endoftext|>"""
===================================================================
HermiteE Series, "Probabilists" (:mod:`numpy.polynomial.hermite_e`)
===================================================================

This module provides a number of objects (mostly functions) useful for
dealing with Hermite_e series, including a `HermiteE` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   HermiteE

Constants
---------
.. autosummary::
   :toctree: generated/

   hermedomain
   hermezero
   hermeone
   hermex

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   hermeadd
   hermesub
   hermemulx
   hermemul
   hermediv
   hermepow
   hermeval
   hermeval2d
   hermeval3d
   hermegrid2d
   hermegrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   hermeder
   hermeint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   hermefromroots
   hermeroots
   hermevander
   hermevander2d
   hermevander3d
   hermegauss
   hermeweight
   hermecompanion
   hermefit
   hermetrim
   hermeline
   herme2poly
   poly2herme

See also
--------
`numpy.polynomial`

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline',
    'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv',
    'hermepow', 'hermeval', 'hermeder', 'hermeint', 'herme2poly',
    'poly2herme', 'hermefromroots', 'hermevander', 'hermefit', 'hermetrim',
    'hermeroots', 'HermiteE', 'hermeval2d', 'hermeval3d', 'hermegrid2d',
    'hermegrid3d', 'hermevander2d', 'hermevander3d', 'hermecompanion',
    'hermegauss', 'hermeweight']

hermetrim = pu.trimcoef


def poly2herme(pol):
    """
    poly2herme(pol)

    Convert a polynomial to a Hermite series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Hermite series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Hermite
        series.

    See Also
    --------
    herme2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite_e import poly2herme
    >>> poly2herme(np.arange(4))
    array([  2.,  10.,   2.,   3.])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), pol[i])
    return res


def herme2poly(c):
    """
    Convert a Hermite series to a polynomial.

    Convert an array representing the coefficients of a Hermite series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Hermite series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2herme

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import herme2poly
    >>> herme2poly([  2.,  10.,   2.,   3.])
    array([0.,  1.,  2.,  3.])

    """
    from .polynomial import polyadd, polysub, polymulx

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1*(i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Hermite
hermedomain = np.array([-1., 1.])

# Hermite coefficients representing zero.
hermezero = np.array([0])

# Hermite coefficients representing one.
hermeone = np.array([1])

# Hermite coefficients representing the identity x.
hermex = np.array([0, 1])


def hermeline(off, scl):
    """
    Hermite series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Hermite series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeline
    >>> from numpy.polynomial.hermite_e import hermeline, hermeval
    >>> hermeval(0,hermeline(3, 2))
    3.0
    >>> hermeval(1,hermeline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def hermefromroots(roots):
    """
    Generate a HermiteE series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in HermiteE form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * He_1(x) + ... +  c_n * He_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in HermiteE form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.chebyshev.chebfromroots

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermefromroots, hermeval
    >>> coef = hermefromroots((-1, 0, 1))
    >>> hermeval((-1, 0, 1), coef)
    array([0., 0., 0.])
    >>> coef = hermefromroots((-1j, 1j))
    >>> hermeval((-1j, 1j), coef)
    array([0.+0.j, 0.+0.j])

    """
    return pu._fromroots(hermeline, hermemul, roots)


def hermeadd(c1, c2):
    """
    Add one Hermite series to another.

    Returns the sum of two Hermite series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Hermite series of their sum.

    See Also
    --------
    hermesub, hermemulx, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Hermite series
    is a Hermite series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeadd
    >>> hermeadd([1, 2, 3], [1, 2, 3, 4])
    array([2.,  4.,  6.,  4.])

    """
    return pu._add(c1, c2)


def hermesub(c1, c2):
    """
    Subtract one Hermite series from another.

    Returns the difference of two Hermite series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their difference.

    See Also
    --------
    hermeadd, hermemulx, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Hermite
    series is a Hermite series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermesub
    >>> hermesub([1, 2, 3, 4], [1, 2, 3])
    array([0., 0., 0., 4.])

    """
    return pu._sub(c1, c2)


def hermemulx(c):
    """Multiply a Hermite series by x.

    Multiply the Hermite series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    hermeadd, hermesub, hermemul, hermediv, hermepow

    Notes
    -----
    The multiplication uses the recursion relationship for Hermite
    polynomials in the form

    .. math::

        xP_i(x) = (P_{i + 1}(x) + iP_{i - 1}(x)))

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemulx
    >>> hermemulx([1, 2, 3])
    array([2.,  7.,  2.,  3.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1] = c[0]
    for i in range(1, len(c)):
        prd[i + 1] = c[i]
        prd[i - 1] += c[i]*i
    return prd


def hermemul(c1, c2):
    """
    Multiply one Hermite series by another.

    Returns the product of two Hermite series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their product.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermediv, hermepow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemul
    >>> hermemul([1, 2, 3], [0, 1, 2])
    array([14.,  15.,  28.,   7.,   6.])

    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0]*xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]*xs
        c1 = c[1]*xs
    else:
        nd = len(c)
        c0 = c[-2]*xs
        c1 = c[-1]*xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermesub(c[-i]*xs, c1*(nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1))
    return hermeadd(c0, hermemulx(c1))


def hermediv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermepow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "reproject" the results onto the Hermite
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermediv
    >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 2.]))

    """
    return pu._div(hermemul, c1, c2)


def hermepow(c, pow, maxpower=16):
    """Raise a Hermite series to a power.

    Returns the Hermite series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Hermite series of power.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermediv

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermepow
    >>> hermepow([1, 2, 3], 2)
    array([23.,  28.,  46.,  12.,   9.])

    """
    return pu._pow(hermemul, c, pow, maxpower)


def hermeder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Hermite_e series.

    Returns the series coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*He_0 + 2*He_1 + 3*He_2``
    while [[1,2],[1,2]] represents ``1*He_0(x)*He_0(y) + 1*He_1(x)*He_0(y)
    + 2*He_0(x)*He_1(y) + 2*He_1(x)*He_1(y)`` if axis=0 is ``x`` and axis=1
    is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Hermite series of the derivative.

    See Also
    --------
    hermeint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeder
    >>> hermeder([ 1.,  1.,  1.,  1.])
    array([1.,  2.,  3.])
    >>> hermeder([-0.25,  1.,  1./2.,  1./3.,  1./4 ], m=2)
    array([1.,  2.,  3.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        return c[:1]*0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = j*c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def hermeint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite_e series.

    Returns the Hermite_e series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Hermite_e series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    hermeder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeint
    >>> hermeint([1, 2, 3]) # integrate once, value 0 at 0.
    array([1., 1., 1., 1.])
    >>> hermeint([1, 2, 3], m=2) # integrate twice, value & deriv 0 at 0
    array([-0.25      ,  1.        ,  0.5       ,  0.33333333,  0.25      ]) # may vary
    >>> hermeint([1, 2, 3], k=1) # integrate once, value 1 at 0.
    array([2., 1., 1., 1.])
    >>> hermeint([1, 2, 3], lbnd=-1) # integrate once, value 0 at -1
    array([-1.,  1.,  1.,  1.])
    >>> hermeint([1, 2, 3], m=2, k=[1, 2], lbnd=-1)
    array([ 1.83333333,  0.        ,  0.5       ,  0.33333333,  0.25      ]) # may vary

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    k = list(k) + [0]*(cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j]/(j + 1)
            tmp[0] += k[i] - hermeval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def hermeval(x, c, tensor=True):
    """
    Evaluate an HermiteE series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * He_0(x) + c_1 * He_1(x) + ... + c_n * He_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    hermeval2d, hermegrid2d, hermeval3d, hermegrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeval
    >>> coef = [1,2,3]
    >>> hermeval(1, coef)
    3.0
    >>> hermeval([[1,2],[3,4]], coef)
    array([[ 3., 14.],
           [31., 54.]])

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1*(nd - 1)
            c1 = tmp + c1*x
    return c0 + c1*x


def hermeval2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * He_i(x) * He_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    hermeval, hermegrid2d, hermeval3d, hermegrid3d
    """
    return pu._valnd(hermeval, c, x, y)


def hermegrid2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * H_i(a) * H_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    hermeval, hermeval2d, hermeval3d, hermegrid3d
    """
    return pu._gridnd(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite_e series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * He_i(x) * He_j(y) * He_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    hermeval, hermeval2d, hermegrid2d, hermegrid3d
    """
    return pu._valnd(hermeval, c, x, y, z)


def hermegrid3d(x, y, z, c):
    """
    Evaluate a 3-D HermiteE series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * He_i(a) * He_j(b) * He_k(c)

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    hermeval, hermeval2d, hermegrid2d, hermeval3d
    """
    return pu._gridnd(hermeval, c, x, y, z)


def hermevander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = He_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the HermiteE polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = hermevander(x, n)``, then ``np.dot(V, c)`` and
    ``hermeval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of HermiteE series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding HermiteE polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite_e import hermevander
    >>> x = np.array([-1, 0, 1])
    >>> hermevander(x, 3)
    array([[ 1., -1.,  0.,  2.],
           [ 1.,  0., -1., -0.],
           [ 1.,  1.,  0., -2.]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x*0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i-1]*x - v[i-2]*(i - 1))
    return np.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = He_i(x) * He_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the HermiteE polynomials.

    If ``V = hermevander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``hermeval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D HermiteE
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    hermevander, hermevander3d, hermeval2d, hermeval3d
    """
    return pu._vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then Hehe pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = He_i(x)*He_j(y)*He_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the HermiteE polynomials.

    If ``V = hermevander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``hermeval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D HermiteE
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    hermevander, hermevander3d, hermeval2d, hermeval3d
    """
    return pu._vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Hermite series to data.

    Return the coefficients of a HermiteE series of degree `deg` that is
    the least squares fit to the data values `y` given at points `x`. If
    `y` is 1-D the returned coefficients will also be 1-D. If `y` is 2-D
    multiple fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * He_1(x) + ... + c_n * He_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Hermite coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full = False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.laguerre.lagfit
    hermeval : Evaluates a Hermite series.
    hermevander : pseudo Vandermonde matrix of Hermite series.
    hermeweight : HermiteE weight function.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the HermiteE series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the pseudo Vandermonde matrix of `x`, the elements of `c`
    are the coefficients to be solved for, and the elements of `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using HermiteE series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the HermiteE
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `hermeweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite_e import hermefit, hermeval
    >>> x = np.linspace(-10, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = hermeval(x, [1, 2, 3]) + err
    >>> hermefit(x, y, 2)
    array([1.02284196, 2.00032805, 2.99978457]) # may vary

    """
    return pu._fit(hermevander, x, y, deg, rcond, full, w)


def hermecompanion(c):
    """
    Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an HermiteE basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of HermiteE series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).
    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0]/c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = np.hstack((1., 1./np.sqrt(np.arange(n - 1, 0, -1))))
    scl = np.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[...] = np.sqrt(np.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl*c[:-1]/c[-1]
    return mat


def hermeroots(c):
    """
    Compute the roots of a HermiteE series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * He_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.chebyshev.chebroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The HermiteE series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeroots, hermefromroots
    >>> coef = hermefromroots([-1, 0, 1])
    >>> coef
    array([0., 2., 0., 1.])
    >>> hermeroots(coef)
    array([-1.,  0.,  1.]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0]/c[1]])

    # rotated companion matrix reduces error
    m = hermecompanion(c)[::-1,::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def _normed_hermite_e_n(x, n):
    """
    Evaluate a normalized HermiteE polynomial.

    Compute the value of the normalized HermiteE polynomial of degree ``n``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    n : int
        Degree of the normalized HermiteE function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard HermiteE functions
    overflow when n >= 207.

    """
    if n == 0:
        return np.full(x.shape, 1/np.sqrt(np.sqrt(2*np.pi)))

    c0 = 0.
    c1 = 1./np.sqrt(np.sqrt(2*np.pi))
    nd = float(n)
    for i in range(n - 1):
        tmp = c0
        c0 = -c1*np.sqrt((nd - 1.)/nd)
        c1 = tmp + c1*x*np.sqrt(1./nd)
        nd = nd - 1.0
    return c0 + c1*x


def hermegauss(deg):
    """
    Gauss-HermiteE quadrature.

    Computes the sample points and weights for Gauss-HermiteE quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x^2/2)`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (He'_n(x_k) * He_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`He_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0]*deg + [1])
    m = hermecompanion(c)
    x = la.eigvalsh(m)

    # improve roots by one application of Newton
    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * np.sqrt(ideg)
    x -= dy/df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = _normed_hermite_e_n(x, ideg - 1)
    fm /= np.abs(fm).max()
    w = 1/(fm * fm)

    # for Hermite_e we can also symmetrize
    w = (w + w[::-1])/2
    x = (x - x[::-1])/2

    # scale w to get the right value
    w *= np.sqrt(2*np.pi) / w.sum()

    return x, w


def hermeweight(x):
    """Weight function of the Hermite_e polynomials.

    The weight function is :math:`\\exp(-x^2/2)` and the interval of
    integration is :math:`[-\\inf, \\inf]`. the HermiteE polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.
    """
    w = np.exp(-.5*x**2)
    return w


#
# HermiteE series class
#

class HermiteE(ABCPolyBase):
    """An HermiteE series class.

    The HermiteE class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        HermiteE coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*He_0(x) + 2*He_1(X) + 3*He_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(hermeadd)
    _sub = staticmethod(hermesub)
    _mul = staticmethod(hermemul)
    _div = staticmethod(hermediv)
    _pow = staticmethod(hermepow)
    _val = staticmethod(hermeval)
    _int = staticmethod(hermeint)
    _der = staticmethod(hermeder)
    _fit = staticmethod(hermefit)
    _line = staticmethod(hermeline)
    _roots = staticmethod(hermeroots)
    _fromroots = staticmethod(hermefromroots)

    # Virtual properties
    domain = np.array(hermedomain)
    window = np.array(hermedomain)
    basis_name = 'He'
<|endoftext|>"""
Utility classes and functions for the polynomial modules.

This module provides: error and warning objects; a polynomial base class;
and some routines used in both the `polynomial` and `chebyshev` modules.

Functions
---------

.. autosummary::
   :toctree: generated/

   as_series    convert list of array_likes into 1-D arrays of common type.
   trimseq      remove trailing zeros.
   trimcoef     remove small trailing coefficients.
   getdomain    return the domain appropriate for a given set of abscissae.
   mapdomain    maps points between domains.
   mapparms     parameters of the linear map between domains.

"""
import operator
import functools
import warnings

import numpy as np

from numpy._core.multiarray import dragon4_positional, dragon4_scientific
from numpy.exceptions import RankWarning

__all__ = [
    'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain', 'mapparms',
    'format_float']

#
# Helper functions to convert inputs to 1-D arrays
#
def trimseq(seq):
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    if len(seq) == 0 or seq[-1] != 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[:i+1]


def as_series(alist, trim=True):
    """
    Return argument as a list of 1-d arrays.

    The returned list contains array(s) of dtype double, complex double, or
    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
    raises a Value Error if it is not first reshaped into either a 1-d or 2-d
    array.

    Parameters
    ----------
    alist : array_like
        A 1- or 2-d array_like
    trim : boolean, optional
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    [a1, a2,...] : list of 1-D arrays
        A copy of the input data as a list of 1-d arrays.

    Raises
    ------
    ValueError
        Raised when `as_series` cannot convert its input to 1-d arrays, or at
        least one of the resulting arrays is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> a = np.arange(4)
    >>> pu.as_series(a)
    [array([0.]), array([1.]), array([2.]), array([3.])]
    >>> b = np.arange(6).reshape((2,3))
    >>> pu.as_series(b)
    [array([0., 1., 2.]), array([3., 4., 5.])]

    >>> pu.as_series((1, np.arange(3), np.arange(2, dtype=np.float16)))
    [array([1.]), array([0., 1., 2.]), array([0., 1.])]

    >>> pu.as_series([2, [1.1, 0.]])
    [array([2.]), array([1.1])]

    >>> pu.as_series([2, [1.1, 0.]], trim=False)
    [array([2.]), array([1.1, 0. ])]

    """
    arrays = [np.array(a, ndmin=1, copy=None) for a in alist]
    for a in arrays:
        if a.size == 0:
            raise ValueError("Coefficient array is empty")
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Coefficient array is not 1-d")
    if trim:
        arrays = [trimseq(a) for a in arrays]

    if any(a.dtype == np.dtype(object) for a in arrays):
        ret = []
        for a in arrays:
            if a.dtype != np.dtype(object):
                tmp = np.empty(len(a), dtype=np.dtype(object))
                tmp[:] = a[:]
                ret.append(tmp)
            else:
                ret.append(a.copy())
    else:
        try:
            dtype = np.common_type(*arrays)
        except Exception as e:
            raise ValueError("Coefficient arrays have no common type") from e
        ret = [np.array(a, copy=True, dtype=dtype) for a in arrays]
    return ret


def trimcoef(c, tol=0):
    """
    Remove "small" "trailing" coefficients from a polynomial.

    "Small" means "small in absolute value" and is controlled by the
    parameter `tol`; "trailing" means highest order coefficient(s), e.g., in
    ``[0, 1, 1, 0, 0]`` (which represents ``0 + x + x**2 + 0*x**3 + 0*x**4``)
    both the 3-rd and 4-th order coefficients would be "trimmed."

    Parameters
    ----------
    c : array_like
        1-d array of coefficients, ordered from lowest order to highest.
    tol : number, optional
        Trailing (i.e., highest order) elements with absolute value less
        than or equal to `tol` (default value is zero) are removed.

    Returns
    -------
    trimmed : ndarray
        1-d array with trailing zeros removed.  If the resulting series
        would be empty, a series containing a single zero is returned.

    Raises
    ------
    ValueError
        If `tol` < 0

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> pu.trimcoef((0,0,3,0,5,0,0))
    array([0.,  0.,  3.,  0.,  5.])
    >>> pu.trimcoef((0,0,1e-3,0,1e-5,0,0),1e-3) # item == tol is trimmed
    array([0.])
    >>> i = complex(0,1) # works for complex
    >>> pu.trimcoef((3e-4,1e-3*(1-i),5e-4,2e-5*(1+i)), 1e-3)
    array([0.0003+0.j   , 0.001 -0.001j])

    """
    if tol < 0:
        raise ValueError("tol must be non-negative")

    [c] = as_series([c])
    [ind] = np.nonzero(np.abs(c) > tol)
    if len(ind) == 0:
        return c[:1]*0
    else:
        return c[:ind[-1] + 1].copy()

def getdomain(x):
    """
    Return a domain suitable for given abscissae.

    Find a domain suitable for a polynomial or Chebyshev series
    defined at the values supplied.

    Parameters
    ----------
    x : array_like
        1-d array of abscissae whose domain will be determined.

    Returns
    -------
    domain : ndarray
        1-d array containing two values.  If the inputs are complex, then
        the two returned points are the lower left and upper right corners
        of the smallest rectangle (aligned with the axes) in the complex
        plane containing the points `x`. If the inputs are real, then the
        two points are the ends of the smallest interval containing the
        points `x`.

    See Also
    --------
    mapparms, mapdomain

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> points = np.arange(4)**2 - 5; points
    array([-5, -4, -1,  4])
    >>> pu.getdomain(points)
    array([-5.,  4.])
    >>> c = np.exp(complex(0,1)*np.pi*np.arange(12)/6) # unit circle
    >>> pu.getdomain(c)
    array([-1.-1.j,  1.+1.j])

    """
    [x] = as_series([x], trim=False)
    if x.dtype.char in np.typecodes['Complex']:
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return np.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        return np.array((x.min(), x.max()))

def mapparms(old, new):
    """
    Linear map parameters between domains.

    Return the parameters of the linear map ``offset + scale*x`` that maps
    `old` to `new` such that ``old[i] -> new[i]``, ``i = 0, 1``.

    Parameters
    ----------
    old, new : array_like
        Domains. Each domain must (successfully) convert to a 1-d array
        containing precisely two values.

    Returns
    -------
    offset, scale : scalars
        The map ``L(x) = offset + scale*x`` maps the first domain to the
        second.

    See Also
    --------
    getdomain, mapdomain

    Notes
    -----
    Also works for complex numbers, and thus can be used to calculate the
    parameters required to map any line in the complex plane to any other
    line therein.

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> pu.mapparms((-1,1),(-1,1))
    (0.0, 1.0)
    >>> pu.mapparms((1,-1),(-1,1))
    (-0.0, -1.0)
    >>> i = complex(0,1)
    >>> pu.mapparms((-i,-1),(1,i))
    ((1+1j), (1-0j))

    """
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1]*new[0] - old[0]*new[1])/oldlen
    scl = newlen/oldlen
    return off, scl

def mapdomain(x, old, new):
    """
    Apply linear map to input points.

    The linear map ``offset + scale*x`` that maps the domain `old` to
    the domain `new` is applied to the points `x`.

    Parameters
    ----------
    x : array_like
        Points to be mapped. If `x` is a subtype of ndarray the subtype
        will be preserved.
    old, new : array_like
        The two domains that determine the map.  Each must (successfully)
        convert to 1-d arrays containing precisely two values.

    Returns
    -------
    x_out : ndarray
        Array of points of the same shape as `x`, after application of the
        linear map between the two domains.

    See Also
    --------
    getdomain, mapparms

    Notes
    -----
    Effectively, this implements:

    .. math::
        x\\_out = new[0] + m(x - old[0])

    where

    .. math::
        m = \\frac{new[1]-new[0]}{old[1]-old[0]}

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> old_domain = (-1,1)
    >>> new_domain = (0,2*np.pi)
    >>> x = np.linspace(-1,1,6); x
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])
    >>> x_out = pu.mapdomain(x, old_domain, new_domain); x_out
    array([ 0.        ,  1.25663706,  2.51327412,  3.76991118,  5.02654825, # may vary
            6.28318531])
    >>> x - pu.mapdomain(x_out, new_domain, old_domain)
    array([0., 0., 0., 0., 0., 0.])

    Also works for complex numbers (and thus can be used to map any line in
    the complex plane to any other line therein).

    >>> i = complex(0,1)
    >>> old = (-1 - i, 1 + i)
    >>> new = (-1 + i, 1 - i)
    >>> z = np.linspace(old[0], old[1], 6); z
    array([-1. -1.j , -0.6-0.6j, -0.2-0.2j,  0.2+0.2j,  0.6+0.6j,  1. +1.j ])
    >>> new_z = pu.mapdomain(z, old, new); new_z
    array([-1.0+1.j , -0.6+0.6j, -0.2+0.2j,  0.2-0.2j,  0.6-0.6j,  1.0-1.j ]) # may vary

    """
    if type(x) not in (int, float, complex) and not isinstance(x, np.generic):
        x = np.asanyarray(x)
    off, scl = mapparms(old, new)
    return off + scl*x


def _nth_slice(i, ndim):
    sl = [np.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _vander_nd(vander_fs, points, degrees):
    r"""
    A generalization of the Vandermonde matrix for N dimensions

    The result is built by combining the results of 1d Vandermonde matrices,

    .. math::
        W[i_0, \ldots, i_M, j_0, \ldots, j_N] = \prod_{k=0}^N{V_k(x_k)[i_0, \ldots, i_M, j_k]}

    where

    .. math::
        N &= \texttt{len(points)} = \texttt{len(degrees)} = \texttt{len(vander\_fs)} \\
        M &= \texttt{points[k].ndim} \\
        V_k &= \texttt{vander\_fs[k]} \\
        x_k &= \texttt{points[k]} \\
        0 \le j_k &\le \texttt{degrees[k]}

    Expanding the one-dimensional :math:`V_k` functions gives:

    .. math::
        W[i_0, \ldots, i_M, j_0, \ldots, j_N] = \prod_{k=0}^N{B_{k, j_k}(x_k[i_0, \ldots, i_M])}

    where :math:`B_{k,m}` is the m'th basis of the polynomial construction used along
    dimension :math:`k`. For a regular polynomial, :math:`B_{k, m}(x) = P_m(x) = x^m`.

    Parameters
    ----------
    vander_fs : Sequence[function(array_like, int) -> ndarray]
        The 1d vander function to use for each axis, such as ``polyvander``
    points : Sequence[array_like]
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
        This must be the same length as `vander_fs`.
    degrees : Sequence[int]
        The maximum degree (inclusive) to use for each axis.
        This must be the same length as `vander_fs`.

    Returns
    -------
    vander_nd : ndarray
        An array of shape ``points[0].shape + tuple(d + 1 for d in degrees)``.
    """
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}")
    if n_dims != len(degrees):
        raise ValueError(
            f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    # convert to the same shape and type
    points = tuple(np.asarray(tuple(points)) + 0.0)

    # produce the vandermonde matrix for each dimension, placing the last
    # axis of each in an independent trailing axis of the output
    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # we checked this wasn't empty already, so no `initial` needed
    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    """
    Like `_vander_nd`, but flattens the last ``len(degrees)`` axes into a single axis

    Used to implement the public ``<type>vander<n>d`` functions.
    """
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[:-len(degrees)] + (-1,))


def _fromroots(line_f, mul_f, roots):
    """
    Helper function used to implement the ``<type>fromroots`` functions.

    Parameters
    ----------
    line_f : function(float, float) -> ndarray
        The ``<type>line`` function, such as ``polyline``
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    roots
        See the ``<type>fromroots`` functions for more detail
    """
    if len(roots) == 0:
        return np.ones(1)
    else:
        [roots] = as_series([roots], trim=False)
        roots.sort()
        p = [line_f(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [mul_f(p[i], p[i+m]) for i in range(m)]
            if r:
                tmp[0] = mul_f(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]


def _valnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>val<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>val<n>d`` functions for more detail
    """
    args = [np.asanyarray(a) for a in args]
    shape0 = args[0].shape
    if not all(a.shape == shape0 for a in args[1:]):
        if len(args) == 3:
            raise ValueError('x, y, z are incompatible')
        elif len(args) == 2:
            raise ValueError('x, y are incompatible')
        else:
            raise ValueError('ordinates are incompatible')
    it = iter(args)
    x0 = next(it)

    # use tensor on only the first
    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c


def _gridnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>grid<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>grid<n>d`` functions for more detail
    """
    for xi in args:
        c = val_f(xi, c)
    return c


def _div(mul_f, c1, c2):
    """
    Helper function used to implement the ``<type>div`` functions.

    Implementation uses repeated subtraction of c2 multiplied by the nth basis.
    For some polynomial types, a more efficient approach may be possible.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> array_like
        The ``<type>mul`` function, such as ``polymul``
    c1, c2
        See the ``<type>div`` functions for more detail
    """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError  # FIXME: add message with details to exception

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1]*0, c1
    elif lc2 == 1:
        return c1/c2[-1], c1[:1]*0
    else:
        quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        for i in range(lc1 - lc2, - 1, -1):
            p = mul_f([0]*i + [1], c2)
            q = rem[-1]/p[-1]
            rem = rem[:-1] - q*p[:-1]
            quo[i] = q
        return quo, trimseq(rem)


def _add(c1, c2):
    """ Helper function used to implement the ``<type>add`` functions. """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[:c2.size] += c2
        ret = c1
    else:
        c2[:c1.size] += c1
        ret = c2
    return trimseq(ret)


def _sub(c1, c2):
    """ Helper function used to implement the ``<type>sub`` functions. """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[:c2.size] -= c2
        ret = c1
    else:
        c2 = -c2
        c2[:c1.size] += c1
        ret = c2
    return trimseq(ret)


def _fit(vander_f, x, y, deg, rcond=None, full=False, w=None):
    """
    Helper function used to implement the ``<type>fit`` functions.

    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    c1, c2
        See the ``<type>fit`` functions for more detail
    """
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van = vander_f(x, lmax)
    else:
        deg = np.sort(deg)
        lmax = deg[-1]
        order = len(deg)
        van = vander_f(x, lmax)[:, deg]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = y.T
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")
        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w
        rhs = rhs * w

    # set rcond
    if rcond is None:
        rcond = len(x)*np.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T, rcond)
    c = (c.T/scl).T

    # Expand c to include non-fitted coefficients which are set to zero
    if deg.ndim > 0:
        if c.ndim == 2:
            cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
        else:
            cc = np.zeros(lmax+1, dtype=c.dtype)
        cc[deg] = c
        c = cc

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c


def _pow(mul_f, c, pow, maxpower):
    """
    Helper function used to implement the ``<type>pow`` functions.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    c : array_like
        1-D array of array of series coefficients
    pow, maxpower
        See the ``<type>pow`` functions for more detail
    """
    # c is a trimmed copy
    [c] = as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return np.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        # This can be made more efficient by using powers of two
        # in the usual way.
        prd = c
        for i in range(2, power + 1):
            prd = mul_f(prd, c)
        return prd


def _as_int(x, desc):
    """
    Like `operator.index`, but emits a custom exception when passed an
    incorrect type

    Parameters
    ----------
    x : int-like
        Value to interpret as an integer
    desc : str
        description to include in any error message

    Raises
    ------
    TypeError : if x is a float or non-numeric
    """
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{desc} must be an integer, received {x}") from e


def format_float(x, parens=False):
    if not np.issubdtype(type(x), np.floating):
        return str(x)

    opts = np.get_printoptions()

    if np.isnan(x):
        return opts['nanstr']
    elif np.isinf(x):
        return opts['infstr']

    exp_format = False
    if x != 0:
        a = np.abs(x)
        if a >= 1.e8 or a < 10**min(0, -(opts['precision']-1)//2):
            exp_format = True

    trim, unique = '0', True
    if opts['floatmode'] == 'fixed':
        trim, unique = 'k', False

    if exp_format:
        s = dragon4_scientific(x, precision=opts['precision'],
                               unique=unique, trim=trim,
                               sign=opts['sign'] == '+')
        if parens:
            s = '(' + s + ')'
    else:
        s = dragon4_positional(x, precision=opts['precision'],
                               fractional=True,
                               unique=unique, trim=trim,
                               sign=opts['sign'] == '+')
    return s
<|endoftext|>"""
==================================================
Laguerre Series (:mod:`numpy.polynomial.laguerre`)
==================================================

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Laguerre

Constants
---------
.. autosummary::
   :toctree: generated/

   lagdomain
   lagzero
   lagone
   lagx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   lagadd
   lagsub
   lagmulx
   lagmul
   lagdiv
   lagpow
   lagval
   lagval2d
   lagval3d
   laggrid2d
   laggrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   lagder
   lagint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   lagfromroots
   lagroots
   lagvander
   lagvander2d
   lagvander3d
   laggauss
   lagweight
   lagcompanion
   lagfit
   lagtrim
   lagline
   lag2poly
   poly2lag

See also
--------
`numpy.polynomial`

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd',
    'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder',
    'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander',
    'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d',
    'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion',
    'laggauss', 'lagweight']

lagtrim = pu.trimcoef


def poly2lag(pol):
    """
    poly2lag(pol)

    Convert a polynomial to a Laguerre series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Laguerre series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Laguerre
        series.

    See Also
    --------
    lag2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import poly2lag
    >>> poly2lag(np.arange(4))
    array([ 23., -63.,  58., -18.])

    """
    [pol] = pu.as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res


def lag2poly(c):
    """
    Convert a Laguerre series to a polynomial.

    Convert an array representing the coefficients of a Laguerre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Laguerre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2lag

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lag2poly
    >>> lag2poly([ 23., -63.,  58., -18.])
    array([0., 1., 2., 3.])

    """
    from .polynomial import polyadd, polysub, polymulx

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1*(i - 1))/i)
            c1 = polyadd(tmp, polysub((2*i - 1)*c1, polymulx(c1))/i)
        return polyadd(c0, polysub(c1, polymulx(c1)))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Laguerre
lagdomain = np.array([0., 1.])

# Laguerre coefficients representing zero.
lagzero = np.array([0])

# Laguerre coefficients representing one.
lagone = np.array([1])

# Laguerre coefficients representing the identity x.
lagx = np.array([1, -1])


def lagline(off, scl):
    """
    Laguerre series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Laguerre series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagline, lagval
    >>> lagval(0,lagline(3, 2))
    3.0
    >>> lagval(1,lagline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off + scl, -scl])
    else:
        return np.array([off])


def lagfromroots(roots):
    """
    Generate a Laguerre series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Laguerre form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Laguerre form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagfromroots, lagval
    >>> coef = lagfromroots((-1, 0, 1))
    >>> lagval((-1, 0, 1), coef)
    array([0.,  0.,  0.])
    >>> coef = lagfromroots((-1j, 1j))
    >>> lagval((-1j, 1j), coef)
    array([0.+0.j, 0.+0.j])

    """
    return pu._fromroots(lagline, lagmul, roots)


def lagadd(c1, c2):
    """
    Add one Laguerre series to another.

    Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Laguerre series of their sum.

    See Also
    --------
    lagsub, lagmulx, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Laguerre series
    is a Laguerre series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagadd
    >>> lagadd([1, 2, 3], [1, 2, 3, 4])
    array([2.,  4.,  6.,  4.])

    """
    return pu._add(c1, c2)


def lagsub(c1, c2):
    """
    Subtract one Laguerre series from another.

    Returns the difference of two Laguerre series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their difference.

    See Also
    --------
    lagadd, lagmulx, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Laguerre
    series is a Laguerre series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagsub
    >>> lagsub([1, 2, 3, 4], [1, 2, 3])
    array([0.,  0.,  0.,  4.])

    """
    return pu._sub(c1, c2)


def lagmulx(c):
    """Multiply a Laguerre series by x.

    Multiply the Laguerre series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    lagadd, lagsub, lagmul, lagdiv, lagpow

    Notes
    -----
    The multiplication uses the recursion relationship for Laguerre
    polynomials in the form

    .. math::

        xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmulx
    >>> lagmulx([1, 2, 3])
    array([-1.,  -1.,  11.,  -9.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]
    prd[1] = -c[0]
    for i in range(1, len(c)):
        prd[i + 1] = -c[i]*(i + 1)
        prd[i] += c[i]*(2*i + 1)
        prd[i - 1] -= c[i]*i
    return prd


def lagmul(c1, c2):
    """
    Multiply one Laguerre series by another.

    Returns the product of two Laguerre series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their product.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagdiv, lagpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Laguerre polynomial basis set.  Thus, to express
    the product as a Laguerre series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmul
    >>> lagmul([1, 2, 3], [0, 1, 2])
    array([  8., -13.,  38., -51.,  36.])

    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0]*xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]*xs
        c1 = c[1]*xs
    else:
        nd = len(c)
        c0 = c[-2]*xs
        c1 = c[-1]*xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i]*xs, (c1*(nd - 1))/nd)
            c1 = lagadd(tmp, lagsub((2*nd - 1)*c1, lagmulx(c1))/nd)
    return lagadd(c0, lagsub(c1, lagmulx(c1)))


def lagdiv(c1, c2):
    """
    Divide one Laguerre series by another.

    Returns the quotient-with-remainder of two Laguerre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Laguerre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagpow

    Notes
    -----
    In general, the (polynomial) division of one Laguerre series by another
    results in quotient and remainder terms that are not in the Laguerre
    polynomial basis set.  Thus, to express these results as a Laguerre
    series, it is necessary to "reproject" the results onto the Laguerre
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagdiv
    >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 1.]))

    """
    return pu._div(lagmul, c1, c2)


def lagpow(c, pow, maxpower=16):
    """Raise a Laguerre series to a power.

    Returns the Laguerre series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Laguerre series of power.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagdiv

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagpow
    >>> lagpow([1, 2, 3], 2)
    array([ 14., -16.,  56., -72.,  54.])

    """
    return pu._pow(lagmul, c, pow, maxpower)


def lagder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Laguerre series.

    Returns the Laguerre series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Laguerre series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Laguerre series of the derivative.

    See Also
    --------
    lagint

    Notes
    -----
    In general, the result of differentiating a Laguerre series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagder
    >>> lagder([ 1.,  1.,  1., -3.])
    array([1.,  2.,  3.])
    >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)
    array([1.,  2.,  3.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)

    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1]*0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def lagint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Laguerre series.

    Returns the Laguerre series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
    represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
    2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.


    Parameters
    ----------
    c : array_like
        Array of Laguerre series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Laguerre series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    lagder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagint
    >>> lagint([1,2,3])
    array([ 1.,  1.,  1., -3.])
    >>> lagint([1,2,3], m=2)
    array([ 1.,  0.,  0., -4.,  3.])
    >>> lagint([1,2,3], k=1)
    array([ 2.,  1.,  1., -3.])
    >>> lagint([1,2,3], lbnd=-1)
    array([11.5,  1. ,  1. , -3. ])
    >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)
    array([ 11.16666667,  -5.        ,  -3.        ,   2.        ]) # may vary

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    k = list(k) + [0]*(cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def lagval(x, c, tensor=True):
    """
    Evaluate a Laguerre series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    lagval2d, laggrid2d, lagval3d, laggrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval
    >>> coef = [1, 2, 3]
    >>> lagval(1, coef)
    -0.5
    >>> lagval([[1, 2],[3, 4]], coef)
    array([[-0.5, -4. ],
