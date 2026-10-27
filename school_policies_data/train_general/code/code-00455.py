    def is_nonnegative(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``GaussianElement``. """
        return False

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY mpz to ``self.dtype``."""
        return K1(a)

    def from_ZZ(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a ZZ_python element to ``self.dtype``."""
        return K1(a)

    def from_QQ(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY mpq to ``self.dtype``."""
        return K1(a)

    def from_QQ_python(K1, a, K0):
        """Convert a QQ_python element to ``self.dtype``."""
        return K1(a)

    def from_AlgebraicField(K1, a, K0):
        """Convert an element from ZZ<I> or QQ<I> to ``self.dtype``."""
        if K0.ext.args[0] == I:
            return K1.from_sympy(K0.to_sympy(a))


class GaussianIntegerRing(GaussianDomain, Ring):
    r"""Ring of Gaussian integers ``ZZ_I``

    The :ref:`ZZ_I` domain represents the `Gaussian integers`_ `\mathbb{Z}[i]`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of integers and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`ZZ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I)
    >>> p
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> p.domain
    ZZ_I

    The :ref:`ZZ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian integers.

    >>> from sympy import factor
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, domain='ZZ_I')
    (x - I)*(x + I)

    The corresponding `field of fractions`_ is the domain of the Gaussian
    rationals :ref:`QQ_I`. Conversely :ref:`ZZ_I` is the `ring of integers`_
    of :ref:`QQ_I`.

    >>> from sympy import ZZ_I, QQ_I
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`ZZ_I` can be used as a constructor.

    >>> ZZ_I(3, 4)
    (3 + 4*I)
    >>> ZZ_I(5)
    (5 + 0*I)

    The domain elements of :ref:`ZZ_I` are instances of
    :py:class:`~.GaussianInteger` which support the rings operations
    ``+,-,*,**``.

    >>> z1 = ZZ_I(5, 1)
    >>> z2 = ZZ_I(2, 3)
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 3*I)
    >>> z1 + z2
    (7 + 4*I)
    >>> z1 * z2
    (7 + 17*I)
    >>> z1 ** 2
    (24 + 10*I)

    Both floor (``//``) and modulo (``%``) division work with
    :py:class:`~.GaussianInteger` (see the :py:meth:`~.Domain.div` method).

    >>> z3, z4 = ZZ_I(5), ZZ_I(1, 3)
    >>> z3 // z4  # floor division
    (1 + -1*I)
    >>> z3 % z4   # modulo division (remainder)
    (1 + -2*I)
    >>> (z3//z4)*z4 + z3%z4 == z3
    True

    True division (``/``) in :ref:`ZZ_I` gives an element of :ref:`QQ_I`. The
    :py:meth:`~.Domain.exquo` method can be used to divide in :ref:`ZZ_I` when
    exact division is possible.

    >>> z1 / z2
    (1 + -1*I)
    >>> ZZ_I.exquo(z1, z2)
    (1 + -1*I)
    >>> z3 / z4
    (1/2 + -3/2*I)
    >>> ZZ_I.exquo(z3, z4)
    Traceback (most recent call last):
        ...
    ExactQuotientFailed: (1 + 3*I) does not divide (5 + 0*I) in ZZ_I

    The :py:meth:`~.Domain.gcd` method can be used to compute the `gcd`_ of any
    two elements.

    >>> ZZ_I.gcd(ZZ_I(10), ZZ_I(2))
    (2 + 0*I)
    >>> ZZ_I.gcd(ZZ_I(5), ZZ_I(2, 1))
    (2 + 1*I)

    .. _Gaussian integers: https://en.wikipedia.org/wiki/Gaussian_integer
    .. _gcd: https://en.wikipedia.org/wiki/Greatest_common_divisor

    """
    dom = ZZ
    mod = DMP([ZZ.one, ZZ.zero, ZZ.one], ZZ)
    dtype = GaussianInteger
    zero = dtype(ZZ(0), ZZ(0))
    one = dtype(ZZ(1), ZZ(0))
    imag_unit = dtype(ZZ(0), ZZ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'ZZ_I'

    is_GaussianRing = True
    is_ZZ_I = True
    is_PID = True

    def __init__(self):  # override Domain.__init__
        """For constructing ZZ_I."""

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, GaussianIntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        return hash('ZZ_I')

    @property
    def has_CharacteristicZero(self):
        return True

    def characteristic(self):
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_field(self):
        """Returns a field associated with ``self``. """
        return QQ_I

    def normalize(self, d, *args):
        """Return first quadrant element associated with ``d``.

        Also multiply the other arguments by the same power of i.
        """
        unit = self.canonical_unit(d)
        d *= unit
        args = tuple(a*unit for a in args)
        return (d,) + args if args else d

    def gcd(self, a, b):
        """Greatest common divisor of a and b over ZZ_I."""
        while b:
            a, b = b, a % b
        return self.normalize(a)

    def gcdex(self, a, b):
        """Return x, y, g such that x * a + y * b = g = gcd(a, b)"""
        x_a = self.one
        x_b = self.zero
        y_a = self.zero
        y_b = self.one
        while b:
            q = a // b
            a, b = b, a - q * b
            x_a, x_b = x_b, x_a - q * x_b
            y_a, y_b = y_b, y_a - q * y_b

        a, x_a, y_a = self.normalize(a, x_a, y_a)
        return x_a, y_a, a

    def lcm(self, a, b):
        """Least common multiple of a and b over ZZ_I."""
        return (a * b) // self.gcd(a, b)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to ZZ_I."""
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to ZZ_I."""
        return K1.new(ZZ.convert(a.x), ZZ.convert(a.y))

ZZ_I = GaussianInteger._parent = GaussianIntegerRing()


class GaussianRationalField(GaussianDomain, Field):
    r"""Field of Gaussian rationals ``QQ_I``

    The :ref:`QQ_I` domain represents the `Gaussian rationals`_ `\mathbb{Q}(i)`
    as a :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    By default a :py:class:`~.Poly` created from an expression with
    coefficients that are combinations of rationals and ``I`` (`\sqrt{-1}`)
    will have the domain :ref:`QQ_I`.

    >>> from sympy import Poly, Symbol, I
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + I/2)
    >>> p
    Poly(x**2 + I/2, x, domain='QQ_I')
    >>> p.domain
    QQ_I

    The polys option ``gaussian=True`` can be used to specify that the domain
    should be :ref:`QQ_I` even if the coefficients do not contain ``I`` or are
    all integers.

    >>> Poly(x**2)
    Poly(x**2, x, domain='ZZ')
    >>> Poly(x**2 + I)
    Poly(x**2 + I, x, domain='ZZ_I')
    >>> Poly(x**2/2)
    Poly(1/2*x**2, x, domain='QQ')
    >>> Poly(x**2, gaussian=True)
    Poly(x**2, x, domain='QQ_I')
    >>> Poly(x**2 + I, gaussian=True)
    Poly(x**2 + I, x, domain='QQ_I')
    >>> Poly(x**2/2, gaussian=True)
    Poly(1/2*x**2, x, domain='QQ_I')

    The :ref:`QQ_I` domain can be used to factorise polynomials that are
    reducible over the Gaussian rationals.

    >>> from sympy import factor, QQ_I
    >>> factor(x**2/4 + 1)
    (x**2 + 4)/4
    >>> factor(x**2/4 + 1, domain='QQ_I')
    (x - 2*I)*(x + 2*I)/4
    >>> factor(x**2/4 + 1, domain=QQ_I)
    (x - 2*I)*(x + 2*I)/4

    It is also possible to specify the :ref:`QQ_I` domain explicitly with
    polys functions like :py:func:`~.apart`.

    >>> from sympy import apart
    >>> apart(1/(1 + x**2))
    1/(x**2 + 1)
    >>> apart(1/(1 + x**2), domain=QQ_I)
    I/(2*(x + I)) - I/(2*(x - I))

    The corresponding `ring of integers`_ is the domain of the Gaussian
    integers :ref:`ZZ_I`. Conversely :ref:`QQ_I` is the `field of fractions`_
    of :ref:`ZZ_I`.

    >>> from sympy import ZZ_I, QQ_I, QQ
    >>> ZZ_I.get_field()
    QQ_I
    >>> QQ_I.get_ring()
    ZZ_I

    When using the domain directly :ref:`QQ_I` can be used as a constructor.

    >>> QQ_I(3, 4)
    (3 + 4*I)
    >>> QQ_I(5)
    (5 + 0*I)
    >>> QQ_I(QQ(2, 3), QQ(4, 5))
    (2/3 + 4/5*I)

    The domain elements of :ref:`QQ_I` are instances of
    :py:class:`~.GaussianRational` which support the field operations
    ``+,-,*,**,/``.

    >>> z1 = QQ_I(5, 1)
    >>> z2 = QQ_I(2, QQ(1, 2))
    >>> z1
    (5 + 1*I)
    >>> z2
    (2 + 1/2*I)
    >>> z1 + z2
    (7 + 3/2*I)
    >>> z1 * z2
    (19/2 + 9/2*I)
    >>> z2 ** 2
    (15/4 + 2*I)

    True division (``/``) in :ref:`QQ_I` gives an element of :ref:`QQ_I` and
    is always exact.

    >>> z1 / z2
    (42/17 + -2/17*I)
    >>> QQ_I.exquo(z1, z2)
    (42/17 + -2/17*I)
    >>> z1 == (z1/z2)*z2
    True

    Both floor (``//``) and modulo (``%``) division can be used with
    :py:class:`~.GaussianRational` (see :py:meth:`~.Domain.div`)
    but division is always exact so there is no remainder.

    >>> z1 // z2
    (42/17 + -2/17*I)
    >>> z1 % z2
    (0 + 0*I)
    >>> QQ_I.div(z1, z2)
    ((42/17 + -2/17*I), (0 + 0*I))
    >>> (z1//z2)*z2 + z1%z2 == z1
    True

    .. _Gaussian rationals: https://en.wikipedia.org/wiki/Gaussian_rational
    """
    dom = QQ
    mod = DMP([QQ.one, QQ.zero, QQ.one], QQ)
    dtype = GaussianRational
    zero = dtype(QQ(0), QQ(0))
    one = dtype(QQ(1), QQ(0))
    imag_unit = dtype(QQ(0), QQ(1))
    units = (one, imag_unit, -one, -imag_unit)  # powers of i

    rep = 'QQ_I'

    is_GaussianField = True
    is_QQ_I = True

    def __init__(self):  # override Domain.__init__
        """For constructing QQ_I."""

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, GaussianRationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute hash code of ``self``. """
        return hash('QQ_I')

    @property
    def has_CharacteristicZero(self):
        return True

    def characteristic(self):
        return 0

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return ZZ_I

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def as_AlgebraicField(self):
        """Get equivalent domain as an ``AlgebraicField``. """
        return AlgebraicField(self.dom, I)

    def numer(self, a):
        """Get the numerator of ``a``."""
        ZZ_I = self.get_ring()
        return ZZ_I.convert(a * self.denom(a))

    def denom(self, a):
        """Get the denominator of ``a``."""
        ZZ = self.dom.get_ring()
        QQ = self.dom
        ZZ_I = self.get_ring()
        denom_ZZ = ZZ.lcm(QQ.denom(a.x), QQ.denom(a.y))
        return ZZ_I(denom_ZZ, ZZ.zero)

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ZZ_I element to QQ_I."""
        return K1.new(a.x, a.y)

    def from_GaussianRationalField(K1, a, K0):
        """Convert a QQ_I element to QQ_I."""
        return a

    def from_ComplexField(K1, a, K0):
        """Convert a ComplexField element to QQ_I."""
        return K1.new(QQ.convert(a.real), QQ.convert(a.imag))


QQ_I = GaussianRational._parent = GaussianRationalField()
<|endoftext|>"""Ground types for various mathematical domains in SymPy. """

import builtins
from sympy.external.gmpy import GROUND_TYPES, factorial, sqrt, is_square, sqrtrem

PythonInteger = builtins.int
PythonReal = builtins.float
PythonComplex = builtins.complex

from .pythonrational import PythonRational

from sympy.core.intfunc import (
    igcdex as python_gcdex,
    igcd2 as python_gcd,
    ilcm as python_lcm,
)

from sympy.core.numbers import (Float as SymPyReal, Integer as SymPyInteger, Rational as SymPyRational)


class _GMPYInteger:
    def __init__(self, obj):
        pass

class _GMPYRational:
    def __init__(self, obj):
        pass


if GROUND_TYPES == 'gmpy':

    from gmpy2 import (
        mpz as GMPYInteger,
        mpq as GMPYRational,
        numer as gmpy_numer,
        denom as gmpy_denom,
        gcdext as gmpy_gcdex,
        gcd as gmpy_gcd,
        lcm as gmpy_lcm,
        qdiv as gmpy_qdiv,
    )
    gcdex = gmpy_gcdex
    gcd = gmpy_gcd
    lcm = gmpy_lcm

elif GROUND_TYPES == 'flint':

    from flint import fmpz as _fmpz

    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None

    def gcd(a, b):
        return a.gcd(b)

    def gcdex(a, b):
        x, y, g = python_gcdex(a, b)
        return _fmpz(x), _fmpz(y), _fmpz(g)

    def lcm(a, b):
        return a.lcm(b)

else:
    GMPYInteger = _GMPYInteger
    GMPYRational = _GMPYRational
    gmpy_numer = None
    gmpy_denom = None
    gmpy_gcdex = None
    gmpy_gcd = None
    gmpy_lcm = None
    gmpy_qdiv = None
    gcdex = python_gcdex
    gcd = python_gcd
    lcm = python_lcm


__all__ = [
    'PythonInteger', 'PythonReal', 'PythonComplex',

    'PythonRational',

    'python_gcdex', 'python_gcd', 'python_lcm',

    'SymPyReal', 'SymPyInteger', 'SymPyRational',

    'GMPYInteger', 'GMPYRational', 'gmpy_numer',
    'gmpy_denom', 'gmpy_gcdex', 'gmpy_gcd', 'gmpy_lcm',
    'gmpy_qdiv',

    'factorial', 'sqrt', 'is_square', 'sqrtrem',

    'GMPYInteger', 'GMPYRational',
]
<|endoftext|>"""Implementation of :class:`GMPYFiniteField` class. """


from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.gmpyintegerring import GMPYIntegerRing

from sympy.utilities import public

@public
class GMPYFiniteField(FiniteField):
    """Finite field based on GMPY integers. """

    alias = 'FF_gmpy'

    def __init__(self, mod, symmetric=True):
        super().__init__(mod, GMPYIntegerRing(), symmetric)
<|endoftext|>"""Implementation of mathematical domains. """

__all__ = [
    'Domain', 'FiniteField', 'IntegerRing', 'RationalField', 'RealField',
    'ComplexField', 'AlgebraicField', 'PolynomialRing', 'FractionField',
    'ExpressionDomain', 'PythonRational',

    'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 'CC', 'EX', 'EXRAW',
]

from .domain import Domain
from .finitefield import FiniteField, FF, GF
from .integerring import IntegerRing, ZZ
from .rationalfield import RationalField, QQ
from .algebraicfield import AlgebraicField
from .gaussiandomains import ZZ_I, QQ_I
from .realfield import RealField, RR
from .complexfield import ComplexField, CC
from .polynomialring import PolynomialRing
from .fractionfield import FractionField
from .expressiondomain import ExpressionDomain, EX
from .expressionrawdomain import EXRAW
from .pythonrational import PythonRational


# This is imported purely for backwards compatibility because some parts of
# the codebase used to import this from here and it's possible that downstream
# does as well:
from sympy.external.gmpy import GROUND_TYPES  # noqa: F401

#
# The rest of these are obsolete and provided only for backwards
# compatibility:
#

from .pythonfinitefield import PythonFiniteField
from .gmpyfinitefield import GMPYFiniteField
from .pythonintegerring import PythonIntegerRing
from .gmpyintegerring import GMPYIntegerRing
from .pythonrationalfield import PythonRationalField
from .gmpyrationalfield import GMPYRationalField

FF_python = PythonFiniteField
FF_gmpy = GMPYFiniteField

ZZ_python = PythonIntegerRing
ZZ_gmpy = GMPYIntegerRing

QQ_python = PythonRationalField
QQ_gmpy = GMPYRationalField

__all__.extend((
    'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing',
    'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField',

    'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 'QQ_python', 'QQ_gmpy',
))
<|endoftext|>"""Implementation of :class:`QuotientRing` class."""


from sympy.polys.agca.modules import FreeModuleQuotientRing
from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, CoercionFailed
from sympy.utilities import public

# TODO
# - successive quotients (when quotient ideals are implemented)
# - poly rings over quotients?
# - division by non-units in integral domains?

@public
class QuotientRingElement:
    """
    Class representing elements of (commutative) quotient rings.

    Attributes:

    - ring - containing ring
    - data - element of ring.ring (i.e. base ring) representing self
    """

    def __init__(self, ring, data):
        self.ring = ring
        self.data = data

    def __str__(self):
        from sympy.printing.str import sstr
        data = self.ring.ring.to_sympy(self.data)
        return sstr(data) + " + " + str(self.ring.base_ideal)

    __repr__ = __str__

    def __bool__(self):
        return not self.ring.is_zero(self)

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            try:
                om = self.ring.convert(om)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data + om.data)

    __radd__ = __add__

    def __neg__(self):
        return self.ring(self.data*self.ring.ring.convert(-1))

    def __sub__(self, om):
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring(self.data*o.data)

    __rmul__ = __mul__

    def __rtruediv__(self, o):
        return self.ring.revert(self)*o

    def __truediv__(self, o):
        if not isinstance(o, self.__class__):
            try:
                o = self.ring.convert(o)
            except (NotImplementedError, CoercionFailed):
                return NotImplemented
        return self.ring.revert(o)*self

    def __pow__(self, oth):
        if oth < 0:
            return self.ring.revert(self) ** -oth
        return self.ring(self.data ** oth)

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.ring != self.ring:
            return False
        return self.ring.is_zero(self - om)

    def __ne__(self, om):
        return not self == om


class QuotientRing(Ring):
    """
    Class representing (commutative) quotient rings.

    You should not usually instantiate this by hand, instead use the constructor
    from the base ring in the construction.

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> I = QQ.old_poly_ring(x).ideal(x**3 + 1)
    >>> QQ.old_poly_ring(x).quotient_ring(I)
    QQ[x]/<x**3 + 1>

    Shorter versions are possible:

    >>> QQ.old_poly_ring(x)/I
    QQ[x]/<x**3 + 1>

    >>> QQ.old_poly_ring(x)/[x**3 + 1]
    QQ[x]/<x**3 + 1>

    Attributes:

    - ring - the base ring
    - base_ideal - the ideal used to form the quotient
    """

    has_assoc_Ring = True
    has_assoc_Field = False
    dtype = QuotientRingElement

    def __init__(self, ring, ideal):
        if not ideal.ring == ring:
            raise ValueError('Ideal must belong to %s, got %s' % (ring, ideal))
        self.ring = ring
        self.base_ideal = ideal
        self.zero = self(self.ring.zero)
        self.one = self(self.ring.one)

    def __str__(self):
        return str(self.ring) + "/" + str(self.base_ideal)

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.ring, self.base_ideal))

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        if not isinstance(a, self.ring.dtype):
            a = self.ring(a)
        # TODO optionally disable reduction?
        return self.dtype(self, self.base_ideal.reduce_element(a))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, QuotientRing) and \
            self.ring == other.ring and self.base_ideal == other.base_ideal

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.ring.convert(a, K0))

    from_ZZ_python = from_ZZ
    from_QQ_python = from_ZZ_python
    from_ZZ_gmpy = from_ZZ_python
    from_QQ_gmpy = from_ZZ_python
    from_RealField = from_ZZ_python
    from_GlobalPolynomialRing = from_ZZ_python
    from_FractionField = from_ZZ_python

    def from_sympy(self, a):
        return self(self.ring.from_sympy(a))

    def to_sympy(self, a):
        return self.ring.to_sympy(a.data)

    def from_QuotientRing(self, a, K0):
        if K0 == self:
            return a

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        """
        Compute a**(-1), if possible.
        """
        I = self.ring.ideal(a.data) + self.base_ideal
        try:
            return self(I.in_terms_of_generators(1)[0])
        except ValueError:  # 1 not in I
            raise NotReversible('%s not a unit in %r' % (a, self))

    def is_zero(self, a):
        return self.base_ideal.contains(a.data)

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        (QQ[x]/<x**2 + 1>)**2
        """
        return FreeModuleQuotientRing(self, rank)
<|endoftext|>"""Implementation of :class:`GMPYRationalField` class. """


from sympy.polys.domains.groundtypes import (
    GMPYRational, SymPyRational,
    gmpy_numer, gmpy_denom, factorial as gmpy_factorial,
)
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class GMPYRationalField(RationalField):
    """Rational field based on GMPY's ``mpq`` type.

    This will be the implementation of :ref:`QQ` if ``gmpy`` or ``gmpy2`` is
    installed. Elements will be of type ``gmpy.mpq``.
    """

    dtype = GMPYRational
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    alias = 'QQ_gmpy'

    def __init__(self):
        pass

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import GMPYIntegerRing
        return GMPYIntegerRing()

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyRational(int(gmpy_numer(a)),
                             int(gmpy_denom(a)))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Rational:
            return GMPYRational(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return GMPYRational(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed("expected ``Rational`` object, got %s" % a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return GMPYRational(a)

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return GMPYRational(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return GMPYRational(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianElement`` object to ``dtype``. """
        if a.y == 0:
            return GMPYRational(a.x)

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return GMPYRational(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return GMPYRational(a) / GMPYRational(b)

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return GMPYRational(a) / GMPYRational(b)

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return GMPYRational(a) / GMPYRational(b), self.zero

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denominator

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return GMPYRational(gmpy_factorial(int(a)))
<|endoftext|>"""
Rational number type based on Python integers.

The PythonRational class from here has been moved to
sympy.external.pythonmpq

This module is just left here for backwards compatibility.
"""


from sympy.core.numbers import Rational
from sympy.core.sympify import _sympy_converter
from sympy.utilities import public
from sympy.external.pythonmpq import PythonMPQ


PythonRational = public(PythonMPQ)


def sympify_pythonrational(arg):
    return Rational(arg.numerator, arg.denominator)
_sympy_converter[PythonRational] = sympify_pythonrational
<|endoftext|>"""Implementation of :class:`IntegerRing` class. """

from sympy.external.gmpy import MPZ, GROUND_TYPES

from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import (
    SymPyInteger,
    factorial,
    gcdex, gcd, lcm, sqrt, is_square, sqrtrem,
)

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

import math

@public
class IntegerRing(Ring, CharacteristicZero, SimpleDomain):
    r"""The domain ``ZZ`` representing the integers `\mathbb{Z}`.

    The :py:class:`IntegerRing` class represents the ring of integers as a
    :py:class:`~.Domain` in the domain system. :py:class:`IntegerRing` is a
    super class of :py:class:`PythonIntegerRing` and
    :py:class:`GMPYIntegerRing` one of which will be the implementation for
    :ref:`ZZ` depending on whether or not ``gmpy`` or ``gmpy2`` is installed.

    See also
    ========

    Domain
    """

    rep = 'ZZ'
    alias = 'ZZ'
    dtype = MPZ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)


    is_IntegerRing = is_ZZ = True
    is_Numerical = True
    is_PID = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self):
        """Allow instantiation of this domain. """

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, IntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute a hash value for this domain. """
        return hash('ZZ')

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return MPZ(a.p)
        elif int_valued(a):
            return MPZ(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def get_field(self):
        r"""Return the associated field of fractions :ref:`QQ`

        Returns
        =======

        :ref:`QQ`:
            The associated field of fractions :ref:`QQ`, a
            :py:class:`~.Domain` representing the rational numbers
            `\mathbb{Q}`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.get_field()
        QQ
        """
        from sympy.polys.domains import QQ
        return QQ

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`.
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import ZZ, sqrt
        >>> ZZ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        return self.get_field().algebraic_field(*extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`ZZ`.

        See :py:meth:`~.Domain.convert`.
        """
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def log(self, a, b):
        r"""Logarithm of *a* to the base *b*.

        Parameters
        ==========

        a: number
        b: number

        Returns
        =======

        $\\lfloor\log(a, b)\\rfloor$:
            Floor of the logarithm of *a* to the base *b*

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.log(ZZ(8), ZZ(2))
        3
        >>> ZZ.log(ZZ(9), ZZ(2))
        3

        Notes
        =====

        This function uses ``math.log`` which is based on ``float`` so it will
        fail for large integer arguments.
        """
        return self.dtype(int(math.log(int(a), b)))

    def from_FF(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_ZZ(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return MPZ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return MPZ(a)

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to GMPY's ``mpz``. """
        return a

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY ``mpq`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return a.numerator

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to GMPY's ``mpz``. """
        p, q = K0.to_rational(a)

        if q == 1:
            # XXX: If MPZ is flint.fmpz and p is a gmpy2.mpz, then we need
            # to convert via int because fmpz and mpz do not know about each
            # other.
            return MPZ(int(p))

    def from_GaussianIntegerRing(K1, a, K0):
        if a.y == 0:
            return a.x

    def from_EX(K1, a, K0):
        """Convert ``Expression`` to GMPY's ``mpz``. """
        if a.is_Integer:
            return K1.from_sympy(a)

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        h, s, t = gcdex(a, b)
        # XXX: This conditional logic should be handled somewhere else.
        if GROUND_TYPES == 'gmpy':
            return s, t, h
        else:
            return h, s, t

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return sqrt(a)

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        An integer is a square if and only if there exists an integer
        ``b`` such that ``b * b == a``.
        """
        return is_square(a)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        if a < 0:
            return None
        root, rem = sqrtrem(a)
        if rem != 0:
            return None
        return root

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return factorial(a)


ZZ = IntegerRing()
<|endoftext|>"""Implementation of :class:`ExpressionDomain` class. """


from sympy.core import sympify, SympifyError
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots
from sympy.utilities import public

eflags = {"deep": False, "mul": True, "power_exp": False, "power_base": False,
              "basic": False, "multinomial": False, "log": False}

@public
class ExpressionDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions. """

    is_SymbolicDomain = is_EX = True

    class Expression(DomainElement, PicklableWithSlots):
        """An arbitrary expression. """

        __slots__ = ('ex',)

        def __init__(self, ex):
            if not isinstance(ex, self.__class__):
                self.ex = sympify(ex)
            else:
                self.ex = ex.ex

        def __repr__(f):
            return 'EX(%s)' % repr(f.ex)

        def __str__(f):
            return 'EX(%s)' % str(f.ex)

        def __hash__(self):
            return hash((self.__class__.__name__, self.ex))

        def parent(self):
            return EX

        def as_expr(f):
            return f.ex

        def numer(f):
            return f.__class__(f.ex.as_numer_denom()[0])

        def denom(f):
            return f.__class__(f.ex.as_numer_denom()[1])

        def simplify(f, ex):
            return f.__class__(ex.cancel().expand(**eflags))

        def __abs__(f):
            return f.__class__(abs(f.ex))

        def __neg__(f):
            return f.__class__(-f.ex)

        def _to_ex(f, g):
            try:
                return f.__class__(g)
            except SympifyError:
                return None

        def __lt__(f, g):
            return f.ex.sort_key() < g.ex.sort_key()

        def __add__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return g
            else:
                return f.simplify(f.ex + g.ex)

        def __radd__(f, g):
            return f.simplify(f.__class__(g).ex + f.ex)

        def __sub__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return -g
            else:
                return f.simplify(f.ex - g.ex)

        def __rsub__(f, g):
            return f.simplify(f.__class__(g).ex - f.ex)

        def __mul__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented

            if EX.zero in (f, g):
                return EX.zero
            elif f.ex.is_Number and g.ex.is_Number:
                return f.__class__(f.ex*g.ex)

            return f.simplify(f.ex*g.ex)

        def __rmul__(f, g):
            return f.simplify(f.__class__(g).ex*f.ex)

        def __pow__(f, n):
            n = f._to_ex(n)

            if n is not None:
                return f.simplify(f.ex**n.ex)
            else:
                return NotImplemented

        def __truediv__(f, g):
            g = f._to_ex(g)

            if g is not None:
                return f.simplify(f.ex/g.ex)
            else:
                return NotImplemented

        def __rtruediv__(f, g):
            return f.simplify(f.__class__(g).ex/f.ex)

        def __eq__(f, g):
            return f.ex == f.__class__(g).ex

        def __ne__(f, g):
            return not f == g

        def __bool__(f):
            return not f.ex.is_zero

        def gcd(f, g):
            from sympy.polys import gcd
            return f.__class__(gcd(f.ex, f.__class__(g).ex))

        def lcm(f, g):
            from sympy.polys import lcm
            return f.__class__(lcm(f.ex, f.__class__(g).ex))

    dtype = Expression

    zero = Expression(0)
    one = Expression(1)

    rep = 'EX'

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, ExpressionDomain):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        return hash("EX")

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        return self.dtype(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_AlgebraicField(K1, a, K0):
        """Convert an ``ANP`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath ``mpc`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_FractionField(K1, a, K0):
        """Convert a ``DMF`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ExpressionDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return a

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self  # XXX: EX is not a ring but we don't have much choice here.

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return a.ex.as_coeff_mul()[0].is_positive

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return a.ex.could_extract_minus_sign()

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return a.ex.as_coeff_mul()[0].is_nonpositive

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return a.ex.as_coeff_mul()[0].is_nonnegative

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numer()

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denom()

    def gcd(self, a, b):
        return self(1)

    def lcm(self, a, b):
        return a.lcm(b)


EX = ExpressionDomain()
<|endoftext|>"""Implementation of :class:`Ring` class. """


from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import ExactQuotientFailed, NotInvertible, NotReversible

from sympy.utilities import public

@public
class Ring(Domain):
    """Represents a ring domain. """

    is_Ring = True

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__floordiv__``.  """
        if a % b:
            raise ExactQuotientFailed(a, b, self)
        else:
            return a // b

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__floordiv__``. """
        return a // b

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies ``__mod__``.  """
        return a % b

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__divmod__``. """
        return divmod(a, b)

    def invert(self, a, b):
        """Returns inversion of ``a mod b``. """
        s, t, h = self.gcdex(a, b)

        if self.is_one(h):
            return s % b
        else:
            raise NotInvertible("zero divisor")

    def revert(self, a):
        """Returns ``a**(-1)`` if possible. """
        if self.is_one(a) or self.is_one(-a):
            return a
        else:
            raise NotReversible('only units are reversible in a ring')

    def is_unit(self, a):
        try:
            self.revert(a)
            return True
        except NotReversible:
            return False

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a

    def denom(self, a):
        """Returns denominator of `a`. """
        return self.one

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over self.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2)
        QQ[x]**2
        """
        raise NotImplementedError

    def ideal(self, *gens):
        """
        Generate an ideal of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x**2)
        <x**2>
        """
        from sympy.polys.agca.ideals import ModuleImplementedIdeal
        return ModuleImplementedIdeal(self, self.free_module(1).submodule(
            *[[x] for x in gens]))

    def quotient_ring(self, e):
        """
        Form a quotient ring of ``self``.

        Here ``e`` can be an ideal or an iterable.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).quotient_ring(QQ.old_poly_ring(x).ideal(x**2))
        QQ[x]/<x**2>
        >>> QQ.old_poly_ring(x).quotient_ring([x**2])
        QQ[x]/<x**2>

        The division operator has been overloaded for this:

        >>> QQ.old_poly_ring(x)/[x**2]
        QQ[x]/<x**2>
        """
        from sympy.polys.agca.ideals import Ideal
        from sympy.polys.domains.quotientring import QuotientRing
        if not isinstance(e, Ideal):
            e = self.ideal(*e)
        return QuotientRing(self, e)

    def __truediv__(self, e):
        return self.quotient_ring(e)
<|endoftext|>"""Implementation of :class:`FractionField` class. """


from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.field import Field
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public

@public
class FractionField(Field, CompositeDomain):
    """A class for representing multivariate rational function fields. """

    is_FractionField = is_Frac = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self, domain_or_field, symbols=None, order=None):
        from sympy.polys.fields import FracField

        if isinstance(domain_or_field, FracField) and symbols is None and order is None:
            field = domain_or_field
        else:
            field = FracField(symbols, domain_or_field, order)

        self.field = field
        self.dtype = field.dtype

        self.gens = field.gens
        self.ngens = field.ngens
        self.symbols = field.symbols
        self.domain = field.domain

        # TODO: remove this
        self.dom = self.domain

    def new(self, element):
        return self.field.field_new(element)

    def of_type(self, element):
        """Check if ``a`` is of type ``dtype``. """
        return self.field.is_element(element)

    @property
    def zero(self):
        return self.field.zero

    @property
    def one(self):
        return self.field.one

    @property
    def order(self):
        return self.field.order

    def __str__(self):
        return str(self.domain) + '(' + ','.join(map(str, self.symbols)) + ')'

    def __hash__(self):
        return hash((self.__class__.__name__, self.field, self.domain, self.symbols))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if not isinstance(other, FractionField):
            return NotImplemented
        return self.field == other.field

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        return self.field.from_expr(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        dom = K1.domain
        conv = dom.convert_from
        if dom.is_ZZ:
            return K1(conv(K0.numer(a), K0)) / K1(conv(K0.denom(a), K0))
        else:
            return K1(conv(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ``GaussianInteger`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K1.domain.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        if K1.domain != K0:
            a = K1.domain.convert_from(a, K0)
        if a is not None:
            return K1.new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        if a.is_ground:
            return K1.convert_from(a.coeff(1), K0.domain)
        try:
            return K1.new(a.set_ring(K1.field.ring))
        except (CoercionFailed, GeneratorsError):
            # XXX: We get here if K1=ZZ(x,y) and K0=QQ[x,y]
            # and the poly a in K0 has non-integer coefficients.
            # It seems that K1.new can handle this but K1.new doesn't work
            # when K0.domain is an algebraic field...
            try:
                return K1.new(a)
            except (CoercionFailed, GeneratorsError):
                return None

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        try:
            return a.set_field(K1.field)
        except (CoercionFailed, GeneratorsError):
            return None

    def get_ring(self):
        """Returns a field associated with ``self``. """
        return self.field.to_ring().to_domain()

    def is_positive(self, a):
        """Returns True if ``LC(a)`` is positive. """
        return self.domain.is_positive(a.numer.LC)

    def is_negative(self, a):
        """Returns True if ``LC(a)`` is negative. """
        return self.domain.is_negative(a.numer.LC)

    def is_nonpositive(self, a):
        """Returns True if ``LC(a)`` is non-positive. """
        return self.domain.is_nonpositive(a.numer.LC)

    def is_nonnegative(self, a):
        """Returns True if ``LC(a)`` is non-negative. """
        return self.domain.is_nonnegative(a.numer.LC)

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numer

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denom

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return self.dtype(self.domain.factorial(a))
<|endoftext|>"""Implementation of :class:`FiniteField` class. """

import operator

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from sympy.core.numbers import int_valued
from sympy.polys.domains.field import Field

from sympy.polys.domains.modularinteger import ModularIntegerFactory
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.galoistools import gf_zassenhaus, gf_irred_p_rabin
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
from sympy.polys.domains.groundtypes import SymPyInteger


if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['FiniteField']


if GROUND_TYPES == 'flint':
    import flint
    # Don't use python-flint < 0.5.0 because nmod was missing some features in
    # previous versions of python-flint and fmpz_mod was not yet added.
    _major, _minor, *_ = flint.__version__.split('.')
    if (int(_major), int(_minor)) < (0, 5):
        flint = None
else:
    flint = None


def _modular_int_factory_nmod(mod):
    # nmod only recognises int
    index = operator.index
    mod = index(mod)
    nmod = flint.nmod
    nmod_poly = flint.nmod_poly

    # flint's nmod is only for moduli up to 2^64-1 (on a 64-bit machine)
    try:
        nmod(0, mod)
    except OverflowError:
        return None, None

    def ctx(x):
        try:
            return nmod(x, mod)
        except TypeError:
            return nmod(index(x), mod)

    def poly_ctx(cs):
        return nmod_poly(cs, mod)

    return ctx, poly_ctx


def _modular_int_factory_fmpz_mod(mod):
    index = operator.index
    fctx = flint.fmpz_mod_ctx(mod)
    fctx_poly = flint.fmpz_mod_poly_ctx(mod)
    fmpz_mod_poly = flint.fmpz_mod_poly

    def ctx(x):
        try:
            return fctx(x)
        except TypeError:
            # x might be Integer
            return fctx(index(x))

    def poly_ctx(cs):
        return fmpz_mod_poly(cs, fctx_poly)

    return ctx, poly_ctx


def _modular_int_factory(mod, dom, symmetric, self):
    # Convert the modulus to ZZ
    try:
        mod = dom.convert(mod)
    except CoercionFailed:
        raise ValueError('modulus must be an integer, got %s' % mod)

    ctx, poly_ctx, is_flint = None, None, False

    # Don't use flint if the modulus is not prime as it often crashes.
    if flint is not None and mod.is_prime():

        is_flint = True

        # Try to use flint's nmod first
        ctx, poly_ctx = _modular_int_factory_nmod(mod)

        if ctx is None:
            # Use fmpz_mod for larger moduli
            ctx, poly_ctx = _modular_int_factory_fmpz_mod(mod)

    if ctx is None:
        # Use the Python implementation if flint is not available or the
        # modulus is not prime.
        ctx = ModularIntegerFactory(mod, dom, symmetric, self)
        poly_ctx = None  # not used

    return ctx, poly_ctx, is_flint


@public
@doctest_depends_on(modules=['python', 'gmpy'])
class FiniteField(Field, SimpleDomain):
    r"""Finite field of prime order :ref:`GF(p)`

    A :ref:`GF(p)` domain represents a `finite field`_ `\mathbb{F}_p` of prime
    order as :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression with integer
    coefficients will have the domain :ref:`ZZ`. However, if the ``modulus=p``
    option is given then the domain will be a finite field instead.

    >>> from sympy import Poly, Symbol
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + 1)
    >>> p
    Poly(x**2 + 1, x, domain='ZZ')
    >>> p.domain
    ZZ
    >>> p2 = Poly(x**2 + 1, modulus=2)
    >>> p2
    Poly(x**2 + 1, x, modulus=2)
    >>> p2.domain
    GF(2)

    It is possible to factorise a polynomial over :ref:`GF(p)` using the
    modulus argument to :py:func:`~.factor` or by specifying the domain
    explicitly. The domain can also be given as a string.

    >>> from sympy import factor, GF
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, modulus=2)
    (x + 1)**2
    >>> factor(x**2 + 1, domain=GF(2))
    (x + 1)**2
    >>> factor(x**2 + 1, domain='GF(2)')
    (x + 1)**2

    It is also possible to use :ref:`GF(p)` with the :py:func:`~.cancel`
    and :py:func:`~.gcd` functions.

    >>> from sympy import cancel, gcd
    >>> cancel((x**2 + 1)/(x + 1))
    (x**2 + 1)/(x + 1)
    >>> cancel((x**2 + 1)/(x + 1), domain=GF(2))
    x + 1
    >>> gcd(x**2 + 1, x + 1)
    1
    >>> gcd(x**2 + 1, x + 1, domain=GF(2))
    x + 1

    When using the domain directly :ref:`GF(p)` can be used as a constructor
    to create instances which then support the operations ``+,-,*,**,/``

    >>> from sympy import GF
    >>> K = GF(5)
    >>> K
    GF(5)
    >>> x = K(3)
    >>> y = K(2)
    >>> x
    3 mod 5
    >>> y
    2 mod 5
    >>> x * y
    1 mod 5
    >>> x / y
    4 mod 5

    Notes
    =====

    It is also possible to create a :ref:`GF(p)` domain of **non-prime**
    order but the resulting ring is **not** a field: it is just the ring of
    the integers modulo ``n``.

    >>> K = GF(9)
    >>> z = K(3)
    >>> z
    3 mod 9
    >>> z**2
    0 mod 9

    It would be good to have a proper implementation of prime power fields
    (``GF(p**n)``) but these are not yet implemented in SymPY.

    .. _finite field: https://en.wikipedia.org/wiki/Finite_field
    """

    rep = 'FF'
    alias = 'FF'

    is_FiniteField = is_FF = True
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    dom = None
    mod = None

    def __init__(self, mod, symmetric=True):
        from sympy.polys.domains import ZZ
        dom = ZZ

        if mod <= 0:
            raise ValueError('modulus must be a positive integer, got %s' % mod)

        ctx, poly_ctx, is_flint = _modular_int_factory(mod, dom, symmetric, self)

        self.dtype = ctx
        self._poly_ctx = poly_ctx
        self._is_flint = is_flint

        self.zero = self.dtype(0)
        self.one = self.dtype(1)
        self.dom = dom
        self.mod = mod
        self.sym = symmetric
        self._tp = type(self.zero)

    @property
    def tp(self):
        return self._tp

    @property
    def is_Field(self):
        is_field = getattr(self, '_is_field', None)
        if is_field is None:
            from sympy.ntheory.primetest import isprime
            self._is_field = is_field = isprime(self.mod)
        return is_field

    def __str__(self):
        return 'GF(%s)' % self.mod

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.mod, self.dom))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, FiniteField) and \
            self.mod == other.mod and self.dom == other.dom

    def characteristic(self):
        """Return the characteristic of this domain. """
        return self.mod

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(self.to_int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to SymPy's ``Integer``. """
        if a.is_Integer:
            return self.dtype(self.dom.dtype(int(a)))
        elif int_valued(a):
            return self.dtype(self.dom.dtype(int(a)))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def to_int(self, a):
        """Convert ``val`` to a Python ``int`` object. """
        aval = int(a)
        if self.sym and aval > self.mod // 2:
            aval -= self.mod
        return aval

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return bool(a)

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return True

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return False

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return not a

    def from_FF(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ(int(a), K0.dom))

    def from_FF_python(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(int(a), K0.dom))

    def from_ZZ(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_ZZ_python(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_QQ(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_QQ_python(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_FF_gmpy(K1, a, K0=None):
        """Convert ``ModularInteger(mpz)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_gmpy(a.val, K0.dom))

    def from_ZZ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpz`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_gmpy(a, K0))

    def from_QQ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpq`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_gmpy(a.numerator)

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to ``dtype``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return K1.dtype(K1.dom.dtype(p))

    def is_square(self, a):
        """Returns True if ``a`` is a quadratic residue modulo p. """
        # a is not a square <=> x**2-a is irreducible
        poly = [int(x) for x in [self.one, self.zero, -a]]
        return not gf_irred_p_rabin(poly, self.mod, self.dom)

    def exsqrt(self, a):
        """Square root modulo p of ``a`` if it is a quadratic residue.

        Explanation
        ===========
        Always returns the square root that is no larger than ``p // 2``.
        """
        # x**2-a is not square-free if a=0 or the field is characteristic 2
        if self.mod == 2 or a == 0:
            return a
        # Otherwise, use square-free factorization routine to factorize x**2-a
        poly = [int(x) for x in [self.one, self.zero, -a]]
        for factor in gf_zassenhaus(poly, self.mod, self.dom):
            if len(factor) == 2 and factor[1] <= self.mod // 2:
                return self.dtype(factor[1])
        return None


FF = GF = FiniteField
<|endoftext|>"""Implementation of :class:`Field` class. """


from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, DomainError
from sympy.utilities import public

@public
class Field(Ring):
    """Represents a field domain. """

    is_Field = True
    is_PID = True

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError('there is no ring associated with %s' % self)

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return a / b

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return a / b

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return a / b, self.zero

    def gcd(self, a, b):
        """
        Returns GCD of ``a`` and ``b``.

        This definition of GCD over fields allows to clear denominators
        in `primitive()`.

        Examples
        ========

        >>> from sympy.polys.domains import QQ
        >>> from sympy import S, gcd, primitive
        >>> from sympy.abc import x

        >>> QQ.gcd(QQ(2, 3), QQ(4, 9))
        2/9
        >>> gcd(S(2)/3, S(4)/9)
        2/9
        >>> primitive(2*x/3 + S(4)/9)
        (2/9, 3*x + 2)

        """
        try:
            ring = self.get_ring()
        except DomainError:
            return self.one

        p = ring.gcd(self.numer(a), self.numer(b))
        q = ring.lcm(self.denom(a), self.denom(b))

        return self.convert(p, ring)/q

    def gcdex(self, a, b):
        """
        Returns x, y, g such that a * x + b * y == g == gcd(a, b)
        """
        d = self.gcd(a, b)

        if a == self.zero:
            if b == self.zero:
                return self.zero, self.one, self.zero
            else:
                return self.zero, d/b, d
        else:
            return d/a, self.zero, d

    def lcm(self, a, b):
        """
        Returns LCM of ``a`` and ``b``.

        >>> from sympy.polys.domains import QQ
        >>> from sympy import S, lcm

        >>> QQ.lcm(QQ(2, 3), QQ(4, 9))
        4/3
        >>> lcm(S(2)/3, S(4)/9)
        4/3

        """

        try:
            ring = self.get_ring()
        except DomainError:
            return a*b

        p = ring.lcm(self.numer(a), self.numer(b))
        q = ring.gcd(self.denom(a), self.denom(b))

        return self.convert(p, ring)/q

    def revert(self, a):
        """Returns ``a**(-1)`` if possible. """
        if a:
            return 1/a
        else:
            raise NotReversible('zero is not reversible')

    def is_unit(self, a):
        """Return true if ``a`` is a invertible"""
        return bool(a)
<|endoftext|>"""Implementation of :class:`PolynomialRing` class. """


from sympy.polys.domains.ring import Ring
from sympy.polys.domains.compositedomain import CompositeDomain

from sympy.polys.polyerrors import CoercionFailed, GeneratorsError
from sympy.utilities import public

@public
class PolynomialRing(Ring, CompositeDomain):
    """A class for representing multivariate polynomial rings. """

    is_PolynomialRing = is_Poly = True

    has_assoc_Ring  = True
    has_assoc_Field = True

    def __init__(self, domain_or_ring, symbols=None, order=None):
        from sympy.polys.rings import PolyRing

        if isinstance(domain_or_ring, PolyRing) and symbols is None and order is None:
            ring = domain_or_ring
        else:
            ring = PolyRing(symbols, domain_or_ring, order)

        self.ring = ring
        self.dtype = ring.dtype

        self.gens = ring.gens
        self.ngens = ring.ngens
        self.symbols = ring.symbols
        self.domain = ring.domain


        if symbols:
            if ring.domain.is_Field and ring.domain.is_Exact and len(symbols)==1:
                self.is_PID = True

        # TODO: remove this
        self.dom = self.domain

    def new(self, element):
        return self.ring.ring_new(element)

    def of_type(self, element):
        """Check if ``a`` is of type ``dtype``. """
        return self.ring.is_element(element)

    @property
    def zero(self):
        return self.ring.zero

    @property
    def one(self):
        return self.ring.one

    @property
    def order(self):
        return self.ring.order

    def __str__(self):
        return str(self.domain) + '[' + ','.join(map(str, self.symbols)) + ']'

    def __hash__(self):
        return hash((self.__class__.__name__, self.ring, self.domain, self.symbols))

    def __eq__(self, other):
        """Returns `True` if two domains are equivalent. """
        if not isinstance(other, PolynomialRing):
            return NotImplemented
        return self.ring == other.ring

    def is_unit(self, a):
        """Returns ``True`` if ``a`` is a unit of ``self``"""
        if not a.is_ground:
            return False
        K = self.domain
        return K.is_unit(K.convert_from(a, self))

    def canonical_unit(self, a):
        u = self.domain.canonical_unit(a.LC)
        return self.ring.ground_new(u)

    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to `dtype`. """
        return self.ring.from_expr(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a `GaussianInteger` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a `GaussianRational` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert an algebraic number to ``dtype``. """
        if K1.domain != K0:
            a = K1.domain.convert_from(a, K0)
        if a is not None:
            return K1.new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to ``dtype``. """
        try:
            return a.set_ring(K1.ring)
        except (CoercionFailed, GeneratorsError):
            return None

    def from_FractionField(K1, a, K0):
        """Convert a rational function to ``dtype``. """
        if K1.domain == K0:
            return K1.ring.from_list([a])

        q, r = K0.numer(a).div(K0.denom(a))

        if r.is_zero:
            return K1.from_PolynomialRing(q, K0.field.ring.to_domain())
        else:
            return None

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert from old poly ring to ``dtype``. """
        if K1.symbols == K0.gens:
            ad = a.to_dict()
            if K1.domain != K0.domain:
                ad = {m: K1.domain.convert(c) for m, c in ad.items()}
            return K1(ad)
        elif a.is_ground and K0.domain == K1:
            return K1.convert_from(a.to_list()[0], K0.domain)

    def get_field(self):
        """Returns a field associated with `self`. """
        return self.ring.to_field().to_domain()

    def is_positive(self, a):
        """Returns True if `LC(a)` is positive. """
        return self.domain.is_positive(a.LC)

    def is_negative(self, a):
        """Returns True if `LC(a)` is negative. """
        return self.domain.is_negative(a.LC)

    def is_nonpositive(self, a):
        """Returns True if `LC(a)` is non-positive. """
        return self.domain.is_nonpositive(a.LC)

    def is_nonnegative(self, a):
        """Returns True if `LC(a)` is non-negative. """
        return self.domain.is_nonnegative(a.LC)

    def gcdex(self, a, b):
        """Extended GCD of `a` and `b`. """
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of `a` and `b`. """
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of `a` and `b`. """
        return a.lcm(b)

    def factorial(self, a):
        """Returns factorial of `a`. """
        return self.dtype(self.domain.factorial(a))
<|endoftext|>"""Implementation of :class:`RationalField` class. """


from sympy.external.gmpy import MPQ

from sympy.polys.domains.groundtypes import SymPyRational, is_square, sqrtrem

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class RationalField(Field, CharacteristicZero, SimpleDomain):
    r"""Abstract base class for the domain :ref:`QQ`.

    The :py:class:`RationalField` class represents the field of rational
    numbers $\mathbb{Q}$ as a :py:class:`~.Domain` in the domain system.
    :py:class:`RationalField` is a superclass of
    :py:class:`PythonRationalField` and :py:class:`GMPYRationalField` one of
    which will be the implementation for :ref:`QQ` depending on whether either
    of ``gmpy`` or ``gmpy2`` is installed or not.

    See also
    ========

    Domain
    """

    rep = 'QQ'
    alias = 'QQ'

    is_RationalField = is_QQ = True
    is_Numerical = True

    has_assoc_Ring = True
    has_assoc_Field = True

    dtype = MPQ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)

    def __init__(self):
        pass

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, RationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Returns hash code of ``self``. """
        return hash('QQ')

    def get_ring(self):
        """Returns ring associated with ``self``. """
        from sympy.polys.domains import ZZ
        return ZZ

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyRational(int(a.numerator), int(a.denominator))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Rational:
            return MPQ(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return MPQ(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed("expected `Rational` object, got %s" % a)

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import QQ, sqrt
        >>> QQ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        from sympy.polys.domains import AlgebraicField
        return AlgebraicField(self, *extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`QQ`.

        See :py:meth:`~.Domain.convert`
        """
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return MPQ(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return MPQ(a)

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return a

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianElement`` object to ``dtype``. """
        if a.y == 0:
            return MPQ(a.x)

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return MPQ(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  """
        return MPQ(a) / MPQ(b)

    def quo(self, a, b):
        """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
        return MPQ(a) / MPQ(b)

    def rem(self, a, b):
        """Remainder of ``a`` and ``b``, implies nothing.  """
        return self.zero

    def div(self, a, b):
        """Division of ``a`` and ``b``, implies ``__truediv__``. """
        return MPQ(a) / MPQ(b), self.zero

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numerator

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denominator

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        A rational number is a square if and only if there exists
        a rational number ``b`` such that ``b * b == a``.
        """
        return is_square(a.numerator) and is_square(a.denominator)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        if a.numerator < 0:  # denominator is always positive
            return None
        p_sqrt, p_rem = sqrtrem(a.numerator)
        if p_rem != 0:
            return None
        q_sqrt, q_rem = sqrtrem(a.denominator)
        if q_rem != 0:
            return None
        return MPQ(p_sqrt, q_sqrt)

QQ = RationalField()
<|endoftext|>"""Implementation of :class:`ComplexField` class. """


from sympy.external.gmpy import SYMPY_INTS
from sympy.core.numbers import Float, I
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.gaussiandomains import QQ_I
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import DomainError, CoercionFailed
from sympy.utilities import public

from mpmath import MPContext


@public
class ComplexField(Field, CharacteristicZero, SimpleDomain):
    """Complex numbers up to the given precision. """

    rep = 'CC'

    is_ComplexField = is_CC = True

    is_Exact = False
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    _default_precision = 53

    @property
    def has_default_precision(self):
        return self.precision == self._default_precision

    @property
    def precision(self):
        return self._context.prec

    @property
    def dps(self):
        return self._context.dps

    @property
    def tolerance(self):
        return self._tolerance

    def __init__(self, prec=None, dps=None, tol=None):
        # XXX: The tolerance parameter is ignored but is kept for backward
        # compatibility for now.

        context = MPContext()

        if prec is None and dps is None:
            context.prec = self._default_precision
        elif dps is None:
            context.prec = prec
        elif prec is None:
            context.dps = dps
        else:
            raise TypeError("Cannot set both prec and dps")

        self._context = context

        self._dtype = context.mpc
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

        # XXX: Neither of these is actually used anywhere.
        self._max_denom = max(2**context.prec // 200, 99)
        self._tolerance = self.one / self._max_denom

    @property
    def tp(self):
        # XXX: Domain treats tp as an alias of dtype. Here we need two separate
        # things: dtype is a callable to make/convert instances. We use tp with
        # isinstance to check if an object is an instance of the domain
        # already.
        return self._dtype

    def dtype(self, x, y=0):
        # XXX: This is needed because mpmath does not recognise fmpz.
        # It might be better to add conversion routines to mpmath and if that
        # happens then this can be removed.
        if isinstance(x, SYMPY_INTS):
            x = int(x)
        if isinstance(y, SYMPY_INTS):
            y = int(y)
        return self._dtype(x, y)

    def __eq__(self, other):
        return isinstance(other, ComplexField) and self.precision == other.precision

    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision))

    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
        return Float(element.real, self.dps) + I*Float(element.imag, self.dps)

    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
        number = expr.evalf(n=self.dps)
        real, imag = number.as_real_imag()

        if real.is_Number and imag.is_Number:
            return self.dtype(real, imag)
        else:
            raise CoercionFailed("expected complex number, got %s" % expr)

    def from_ZZ(self, element, base):
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        return self.dtype(int(element))

    def from_ZZ_python(self, element, base):
        return self.dtype(element)

    def from_QQ(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_QQ_python(self, element, base):
        return self.dtype(element.numerator) / element.denominator

    def from_QQ_gmpy(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_GaussianIntegerRing(self, element, base):
        return self.dtype(int(element.x), int(element.y))

    def from_GaussianRationalField(self, element, base):
        x = element.x
        y = element.y
        return (self.dtype(int(x.numerator)) / int(x.denominator) +
                self.dtype(0, int(y.numerator)) / int(y.denominator))

    def from_AlgebraicField(self, element, base):
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        return self.dtype(element)

    def from_ComplexField(self, element, base):
        return self.dtype(element)

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        raise DomainError("there is no ring associated with %s" % self)

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        return QQ_I

    def is_negative(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def is_positive(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def is_nonnegative(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def is_nonpositive(self, element):
        """Returns ``False`` for any ``ComplexElement``. """
        return False

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return self.one

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a*b

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return self._context.almosteq(a, b, tolerance)

    def is_square(self, a):
        """Returns ``True``. Every complex number has a complex square root."""
        return True

    def exsqrt(self, a):
        r"""Returns the principal complex square root of ``a``.

        Explanation
        ===========
        The argument of the principal square root is always within
        $(-\frac{\pi}{2}, \frac{\pi}{2}]$. The square root may be
        slightly inaccurate due to floating point rounding error.
        """
        return a ** 0.5

CC = ComplexField()
<|endoftext|>"""Implementation of :class:`CharacteristicZero` class. """


from sympy.polys.domains.domain import Domain
from sympy.utilities import public

@public
class CharacteristicZero(Domain):
    """Domain that has infinite number of elements. """

    has_CharacteristicZero = True

    def characteristic(self):
        """Return the characteristic of this domain. """
        return 0
<|endoftext|>"""Implementation of :class:`RealField` class. """


from sympy.external.gmpy import SYMPY_INTS, MPQ
from sympy.core.numbers import Float
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

from mpmath import MPContext
from mpmath.libmp import to_rational as _mpmath_to_rational


def to_rational(s, max_denom, limit=True):

    p, q = _mpmath_to_rational(s._mpf_)

    # Needed for GROUND_TYPES=flint if gmpy2 is installed because mpmath's
    # to_rational() function returns a gmpy2.mpz instance and if MPQ is
    # flint.fmpq then MPQ(p, q) will fail.
    p = int(p)
    q = int(q)

    if not limit or q <= max_denom:
        return p, q

    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = p, q

    while True:
        a = n//d
        q2 = q0 + a*q1
        if q2 > max_denom:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
        n, d = d, n - a*d

    k = (max_denom - q0)//q1

    number = MPQ(p, q)
    bound1 = MPQ(p0 + k*p1, q0 + k*q1)
    bound2 = MPQ(p1, q1)

    if not bound2 or not bound1:
        return p, q
    elif abs(bound2 - number) <= abs(bound1 - number):
        return bound2.numerator, bound2.denominator
    else:
        return bound1.numerator, bound1.denominator


@public
class RealField(Field, CharacteristicZero, SimpleDomain):
    """Real numbers up to the given precision. """

    rep = 'RR'

    is_RealField = is_RR = True

    is_Exact = False
    is_Numerical = True
    is_PID = False

    has_assoc_Ring = False
    has_assoc_Field = True

    _default_precision = 53

    @property
    def has_default_precision(self):
        return self.precision == self._default_precision

    @property
    def precision(self):
        return self._context.prec

    @property
    def dps(self):
        return self._context.dps

    @property
    def tolerance(self):
        return self._tolerance

    def __init__(self, prec=None, dps=None, tol=None):
        # XXX: The tol parameter is ignored but is kept for now for backwards
        # compatibility.

        context = MPContext()

        if prec is None and dps is None:
            context.prec = self._default_precision
        elif dps is None:
            context.prec = prec
        elif prec is None:
            context.dps = dps
        else:
            raise TypeError("Cannot set both prec and dps")

        self._context = context

        self._dtype = context.mpf
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

        # Only max_denom here is used for anything and is only used for
        # to_rational.
        self._max_denom = max(2**context.prec // 200, 99)
        self._tolerance = self.one / self._max_denom

    @property
    def tp(self):
        # XXX: Domain treats tp as an alias of dtype. Here we need to two
        # separate things: dtype is a callable to make/convert instances.
        # We use tp with isinstance to check if an object is an instance
        # of the domain already.
        return self._dtype

    def dtype(self, arg):
        # XXX: This is needed because mpmath does not recognise fmpz.
        # It might be better to add conversion routines to mpmath and if that
        # happens then this can be removed.
        if isinstance(arg, SYMPY_INTS):
            arg = int(arg)
        return self._dtype(arg)

    def __eq__(self, other):
        return isinstance(other, RealField) and self.precision == other.precision

    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision))

    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
        return Float(element, self.dps)

    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
        number = expr.evalf(n=self.dps)

        if number.is_Number:
            return self.dtype(number)
        else:
            raise CoercionFailed("expected real number, got %s" % expr)

    def from_ZZ(self, element, base):
        return self.dtype(element)

    def from_ZZ_python(self, element, base):
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        return self.dtype(int(element))

    # XXX: We need to convert the denominators to int here because mpmath does
    # not recognise mpz. Ideally mpmath would handle this and if it changed to
    # do so then the calls to int here could be removed.

    def from_QQ(self, element, base):
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_python(self, element, base):
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_gmpy(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_AlgebraicField(self, element, base):
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        return self.dtype(element)

    def from_ComplexField(self, element, base):
        if not element.imag:
            return self.dtype(element.real)

    def to_rational(self, element, limit=True):
        """Convert a real number to rational number. """
        return to_rational(element, self._max_denom, limit=limit)

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        from sympy.polys.domains import QQ
        return QQ

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return self.one

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a*b

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return self._context.almosteq(a, b, tolerance)

    def is_square(self, a):
        """Returns ``True`` if ``a >= 0`` and ``False`` otherwise. """
        return a >= 0

    def exsqrt(self, a):
        """Non-negative square root for ``a >= 0`` and ``None`` otherwise.

        Explanation
        ===========
        The square root may be slightly inaccurate due to floating point
        rounding error.
        """
        return a ** 0.5 if a >= 0 else None


RR = RealField()
<|endoftext|>"""Prime ideals in number fields. """

from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace


def _check_formal_conditions_for_maximal_order(submodule):
    r"""
    Several functions in this module accept an argument which is to be a
    :py:class:`~.Submodule` representing the maximal order in a number field,
    such as returned by the :py:func:`~sympy.polys.numberfields.basis.round_two`
    algorithm.

    We do not attempt to check that the given ``Submodule`` actually represents
    a maximal order, but we do check a basic set of formal conditions that the
    ``Submodule`` must satisfy, at a minimum. The purpose is to catch an
    obviously ill-formed argument.
    """
    prefix = 'The submodule representing the maximal order should '
    cond = None
    if not submodule.is_power_basis_submodule():
        cond = 'be a direct submodule of a power basis.'
    elif not submodule.starts_with_unity():
        cond = 'have 1 as its first generator.'
    elif not submodule.is_sq_maxrank_HNF():
        cond = 'have square matrix, of maximal rank, in Hermite Normal Form.'
    if cond is not None:
        raise StructureError(prefix + cond)


class PrimeIdeal(IntegerPowerable):
    r"""
    A prime ideal in a ring of algebraic integers.
    """

    def __init__(self, ZK, p, alpha, f, e=None):
        """
        Parameters
        ==========

        ZK : :py:class:`~.Submodule`
            The maximal order where this ideal lives.
        p : int
            The rational prime this ideal divides.
        alpha : :py:class:`~.PowerBasisElement`
            Such that the ideal is equal to ``p*ZK + alpha*ZK``.
        f : int
            The inertia degree.
        e : int, ``None``, optional
            The ramification index, if already known. If ``None``, we will
            compute it here.

        """
        _check_formal_conditions_for_maximal_order(ZK)
        self.ZK = ZK
        self.p = p
        self.alpha = alpha
        self.f = f
        self._test_factor = None
        self.e = e if e is not None else self.valuation(p * ZK)

    def __str__(self):
        if self.is_inert:
            return f'({self.p})'
        return f'({self.p}, {self.alpha.as_expr()})'

    @property
    def is_inert(self):
        """
        Say whether the rational prime we divide is inert, i.e. stays prime in
        our ring of integers.
        """
        return self.f == self.ZK.n

    def repr(self, field_gen=None, just_gens=False):
        """
        Print a representation of this prime ideal.

        Examples
        ========

        >>> from sympy import cyclotomic_poly, QQ
        >>> from sympy.abc import x, zeta
        >>> T = cyclotomic_poly(7, x)
        >>> K = QQ.algebraic_field((T, zeta))
        >>> P = K.primes_above(11)
        >>> print(P[0].repr())
        [ (11, x**3 + 5*x**2 + 4*x - 1) e=1, f=3 ]
        >>> print(P[0].repr(field_gen=zeta))
        [ (11, zeta**3 + 5*zeta**2 + 4*zeta - 1) e=1, f=3 ]
        >>> print(P[0].repr(field_gen=zeta, just_gens=True))
        (11, zeta**3 + 5*zeta**2 + 4*zeta - 1)

        Parameters
        ==========

        field_gen : :py:class:`~.Symbol`, ``None``, optional (default=None)
            The symbol to use for the generator of the field. This will appear
            in our representation of ``self.alpha``. If ``None``, we use the
            variable of the defining polynomial of ``self.ZK``.
        just_gens : bool, optional (default=False)
            If ``True``, just print the "(p, alpha)" part, showing "just the
            generators" of the prime ideal. Otherwise, print a string of the
            form "[ (p, alpha) e=..., f=... ]", giving the ramification index
            and inertia degree, along with the generators.

        """
        field_gen = field_gen or self.ZK.parent.T.gen
        p, alpha, e, f = self.p, self.alpha, self.e, self.f
        alpha_rep = str(alpha.numerator(x=field_gen).as_expr())
        if alpha.denom > 1:
            alpha_rep = f'({alpha_rep})/{alpha.denom}'
        gens = f'({p}, {alpha_rep})'
        if just_gens:
            return gens
        return f'[ {gens} e={e}, f={f} ]'

    def __repr__(self):
        return self.repr()

    def as_submodule(self):
        r"""
        Represent this prime ideal as a :py:class:`~.Submodule`.

        Explanation
        ===========

        The :py:class:`~.PrimeIdeal` class serves to bundle information about
        a prime ideal, such as its inertia degree, ramification index, and
        two-generator representation, as well as to offer helpful methods like
        :py:meth:`~.PrimeIdeal.valuation` and
        :py:meth:`~.PrimeIdeal.test_factor`.

        However, in order to be added and multiplied by other ideals or
        rational numbers, it must first be converted into a
        :py:class:`~.Submodule`, which is a class that supports these
        operations.

        In many cases, the user need not perform this conversion deliberately,
        since it is automatically performed by the arithmetic operator methods
        :py:meth:`~.PrimeIdeal.__add__` and :py:meth:`~.PrimeIdeal.__mul__`.

        Raising a :py:class:`~.PrimeIdeal` to a non-negative integer power is
        also supported.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly, prime_decomp
        >>> T = Poly(cyclotomic_poly(7))
        >>> P0 = prime_decomp(7, T)[0]
        >>> print(P0**6 == 7*P0.ZK)
        True

        Note that, on both sides of the equation above, we had a
        :py:class:`~.Submodule`. In the next equation we recall that adding
        ideals yields their GCD. This time, we need a deliberate conversion
        to :py:class:`~.Submodule` on the right:

        >>> print(P0 + 7*P0.ZK == P0.as_submodule())
        True

        Returns
        =======

        :py:class:`~.Submodule`
            Will be equal to ``self.p * self.ZK + self.alpha * self.ZK``.

        See Also
        ========

        __add__
        __mul__

        """
        M = self.p * self.ZK + self.alpha * self.ZK
        # Pre-set expensive boolean properties whose value we already know:
        M._starts_with_unity = False
        M._is_sq_maxrank_HNF = True
        return M

    def __eq__(self, other):
        if isinstance(other, PrimeIdeal):
            return self.as_submodule() == other.as_submodule()
        return NotImplemented

    def __add__(self, other):
        """
        Convert to a :py:class:`~.Submodule` and add to another
        :py:class:`~.Submodule`.

        See Also
        ========

        as_submodule

        """
        return self.as_submodule() + other

    __radd__ = __add__

    def __mul__(self, other):
        """
        Convert to a :py:class:`~.Submodule` and multiply by another
        :py:class:`~.Submodule` or a rational number.

        See Also
        ========

        as_submodule

        """
        return self.as_submodule() * other

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.ZK

    def _first_power(self):
        return self

    def test_factor(self):
        r"""
        Compute a test factor for this prime ideal.

        Explanation
        ===========

        Write $\mathfrak{p}$ for this prime ideal, $p$ for the rational prime
        it divides. Then, for computing $\mathfrak{p}$-adic valuations it is
        useful to have a number $\beta \in \mathbb{Z}_K$ such that
        $p/\mathfrak{p} = p \mathbb{Z}_K + \beta \mathbb{Z}_K$.

        Essentially, this is the same as the number $\Psi$ (or the "reagent")
        from Kummer's 1847 paper (*Ueber die Zerlegung...*, Crelle vol. 35) in
        which ideal divisors were invented.
        """
        if self._test_factor is None:
            self._test_factor = _compute_test_factor(self.p, [self.alpha], self.ZK)
        return self._test_factor

    def valuation(self, I):
        r"""
        Compute the $\mathfrak{p}$-adic valuation of integral ideal I at this
        prime ideal.

        Parameters
        ==========

        I : :py:class:`~.Submodule`

        See Also
        ========

        prime_valuation

        """
        return prime_valuation(I, self)

    def reduce_element(self, elt):
        """
        Reduce a :py:class:`~.PowerBasisElement` to a "small representative"
        modulo this prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.PowerBasisElement`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.PowerBasisElement`
            The reduced element.

        See Also
        ========

        reduce_ANP
        reduce_alg_num
        .Submodule.reduce_element

        """
        return self.as_submodule().reduce_element(elt)

    def reduce_ANP(self, a):
        """
        Reduce an :py:class:`~.ANP` to a "small representative" modulo this
        prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.ANP`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.ANP`
            The reduced element.

        See Also
        ========

        reduce_element
        reduce_alg_num
        .Submodule.reduce_element

        """
        elt = self.ZK.parent.element_from_ANP(a)
        red = self.reduce_element(elt)
        return red.to_ANP()

    def reduce_alg_num(self, a):
        """
        Reduce an :py:class:`~.AlgebraicNumber` to a "small representative"
        modulo this prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.AlgebraicNumber`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.AlgebraicNumber`
            The reduced element.

        See Also
        ========

        reduce_element
        reduce_ANP
        .Submodule.reduce_element

        """
        elt = self.ZK.parent.element_from_alg_num(a)
        red = self.reduce_element(elt)
        return a.field_element(list(reversed(red.QQ_col.flat())))


def _compute_test_factor(p, gens, ZK):
    r"""
    Compute the test factor for a :py:class:`~.PrimeIdeal` $\mathfrak{p}$.

    Parameters
    ==========

    p : int
        The rational prime $\mathfrak{p}$ divides

    gens : list of :py:class:`PowerBasisElement`
        A complete set of generators for $\mathfrak{p}$ over *ZK*, EXCEPT that
        an element equivalent to rational *p* can and should be omitted (since
        it has no effect except to waste time).

    ZK : :py:class:`~.Submodule`
        The maximal order where the prime ideal $\mathfrak{p}$ lives.

    Returns
    =======

    :py:class:`~.PowerBasisElement`

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
    (See Proposition 4.8.15.)

    """
    _check_formal_conditions_for_maximal_order(ZK)
    E = ZK.endomorphism_ring()
    matrices = [E.inner_endomorphism(g).matrix(modulus=p) for g in gens]
    B = DomainMatrix.zeros((0, ZK.n), FF(p)).vstack(*matrices)
    # A nonzero element of the nullspace of B will represent a
    # lin comb over the omegas which (i) is not a multiple of p
    # (since it is nonzero over FF(p)), while (ii) is such that
    # its product with each g in gens _is_ a multiple of p (since
    # B represents multiplication by these generators). Theory
    # predicts that such an element must exist, so nullspace should
    # be non-trivial.
    x = B.nullspace()[0, :].transpose()
    beta = ZK.parent(ZK.matrix * x.convert_to(ZZ), denom=ZK.denom)
    return beta


@public
def prime_valuation(I, P):
    r"""
    Compute the *P*-adic valuation for an integral ideal *I*.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.polys.numberfields import prime_valuation
    >>> K = QQ.cyclotomic_field(5)
    >>> P = K.primes_above(5)
    >>> ZK = K.maximal_order()
    >>> print(prime_valuation(25*ZK, P[0]))
    8

    Parameters
    ==========

    I : :py:class:`~.Submodule`
        An integral ideal whose valuation is desired.

    P : :py:class:`~.PrimeIdeal`
        The prime at which to compute the valuation.

    Returns
    =======

    int

    See Also
    ========

    .PrimeIdeal.valuation

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 4.8.17.)

    """
    p, ZK = P.p, P.ZK
    n, W, d = ZK.n, ZK.matrix, ZK.denom

    A = W.convert_to(QQ).inv() * I.matrix * d / I.denom
    # Although A must have integer entries, given that I is an integral ideal,
    # as a DomainMatrix it will still be over QQ, so we convert back:
    A = A.convert_to(ZZ)
    D = A.det()
    if D % p != 0:
        return 0

    beta = P.test_factor()

    f = d ** n // W.det()
    need_complete_test = (f % p == 0)
    v = 0
    while True:
        # Entering the loop, the cols of A represent lin combs of omegas.
        # Turn them into lin combs of thetas:
        A = W * A
        # And then one column at a time...
        for j in range(n):
            c = ZK.parent(A[:, j], denom=d)
            c *= beta
            # ...turn back into lin combs of omegas, after multiplying by beta:
            c = ZK.represent(c).flat()
            for i in range(n):
                A[i, j] = c[i]
        if A[n - 1, n - 1].element % p != 0:
            break
        A = A / p
        # As noted above, domain converts to QQ even when division goes evenly.
        # So must convert back, even when we don't "need_complete_test".
        if need_complete_test:
            # In this case, having a non-integer entry is actually just our
            # halting condition.
            try:
                A = A.convert_to(ZZ)
            except CoercionFailed:
                break
        else:
            # In this case theory says we should not have any non-integer entries.
            A = A.convert_to(ZZ)
        v += 1
    return v


def _two_elt_rep(gens, ZK, p, f=None, Np=None):
    r"""
    Given a set of *ZK*-generators of a prime ideal, compute a set of just two
    *ZK*-generators for the same ideal, one of which is *p* itself.

    Parameters
    ==========

    gens : list of :py:class:`PowerBasisElement`
        Generators for the prime ideal over *ZK*, the ring of integers of the
        field $K$.

    ZK : :py:class:`~.Submodule`
        The maximal order in $K$.

    p : int
        The rational prime divided by the prime ideal.

    f : int, optional
        The inertia degree of the prime ideal, if known.

    Np : int, optional
        The norm $p^f$ of the prime ideal, if known.
        NOTE: There is no reason to supply both *f* and *Np*. Either one will
        save us from having to compute the norm *Np* ourselves. If both are known,
        *Np* is preferred since it saves one exponentiation.

    Returns
    =======

    :py:class:`~.PowerBasisElement` representing a single algebraic integer
    alpha such that the prime ideal is equal to ``p*ZK + alpha*ZK``.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
    (See Algorithm 4.7.10.)

    """
    _check_formal_conditions_for_maximal_order(ZK)
    pb = ZK.parent
    T = pb.T
    # Detect the special cases in which either (a) all generators are multiples
    # of p, or (b) there are no generators (so `all` is vacuously true):
    if all((g % p).equiv(0) for g in gens):
        return pb.zero()

    if Np is None:
        if f is not None:
            Np = p**f
        else:
            Np = abs(pb.submodule_from_gens(gens).matrix.det())

    omega = ZK.basis_element_pullbacks()
    beta = [p*om for om in omega[1:]]  # note: we omit omega[0] == 1
    beta += gens
    search = coeff_search(len(beta), 1)
    for c in search:
        alpha = sum(ci*betai for ci, betai in zip(c, beta))
        # Note: It may be tempting to reduce alpha mod p here, to try to work
        # with smaller numbers, but must not do that, as it can result in an
        # infinite loop! E.g. try factoring 2 in Q(sqrt(-7)).
        n = alpha.norm(T) // Np
        if n % p != 0:
            # Now can reduce alpha mod p.
            return alpha % p


def _prime_decomp_easy_case(p, ZK):
    r"""
    Compute the decomposition of rational prime *p* in the ring of integers
    *ZK* (given as a :py:class:`~.Submodule`), in the "easy case", i.e. the
    case where *p* does not divide the index of $\theta$ in *ZK*, where
    $\theta$ is the generator of the ``PowerBasis`` of which *ZK* is a
    ``Submodule``.
    """
    T = ZK.parent.T
    T_bar = Poly(T, modulus=p)
    lc, fl = T_bar.factor_list()
    if len(fl) == 1 and fl[0][1] == 1:
        return [PrimeIdeal(ZK, p, ZK.parent.zero(), ZK.n, 1)]
    return [PrimeIdeal(ZK, p,
                       ZK.parent.element_from_poly(Poly(t, domain=ZZ)),
                       t.degree(), e)
            for t, e in fl]


def _prime_decomp_compute_kernel(I, p, ZK):
    r"""
    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``ZK/pZK``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    Pair ``(N, G)``, where:

        ``N`` is a :py:class:`~.Module` representing the kernel of the map
        ``a |--> a**p - a`` on ``(O/pO)/I``, guaranteed to be a module with
        unity.

        ``G`` is a :py:class:`~.Module` representing a basis for the separable
        algebra ``A = O/I`` (see Cohen).

    """
    W = I.matrix
    n, r = W.shape
    # Want to take the Fp-basis given by the columns of I, adjoin (1, 0, ..., 0)
    # (which we know is not already in there since I is a basis for a prime ideal)
    # and then supplement this with additional columns to make an invertible n x n
    # matrix. This will then represent a full basis for ZK, whose first r columns
    # are pullbacks of the basis for I.
    if r == 0:
        B = W.eye(n, ZZ)
    else:
        B = W.hstack(W.eye(n, ZZ)[:, 0])
    if B.shape[1] < n:
        B = supplement_a_subspace(B.convert_to(FF(p))).convert_to(ZZ)

    G = ZK.submodule_from_matrix(B)
    # Must compute G's multiplication table _before_ discarding the first r
    # columns. (See Step 9 in Alg 6.2.9 in Cohen, where the betas are actually
    # needed in order to represent each product of gammas. However, once we've
    # found the representations, then we can ignore the betas.)
    G.compute_mult_tab()
    G = G.discard_before(r)

    phi = ModuleEndomorphism(G, lambda x: x**p - x)
    N = phi.kernel(modulus=p)
    assert N.starts_with_unity()
    return N, G


def _prime_decomp_maximal_ideal(I, p, ZK):
    r"""
    We have reached the case where we have a maximal (hence prime) ideal *I*,
    which we know because the quotient ``O/I`` is a field.

    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``O/pO``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    :py:class:`~.PrimeIdeal` instance representing this prime

    """
    m, n = I.matrix.shape
    f = m - n
    G = ZK.matrix * I.matrix
    gens = [ZK.parent(G[:, j], denom=ZK.denom) for j in range(G.shape[1])]
    alpha = _two_elt_rep(gens, ZK, p, f=f)
    return PrimeIdeal(ZK, p, alpha, f)


def _prime_decomp_split_ideal(I, p, N, G, ZK):
    r"""
    Perform the step in the prime decomposition algorithm where we have determined
    the quotient ``ZK/I`` is _not_ a field, and we want to perform a non-trivial
    factorization of *I* by locating an idempotent element of ``ZK/I``.
    """
    assert I.parent == ZK and G.parent is ZK and N.parent is G
    # Since ZK/I is not a field, the kernel computed in the previous step contains
    # more than just the prime field Fp, and our basis N for the nullspace therefore
    # contains at least a second column (which represents an element outside Fp).
    # Let alpha be such an element:
    alpha = N(1).to_parent()
    assert alpha.module is G

    alpha_powers = []
    m = find_min_poly(alpha, FF(p), powers=alpha_powers)
    # TODO (future work):
    #  We don't actually need full factorization, so might use a faster method
    #  to just break off a single non-constant factor m1?
    lc, fl = m.factor_list()
    m1 = fl[0][0]
    m2 = m.quo(m1)
    U, V, g = m1.gcdex(m2)
    # Sanity check: theory says m is squarefree, so m1, m2 should be coprime:
    assert g == 1
    E = list(reversed(Poly(U * m1, domain=ZZ).rep.to_list()))
    eps1 = sum(E[i]*alpha_powers[i] for i in range(len(E)))
    eps2 = 1 - eps1
    idemps = [eps1, eps2]
    factors = []
    for eps in idemps:
        e = eps.to_parent()
        assert e.module is ZK
        D = I.matrix.convert_to(FF(p)).hstack(*[
            (e * om).column(domain=FF(p)) for om in ZK.basis_elements()
        ])
        W = D.columnspace().convert_to(ZZ)
        H = ZK.submodule_from_matrix(W)
        factors.append(H)
    return factors


@public
def prime_decomp(p, T=None, ZK=None, dK=None, radical=None):
    r"""
    Compute the decomposition of rational prime *p* in a number field.

    Explanation
    ===========

    Ordinarily this should be accessed through the
    :py:meth:`~.AlgebraicField.primes_above` method of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x, theta
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.algebraic_field((T, theta))
    >>> print(K.primes_above(2))
    [[ (2, x**2 + 1) e=1, f=1 ], [ (2, (x**2 + 3*x + 2)/2) e=1, f=1 ],
     [ (2, (3*x**2 + 3*x)/2) e=1, f=1 ]]

    Parameters
    ==========

    p : int
        The rational prime whose decomposition is desired.

    T : :py:class:`~.Poly`, optional
        Monic irreducible polynomial defining the number field $K$ in which to
        factor. NOTE: at least one of *T* or *ZK* must be provided.

    ZK : :py:class:`~.Submodule`, optional
        The maximal order for $K$, if already known.
        NOTE: at least one of *T* or *ZK* must be provided.

    dK : int, optional
        The discriminant of the field $K$, if already known.

    radical : :py:class:`~.Submodule`, optional
        The nilradical mod *p* in the integers of $K$, if already known.

    Returns
    =======

    List of :py:class:`~.PrimeIdeal` instances.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 6.2.9.)

    """
    if T is None and ZK is None:
        raise ValueError('At least one of T or ZK must be provided.')
    if ZK is not None:
        _check_formal_conditions_for_maximal_order(ZK)
    if T is None:
        T = ZK.parent.T
    radicals = {}
    if dK is None or ZK is None:
        ZK, dK = round_two(T, radicals=radicals)
    dT = T.discriminant()
    f_squared = dT // dK
    if f_squared % p != 0:
        return _prime_decomp_easy_case(p, ZK)
    radical = radical or radicals.get(p) or nilradical_mod_p(ZK, p)
    stack = [radical]
    primes = []
    while stack:
        I = stack.pop()
        N, G = _prime_decomp_compute_kernel(I, p, ZK)
        if N.n == 1:
            P = _prime_decomp_maximal_ideal(I, p, ZK)
            primes.append(P)
        else:
            I1, I2 = _prime_decomp_split_ideal(I, p, N, G, ZK)
            stack.extend([I1, I2])
    return primes
<|endoftext|>"""
Compute Galois groups of polynomials.

We use algorithms from [1], with some modifications to use lookup tables for
resolvents.

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict
import random

from sympy.core.symbol import Dummy, symbols
from sympy.ntheory.primetest import is_square
from sympy.polys.domains import ZZ
from sympy.polys.densebasic import dup_random
from sympy.polys.densetools import dup_eval
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p
from sympy.polys.numberfields.galois_resolvents import (
    GaloisGroupException, get_resolvent_by_lookup, define_resolvents,
    Resolvent,
)
from sympy.polys.numberfields.utilities import coeff_search
from sympy.polys.polytools import (Poly, poly_from_expr,
                                   PolificationFailed, ComputationFailed)
from sympy.polys.sqfreetools import dup_sqf_p
from sympy.utilities import public


class MaxTriesException(GaloisGroupException):
    ...


def tschirnhausen_transformation(T, max_coeff=10, max_tries=30, history=None,
                                 fixed_order=True):
    r"""
    Given a univariate, monic, irreducible polynomial over the integers, find
    another such polynomial defining the same number field.

    Explanation
    ===========

    See Alg 6.3.4 of [1].

    Parameters
    ==========

    T : Poly
        The given polynomial
    max_coeff : int
        When choosing a transformation as part of the process,
        keep the coeffs between plus and minus this.
    max_tries : int
        Consider at most this many transformations.
    history : set, None, optional (default=None)
        Pass a set of ``Poly.rep``'s in order to prevent any of these
        polynomials from being returned as the polynomial ``U`` i.e. the
        transformation of the given polynomial *T*. The given poly *T* will
        automatically be added to this set, before we try to find a new one.
    fixed_order : bool, default True
        If ``True``, work through candidate transformations A(x) in a fixed
        order, from small coeffs to large, resulting in deterministic behavior.
        If ``False``, the A(x) are chosen randomly, while still working our way
        up from small coefficients to larger ones.

    Returns
    =======

    Pair ``(A, U)``

        ``A`` and ``U`` are ``Poly``, ``A`` is the
        transformation, and ``U`` is the transformed polynomial that defines
        the same number field as *T*. The polynomial ``A`` maps the roots of
        *T* to the roots of ``U``.

    Raises
    ======

    MaxTriesException
        if could not find a polynomial before exceeding *max_tries*.

    """
    X = Dummy('X')
    n = T.degree()
    if history is None:
        history = set()
    history.add(T.rep)

    if fixed_order:
        coeff_generators = {}
        deg_coeff_sum = 3
        current_degree = 2

    def get_coeff_generator(degree):
        gen = coeff_generators.get(degree, coeff_search(degree, 1))
        coeff_generators[degree] = gen
        return gen

    for i in range(max_tries):

        # We never use linear A(x), since applying a fixed linear transformation
        # to all roots will only multiply the discriminant of T by a square
        # integer. This will change nothing important. In particular, if disc(T)
        # was zero before, it will still be zero now, and typically we apply
        # the transformation in hopes of replacing T by a squarefree poly.

        if fixed_order:
            # If d is degree and c max coeff, we move through the dc-space
            # along lines of constant sum. First d + c = 3 with (d, c) = (2, 1).
            # Then d + c = 4 with (d, c) = (3, 1), (2, 2). Then d + c = 5 with
            # (d, c) = (4, 1), (3, 2), (2, 3), and so forth. For a given (d, c)
            # we go though all sets of coeffs where max = c, before moving on.
            gen = get_coeff_generator(current_degree)
            coeffs = next(gen)
            m = max(abs(c) for c in coeffs)
            if current_degree + m > deg_coeff_sum:
                if current_degree == 2:
                    deg_coeff_sum += 1
                    current_degree = deg_coeff_sum - 1
                else:
                    current_degree -= 1
                gen = get_coeff_generator(current_degree)
                coeffs = next(gen)
            a = [ZZ(1)] + [ZZ(c) for c in coeffs]

        else:
            # We use a progressive coeff bound, up to the max specified, since it
            # is preferable to succeed with smaller coeffs.
            # Give each coeff bound five tries, before incrementing.
            C = min(i//5 + 1, max_coeff)
            d = random.randint(2, n - 1)
            a = dup_random(d, -C, C, ZZ)

        A = Poly(a, T.gen)
        U = Poly(T.resultant(X - A), X)
        if U.rep not in history and dup_sqf_p(U.rep.to_list(), ZZ):
            return A, U
    raise MaxTriesException


def has_square_disc(T):
    """Convenience to check if a Poly or dup has square discriminant. """
    d = T.discriminant() if isinstance(T, Poly) else dup_discriminant(T, ZZ)
    return is_square(d)


def _galois_group_degree_3(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 3.

    Explanation
    ===========

    Uses Prop 6.3.5 of [1].

    """
    from sympy.combinatorics.galois import S3TransitiveSubgroups
    return ((S3TransitiveSubgroups.A3, True) if has_square_disc(T)
            else (S3TransitiveSubgroups.S3, False))


def _galois_group_degree_4_root_approx(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Follows Alg 6.3.7 of [1], using a pure root approximation approach.

    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    X = symbols('X0 X1 X2 X3')
    # We start by considering the resolvent for the form
    #   F = X0*X2 + X1*X3
    # and the group G = S4. In this case, the stabilizer H is D4 = < (0123), (02) >,
    # and a set of representatives of G/H is {I, (01), (03)}
    F1 = X[0]*X[2] + X[1]*X[3]
    s1 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]
    R1 = Resolvent(F1, X, s1)

    # In the second half of the algorithm (if we reach it), we use another
    # form and set of coset representatives. However, we may need to permute
    # them first, so cannot form their resolvent now.
    F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s2_pre = [
        Permutation(3),
        Permutation(3)(0, 2)
    ]

    history = set()
    for i in range(max_tries):
        if i > 0:
            # If we're retrying, need a new polynomial T.
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)

        R_dup, _, i0 = R1.eval_for_poly(T, find_integer_root=True)
        # If R is not squarefree, must retry.
        if not dup_sqf_p(R_dup, ZZ):
            continue

        # By Prop 6.3.1 of [1], Gal(T) is contained in A4 iff disc(T) is square.
        sq_disc = has_square_disc(T)

        if i0 is None:
            # By Thm 6.3.3 of [1], Gal(T) is not conjugate to any subgroup of the
            # stabilizer H = D4 that we chose. This means Gal(T) is either A4 or S4.
            return ((S4TransitiveSubgroups.A4, True) if sq_disc
                    else (S4TransitiveSubgroups.S4, False))

        # Gal(T) is conjugate to a subgroup of H = D4, so it is either V, C4
        # or D4 itself.

        if sq_disc:
            # Neither C4 nor D4 is contained in A4, so Gal(T) must be V.
            return (S4TransitiveSubgroups.V, True)

        # Gal(T) can only be D4 or C4.
        # We will now use our second resolvent, with G being that conjugate of D4 that
        # Gal(T) is contained in. To determine the right conjugate, we will need
        # the permutation corresponding to the integer root we found.
        sigma = s1[i0]
        # Applying sigma means permuting the args of F, and
        # conjugating the set of coset representatives.
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma*tau*sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        # If d is zero (R has a repeated root), must retry.
        if d == 0:
            continue
        if is_square(d):
            return (S4TransitiveSubgroups.C4, False)
        else:
            return (S4TransitiveSubgroups.D4, False)

    raise MaxTriesException


def _galois_group_degree_4_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Based on Alg 6.3.6 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 0)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    # Compute list L of degrees of irreducible factors of R, in increasing order:
    fl = dup_factor_list(R_dup, ZZ)
    L = sorted(sum([
        [len(r) - 1] * e for r, e in fl[1]
    ], []))

    if L == [6]:
        return ((S4TransitiveSubgroups.A4, True) if has_square_disc(T)
            else (S4TransitiveSubgroups.S4, False))

    if L == [1, 1, 4]:
        return (S4TransitiveSubgroups.C4, False)

    if L == [2, 2, 2]:
        return (S4TransitiveSubgroups.V, True)

    assert L == [2, 4]
    return (S4TransitiveSubgroups.D4, False)


def _galois_group_degree_5_hybrid(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses a hybrid approach, combining resolvent
    coeff lookup, with root approximation.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    from sympy.combinatorics.permutations import Permutation

    X5 = symbols("X0,X1,X2,X3,X4")
    res = define_resolvents()
    F51, _, s51 = res[(5, 1)]
    F51 = F51.as_expr(*X5)
    R51 = Resolvent(F51, X5, s51)

    history = set()
    reached_second_stage = False
    for i in range(max_tries):
        if i > 0:
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)
        R51_dup = get_resolvent_by_lookup(T, 1)
        if not dup_sqf_p(R51_dup, ZZ):
            continue

        # First stage
        # If we have not yet reached the second stage, then the group still
        # might be S5, A5, or M20, so must test for that.
        if not reached_second_stage:
            sq_disc = has_square_disc(T)

            if dup_irreducible_p(R51_dup, ZZ):
                return ((S5TransitiveSubgroups.A5, True) if sq_disc
                        else (S5TransitiveSubgroups.S5, False))

            if not sq_disc:
                return (S5TransitiveSubgroups.M20, False)

        # Second stage
        reached_second_stage = True
        # R51 must have an integer root for T.
        # To choose our second resolvent, we need to know which conjugate of
        # F51 is a root.
        rounded_roots = R51.round_roots_to_integers_for_poly(T)
        # These are integers, and candidates to be roots of R51.
        # We find the first one that actually is a root.
        for permutation_index, candidate_root in rounded_roots.items():
            if not dup_eval(R51_dup, candidate_root, ZZ):
                break

        X = X5
        F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[4]**2 + X[4]*X[0]**2
        s2_pre = [
            Permutation(4),
            Permutation(4)(0, 1)(2, 4)
        ]

        i0 = permutation_index
        sigma = s51[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma*tau*sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)

        if d == 0:
            continue
        if is_square(d):
            return (S5TransitiveSubgroups.C5, True)
        else:
            return (S5TransitiveSubgroups.D5, True)

    raise MaxTriesException


def _galois_group_degree_5_lookup_ext_factor(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses resolvent coeff lookup, plus
    factorization over an algebraic extension.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups

    _T = T

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    sq_disc = has_square_disc(T)

    if dup_irreducible_p(R_dup, ZZ):
        return ((S5TransitiveSubgroups.A5, True) if sq_disc
                else (S5TransitiveSubgroups.S5, False))

    if not sq_disc:
        return (S5TransitiveSubgroups.M20, False)

    # If we get this far, Gal(T) can only be D5 or C5.
    # But for Gal(T) to have order 5, T must already split completely in
    # the extension field obtained by adjoining a single one of its roots.
    fl = Poly(_T, domain=ZZ.alg_field_from_poly(_T)).factor_list()[1]
    if len(fl) == 5:
        return (S5TransitiveSubgroups.C5, True)
    else:
        return (S5TransitiveSubgroups.D5, True)


def _galois_group_degree_6_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 6.

    Explanation
    ===========

    Based on Alg 6.3.10 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S6TransitiveSubgroups

    # First resolvent:

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    fl = dup_factor_list(R_dup, ZZ)

    # Group the factors by degree.
    factors_by_deg = defaultdict(list)
    for r, _ in fl[1]:
        factors_by_deg[len(r) - 1].append(r)

    L = sorted(sum([
        [d] * len(ff) for d, ff in factors_by_deg.items()
    ], []))

    T_has_sq_disc = has_square_disc(T)

    if L == [1, 2, 3]:
        f1 = factors_by_deg[3][0]
        return ((S6TransitiveSubgroups.C6, False) if has_square_disc(f1)
                else (S6TransitiveSubgroups.D6, False))

    elif L == [3, 3]:
        f1, f2 = factors_by_deg[3]
        any_square = has_square_disc(f1) or has_square_disc(f2)
        return ((S6TransitiveSubgroups.G18, False) if any_square
                else (S6TransitiveSubgroups.G36m, False))

    elif L == [2, 4]:
        if T_has_sq_disc:
            return (S6TransitiveSubgroups.S4p, True)
        else:
            f1 = factors_by_deg[4][0]
            return ((S6TransitiveSubgroups.A4xC2, False) if has_square_disc(f1)
                    else (S6TransitiveSubgroups.S4xC2, False))

    elif L == [1, 1, 4]:
        return ((S6TransitiveSubgroups.A4, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S4m, False))

    elif L == [1, 5]:
        return ((S6TransitiveSubgroups.PSL2F5, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.PGL2F5, False))

    elif L == [1, 1, 1, 3]:
        return (S6TransitiveSubgroups.S3, False)

    assert L == [6]

    # Second resolvent:

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 2)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    T_has_sq_disc = has_square_disc(T)

    if dup_irreducible_p(R_dup, ZZ):
        return ((S6TransitiveSubgroups.A6, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S6, False))
    else:
        return ((S6TransitiveSubgroups.G36p, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.G72, False))


@public
def galois_group(f, *gens, by_name=False, max_tries=30, randomize=False, **args):
    r"""
    Compute the Galois group for polynomials *f* up to degree 6.

    Examples
    ========

    >>> from sympy import galois_group
    >>> from sympy.abc import x
    >>> f = x**4 + 1
    >>> G, alt = galois_group(f)
    >>> print(G)
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    The group is returned along with a boolean, indicating whether it is
    contained in the alternating group $A_n$, where $n$ is the degree of *T*.
    Along with other group properties, this can help determine which group it
    is:

    >>> alt
    True
    >>> G.order()
    4

    Alternatively, the group can be returned by name:

    >>> G_name, _ = galois_group(f, by_name=True)
    >>> print(G_name)
    S4TransitiveSubgroups.V

    The group itself can then be obtained by calling the name's
    ``get_perm_group()`` method:

    >>> G_name.get_perm_group()
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    Group names are values of the enum classes
    :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`,
    :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`,
    etc.

    Parameters
    ==========

    f : Expr
        Irreducible polynomial over :ref:`ZZ` or :ref:`QQ`, whose Galois group
        is to be determined.
    gens : optional list of symbols
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.
    by_name : bool, default False
        If ``True``, the Galois group will be returned by name.
        Otherwise it will be returned as a :py:class:`~.PermutationGroup`.
    max_tries : int, default 30
        Make at most this many attempts in those steps that involve
        generating Tschirnhausen transformations.
    randomize : bool, default False
        If ``True``, then use random coefficients when generating Tschirnhausen
        transformations. Otherwise try transformations in a fixed order. Both
        approaches start with small coefficients and degrees and work upward.
    args : optional
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.

    Returns
    =======

    Pair ``(G, alt)``
        The first element ``G`` indicates the Galois group. It is an instance
        of one of the :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`
        :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`, etc. enum
        classes if *by_name* was ``True``, and a :py:class:`~.PermutationGroup`
        if ``False``.

        The second element is a boolean, saying whether the group is contained
        in the alternating group $A_n$ ($n$ the degree of *T*).

    Raises
    ======

    ValueError
        if *f* is of an unsupported degree.

    MaxTriesException
        if could not complete before exceeding *max_tries* in those steps
        that involve generating Tschirnhausen transformations.

    See Also
    ========

    .Poly.galois_group

    """
    gens = gens or []
    args = args or {}

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('galois_group', 1, exc)

    return F.galois_group(by_name=by_name, max_tries=max_tries,
                          randomize=randomize)
<|endoftext|>"""Minimal polynomials for algebraic numbers."""

from functools import reduce

from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets

from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
    NotAlgebraic,
    GeneratorsError,
)
from sympy.polys.polytools import (
    Poly, PurePoly, invert, factor_list, groebner, resultant,
    degree, poly_from_expr, parallel_poly_from_expr, lcm
)
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
    numbered_symbols, public, sift
)


def _choose_factor(factors, x, v, dom=QQ, prec=200, bound=5):
    """
    Return a factor having root ``v``
    It is assumed that one of the factors has root ``v``.
    """

    if isinstance(factors[0], tuple):
        factors = [f[0] for f in factors]
    if len(factors) == 1:
        return factors[0]

    prec1 = 10
    points = {}
    symbols = dom.symbols if hasattr(dom, 'symbols') else []
    while prec1 <= prec:
        # when dealing with non-Rational numbers we usually evaluate
        # with `subs` argument but we only need a ballpark evaluation
        fe = [f.as_expr().xreplace({x:v}) for f in factors]
        if v.is_number:
            fe = [f.n(prec) for f in fe]

        # assign integers [0, n) to symbols (if any)
        for n in subsets(range(bound), k=len(symbols), repetition=True):
            for s, i in zip(symbols, n):
                points[s] = i

            # evaluate the expression at these points
            candidates = [(abs(f.subs(points).n(prec1)), i)
                for i,f in enumerate(fe)]

            # if we get invalid numbers (e.g. from division by zero)
            # we try again
            if any(i in _illegal for i, _ in candidates):
                continue

            # find the smallest two -- if they differ significantly
            # then we assume we have found the factor that becomes
            # 0 when v is substituted into it
            can = sorted(candidates)
            (a, ix), (b, _) = can[:2]
            if b > a * 10**6:  # XXX what to use?
                return factors[ix]

        prec1 *= 2

    raise NotImplementedError("multiple candidates for the minimal polynomial of %s" % v)


def _is_sum_surds(p):
    return all(f.is_Rational or f.is_Pow and
        f.base.is_Rational and (2*f.exp).is_Integer and f.is_extended_real
        for t in Add.make_args(p) for f in Mul.make_args(t))


def _separate_sq(p):
    """
    helper function for ``_minimal_polynomial_sq``

    It selects a rational ``g`` such that the polynomial ``p``
    consists of a sum of terms whose surds squared have gcd equal to ``g``
    and a sum of terms with surds squared prime with ``g``;
    then it takes the field norm to eliminate ``sqrt(g)``

    See simplify.simplify.split_surds and polytools.sqf_norm.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.minpoly import _separate_sq
    >>> p= -x + sqrt(2) + sqrt(3) + sqrt(7)
    >>> p = _separate_sq(p); p
    -x**2 + 2*sqrt(3)*x + 2*sqrt(7)*x - 2*sqrt(21) - 8
    >>> p = _separate_sq(p); p
    -x**4 + 4*sqrt(7)*x**3 - 32*x**2 + 8*sqrt(7)*x + 20
    >>> p = _separate_sq(p); p
    -x**8 + 48*x**6 - 536*x**4 + 1728*x**2 - 400

    """
    def is_sqrt(expr):
        return expr.is_Pow and expr.exp is S.Half
    # p = c1*sqrt(q1) + ... + cn*sqrt(qn) -> a = [(c1, q1), .., (cn, qn)]
    a = []
    for y in p.args:
        if not y.is_Mul:
            if is_sqrt(y):
                a.append((S.One, y**2))
            elif y.is_Atom:
                a.append((y, S.One))
            elif y.is_Pow and y.exp.is_integer:
                a.append((y, S.One))
            else:
                raise NotImplementedError
        else:
            T, F = sift(y.args, is_sqrt, binary=True)
            a.append((Mul(*F), Mul(*T)**2))
    a.sort(key=lambda z: z[1])
    if a[-1][1] is S.One:
        # there are no surds
        return p
    surds = [z for y, z in a]
    for i in range(len(surds)):
        if surds[i] != 1:
            break
    from sympy.simplify.radsimp import _split_gcd
    g, b1, b2 = _split_gcd(*surds[i:])
    a1 = []
    a2 = []
    for y, z in a:
        if z in b1:
            a1.append(y*z**S.Half)
        else:
            a2.append(y*z**S.Half)
    p1 = Add(*a1)
    p2 = Add(*a2)
    p = _mexpand(p1**2) - _mexpand(p2**2)
    return p

def _minimal_polynomial_sq(p, n, x):
    """
    Returns the minimal polynomial for the ``nth-root`` of a sum of surds
    or ``None`` if it fails.

    Parameters
    ==========

    p : sum of surds
    n : positive integer
    x : variable of the returned polynomial

    Examples
    ========

    >>> from sympy.polys.numberfields.minpoly import _minimal_polynomial_sq
    >>> from sympy import sqrt
    >>> from sympy.abc import x
    >>> q = 1 + sqrt(2) + sqrt(3)
    >>> _minimal_polynomial_sq(q, 3, x)
    x**12 - 4*x**9 - 4*x**6 + 16*x**3 - 8

    """
    p = sympify(p)
    n = sympify(n)
    if not n.is_Integer or not n > 0 or not _is_sum_surds(p):
        return None
    pn = p**Rational(1, n)
    # eliminate the square roots
    p -= x
    while 1:
        p1 = _separate_sq(p)
        if p1 is p:
            p = p1.subs({x:x**n})
            break
        else:
            p = p1

    # _separate_sq eliminates field extensions in a minimal way, so that
    # if n = 1 then `p = constant*(minimal_polynomial(p))`
    # if n > 1 it contains the minimal polynomial as a factor.
    if n == 1:
        p1 = Poly(p)
        if p.coeff(x**p1.degree(x)) < 0:
            p = -p
        p = p.primitive()[1]
        return p
    # by construction `p` has root `pn`
    # the minimal polynomial is the factor vanishing in x = pn
    factors = factor_list(p)[1]

    result = _choose_factor(factors, x, pn)
    return result

def _minpoly_op_algebraic_element(op, ex1, ex2, x, dom, mp1=None, mp2=None):
    """
    return the minimal polynomial for ``op(ex1, ex2)``

    Parameters
    ==========

    op : operation ``Add`` or ``Mul``
    ex1, ex2 : expressions for the algebraic elements
    x : indeterminate of the polynomials
    dom: ground domain
    mp1, mp2 : minimal polynomials for ``ex1`` and ``ex2`` or None

    Examples
    ========

    >>> from sympy import sqrt, Add, Mul, QQ
    >>> from sympy.polys.numberfields.minpoly import _minpoly_op_algebraic_element
    >>> from sympy.abc import x, y
    >>> p1 = sqrt(sqrt(2) + 1)
    >>> p2 = sqrt(sqrt(2) - 1)
    >>> _minpoly_op_algebraic_element(Mul, p1, p2, x, QQ)
    x - 1
    >>> q1 = sqrt(y)
    >>> q2 = 1 / y
    >>> _minpoly_op_algebraic_element(Add, q1, q2, x, QQ.frac_field(y))
    x**2*y**2 - 2*x*y - y**3 + 1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Resultant
    .. [2] I.M. Isaacs, Proc. Amer. Math. Soc. 25 (1970), 638
           "Degrees of sums in a separable field extension".

    """
    y = Dummy(str(x))
    if mp1 is None:
        mp1 = _minpoly_compose(ex1, x, dom)
    if mp2 is None:
        mp2 = _minpoly_compose(ex2, y, dom)
    else:
        mp2 = mp2.subs({x: y})

    if op is Add:
        # mp1a = mp1.subs({x: x - y})
        if dom == QQ:
            R, X = ring('X', QQ)
            p1 = R(dict_from_expr(mp1)[0])
            p2 = R(dict_from_expr(mp2)[0])
        else:
            (p1, p2), _ = parallel_poly_from_expr((mp1, x - y), x, y)
            r = p1.compose(p2)
            mp1a = r.as_expr()

    elif op is Mul:
        mp1a = _muly(mp1, x, y)
    else:
        raise NotImplementedError('option not available')

    if op is Mul or dom != QQ:
        r = resultant(mp1a, mp2, gens=[y, x])
    else:
        r = rs_compose_add(p1, p2)
        r = expr_from_dict(r.as_expr_dict(), x)

    deg1 = degree(mp1, x)
    deg2 = degree(mp2, y)
    if op is Mul and deg1 == 1 or deg2 == 1:
        # if deg1 = 1, then mp1 = x - a; mp1a = x - y - a;
        # r = mp2(x - a), so that `r` is irreducible
        return r

    r = Poly(r, x, domain=dom)
    _, factors = r.factor_list()
    res = _choose_factor(factors, x, op(ex1, ex2), dom)
    return res.as_expr()


def _invertx(p, x):
    """
    Returns ``expand_mul(x**degree(p, x)*p.subs(x, 1/x))``
    """
    p1 = poly_from_expr(p, x)[0]

    n = degree(p1)
    a = [c * x**(n - i) for (i,), c in p1.terms()]
    return Add(*a)


def _muly(p, x, y):
    """
    Returns ``_mexpand(y**deg*p.subs({x:x / y}))``
    """
    p1 = poly_from_expr(p, x)[0]

    n = degree(p1)
    a = [c * x**i * y**(n - i) for (i,), c in p1.terms()]
    return Add(*a)


def _minpoly_pow(ex, pw, x, dom, mp=None):
    """
    Returns ``minpoly(ex**pw, x)``

    Parameters
    ==========

    ex : algebraic element
    pw : rational number
    x : indeterminate of the polynomial
    dom: ground domain
    mp : minimal polynomial of ``p``

    Examples
    ========

    >>> from sympy import sqrt, QQ, Rational
    >>> from sympy.polys.numberfields.minpoly import _minpoly_pow, minpoly
    >>> from sympy.abc import x, y
    >>> p = sqrt(1 + sqrt(2))
    >>> _minpoly_pow(p, 2, x, QQ)
    x**2 - 2*x - 1
    >>> minpoly(p**2, x)
    x**2 - 2*x - 1
    >>> _minpoly_pow(y, Rational(1, 3), x, QQ.frac_field(y))
    x**3 - y
    >>> minpoly(y**Rational(1, 3), x)
    x**3 - y

    """
    pw = sympify(pw)
    if not mp:
        mp = _minpoly_compose(ex, x, dom)
    if not pw.is_rational:
        raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    if pw < 0:
        if mp == x:
            raise ZeroDivisionError('%s is zero' % ex)
        mp = _invertx(mp, x)
        if pw == -1:
            return mp
        pw = -pw
        ex = 1/ex

    y = Dummy(str(x))
    mp = mp.subs({x: y})
    n, d = pw.as_numer_denom()
    res = Poly(resultant(mp, x**d - y**n, gens=[y]), x, domain=dom)
    _, factors = res.factor_list()
    res = _choose_factor(factors, x, ex**pw, dom)
    return res.as_expr()


def _minpoly_add(x, dom, *a):
    """
    returns ``minpoly(Add(*a), dom, x)``
    """
    mp = _minpoly_op_algebraic_element(Add, a[0], a[1], x, dom)
    p = a[0] + a[1]
    for px in a[2:]:
        mp = _minpoly_op_algebraic_element(Add, p, px, x, dom, mp1=mp)
        p = p + px
    return mp


def _minpoly_mul(x, dom, *a):
    """
    returns ``minpoly(Mul(*a), dom, x)``
    """
    mp = _minpoly_op_algebraic_element(Mul, a[0], a[1], x, dom)
    p = a[0] * a[1]
    for px in a[2:]:
        mp = _minpoly_op_algebraic_element(Mul, p, px, x, dom, mp1=mp)
        p = p * px
    return mp


def _minpoly_sin(ex, x):
    """
    Returns the minimal polynomial of ``sin(ex)``
    see https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            n = c.q
            q = sympify(n)
            if q.is_prime:
                # for a = pi*p/q with q odd prime, using chebyshevt
                # write sin(q*a) = mp(sin(a))*sin(a);
                # the roots of mp(x) are sin(pi*p/q) for p = 1,..., q - 1
                a = dup_chebyshevt(n, ZZ)
                return Add(*[x**(n - i - 1)*a[i] for i in range(n)])
            if c.p == 1:
                if q == 9:
                    return 64*x**6 - 96*x**4 + 36*x**2 - 3

            if n % 2 == 1:
                # for a = pi*p/q with q odd, use
                # sin(q*a) = 0 to see that the minimal polynomial must be
                # a factor of dup_chebyshevt(n, ZZ)
                a = dup_chebyshevt(n, ZZ)
                a = [x**(n - i)*a[i] for i in range(n + 1)]
                r = Add(*a)
                _, factors = factor_list(r)
                res = _choose_factor(factors, x, ex)
                return res

            expr = ((1 - cos(2*c*pi))/2)**S.Half
            res = _minpoly_compose(expr, x, QQ)
            return res

    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_cos(ex, x):
    """
    Returns the minimal polynomial of ``cos(ex)``
    see https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            if c.p == 1:
                if c.q == 7:
                    return 8*x**3 - 4*x**2 - 4*x + 1
                if c.q == 9:
                    return 8*x**3 - 6*x - 1
            elif c.p == 2:
                q = sympify(c.q)
                if q.is_prime:
                    s = _minpoly_sin(ex, x)
                    return _mexpand(s.subs({x:sqrt((1 - x)/2)}))

            # for a = pi*p/q, cos(q*a) =T_q(cos(a)) = (-1)**p
            n = int(c.q)
            a = dup_chebyshevt(n, ZZ)
            a = [x**(n - i)*a[i] for i in range(n + 1)]
            r = Add(*a) - (-1)**c.p
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res

    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_tan(ex, x):
    """
    Returns the minimal polynomial of ``tan(ex)``
    see https://github.com/sympy/sympy/issues/21430
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            c = c * 2
            n = int(c.q)
            a = n if c.p % 2 == 0 else 1
            terms = []
            for k in range((c.p+1)%2, n+1, 2):
                terms.append(a*x**k)
                a = -(a*(n-k-1)*(n-k)) // ((k+1)*(k+2))

            r = Add(*terms)
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res

    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_exp(ex, x):
    """
    Returns the minimal polynomial of ``exp(ex)``
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a == I*pi:
        if c.is_rational:
            q = sympify(c.q)
            if c.p == 1 or c.p == -1:
                if q == 3:
                    return x**2 - x + 1
                if q == 4:
                    return x**4 + 1
                if q == 6:
                    return x**4 - x**2 + 1
                if q == 8:
                    return x**8 + 1
                if q == 9:
                    return x**6 - x**3 + 1
                if q == 10:
                    return x**8 - x**6 + x**4 - x**2 + 1
                if q.is_prime:
                    s = 0
                    for i in range(q):
                        s += (-x)**i
                    return s

            # x**(2*q) = product(factors)
            factors = [cyclotomic_poly(i, x) for i in divisors(2*q)]
            mp = _choose_factor(factors, x, ex)
            return mp
        else:
            raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_rootof(ex, x):
    """
    Returns the minimal polynomial of a ``CRootOf`` object.
    """
    p = ex.expr
    p = p.subs({ex.poly.gens[0]:x})
    _, factors = factor_list(p, x)
    result = _choose_factor(factors, x, ex)
    return result


def _minpoly_compose(ex, x, dom):
    """
    Computes the minimal polynomial of an algebraic element
    using operations on minimal polynomials

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, Rational
    >>> from sympy.abc import x, y
    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=True)
    x**2 - 2*x - 1
    >>> minimal_polynomial(sqrt(y) + 1/y, x, compose=True)
    x**2*y**2 - 2*x*y - y**3 + 1

    """
    if ex.is_Rational:
        return ex.q*x - ex.p
    if ex is I:
        _, factors = factor_list(x**2 + 1, x, domain=dom)
        return x**2 + 1 if len(factors) == 1 else x - I

    if ex is S.GoldenRatio:
        _, factors = factor_list(x**2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x**2 - x - 1
        else:
            return _choose_factor(factors, x, (1 + sqrt(5))/2, dom=dom)

    if ex is S.TribonacciConstant:
        _, factors = factor_list(x**3 - x**2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x**3 - x**2 - x - 1
        else:
            fac = (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3
            return _choose_factor(factors, x, fac, dom=dom)

    if hasattr(dom, 'symbols') and ex in dom.symbols:
        return x - ex

    if dom.is_QQ and _is_sum_surds(ex):
        # eliminate the square roots
        v = ex
        ex -= x
        while 1:
            ex1 = _separate_sq(ex)
            if ex1 is ex:
                return _choose_factor(factor_list(ex)[1], x, v)
            else:
                ex = ex1

    if ex.is_Add:
        res = _minpoly_add(x, dom, *ex.args)
    elif ex.is_Mul:
        f = Factors(ex).factors
        r = sift(f.items(), lambda itx: itx[0].is_Rational and itx[1].is_Rational)
        if r[True] and dom == QQ:
            ex1 = Mul(*[bx**ex for bx, ex in r[False] + r[None]])
            r1 = dict(r[True])
            dens = [y.q for y in r1.values()]
            lcmdens = reduce(lcm, dens, 1)
            neg1 = S.NegativeOne
            expn1 = r1.pop(neg1, S.Zero)
            nums = [base**(y.p*lcmdens // y.q) for base, y in r1.items()]
            ex2 = Mul(*nums)
            mp1 = minimal_polynomial(ex1, x)
            # use the fact that in SymPy canonicalization products of integers
            # raised to rational powers are organized in relatively prime
            # bases, and that in ``base**(n/d)`` a perfect power is
            # simplified with the root
            # Powers of -1 have to be treated separately to preserve sign.
            mp2 = ex2.q*x**lcmdens - ex2.p*neg1**(expn1*lcmdens)
            ex2 = neg1**expn1 * ex2**Rational(1, lcmdens)
            res = _minpoly_op_algebraic_element(Mul, ex1, ex2, x, dom, mp1=mp1, mp2=mp2)
        else:
            res = _minpoly_mul(x, dom, *ex.args)
    elif ex.is_Pow:
        res = _minpoly_pow(ex.base, ex.exp, x, dom)
    elif ex.__class__ is sin:
        res = _minpoly_sin(ex, x)
    elif ex.__class__ is cos:
        res = _minpoly_cos(ex, x)
    elif ex.__class__ is tan:
        res = _minpoly_tan(ex, x)
    elif ex.__class__ is exp:
        res = _minpoly_exp(ex, x)
    elif ex.__class__ is CRootOf:
        res = _minpoly_rootof(ex, x)
    else:
        raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    return res


@public
def minimal_polynomial(ex, x=None, compose=True, polys=False, domain=None):
    """
    Computes the minimal polynomial of an algebraic element.

    Parameters
    ==========

    ex : Expr
        Element or expression whose minimal polynomial is to be calculated.

    x : Symbol, optional
        Independent variable of the minimal polynomial

    compose : boolean, optional (default=True)
        Method to use for computing minimal polynomial. If ``compose=True``
        (default) then ``_minpoly_compose`` is used, if ``compose=False`` then
        groebner bases are used.

    polys : boolean, optional (default=False)
        If ``True`` returns a ``Poly`` object else an ``Expr`` object.

    domain : Domain, optional
        Ground domain

    Notes
    =====

    By default ``compose=True``, the minimal polynomial of the subexpressions of ``ex``
    are computed, then the arithmetic operations on them are performed using the resultant
    and factorization.
    If ``compose=False``, a bottom-up algorithm is used with ``groebner``.
    The default algorithm stalls less frequently.

    If no ground domain is given, it will be generated automatically from the expression.

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, solve, QQ
    >>> from sympy.abc import x, y

    >>> minimal_polynomial(sqrt(2), x)
    x**2 - 2
    >>> minimal_polynomial(sqrt(2), x, domain=QQ.algebraic_field(sqrt(2)))
    x - sqrt(2)
    >>> minimal_polynomial(sqrt(2) + sqrt(3), x)
    x**4 - 10*x**2 + 1
    >>> minimal_polynomial(solve(x**3 + x + 3)[0], x)
    x**3 + x + 3
    >>> minimal_polynomial(sqrt(y), x)
    x**2 - y

    """

    ex = sympify(ex)
    if ex.is_number:
        # not sure if it's always needed but try it for numbers (issue 8354)
        ex = _mexpand(ex, recursive=True)
    for expr in preorder_traversal(ex):
        if expr.is_AlgebraicNumber:
            compose = False
            break

    if x is not None:
        x, cls = sympify(x), Poly
    else:
        x, cls = Dummy('x'), PurePoly

    if not domain:
        if ex.free_symbols:
            domain = FractionField(QQ, list(ex.free_symbols))
        else:
            domain = QQ
    if hasattr(domain, 'symbols') and x in domain.symbols:
        raise GeneratorsError("the variable %s is an element of the ground "
                              "domain %s" % (x, domain))

    if compose:
        result = _minpoly_compose(ex, x, domain)
        result = result.primitive()[1]
        c = result.coeff(x**degree(result, x))
        if c.is_negative:
            result = expand_mul(-result)
        return cls(result, x, field=True) if polys else result.collect(x)

    if not domain.is_QQ:
        raise NotImplementedError("groebner method only works for QQ")

    result = _minpoly_groebner(ex, x, cls)
    return cls(result, x, field=True) if polys else result.collect(x)


def _minpoly_groebner(ex, x, cls):
    """
    Computes the minimal polynomial of an algebraic number
    using Groebner bases

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, Rational
    >>> from sympy.abc import x
    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=False)
    x**2 - 2*x - 1

    """

    generator = numbered_symbols('a', cls=Dummy)
    mapping, symbols = {}, {}

    def update_mapping(ex, exp, base=None):
        a = next(generator)
        symbols[ex] = a

        if base is not None:
            mapping[ex] = a**exp + base
        else:
            mapping[ex] = exp.as_expr(a)

        return a

    def bottom_up_scan(ex):
        """
        Transform a given algebraic expression *ex* into a multivariate
        polynomial, by introducing fresh variables with defining equations.

        Explanation
        ===========

        The critical elements of the algebraic expression *ex* are root
        extractions, instances of :py:class:`~.AlgebraicNumber`, and negative
        powers.

        When we encounter a root extraction or an :py:class:`~.AlgebraicNumber`
        we replace this expression with a fresh variable ``a_i``, and record
        the defining polynomial for ``a_i``. For example, if ``a_0**(1/3)``
        occurs, we will replace it with ``a_1``, and record the new defining
        polynomial ``a_1**3 - a_0``.

        When we encounter a negative power we transform it into a positive
        power by algebraically inverting the base. This means computing the
        minimal polynomial in ``x`` for the base, inverting ``x`` modulo this
        poly (which generates a new polynomial) and then substituting the
        original base expression for ``x`` in this last polynomial.

        We return the transformed expression, and we record the defining
        equations for new symbols using the ``update_mapping()`` function.

        """
        if ex.is_Atom:
            if ex is S.ImaginaryUnit:
                if ex not in mapping:
                    return update_mapping(ex, 2, 1)
                else:
                    return symbols[ex]
            elif ex.is_Rational:
                return ex
        elif ex.is_Add:
            return Add(*[ bottom_up_scan(g) for g in ex.args ])
        elif ex.is_Mul:
            return Mul(*[ bottom_up_scan(g) for g in ex.args ])
        elif ex.is_Pow:
            if ex.exp.is_Rational:
                if ex.exp < 0:
                    minpoly_base = _minpoly_groebner(ex.base, x, cls)
                    inverse = invert(x, minpoly_base).as_expr()
                    base_inv = inverse.subs(x, ex.base).expand()

                    if ex.exp == -1:
                        return bottom_up_scan(base_inv)
                    else:
                        ex = base_inv**(-ex.exp)
                if not ex.exp.is_Integer:
                    base, exp = (
                        ex.base**ex.exp.p).expand(), Rational(1, ex.exp.q)
                else:
                    base, exp = ex.base, ex.exp
                base = bottom_up_scan(base)
                expr = base**exp

                if expr not in mapping:
                    if exp.is_Integer:
                        return expr.expand()
                    else:
                        return update_mapping(expr, 1 / exp, -base)
                else:
                    return symbols[expr]
        elif ex.is_AlgebraicNumber:
            if ex not in mapping:
                return update_mapping(ex, ex.minpoly_of_element())
            else:
                return symbols[ex]

        raise NotAlgebraic("%s does not seem to be an algebraic number" % ex)

    def simpler_inverse(ex):
        """
        Returns True if it is more likely that the minimal polynomial
        algorithm works better with the inverse
        """
        if ex.is_Pow:
            if (1/ex.exp).is_integer and ex.exp < 0:
                if ex.base.is_Add:
                    return True
        if ex.is_Mul:
            hit = True
            for p in ex.args:
                if p.is_Add:
                    return False
                if p.is_Pow:
                    if p.base.is_Add and p.exp > 0:
                        return False

            if hit:
                return True
        return False

    inverted = False
    ex = expand_multinomial(ex)
    if ex.is_AlgebraicNumber:
        return ex.minpoly_of_element().as_expr(x)
    elif ex.is_Rational:
        result = ex.q*x - ex.p
    else:
        inverted = simpler_inverse(ex)
        if inverted:
            ex = ex**-1
        res = None
        if ex.is_Pow and (1/ex.exp).is_Integer:
            n = 1/ex.exp
            res = _minimal_polynomial_sq(ex.base, n, x)

        elif _is_sum_surds(ex):
            res = _minimal_polynomial_sq(ex, S.One, x)

        if res is not None:
            result = res

        if res is None:
            bus = bottom_up_scan(ex)
            F = [x - bus] + list(mapping.values())
            G = groebner(F, list(symbols.values()) + [x], order='lex')

            _, factors = factor_list(G[-1])
            # by construction G[-1] has root `ex`
            result = _choose_factor(factors, x, ex)
    if inverted:
        result = _invertx(result, x)
        if result.coeff(x**degree(result, x)) < 0:
            result = expand_mul(-result)

    return result


@public
def minpoly(ex, x=None, compose=True, polys=False, domain=None):
    """This is a synonym for :py:func:`~.minimal_polynomial`."""
    return minimal_polynomial(ex, x=x, compose=compose, polys=polys, domain=domain)
<|endoftext|>"""Utilities for algebraic number theory. """

from sympy.core.sympify import sympify
from sympy.ntheory.factor_ import factorint
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.minpoly import minpoly
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.utilities.decorator import public
from sympy.utilities.lambdify import lambdify

from mpmath import mp


def is_rat(c):
    r"""
    Test whether an argument is of an acceptable type to be used as a rational
    number.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int``, :ref:`ZZ`, or :ref:`QQ`.

    See Also
    ========

    is_int

    """
    # ``c in QQ`` is too accepting (e.g. ``3.14 in QQ`` is ``True``),
    # ``QQ.of_type(c)`` is too demanding (e.g. ``QQ.of_type(3)`` is ``False``).
    #
    # Meanwhile, if gmpy2 is installed then ``ZZ.of_type()`` accepts only
    # ``mpz``, not ``int``, so we need another clause to ensure ``int`` is
    # accepted.
    return isinstance(c, int) or ZZ.of_type(c) or QQ.of_type(c)


def is_int(c):
    r"""
    Test whether an argument is of an acceptable type to be used as an integer.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int`` or :ref:`ZZ`.

    See Also
    ========

    is_rat

    """
    # If gmpy2 is installed then ``ZZ.of_type()`` accepts only
    # ``mpz``, not ``int``, so we need another clause to ensure ``int`` is
    # accepted.
    return isinstance(c, int) or ZZ.of_type(c)


def get_num_denom(c):
    r"""
    Given any argument on which :py:func:`~.is_rat` is ``True``, return the
    numerator and denominator of this number.

    See Also
    ========

    is_rat

    """
    r = QQ(c)
    return r.numerator, r.denominator


@public
def extract_fundamental_discriminant(a):
    r"""
    Extract a fundamental discriminant from an integer *a*.

    Explanation
    ===========

    Given any rational integer *a* that is 0 or 1 mod 4, write $a = d f^2$,
    where $d$ is either 1 or a fundamental discriminant, and return a pair
    of dictionaries ``(D, F)`` giving the prime factorizations of $d$ and $f$
    respectively, in the same format returned by :py:func:`~.factorint`.

    A fundamental discriminant $d$ is different from unity, and is either
    1 mod 4 and squarefree, or is 0 mod 4 and such that $d/4$ is squarefree
    and 2 or 3 mod 4. This is the same as being the discriminant of some
    quadratic field.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import extract_fundamental_discriminant
    >>> print(extract_fundamental_discriminant(-432))
    ({3: 1, -1: 1}, {2: 2, 3: 1})

    For comparison:

    >>> from sympy import factorint
    >>> print(factorint(-432))
    {2: 4, 3: 3, -1: 1}

    Parameters
    ==========

    a: int, must be 0 or 1 mod 4

    Returns
    =======

    Pair ``(D, F)``  of dictionaries.

    Raises
    ======

    ValueError
        If *a* is not 0 or 1 mod 4.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Prop. 5.1.3)

    """
    if a % 4 not in [0, 1]:
        raise ValueError('To extract fundamental discriminant, number must be 0 or 1 mod 4.')
    if a == 0:
        return {}, {0: 1}
    if a == 1:
        return {}, {}
    a_factors = factorint(a)
    D = {}
    F = {}
    # First pass: just make d squarefree, and a/d a perfect square.
    # We'll count primes (and units! i.e. -1) that are 3 mod 4 and present in d.
    num_3_mod_4 = 0
    for p, e in a_factors.items():
        if e % 2 == 1:
            D[p] = 1
            if p % 4 == 3:
                num_3_mod_4 += 1
            if e >= 3:
                F[p] = (e - 1) // 2
        else:
            F[p] = e // 2
    # Second pass: if d is cong. to 2 or 3 mod 4, then we must steal away
    # another factor of 4 from f**2 and give it to d.
    even = 2 in D
    if even or num_3_mod_4 % 2 == 1:
        e2 = F[2]
        assert e2 > 0
        if e2 == 1:
            del F[2]
        else:
            F[2] = e2 - 1
        D[2] = 3 if even else 2
    return D, F


@public
class AlgIntPowers:
    r"""
    Compute the powers of an algebraic integer.

    Explanation
    ===========

    Given an algebraic integer $\theta$ by its monic irreducible polynomial
    ``T`` over :ref:`ZZ`, this class computes representations of arbitrarily
    high powers of $\theta$, as :ref:`ZZ`-linear combinations over
    $\{1, \theta, \ldots, \theta^{n-1}\}$, where $n = \deg(T)$.

    The representations are computed using the linear recurrence relations for
    powers of $\theta$, derived from the polynomial ``T``. See [1], Sec. 4.2.2.

    Optionally, the representations may be reduced with respect to a modulus.

    Examples
    ========

    >>> from sympy import Poly, cyclotomic_poly
    >>> from sympy.polys.numberfields.utilities import AlgIntPowers
    >>> T = Poly(cyclotomic_poly(5))
    >>> zeta_pow = AlgIntPowers(T)
    >>> print(zeta_pow[0])
    [1, 0, 0, 0]
    >>> print(zeta_pow[1])
    [0, 1, 0, 0]
    >>> print(zeta_pow[4])  # doctest: +SKIP
    [-1, -1, -1, -1]
    >>> print(zeta_pow[24])  # doctest: +SKIP
    [-1, -1, -1, -1]

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    """

    def __init__(self, T, modulus=None):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`
            The monic irreducible polynomial over :ref:`ZZ` defining the
            algebraic integer.

        modulus : int, None, optional
            If not ``None``, all representations will be reduced w.r.t. this.

        """
        self.T = T
        self.modulus = modulus
        self.n = T.degree()
        self.powers_n_and_up = [[-c % self for c in reversed(T.rep.to_list())][:-1]]
        self.max_so_far = self.n

    def red(self, exp):
        return exp if self.modulus is None else exp % self.modulus

    def __rmod__(self, other):
        return self.red(other)

    def compute_up_through(self, e):
        m = self.max_so_far
        if e <= m: return
        n = self.n
        r = self.powers_n_and_up
        c = r[0]
        for k in range(m+1, e+1):
            b = r[k-1-n][n-1]
            r.append(
                [c[0]*b % self] + [
                    (r[k-1-n][i-1] + c[i]*b) % self for i in range(1, n)
                ]
            )
        self.max_so_far = e

    def get(self, e):
        n = self.n
        if e < 0:
            raise ValueError('Exponent must be non-negative.')
        elif e < n:
            return [1 if i == e else 0 for i in range(n)]
        else:
            self.compute_up_through(e)
            return self.powers_n_and_up[e - n]

    def __getitem__(self, item):
        return self.get(item)


@public
def coeff_search(m, R):
    r"""
    Generate coefficients for searching through polynomials.

    Explanation
    ===========

    Lead coeff is always non-negative. Explore all combinations with coeffs
    bounded in absolute value before increasing the bound. Skip the all-zero
    list, and skip any repeats. See examples.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import coeff_search
    >>> cs = coeff_search(2, 1)
    >>> C = [next(cs) for i in range(13)]
    >>> print(C)
    [[1, 1], [1, 0], [1, -1], [0, 1], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2],
     [1, 2], [1, -2], [0, 2], [3, 3]]

    Parameters
    ==========

    m : int
        Length of coeff list.
    R : int
        Initial max abs val for coeffs (will increase as search proceeds).

    Returns
    =======

    generator
        Infinite generator of lists of coefficients.

    """
    R0 = R
    c = [R] * m
    while True:
        if R == R0 or R in c or -R in c:
            yield c[:]
        j = m - 1
        while c[j] == -R:
            j -= 1
        c[j] -= 1
        for i in range(j + 1, m):
            c[i] = R
        for j in range(m):
            if c[j] != 0:
                break
        else:
            R += 1
            c = [R] * m


def supplement_a_subspace(M):
    r"""
    Extend a basis for a subspace to a basis for the whole space.

    Explanation
    ===========

    Given an $n \times r$ matrix *M* of rank $r$ (so $r \leq n$), this function
    computes an invertible $n \times n$ matrix $B$ such that the first $r$
    columns of $B$ equal *M*.

    This operation can be interpreted as a way of extending a basis for a
    subspace, to give a basis for the whole space.

    To be precise, suppose you have an $n$-dimensional vector space $V$, with
    basis $\{v_1, v_2, \ldots, v_n\}$, and an $r$-dimensional subspace $W$ of
    $V$, spanned by a basis $\{w_1, w_2, \ldots, w_r\}$, where the $w_j$ are
    given as linear combinations of the $v_i$. If the columns of *M* represent
    the $w_j$ as such linear combinations, then the columns of the matrix $B$
    computed by this function give a new basis $\{u_1, u_2, \ldots, u_n\}$ for
    $V$, again relative to the $\{v_i\}$ basis, and such that $u_j = w_j$
    for $1 \leq j \leq r$.

    Examples
    ========

    Note: The function works in terms of columns, so in these examples we
    print matrix transposes in order to make the columns easier to inspect.

    >>> from sympy.polys.matrices import DM
    >>> from sympy import QQ, FF
    >>> from sympy.polys.numberfields.utilities import supplement_a_subspace
    >>> M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()
    >>> print(supplement_a_subspace(M).to_Matrix().transpose())
    Matrix([[1, 7, 0], [2, 3, 4], [1, 0, 0]])

    >>> M2 = M.convert_to(FF(7))
    >>> print(M2.to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3]])
    >>> print(supplement_a_subspace(M2).to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3], [0, 1, 0]])

    Parameters
    ==========

    M : :py:class:`~.DomainMatrix`
        The columns give the basis for the subspace.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        This matrix is invertible and its first $r$ columns equal *M*.

    Raises
    ======

    DMRankError
        If *M* was not of maximal rank.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*
       (See Sec. 2.3.2.)

    """
    n, r = M.shape
    # Let In be the n x n identity matrix.
    # Form the augmented matrix [M | In] and compute RREF.
    Maug = M.hstack(M.eye(n, M.domain))
    R, pivots = Maug.rref()
    if pivots[:r] != tuple(range(r)):
        raise DMRankError('M was not of maximal rank')
    # Let J be the n x r matrix equal to the first r columns of In.
    # Since M is of rank r, RREF reduces [M | In] to [J | A], where A is the product of
    # elementary matrices Ei corresp. to the row ops performed by RREF. Since the Ei are
    # invertible, so is A. Let B = A^(-1).
    A = R[:, r:]
    B = A.inv()
    # Then B is the desired matrix. It is invertible, since B^(-1) == A.
    # And A * [M | In] == [J | A]
    #  => A * M == J
    #  => M == B * J == the first r columns of B.
    return B


@public
def isolate(alg, eps=None, fast=False):
    """
    Find a rational isolating interval for a real algebraic number.

    Examples
    ========

    >>> from sympy import isolate, sqrt, Rational
    >>> print(isolate(sqrt(2)))  # doctest: +SKIP
    (1, 2)
    >>> print(isolate(sqrt(2), eps=Rational(1, 100)))
    (24/17, 17/12)

    Parameters
    ==========

    alg : str, int, :py:class:`~.Expr`
        The algebraic number to be isolated. Must be a real number, to use this
        particular function. However, see also :py:meth:`.Poly.intervals`,
        which isolates complex roots when you pass ``all=True``.
    eps : positive element of :ref:`QQ`, None, optional (default=None)
        Precision to be passed to :py:meth:`.Poly.refine_root`
    fast : boolean, optional (default=False)
        Say whether fast refinement procedure should be used.
        (Will be passed to :py:meth:`.Poly.refine_root`.)

    Returns
    =======

    Pair of rational numbers defining an isolating interval for the given
    algebraic number.

    See Also
    ========

    .Poly.intervals

    """
    alg = sympify(alg)

    if alg.is_Rational:
        return (alg, alg)
    elif not alg.is_real:
        raise NotImplementedError(
            "complex algebraic numbers are not supported")

    func = lambdify((), alg, modules="mpmath", printer=IntervalPrinter())

    poly = minpoly(alg, polys=True)
    intervals = poly.intervals(sqf=True)

    dps, done = mp.dps, False

    try:
        while not done:
            alg = func()

            for a, b in intervals:
                if a <= alg.a and alg.b <= b:
                    done = True
                    break
            else:
                mp.dps *= 2
    finally:
        mp.dps = dps

    if eps is not None:
        a, b = poly.refine_root(a, b, eps=eps, fast=fast)

    return (a, b)
<|endoftext|>"""Computational algebraic field theory. """

__all__ = [
    'minpoly', 'minimal_polynomial',

    'field_isomorphism', 'primitive_element', 'to_number_field',

    'isolate',

    'round_two',

    'prime_decomp', 'prime_valuation',

    'galois_group',
]

from .minpoly import minpoly, minimal_polynomial

from .subfield import field_isomorphism, primitive_element, to_number_field

from .utilities import isolate

from .basis import round_two

from .primes import prime_decomp, prime_valuation

from .galoisgroups import galois_group
<|endoftext|>r"""
Functions in ``polys.numberfields.subfield`` solve the "Subfield Problem" and
allied problems, for algebraic number fields.

Following Cohen (see [Cohen93]_ Section 4.5), we can define the main problem as
follows:

* **Subfield Problem:**

  Given two number fields $\mathbb{Q}(\alpha)$, $\mathbb{Q}(\beta)$
  via the minimal polynomials for their generators $\alpha$ and $\beta$, decide
  whether one field is isomorphic to a subfield of the other.

From a solution to this problem flow solutions to the following problems as
well:

* **Primitive Element Problem:**

  Given several algebraic numbers
  $\alpha_1, \ldots, \alpha_m$, compute a single algebraic number $\theta$
  such that $\mathbb{Q}(\alpha_1, \ldots, \alpha_m) = \mathbb{Q}(\theta)$.

* **Field Isomorphism Problem:**

  Decide whether two number fields
  $\mathbb{Q}(\alpha)$, $\mathbb{Q}(\beta)$ are isomorphic.

* **Field Membership Problem:**

  Given two algebraic numbers $\alpha$,
  $\beta$, decide whether $\alpha \in \mathbb{Q}(\beta)$, and if so write
  $\alpha = f(\beta)$ for some $f(x) \in \mathbb{Q}[x]$.
"""

from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public

from mpmath import MPContext


def is_isomorphism_possible(a, b):
    """Necessary but not sufficient test for isomorphism. """
    n = a.minpoly.degree()
    m = b.minpoly.degree()

    if m % n != 0:
        return False

    if n == m:
        return True

    da = a.minpoly.discriminant()
    db = b.minpoly.discriminant()

    i, k, half = 1, m//n, db//2

    while True:
        p = sieve[i]
        P = p**k

        if P > half:
            break

        if ((da % p) % 2) and not (db % P):
            return False

        i += 1

    return True


def field_isomorphism_pslq(a, b):
    """Construct field isomorphism using PSLQ algorithm. """
    if not a.root.is_real or not b.root.is_real:
        raise NotImplementedError("PSLQ doesn't support complex coefficients")

    f = a.minpoly
    g = b.minpoly.replace(f.gen)

    n, m, prev = 100, b.minpoly.degree(), None
    ctx = MPContext()

    for i in range(1, 5):
        A = a.root.evalf(n)
        B = b.root.evalf(n)

        basis = [1, B] + [ B**i for i in range(2, m) ] + [-A]

        ctx.dps = n
        coeffs = ctx.pslq(basis, maxcoeff=10**10, maxsteps=1000)

        if coeffs is None:
            # PSLQ can't find an integer linear combination. Give up.
            break

        if coeffs != prev:
            prev = coeffs
        else:
            # Increasing precision didn't produce anything new. Give up.
            break

        # We have
        #   c0 + c1*B + c2*B^2 + ... + cm-1*B^(m-1) - cm*A ~ 0.
        # So bring cm*A to the other side, and divide through by cm,
        # for an approximate representation of A as a polynomial in B.
        # (We know cm != 0 since `b.minpoly` is irreducible.)
        coeffs = [S(c)/coeffs[-1] for c in coeffs[:-1]]

        # Throw away leading zeros.
        while not coeffs[-1]:
            coeffs.pop()

        coeffs = list(reversed(coeffs))
        h = Poly(coeffs, f.gen, domain='QQ')

        # We only have A ~ h(B). We must check whether the relation is exact.
        if f.compose(h).rem(g).is_zero:
            # Now we know that h(b) is in fact equal to _some conjugate of_ a.
            # But from the very precise approximation A ~ h(B) we can assume
            # the conjugate is a itself.
            return coeffs
        else:
            n *= 2

    return None


def field_isomorphism_factor(a, b):
    """Construct field isomorphism via factorization. """
    _, factors = factor_list(a.minpoly, extension=b)
    for f, _ in factors:
        if f.degree() == 1:
            # Any linear factor f(x) represents some conjugate of a in QQ(b).
            # We want to know whether this linear factor represents a itself.
            # Let f = x - c
            c = -f.rep.TC()
            # Write c as polynomial in b
            coeffs = c.to_sympy_list()
            d, terms = len(coeffs) - 1, []
            for i, coeff in enumerate(coeffs):
                terms.append(coeff*b.root**(d - i))
            r = Add(*terms)
            # Check whether we got the number a
            if a.minpoly.same_root(r, a):
                return coeffs

    # If none of the linear factors represented a in QQ(b), then in fact a is
    # not an element of QQ(b).
    return None


@public
def field_isomorphism(a, b, *, fast=True):
    r"""
    Find an embedding of one number field into another.

    Explanation
    ===========

    This function looks for an isomorphism from $\mathbb{Q}(a)$ onto some
    subfield of $\mathbb{Q}(b)$. Thus, it solves the Subfield Problem.

    Examples
    ========

    >>> from sympy import sqrt, field_isomorphism, I
    >>> print(field_isomorphism(3, sqrt(2)))  # doctest: +SKIP
    [3]
    >>> print(field_isomorphism( I*sqrt(3), I*sqrt(3)/2))  # doctest: +SKIP
    [2, 0]

    Parameters
    ==========

    a : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    b : :py:class:`~.Expr`
        Any expression representing an algebraic number.
    fast : boolean, optional (default=True)
        If ``True``, we first attempt a potentially faster way of computing the
        isomorphism, falling back on a slower method if this fails. If
        ``False``, we go directly to the slower method, which is guaranteed to
        return a result.

    Returns
    =======

    List of rational numbers, or None
        If $\mathbb{Q}(a)$ is not isomorphic to some subfield of
        $\mathbb{Q}(b)$, then return ``None``. Otherwise, return a list of
        rational numbers representing an element of $\mathbb{Q}(b)$ to which
        $a$ may be mapped, in order to define a monomorphism, i.e. an
        isomorphism from $\mathbb{Q}(a)$ to some subfield of $\mathbb{Q}(b)$.
        The elements of the list are the coefficients of falling powers of $b$.

    """
    a, b = sympify(a), sympify(b)

    if not a.is_AlgebraicNumber:
        a = AlgebraicNumber(a)

    if not b.is_AlgebraicNumber:
        b = AlgebraicNumber(b)

    a = a.to_primitive_element()
    b = b.to_primitive_element()

    if a == b:
        return a.coeffs()

    n = a.minpoly.degree()
    m = b.minpoly.degree()

    if n == 1:
        return [a.root]

    if m % n != 0:
        return None

    if fast:
        try:
            result = field_isomorphism_pslq(a, b)

            if result is not None:
                return result
        except NotImplementedError:
            pass

    return field_isomorphism_factor(a, b)


def _switch_domain(g, K):
    # An algebraic relation f(a, b) = 0 over Q can also be written
    # g(b) = 0 where g is in Q(a)[x] and h(a) = 0 where h is in Q(b)[x].
    # This function transforms g into h where Q(b) = K.
    frep = g.rep.inject()
    hrep = frep.eject(K, front=True)

    return g.new(hrep, g.gens[0])


def _linsolve(p):
    # Compute root of linear polynomial.
    c, d = p.rep.to_list()
    return -d/c


@public
def primitive_element(extension, x=None, *, ex=False, polys=False):
    r"""
    Find a single generator for a number field given by several generators.

    Explanation
    ===========

    The basic problem is this: Given several algebraic numbers
    $\alpha_1, \alpha_2, \ldots, \alpha_n$, find a single algebraic number
    $\theta$ such that
    $\mathbb{Q}(\alpha_1, \alpha_2, \ldots, \alpha_n) = \mathbb{Q}(\theta)$.

    This function actually guarantees that $\theta$ will be a linear
    combination of the $\alpha_i$, with non-negative integer coefficients.

    Furthermore, if desired, this function will tell you how to express each
    $\alpha_i$ as a $\mathbb{Q}$-linear combination of the powers of $\theta$.

    Examples
    ========

    >>> from sympy import primitive_element, sqrt, S, minpoly, simplify
    >>> from sympy.abc import x
    >>> f, lincomb, reps = primitive_element([sqrt(2), sqrt(3)], x, ex=True)

    Then ``lincomb`` tells us the primitive element as a linear combination of
    the given generators ``sqrt(2)`` and ``sqrt(3)``.

    >>> print(lincomb)
    [1, 1]

    This means the primtiive element is $\sqrt{2} + \sqrt{3}$.
    Meanwhile ``f`` is the minimal polynomial for this primitive element.

    >>> print(f)
    x**4 - 10*x**2 + 1
    >>> print(minpoly(sqrt(2) + sqrt(3), x))
    x**4 - 10*x**2 + 1

    Finally, ``reps`` (which was returned only because we set keyword arg
    ``ex=True``) tells us how to recover each of the generators $\sqrt{2}$ and
    $\sqrt{3}$ as $\mathbb{Q}$-linear combinations of the powers of the
    primitive element $\sqrt{2} + \sqrt{3}$.

    >>> print([S(r) for r in reps[0]])
    [1/2, 0, -9/2, 0]
    >>> theta = sqrt(2) + sqrt(3)
    >>> print(simplify(theta**3/2 - 9*theta/2))
    sqrt(2)
    >>> print([S(r) for r in reps[1]])
    [-1/2, 0, 11/2, 0]
    >>> print(simplify(-theta**3/2 + 11*theta/2))
    sqrt(3)

    Parameters
    ==========

    extension : list of :py:class:`~.Expr`
        Each expression must represent an algebraic number $\alpha_i$.
    x : :py:class:`~.Symbol`, optional (default=None)
        The desired symbol to appear in the computed minimal polynomial for the
        primitive element $\theta$. If ``None``, we use a dummy symbol.
    ex : boolean, optional (default=False)
        If and only if ``True``, compute the representation of each $\alpha_i$
        as a $\mathbb{Q}$-linear combination over the powers of $\theta$.
    polys : boolean, optional (default=False)
        If ``True``, return the minimal polynomial as a :py:class:`~.Poly`.
        Otherwise return it as an :py:class:`~.Expr`.

    Returns
    =======

    Pair (f, coeffs) or triple (f, coeffs, reps), where:
        ``f`` is the minimal polynomial for the primitive element.
        ``coeffs`` gives the primitive element as a linear combination of the
        given generators.
        ``reps`` is present if and only if argument ``ex=True`` was passed,
        and is a list of lists of rational numbers. Each list gives the
        coefficients of falling powers of the primitive element, to recover
        one of the original, given generators.

    """
    if not extension:
        raise ValueError("Cannot compute primitive element for empty extension")
    extension = [_sympify(ext) for ext in extension]

    if x is not None:
        x, cls = sympify(x), Poly
    else:
        x, cls = Dummy('x'), PurePoly

    def _canonicalize(f):
        _, f = f.primitive()
        if f.LC() < 0:
            f = -f
        return f

    if not ex:
        gen, coeffs = extension[0], [1]
        g = minimal_polynomial(gen, x, polys=True)
        for ext in extension[1:]:
            if ext.is_Rational:
                coeffs.append(0)
                continue
            _, factors = factor_list(g, extension=ext)
            g = _choose_factor(factors, x, gen)
            [s], _, g = g.sqf_norm()
            gen += s*ext
            coeffs.append(s)

        g = _canonicalize(g)
        if not polys:
            return g.as_expr(), coeffs
        else:
            return cls(g), coeffs

    gen, coeffs = extension[0], [1]
    f = minimal_polynomial(gen, x, polys=True)
    K = QQ.algebraic_field((f, gen))  # incrementally constructed field
    reps = [K.unit]  # representations of extension elements in K
    for ext in extension[1:]:
        if ext.is_Rational:
            coeffs.append(0)    # rational ext is not included in the expression of a primitive element
            reps.append(K.convert(ext))    # but it is included in reps
            continue
        p = minimal_polynomial(ext, x, polys=True)
        L = QQ.algebraic_field((p, ext))
        _, factors = factor_list(f, domain=L)
        f = _choose_factor(factors, x, gen)
        [s], g, f = f.sqf_norm()
        gen += s*ext
        coeffs.append(s)
        K = QQ.algebraic_field((f, gen))
        h = _switch_domain(g, K)
        erep = _linsolve(h.gcd(p))  # ext as element of K
        ogen = K.unit - s*erep  # old gen as element of K
        reps = [dup_eval(_.to_list(), ogen, K) for _ in reps] + [erep]

    if K.ext.root.is_Rational:  # all extensions are rational
        H = [K.convert(_).rep for _ in extension]
        coeffs = [0]*len(extension)
        f = cls(x, domain=QQ)
    else:
        H = [_.to_list() for _ in reps]

    f = _canonicalize(f)
    if not polys:
        return f.as_expr(), coeffs, H
    else:
        return f, coeffs, H


@public
def to_number_field(extension, theta=None, *, gen=None, alias=None):
    r"""
    Express one algebraic number in the field generated by another.

    Explanation
    ===========

    Given two algebraic numbers $\eta, \theta$, this function either expresses
    $\eta$ as an element of $\mathbb{Q}(\theta)$, or else raises an exception
    if $\eta \not\in \mathbb{Q}(\theta)$.

    This function is essentially just a convenience, utilizing
    :py:func:`~.field_isomorphism` (our solution of the Subfield Problem) to
    solve this, the Field Membership Problem.

    As an additional convenience, this function allows you to pass a list of
    algebraic numbers $\alpha_1, \alpha_2, \ldots, \alpha_n$ instead of $\eta$.
    It then computes $\eta$ for you, as a solution of the Primitive Element
    Problem, using :py:func:`~.primitive_element` on the list of $\alpha_i$.

    Examples
    ========

    >>> from sympy import sqrt, to_number_field
    >>> eta = sqrt(2)
    >>> theta = sqrt(2) + sqrt(3)
    >>> a = to_number_field(eta, theta)
    >>> print(type(a))
    <class 'sympy.core.numbers.AlgebraicNumber'>
    >>> a.root
    sqrt(2) + sqrt(3)
    >>> print(a)
    sqrt(2)
    >>> a.coeffs()
    [1/2, 0, -9/2, 0]

    We get an :py:class:`~.AlgebraicNumber`, whose ``.root`` is $\theta$, whose
    value is $\eta$, and whose ``.coeffs()`` show how to write $\eta$ as a
    $\mathbb{Q}$-linear combination in falling powers of $\theta$.

    Parameters
    ==========

    extension : :py:class:`~.Expr` or list of :py:class:`~.Expr`
        Either the algebraic number that is to be expressed in the other field,
        or else a list of algebraic numbers, a primitive element for which is
        to be expressed in the other field.
    theta : :py:class:`~.Expr`, None, optional (default=None)
        If an :py:class:`~.Expr` representing an algebraic number, behavior is
        as described under **Explanation**. If ``None``, then this function
        reduces to a shorthand for calling :py:func:`~.primitive_element` on
        ``extension`` and turning the computed primitive element into an
        :py:class:`~.AlgebraicNumber`.
    gen : :py:class:`~.Symbol`, None, optional (default=None)
        If provided, this will be used as the generator symbol for the minimal
        polynomial in the returned :py:class:`~.AlgebraicNumber`.
    alias : str, :py:class:`~.Symbol`, None, optional (default=None)
        If provided, this will be used as the alias symbol for the returned
        :py:class:`~.AlgebraicNumber`.

    Returns
    =======

    AlgebraicNumber
        Belonging to $\mathbb{Q}(\theta)$ and equaling $\eta$.

    Raises
    ======

    IsomorphismFailed
        If $\eta \not\in \mathbb{Q}(\theta)$.

    See Also
    ========

    field_isomorphism
    primitive_element

    """
    if hasattr(extension, '__iter__'):
        extension = list(extension)
    else:
        extension = [extension]

    if len(extension) == 1 and isinstance(extension[0], tuple):
        return AlgebraicNumber(extension[0], alias=alias)

    minpoly, coeffs = primitive_element(extension, gen, polys=True)
    root = sum(coeff*ext for coeff, ext in zip(coeffs, extension))

    if theta is None:
        return AlgebraicNumber((minpoly, root), alias=alias)
    else:
        theta = sympify(theta)

        if not theta.is_AlgebraicNumber:
            theta = AlgebraicNumber(theta, gen=gen, alias=alias)

        coeffs = field_isomorphism(root, theta)

        if coeffs is not None:
            return AlgebraicNumber(theta, coeffs, alias=alias)
        else:
            raise IsomorphismFailed(
                "%s is not in a subfield of %s" % (root, theta.root))
<|endoftext|>"""Computing integral bases for number fields. """

from sympy.polys.polytools import Poly
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.utilities.decorator import public
from .modules import ModuleEndomorphism, ModuleHomomorphism, PowerBasis
from .utilities import extract_fundamental_discriminant


def _apply_Dedekind_criterion(T, p):
    r"""
    Apply the "Dedekind criterion" to test whether the order needs to be
    enlarged relative to a given prime *p*.
    """
    x = T.gen
    T_bar = Poly(T, modulus=p)
    lc, fl = T_bar.factor_list()
    assert lc == 1
    g_bar = Poly(1, x, modulus=p)
    for ti_bar, _ in fl:
        g_bar *= ti_bar
    h_bar = T_bar // g_bar
    g = Poly(g_bar, domain=ZZ)
    h = Poly(h_bar, domain=ZZ)
    f = (g * h - T) // p
    f_bar = Poly(f, modulus=p)
    Z_bar = f_bar
    for b in [g_bar, h_bar]:
        Z_bar = Z_bar.gcd(b)
    U_bar = T_bar // Z_bar
    m = Z_bar.degree()
    return U_bar, m


def nilradical_mod_p(H, p, q=None):
    r"""
    Compute the nilradical mod *p* for a given order *H*, and prime *p*.

    Explanation
    ===========

    This is the ideal $I$ in $H/pH$ consisting of all elements some positive
    power of which is zero in this quotient ring, i.e. is a multiple of *p*.

    Parameters
    ==========

    H : :py:class:`~.Submodule`
        The given order.
    p : int
        The rational prime.
    q : int, optional
        If known, the smallest power of *p* that is $>=$ the dimension of *H*.
        If not provided, we compute it here.

    Returns
    =======

    :py:class:`~.Module` representing the nilradical mod *p* in *H*.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.
    (See Lemma 6.1.6.)

    """
    n = H.n
    if q is None:
        q = p
        while q < n:
            q *= p
    phi = ModuleEndomorphism(H, lambda x: x**q)
    return phi.kernel(modulus=p)


def _second_enlargement(H, p, q):
    r"""
    Perform the second enlargement in the Round Two algorithm.
    """
    Ip = nilradical_mod_p(H, p, q=q)
    B = H.parent.submodule_from_matrix(H.matrix * Ip.matrix, denom=H.denom)
    C = B + p*H
    E = C.endomorphism_ring()
    phi = ModuleHomomorphism(H, E, lambda x: E.inner_endomorphism(x))
    gamma = phi.kernel(modulus=p)
    G = H.parent.submodule_from_matrix(H.matrix * gamma.matrix, denom=H.denom * p)
    H1 = G + H
    return H1, Ip


@public
def round_two(T, radicals=None):
    r"""
    Zassenhaus's "Round 2" algorithm.

    Explanation
    ===========

    Carry out Zassenhaus's "Round 2" algorithm on an irreducible polynomial
    *T* over :ref:`ZZ` or :ref:`QQ`. This computes an integral basis and the
    discriminant for the field $K = \mathbb{Q}[x]/(T(x))$.

    Alternatively, you may pass an :py:class:`~.AlgebraicField` instance, in
    place of the polynomial *T*, in which case the algorithm is applied to the
    minimal polynomial for the field's primitive element.

    Ordinarily this function need not be called directly, as one can instead
    access the :py:meth:`~.AlgebraicField.maximal_order`,
    :py:meth:`~.AlgebraicField.integral_basis`, and
    :py:meth:`~.AlgebraicField.discriminant` methods of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    Working through an AlgebraicField:

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.alg_field_from_poly(T, "theta")
    >>> print(K.maximal_order())
    Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2
    >>> print(K.discriminant())
    -503
    >>> print(K.integral_basis(fmt='sympy'))
    [1, theta, theta/2 + theta**2/2]

    Calling directly:

    >>> from sympy import Poly
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.basis import round_two
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> print(round_two(T))
    (Submodule[[2, 0, 0], [0, 2, 0], [0, 1, 1]]/2, -503)

    The nilradicals mod $p$ that are sometimes computed during the Round Two
    algorithm may be useful in further calculations. Pass a dictionary under
    `radicals` to receive these:

    >>> T = Poly(x**3 + 3*x**2 + 5)
    >>> rad = {}
    >>> ZK, dK = round_two(T, radicals=rad)
    >>> print(rad)
    {3: Submodule[[-1, 1, 0], [-1, 0, 1]]}

    Parameters
    ==========

    T : :py:class:`~.Poly`, :py:class:`~.AlgebraicField`
        Either (1) the irreducible polynomial over :ref:`ZZ` or :ref:`QQ`
        defining the number field, or (2) an :py:class:`~.AlgebraicField`
        representing the number field itself.

    radicals : dict, optional
        This is a way for any $p$-radicals (if computed) to be returned by
        reference. If desired, pass an empty dictionary. If the algorithm
        reaches the point where it computes the nilradical mod $p$ of the ring
        of integers $Z_K$, then an $\mathbb{F}_p$-basis for this ideal will be
        stored in this dictionary under the key ``p``. This can be useful for
        other algorithms, such as prime decomposition.

    Returns
    =======

    Pair ``(ZK, dK)``, where:

        ``ZK`` is a :py:class:`~sympy.polys.numberfields.modules.Submodule`
        representing the maximal order.

        ``dK`` is the discriminant of the field $K = \mathbb{Q}[x]/(T(x))$.

    See Also
    ========

    .AlgebraicField.maximal_order
    .AlgebraicField.integral_basis
    .AlgebraicField.discriminant

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    """
    K = None
    if isinstance(T, AlgebraicField):
        K, T = T, T.ext.minpoly_of_element()
    if (   not T.is_univariate
        or not T.is_irreducible
        or T.domain not in [ZZ, QQ]):
        raise ValueError('Round 2 requires an irreducible univariate polynomial over ZZ or QQ.')
    T, _ = T.make_monic_over_integers_by_scaling_roots()
    n = T.degree()
    D = T.discriminant()
    D_modulus = ZZ.from_sympy(abs(D))
    # D must be 0 or 1 mod 4 (see Cohen Sec 4.4), which ensures we can write
    # it in the form D = D_0 * F**2, where D_0 is 1 or a fundamental discriminant.
    _, F = extract_fundamental_discriminant(D)
    Ztheta = PowerBasis(K or T)
    H = Ztheta.whole_submodule()
    nilrad = None
    while F:
        # Next prime:
        p, e = F.popitem()
        U_bar, m = _apply_Dedekind_criterion(T, p)
        if m == 0:
            continue
        # For a given prime p, the first enlargement of the order spanned by
        # the current basis can be done in a simple way:
        U = Ztheta.element_from_poly(Poly(U_bar, domain=ZZ))
        # TODO:
        #  Theory says only first m columns of the U//p*H term below are needed.
        #  Could be slightly more efficient to use only those. Maybe `Submodule`
        #  class should support a slice operator?
        H = H.add(U // p * H, hnf_modulus=D_modulus)
        if e <= m:
            continue
        # A second, and possibly more, enlargements for p will be needed.
        # These enlargements require a more involved procedure.
        q = p
        while q < n:
            q *= p
        H1, nilrad = _second_enlargement(H, p, q)
        while H1 != H:
            H = H1
            H1, nilrad = _second_enlargement(H, p, q)
    # Note: We do not store all nilradicals mod p, only the very last. This is
    # because, unless computed against the entire integral basis, it might not
    # be accurate. (In other words, if H was not already equal to ZK when we
    # passed it to `_second_enlargement`, then we can't trust the nilradical
    # so computed.) Example: if T(x) = x ** 3 + 15 * x ** 2 - 9 * x + 13, then
    # F is divisible by 2, 3, and 7, and the nilradical mod 2 as computed above
    # will not be accurate for the full, maximal order ZK.
    if nilrad is not None and isinstance(radicals, dict):
        radicals[p] = nilrad
    ZK = H
    # Pre-set expensive boolean properties which we already know to be true:
    ZK._starts_with_unity = True
    ZK._is_sq_maxrank_HNF = True
    dK = (D * ZK.matrix.det() ** 2) // ZK.denom ** (2 * n)
    return ZK, dK
<|endoftext|>r"""Modules in number fields.

The classes defined here allow us to work with finitely generated, free
