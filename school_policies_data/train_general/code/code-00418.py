        F-number of a given lens.

    c: sympifiable
        Circle of Confusion (CoC) of a given image format.

    Example
    =======

    >>> from sympy.physics.optics import hyperfocal_distance
    >>> round(hyperfocal_distance(f = 0.5, N = 8, c = 0.0033), 2)
    9.47
    """

    f = sympify(f)
    N = sympify(N)
    c = sympify(c)

    return (1/(N * c))*(f**2)

def transverse_magnification(si, so):
    """

    Calculates the transverse magnification upon reflection in a mirror,
    which is the ratio of the image size to the object size.

    Parameters
    ==========

    so: sympifiable
        Lens-object distance.

    si: sympifiable
        Lens-image distance.

    Example
    =======

    >>> from sympy.physics.optics import transverse_magnification
    >>> transverse_magnification(30, 15)
    -2

    """

    si = sympify(si)
    so = sympify(so)

    return (-(si/so))
<|endoftext|>#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The module implements routines to model the polarization of optical fields
and can be used to calculate the effects of polarization optical elements on
the fields.

- Jones vectors.

- Stokes vectors.

- Jones matrices.

- Mueller matrices.

Examples
========

We calculate a generic Jones vector:

>>> from sympy import symbols, pprint, zeros, simplify
>>> from sympy.physics.optics.polarization import (jones_vector, stokes_vector,
...     half_wave_retarder, polarizing_beam_splitter, jones_2_stokes)

>>> psi, chi, p, I0 = symbols("psi, chi, p, I0", real=True)
>>> x0 = jones_vector(psi, chi)
>>> pprint(x0, use_unicode=True)
⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
⎢                                ⎥
⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

And the more general Stokes vector:

>>> s0 = stokes_vector(psi, chi, p, I0)
>>> pprint(s0, use_unicode=True)
⎡          I₀          ⎤
⎢                      ⎥
⎢I₀⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
⎢                      ⎥
⎢I₀⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
⎢                      ⎥
⎣    I₀⋅p⋅sin(2⋅χ)     ⎦

We calculate how the Jones vector is modified by a half-wave plate:

>>> alpha = symbols("alpha", real=True)
>>> HWP = half_wave_retarder(alpha)
>>> x1 = simplify(HWP*x0)

We calculate the very common operation of passing a beam through a half-wave
plate and then through a polarizing beam-splitter. We do this by putting this
Jones vector as the first entry of a two-Jones-vector state that is transformed
by a 4x4 Jones matrix modelling the polarizing beam-splitter to get the
transmitted and reflected Jones vectors:

>>> PBS = polarizing_beam_splitter()
>>> X1 = zeros(4, 1)
>>> X1[:2, :] = x1
>>> X2 = PBS*X1
>>> transmitted_port = X2[:2, :]
>>> reflected_port = X2[2:, :]

This allows us to calculate how the power in both ports depends on the initial
polarization:

>>> transmitted_power = jones_2_stokes(transmitted_port)[0]
>>> reflected_power = jones_2_stokes(reflected_port)[0]
>>> print(transmitted_power)
cos(-2*alpha + chi + psi)**2/2 + cos(2*alpha + chi - psi)**2/2


>>> print(reflected_power)
sin(-2*alpha + chi + psi)**2/2 + sin(2*alpha + chi - psi)**2/2

Please see the description of the individual functions for further
details and examples.

References
==========

.. [1] https://en.wikipedia.org/wiki/Jones_calculus
.. [2] https://en.wikipedia.org/wiki/Mueller_calculus
.. [3] https://en.wikipedia.org/wiki/Stokes_parameters

"""

from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct


def jones_vector(psi, chi):
    """A Jones vector corresponding to a polarization ellipse with `psi` tilt,
    and `chi` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the `x` axis.

    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.


    Returns
    =======

    Matrix :
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> psi, chi = symbols("psi, chi", real=True)

    A general Jones vector.

    >>> pprint(jones_vector(psi, chi), use_unicode=True)
    ⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
    ⎢                                ⎥
    ⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

    Horizontal polarization.

    >>> pprint(jones_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization.

    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)
    ⎡0⎤
    ⎢ ⎥
    ⎣1⎦

    Diagonal polarization.

    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)
    ⎡√2⎤
    ⎢──⎥
    ⎢2 ⎥
    ⎢  ⎥
    ⎢√2⎥
    ⎢──⎥
    ⎣2 ⎦

    Anti-diagonal polarization.

    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢-√2 ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Right-hand circular polarization.

    >>> pprint(jones_vector(0, pi/4), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢√2⋅ⅈ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Left-hand circular polarization.

    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)
    ⎡  √2  ⎤
    ⎢  ──  ⎥
    ⎢  2   ⎥
    ⎢      ⎥
    ⎢-√2⋅ⅈ ⎥
    ⎢──────⎥
    ⎣  2   ⎦

    """
    return Matrix([-I*sin(chi)*sin(psi) + cos(chi)*cos(psi),
                   I*sin(chi)*cos(psi) + sin(psi)*cos(chi)])


def stokes_vector(psi, chi, p=1, I=1):
    """A Stokes vector corresponding to a polarization ellipse with ``psi``
    tilt, and ``chi`` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the ``x`` axis.
    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.
    p : numeric type or SymPy Symbol
        The degree of polarization.
    I : numeric type or SymPy Symbol
        The intensity of the field.


    Returns
    =======

    Matrix :
        A Stokes vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import stokes_vector
    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)
    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)
    ⎡          I          ⎤
    ⎢                     ⎥
    ⎢I⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
    ⎢                     ⎥
    ⎢I⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
    ⎢                     ⎥
    ⎣    I⋅p⋅sin(2⋅χ)     ⎦


    Horizontal polarization

    >>> pprint(stokes_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization

    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣0 ⎦

    Diagonal polarization

    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎣0⎦

    Anti-diagonal polarization

    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎣0 ⎦

    Right-hand circular polarization

    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣1⎦

    Left-hand circular polarization

    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣-1⎦

    Unpolarized light

    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    """
    S0 = I
    S1 = I*p*cos(2*psi)*cos(2*chi)
    S2 = I*p*sin(2*psi)*cos(2*chi)
    S3 = I*p*sin(2*chi)
    return Matrix([S0, S1, S2, S3])


def jones_2_stokes(e):
    """Return the Stokes vector for a Jones vector ``e``.

    Parameters
    ==========

    e : SymPy Matrix
        A Jones vector.

    Returns
    =======

    SymPy Matrix
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> from sympy.physics.optics.polarization import jones_2_stokes
    >>> H = jones_vector(0, 0)
    >>> V = jones_vector(pi/2, 0)
    >>> D = jones_vector(pi/4, 0)
    >>> A = jones_vector(-pi/4, 0)
    >>> R = jones_vector(0, pi/4)
    >>> L = jones_vector(0, -pi/4)
    >>> pprint([jones_2_stokes(e) for e in [H, V, D, A, R, L]],
    ...         use_unicode=True)
    ⎡⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤⎤
    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥
    ⎢⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥  ⎢0⎥  ⎢0 ⎥⎥
    ⎢⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥⎥
    ⎢⎢0⎥  ⎢0 ⎥  ⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥⎥
    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥
    ⎣⎣0⎦  ⎣0 ⎦  ⎣0⎦  ⎣0 ⎦  ⎣1⎦  ⎣-1⎦⎦

    """
    ex, ey = e
    return Matrix([Abs(ex)**2 + Abs(ey)**2,
                   Abs(ex)**2 - Abs(ey)**2,
                   2*re(ex*ey.conjugate()),
                   -2*im(ex*ey.conjugate())])


def linear_polarizer(theta=0):
    """A linear polarizer Jones matrix with transmission axis at
    an angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the transmission axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the polarizer.

    Examples
    ========

    A generic polarizer.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import linear_polarizer
    >>> theta = symbols("theta", real=True)
    >>> J = linear_polarizer(theta)
    >>> pprint(J, use_unicode=True)
    ⎡      2                     ⎤
    ⎢   cos (θ)     sin(θ)⋅cos(θ)⎥
    ⎢                            ⎥
    ⎢                     2      ⎥
    ⎣sin(θ)⋅cos(θ)     sin (θ)   ⎦


    """
    M = Matrix([[cos(theta)**2, sin(theta)*cos(theta)],
                [sin(theta)*cos(theta), sin(theta)**2]])
    return M


def phase_retarder(theta=0, delta=0):
    """A phase retarder Jones matrix with retardance ``delta`` at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.
    delta : numeric type or SymPy Symbol
        The phase difference between the fast and slow axes of the
        transmitted light.

    Returns
    =======

    SymPy Matrix :
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic retarder.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import phase_retarder
    >>> theta, delta = symbols("theta, delta", real=True)
    >>> R = phase_retarder(theta, delta)
    >>> pprint(R, use_unicode=True)
    ⎡                          -ⅈ⋅δ               -ⅈ⋅δ               ⎤
    ⎢                          ─────              ─────              ⎥
    ⎢⎛ ⅈ⋅δ    2         2   ⎞    2    ⎛     ⅈ⋅δ⎞    2                ⎥
    ⎢⎝ℯ   ⋅sin (θ) + cos (θ)⎠⋅ℯ       ⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                                ⎥
    ⎢            -ⅈ⋅δ                                           -ⅈ⋅δ ⎥
    ⎢            ─────                                          ─────⎥
    ⎢⎛     ⅈ⋅δ⎞    2                  ⎛ ⅈ⋅δ    2         2   ⎞    2  ⎥
    ⎣⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝ℯ   ⋅cos (θ) + sin (θ)⎠⋅ℯ     ⎦

    """
    R = Matrix([[cos(theta)**2 + exp(I*delta)*sin(theta)**2,
                (1-exp(I*delta))*cos(theta)*sin(theta)],
                [(1-exp(I*delta))*cos(theta)*sin(theta),
                sin(theta)**2 + exp(I*delta)*cos(theta)**2]])
    return R*exp(-I*delta/2)


def half_wave_retarder(theta):
    """A half-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic half-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import half_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> HWP = half_wave_retarder(theta)
    >>> pprint(HWP, use_unicode=True)
    ⎡   ⎛     2         2   ⎞                        ⎤
    ⎢-ⅈ⋅⎝- sin (θ) + cos (θ)⎠    -2⋅ⅈ⋅sin(θ)⋅cos(θ)  ⎥
    ⎢                                                ⎥
    ⎢                             ⎛   2         2   ⎞⎥
    ⎣   -2⋅ⅈ⋅sin(θ)⋅cos(θ)     -ⅈ⋅⎝sin (θ) - cos (θ)⎠⎦

    """
    return phase_retarder(theta, pi)


def quarter_wave_retarder(theta):
    """A quarter-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic quarter-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import quarter_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> QWP = quarter_wave_retarder(theta)
    >>> pprint(QWP, use_unicode=True)
    ⎡                       -ⅈ⋅π            -ⅈ⋅π               ⎤
    ⎢                       ─────           ─────              ⎥
    ⎢⎛     2         2   ⎞    4               4                ⎥
    ⎢⎝ⅈ⋅sin (θ) + cos (θ)⎠⋅ℯ       (1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                          ⎥
    ⎢         -ⅈ⋅π                                        -ⅈ⋅π ⎥
    ⎢         ─────                                       ─────⎥
    ⎢           4                  ⎛   2           2   ⎞    4  ⎥
    ⎣(1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝sin (θ) + ⅈ⋅cos (θ)⎠⋅ℯ     ⎦

    """
    return phase_retarder(theta, pi/2)


def transmissive_filter(T):
    """An attenuator Jones matrix with transmittance ``T``.

    Parameters
    ==========

    T : numeric type or SymPy Symbol
        The transmittance of the attenuator.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import transmissive_filter
    >>> T = symbols("T", real=True)
    >>> NDF = transmissive_filter(T)
    >>> pprint(NDF, use_unicode=True)
    ⎡√T  0 ⎤
    ⎢      ⎥
    ⎣0   √T⎦

    """
    return Matrix([[sqrt(T), 0], [0, sqrt(T)]])


def reflective_filter(R):
    """A reflective filter Jones matrix with reflectance ``R``.

    Parameters
    ==========

    R : numeric type or SymPy Symbol
        The reflectance of the filter.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import reflective_filter
    >>> R = symbols("R", real=True)
    >>> pprint(reflective_filter(R), use_unicode=True)
    ⎡√R   0 ⎤
    ⎢       ⎥
    ⎣0   -√R⎦

    """
    return Matrix([[sqrt(R), 0], [0, -sqrt(R)]])


def mueller_matrix(J):
    """The Mueller matrix corresponding to Jones matrix `J`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.

    Returns
    =======

    SymPy Matrix
        The corresponding Mueller matrix.

    Examples
    ========

    Generic optical components.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import (mueller_matrix,
    ...     linear_polarizer, half_wave_retarder, quarter_wave_retarder)
    >>> theta = symbols("theta", real=True)

    A linear_polarizer

    >>> pprint(mueller_matrix(linear_polarizer(theta)), use_unicode=True)
    ⎡            cos(2⋅θ)      sin(2⋅θ)     ⎤
    ⎢  1/2       ────────      ────────    0⎥
    ⎢               2             2         ⎥
    ⎢                                       ⎥
    ⎢cos(2⋅θ)  cos(4⋅θ)   1    sin(4⋅θ)     ⎥
    ⎢────────  ──────── + ─    ────────    0⎥
    ⎢   2         4       4       4         ⎥
    ⎢                                       ⎥
    ⎢sin(2⋅θ)    sin(4⋅θ)    1   cos(4⋅θ)   ⎥
    ⎢────────    ────────    ─ - ────────  0⎥
    ⎢   2           4        4      4       ⎥
    ⎢                                       ⎥
    ⎣   0           0             0        0⎦

    A half-wave plate

    >>> pprint(mueller_matrix(half_wave_retarder(theta)), use_unicode=True)
    ⎡1              0                           0               0 ⎤
    ⎢                                                             ⎥
    ⎢        4           2                                        ⎥
    ⎢0  8⋅sin (θ) - 8⋅sin (θ) + 1           sin(4⋅θ)            0 ⎥
    ⎢                                                             ⎥
    ⎢                                     4           2           ⎥
    ⎢0          sin(4⋅θ)           - 8⋅sin (θ) + 8⋅sin (θ) - 1  0 ⎥
    ⎢                                                             ⎥
    ⎣0              0                           0               -1⎦

    A quarter-wave plate

    >>> pprint(mueller_matrix(quarter_wave_retarder(theta)), use_unicode=True)
    ⎡1       0             0            0    ⎤
    ⎢                                        ⎥
    ⎢   cos(4⋅θ)   1    sin(4⋅θ)             ⎥
    ⎢0  ──────── + ─    ────────    -sin(2⋅θ)⎥
    ⎢      2       2       2                 ⎥
    ⎢                                        ⎥
    ⎢     sin(4⋅θ)    1   cos(4⋅θ)           ⎥
    ⎢0    ────────    ─ - ────────  cos(2⋅θ) ⎥
    ⎢        2        2      2               ⎥
    ⎢                                        ⎥
    ⎣0    sin(2⋅θ)     -cos(2⋅θ)        0    ⎦

    """
    A = Matrix([[1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 1, 1, 0],
                [0, -I, I, 0]])

    return simplify(A*TensorProduct(J, J.conjugate())*A.inv())


def polarizing_beam_splitter(Tp=1, Rs=1, Ts=0, Rp=0, phia=0, phib=0):
    r"""A polarizing beam splitter Jones matrix at angle `theta`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.
    Tp : numeric type or SymPy Symbol
        The transmissivity of the P-polarized component.
    Rs : numeric type or SymPy Symbol
        The reflectivity of the S-polarized component.
    Ts : numeric type or SymPy Symbol
        The transmissivity of the S-polarized component.
    Rp : numeric type or SymPy Symbol
        The reflectivity of the P-polarized component.
    phia : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode a.
    phib : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode b.


    Returns
    =======

    SymPy Matrix
        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector
        whose first two entries are the Jones vector on one of the PBS ports,
        and the last two entries the Jones vector on the other port.

    Examples
    ========

    Generic polarizing beam-splitter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter
    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)
    >>> phia, phib = symbols("phi_a, phi_b", real=True)
    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)
    >>> pprint(PBS, use_unicode=False)
    [   ____                           ____                    ]
    [ \/ Tp            0           I*\/ Rp           0         ]
    [                                                          ]
    [                  ____                       ____  I*phi_a]
    [   0            \/ Ts            0      -I*\/ Rs *e       ]
    [                                                          ]
    [    ____                         ____                     ]
    [I*\/ Rp           0            \/ Tp            0         ]
    [                                                          ]
    [               ____  I*phi_b                    ____      ]
    [   0      -I*\/ Rs *e            0            \/ Ts       ]

    """
    PBS = Matrix([[sqrt(Tp), 0, I*sqrt(Rp), 0],
                  [0, sqrt(Ts), 0, -I*sqrt(Rs)*exp(I*phia)],
                  [I*sqrt(Rp), 0, sqrt(Tp), 0],
                  [0, -I*sqrt(Rs)*exp(I*phib), 0, sqrt(Ts)]])
    return PBS
<|endoftext|>__all__ = [
    'TWave',

    'RayTransferMatrix', 'FreeSpace', 'FlatRefraction', 'CurvedRefraction',
    'FlatMirror', 'CurvedMirror', 'ThinLens', 'GeometricRay', 'BeamParameter',
    'waist2rayleigh', 'rayleigh2waist', 'geometric_conj_ab',
    'geometric_conj_af', 'geometric_conj_bf', 'gaussian_conj',
    'conjugate_gauss_beams',

    'Medium',

    'refraction_angle', 'deviation', 'fresnel_coefficients', 'brewster_angle',
    'critical_angle', 'lens_makers_formula', 'mirror_formula', 'lens_formula',
    'hyperfocal_distance', 'transverse_magnification',

    'jones_vector', 'stokes_vector', 'jones_2_stokes', 'linear_polarizer',
    'phase_retarder', 'half_wave_retarder', 'quarter_wave_retarder',
    'transmissive_filter', 'reflective_filter', 'mueller_matrix',
    'polarizing_beam_splitter',
]
from .waves import TWave

from .gaussopt import (RayTransferMatrix, FreeSpace, FlatRefraction,
        CurvedRefraction, FlatMirror, CurvedMirror, ThinLens, GeometricRay,
        BeamParameter, waist2rayleigh, rayleigh2waist, geometric_conj_ab,
        geometric_conj_af, geometric_conj_bf, gaussian_conj,
        conjugate_gauss_beams)

from .medium import Medium

from .utils import (refraction_angle, deviation, fresnel_coefficients,
        brewster_angle, critical_angle, lens_makers_formula, mirror_formula,
        lens_formula, hyperfocal_distance, transverse_magnification)

from .polarization import (jones_vector, stokes_vector, jones_2_stokes,
        linear_polarizer, phase_retarder, half_wave_retarder,
        quarter_wave_retarder, transmissive_filter, reflective_filter,
        mueller_matrix, polarizing_beam_splitter)
<|endoftext|>"""
**Contains**

* Medium
"""
from sympy.physics.units import second, meter, kilogram, ampere

__all__ = ['Medium']

from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units import speed_of_light, u0, e0


c = speed_of_light.convert_to(meter/second)
_e0mksa = e0.convert_to(ampere**2*second**4/(kilogram*meter**3))
_u0mksa = u0.convert_to(meter*kilogram/(ampere**2*second**2))


class Medium(Basic):

    """
    This class represents an optical medium. The prime reason to implement this is
    to facilitate refraction, Fermat's principle, etc.

    Explanation
    ===========

    An optical medium is a material through which electromagnetic waves propagate.
    The permittivity and permeability of the medium define how electromagnetic
    waves propagate in it.


    Parameters
    ==========

    name: string
        The display name of the Medium.

    permittivity: Sympifyable
        Electric permittivity of the space.

    permeability: Sympifyable
        Magnetic permeability of the space.

    n: Sympifyable
        Index of refraction of the medium.


    Examples
    ========

    >>> from sympy.abc import epsilon, mu
    >>> from sympy.physics.optics import Medium
    >>> m1 = Medium('m1')
    >>> m2 = Medium('m2', epsilon, mu)
    >>> m1.intrinsic_impedance
    149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)
    >>> m2.refractive_index
    299792458*meter*sqrt(epsilon*mu)/second


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Optical_medium

    """

    def __new__(cls, name, permittivity=None, permeability=None, n=None):
        if not isinstance(name, Str):
            name = Str(name)

        permittivity = _sympify(permittivity) if permittivity is not None else permittivity
        permeability = _sympify(permeability) if permeability is not None else permeability
        n = _sympify(n) if n is not None else n

        if n is not None:
            if permittivity is not None and permeability is None:
                permeability = n**2/(c**2*permittivity)
                return MediumPP(name, permittivity, permeability)
            elif permeability is not None and permittivity is None:
                permittivity = n**2/(c**2*permeability)
                return MediumPP(name, permittivity, permeability)
            elif permittivity is not None and permittivity is not None:
                raise ValueError("Specifying all of permittivity, permeability, and n is not allowed")
            else:
                return MediumN(name, n)
        elif permittivity is not None and permeability is not None:
            return MediumPP(name, permittivity, permeability)
        elif permittivity is None and permeability is None:
            return MediumPP(name, _e0mksa, _u0mksa)
        else:
            raise ValueError("Arguments are underspecified. Either specify n or any two of permittivity, "
                             "permeability, and n")

    @property
    def name(self):
        return self.args[0]

    @property
    def speed(self):
        """
        Returns speed of the electromagnetic wave travelling in the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.speed
        299792458*meter/second
        >>> m2 = Medium('m2', n=1)
        >>> m.speed == m2.speed
        True

        """
        return c / self.n

    @property
    def refractive_index(self):
        """
        Returns refractive index of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.refractive_index
        1

        """
        return (c/self.speed)


class MediumN(Medium):

    """
    Represents an optical medium for which only the refractive index is known.
    Useful for simple ray optics.

    This class should never be instantiated directly.
    Instead it should be instantiated indirectly by instantiating Medium with
    only n specified.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> m = Medium('m', n=2)
    >>> m
    MediumN(Str('m'), 2)
    """

    def __new__(cls, name, n):
        obj = super(Medium, cls).__new__(cls, name, n)
        return obj

    @property
    def n(self):
        return self.args[1]


class MediumPP(Medium):
    """
    Represents an optical medium for which the permittivity and permeability are known.

    This class should never be instantiated directly. Instead it should be
    instantiated indirectly by instantiating Medium with any two of
    permittivity, permeability, and n specified, or by not specifying any
    of permittivity, permeability, or n, in which case default values for
    permittivity and permeability will be used.

    Examples
    ========
    >>> from sympy.physics.optics import Medium
    >>> from sympy.abc import epsilon, mu
    >>> m1 = Medium('m1', permittivity=epsilon, permeability=mu)
    >>> m1
    MediumPP(Str('m1'), epsilon, mu)
    >>> m2 = Medium('m2')
    >>> m2
    MediumPP(Str('m2'), 625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3), pi*kilogram*meter/(2500000*ampere**2*second**2))
    """


    def __new__(cls, name, permittivity, permeability):
        obj = super(Medium, cls).__new__(cls, name, permittivity, permeability)
        return obj

    @property
    def intrinsic_impedance(self):
        """
        Returns intrinsic impedance of the medium.

        Explanation
        ===========

        The intrinsic impedance of a medium is the ratio of the
        transverse components of the electric and magnetic fields
        of the electromagnetic wave travelling in the medium.
        In a region with no electrical conductivity it simplifies
        to the square root of ratio of magnetic permeability to
        electric permittivity.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.intrinsic_impedance
        149896229*pi*kilogram*meter**2/(1250000*ampere**2*second**3)

        """
        return sqrt(self.permeability / self.permittivity)

    @property
    def permittivity(self):
        """
        Returns electric permittivity of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.permittivity
        625000*ampere**2*second**4/(22468879468420441*pi*kilogram*meter**3)

        """
        return self.args[1]

    @property
    def permeability(self):
        """
        Returns magnetic permeability of the medium.

        Examples
        ========

        >>> from sympy.physics.optics import Medium
        >>> m = Medium('m')
        >>> m.permeability
        pi*kilogram*meter/(2500000*ampere**2*second**2)

        """
        return self.args[2]

    @property
    def n(self):
        return c*sqrt(self.permittivity*self.permeability)
<|endoftext|>"""
This module has all the classes and functions related to waves in optics.

**Contains**

* TWave
"""

__all__ = ['TWave']

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function
from sympy.core.numbers import (Number, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.physics.units import speed_of_light, meter, second


c = speed_of_light.convert_to(meter/second)


class TWave(Expr):

    r"""
    This is a simple transverse sine wave travelling in a one-dimensional space.
    Basic properties are required at the time of creation of the object,
    but they can be changed later with respective methods provided.

    Explanation
    ===========

    It is represented as :math:`A \times cos(k*x - \omega \times t + \phi )`,
    where :math:`A` is the amplitude, :math:`\omega` is the angular frequency,
    :math:`k` is the wavenumber (spatial frequency), :math:`x` is a spatial variable
    to represent the position on the dimension on which the wave propagates,
    and :math:`\phi` is the phase angle of the wave.


    Arguments
    =========

    amplitude : Sympifyable
        Amplitude of the wave.
    frequency : Sympifyable
        Frequency of the wave.
    phase : Sympifyable
        Phase angle of the wave.
    time_period : Sympifyable
        Time period of the wave.
    n : Sympifyable
        Refractive index of the medium.

    Raises
    =======

    ValueError : When neither frequency nor time period is provided
        or they are not consistent.
    TypeError : When anything other than TWave objects is added.


    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.optics import TWave
    >>> A1, phi1, A2, phi2, f = symbols('A1, phi1, A2, phi2, f')
    >>> w1 = TWave(A1, f, phi1)
    >>> w2 = TWave(A2, f, phi2)
    >>> w3 = w1 + w2  # Superposition of two waves
    >>> w3
    TWave(sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2), f,
        atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2)), 1/f, n)
    >>> w3.amplitude
    sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2)
    >>> w3.phase
    atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2))
    >>> w3.speed
    299792458*meter/(second*n)
    >>> w3.angular_velocity
    2*pi*f

    """

    def __new__(
            cls,
            amplitude,
            frequency=None,
            phase=S.Zero,
            time_period=None,
            n=Symbol('n')):
        if time_period is not None:
            time_period = _sympify(time_period)
            _frequency = S.One/time_period
        if frequency is not None:
            frequency = _sympify(frequency)
            _time_period = S.One/frequency
            if time_period is not None:
                if frequency != S.One/time_period:
                    raise ValueError("frequency and time_period should be consistent.")
        if frequency is None and time_period is None:
            raise ValueError("Either frequency or time period is needed.")
        if frequency is None:
            frequency = _frequency
        if time_period is None:
            time_period = _time_period

        amplitude = _sympify(amplitude)
        phase = _sympify(phase)
        n = sympify(n)
        obj = Basic.__new__(cls, amplitude, frequency, phase, time_period, n)
        return obj

    @property
    def amplitude(self):
        """
        Returns the amplitude of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.amplitude
        A
        """
        return self.args[0]

    @property
    def frequency(self):
        """
        Returns the frequency of the wave,
        in cycles per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.frequency
        f
        """
        return self.args[1]

    @property
    def phase(self):
        """
        Returns the phase angle of the wave,
        in radians.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.phase
        phi
        """
        return self.args[2]

    @property
    def time_period(self):
        """
        Returns the temporal period of the wave,
        in seconds per cycle.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.time_period
        1/f
        """
        return self.args[3]

    @property
    def n(self):
        """
        Returns the refractive index of the medium
        """
        return self.args[4]

    @property
    def wavelength(self):
        """
        Returns the wavelength (spatial period) of the wave,
        in meters per cycle.
        It depends on the medium of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavelength
        299792458*meter/(second*f*n)
        """
        return c/(self.frequency*self.n)


    @property
    def speed(self):
        """
        Returns the propagation speed of the wave,
        in meters per second.
        It is dependent on the propagation medium.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.speed
        299792458*meter/(second*n)
        """
        return self.wavelength*self.frequency

    @property
    def angular_velocity(self):
        """
        Returns the angular velocity of the wave,
        in radians per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.angular_velocity
        2*pi*f
        """
        return 2*pi*self.frequency

    @property
    def wavenumber(self):
        """
        Returns the wavenumber of the wave,
        in radians per meter.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavenumber
        pi*second*f*n/(149896229*meter)
        """
        return 2*pi/self.wavelength

    def __str__(self):
        """String representation of a TWave."""
        from sympy.printing import sstr
        return type(self).__name__ + sstr(self.args)

    __repr__ = __str__

    def __add__(self, other):
        """
        Addition of two waves will result in their superposition.
        The type of interference will depend on their phase angles.
        """
        if isinstance(other, TWave):
            if self.frequency == other.frequency and self.wavelength == other.wavelength:
                return TWave(sqrt(self.amplitude**2 + other.amplitude**2 + 2 *
                                  self.amplitude*other.amplitude*cos(
                                      self.phase - other.phase)),
                             self.frequency,
                             atan2(self.amplitude*sin(self.phase)
                             + other.amplitude*sin(other.phase),
                             self.amplitude*cos(self.phase)
                             + other.amplitude*cos(other.phase))
                             )
            else:
                raise NotImplementedError("Interference of waves with different frequencies"
                    " has not been implemented.")
        else:
            raise TypeError(type(other).__name__ + " and TWave objects cannot be added.")

    def __mul__(self, other):
        """
        Multiplying a wave by a scalar rescales the amplitude of the wave.
        """
        other = sympify(other)
        if isinstance(other, Number):
            return TWave(self.amplitude*other, *self.args[1:])
        else:
            raise TypeError(type(other).__name__ + " and TWave objects cannot be multiplied.")

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __neg__(self):
        return self.__mul__(-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return (-self).__radd__(other)

    def _eval_rewrite_as_sin(self, *args, **kwargs):
        return self.amplitude*sin(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase + pi/2, evaluate=False)

    def _eval_rewrite_as_cos(self, *args, **kwargs):
        return self.amplitude*cos(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase)

    def _eval_rewrite_as_pde(self, *args, **kwargs):
        mu, epsilon, x, t = symbols('mu, epsilon, x, t')
        E = Function('E')
        return Derivative(E(x, t), x, 2) + mu*epsilon*Derivative(E(x, t), t, 2)

    def _eval_rewrite_as_exp(self, *args, **kwargs):
        return self.amplitude*exp(I*(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase))
<|endoftext|>"""
    Module to handle gamma matrices expressed as tensor objects.

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i = tensor_indices('i', LorentzIndex)
    >>> G(i)
    GammaMatrix(i)

    Note that there is already an instance of GammaMatrixHead in four dimensions:
    GammaMatrix, which is simply declare as

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i = tensor_indices('i', LorentzIndex)
    >>> GammaMatrix(i)
    GammaMatrix(i)

    To access the metric tensor

    >>> LorentzIndex.metric
    metric(LorentzIndex,LorentzIndex)

"""
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex,\
    TensMul, TensAdd, tensor_mul, Tensor, TensorHead, TensorSymmetry


# DiracSpinorIndex = TensorIndexType('DiracSpinorIndex', dim=4, dummy_name="S")


LorentzIndex = TensorIndexType('LorentzIndex', dim=4, dummy_name="L")


GammaMatrix = TensorHead("GammaMatrix", [LorentzIndex],
                         TensorSymmetry.no_symmetry(1), comm=None)


def extract_type_tens(expression, component):
    """
    Extract from a ``TensExpr`` all tensors with `component`.

    Returns two tensor expressions:

    * the first contains all ``Tensor`` of having `component`.
    * the second contains all remaining.


    """
    if isinstance(expression, Tensor):
        sp = [expression]
    elif isinstance(expression, TensMul):
        sp = expression.args
    else:
        raise ValueError('wrong type')

    # Collect all gamma matrices of the same dimension
    new_expr = S.One
    residual_expr = S.One
    for i in sp:
        if isinstance(i, Tensor) and i.component == component:
            new_expr *= i
        else:
            residual_expr *= i
    return new_expr, residual_expr


def simplify_gamma_expression(expression):
    extracted_expr, residual_expr = extract_type_tens(expression, GammaMatrix)
    res_expr = _simplify_single_line(extracted_expr)
    return res_expr * residual_expr


def simplify_gpgp(ex, sort=True):
    """
    simplify products ``G(i)*p(-i)*G(j)*p(-j) -> p(i)*p(-i)``

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, simplify_gpgp
    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads
    >>> p, q = tensor_heads('p, q', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> ps = p(i0)*G(-i0)
    >>> qs = q(i0)*G(-i0)
    >>> simplify_gpgp(ps*qs*qs)
    GammaMatrix(-L_0)*p(L_0)*q(L_1)*q(-L_1)
    """
    def _simplify_gpgp(ex):
        components = ex.components
        a = []
        comp_map = []
        for i, comp in enumerate(components):
            comp_map.extend([i]*comp.rank)
        dum = [(i[0], i[1], comp_map[i[0]], comp_map[i[1]]) for i in ex.dum]
        for i in range(len(components)):
            if components[i] != GammaMatrix:
                continue
            for dx in dum:
                if dx[2] == i:
                    p_pos1 = dx[3]
                elif dx[3] == i:
                    p_pos1 = dx[2]
                else:
                    continue
                comp1 = components[p_pos1]
                if comp1.comm == 0 and comp1.rank == 1:
                    a.append((i, p_pos1))
        if not a:
            return ex
        elim = set()
        tv = []
        hit = True
        coeff = S.One
        ta = None
        while hit:
            hit = False
            for i, ai in enumerate(a[:-1]):
                if ai[0] in elim:
                    continue
                if ai[0] != a[i + 1][0] - 1:
                    continue
                if components[ai[1]] != components[a[i + 1][1]]:
                    continue
                elim.add(ai[0])
                elim.add(ai[1])
                elim.add(a[i + 1][0])
                elim.add(a[i + 1][1])
                if not ta:
                    ta = ex.split()
                    mu = TensorIndex('mu', LorentzIndex)
                hit = True
                if i == 0:
                    coeff = ex.coeff
                tx = components[ai[1]](mu)*components[ai[1]](-mu)
                if len(a) == 2:
                    tx *= 4  # eye(4)
                tv.append(tx)
                break

        if tv:
            a = [x for j, x in enumerate(ta) if j not in elim]
            a.extend(tv)
            t = tensor_mul(*a)*coeff
            # t = t.replace(lambda x: x.is_Matrix, lambda x: 1)
            return t
        else:
            return ex

    if sort:
        ex = ex.sorted_components()
    # this would be better off with pattern matching
    while 1:
        t = _simplify_gpgp(ex)
        if t != ex:
            ex = t
        else:
            return t


def gamma_trace(t):
    """
    trace of a single line of gamma matrices

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        gamma_trace, LorentzIndex
    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads
    >>> p, q = tensor_heads('p, q', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> ps = p(i0)*G(-i0)
    >>> qs = q(i0)*G(-i0)
    >>> gamma_trace(G(i0)*G(i1))
    4*metric(i0, i1)
    >>> gamma_trace(ps*ps) - 4*p(i0)*p(-i0)
    0
    >>> gamma_trace(ps*qs + ps*ps) - 4*p(i0)*p(-i0) - 4*p(i0)*q(-i0)
    0

    """
    if isinstance(t, TensAdd):
        res = TensAdd(*[gamma_trace(x) for x in t.args])
        return res
    t = _simplify_single_line(t)
    res = _trace_single_line(t)
    return res


def _simplify_single_line(expression):
    """
    Simplify single-line product of gamma matrices.

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, _simplify_single_line
    >>> from sympy.tensor.tensor import tensor_indices, TensorHead
    >>> p = TensorHead('p', [LorentzIndex])
    >>> i0,i1 = tensor_indices('i0:2', LorentzIndex)
    >>> _simplify_single_line(G(i0)*G(i1)*p(-i1)*G(-i0)) + 2*G(i0)*p(-i0)
    0

    """
    t1, t2 = extract_type_tens(expression, GammaMatrix)
    if t1 != 1:
        t1 = kahane_simplify(t1)
    res = t1*t2
    return res


def _trace_single_line(t):
    """
    Evaluate the trace of a single gamma matrix line inside a ``TensExpr``.

    Notes
    =====

    If there are ``DiracSpinorIndex.auto_left`` and ``DiracSpinorIndex.auto_right``
    indices trace over them; otherwise traces are not implied (explain)


    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, _trace_single_line
    >>> from sympy.tensor.tensor import tensor_indices, TensorHead
    >>> p = TensorHead('p', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> _trace_single_line(G(i0)*G(i1))
    4*metric(i0, i1)
    >>> _trace_single_line(G(i0)*p(-i0)*G(i1)*p(-i1)) - 4*p(i0)*p(-i0)
    0

    """
    def _trace_single_line1(t):
        t = t.sorted_components()
        components = t.components
        ncomps = len(components)
        g = LorentzIndex.metric
        # gamma matirices are in a[i:j]
        hit = 0
        for i in range(ncomps):
            if components[i] == GammaMatrix:
                hit = 1
                break

        for j in range(i + hit, ncomps):
            if components[j] != GammaMatrix:
                break
        else:
            j = ncomps
        numG = j - i
        if numG == 0:
            tcoeff = t.coeff
            return t.nocoeff if tcoeff else t
        if numG % 2 == 1:
            return TensMul.from_data(S.Zero, [], [], [])
        elif numG > 4:
            # find the open matrix indices and connect them:
            a = t.split()
            ind1 = a[i].get_indices()[0]
            ind2 = a[i + 1].get_indices()[0]
            aa = a[:i] + a[i + 2:]
            t1 = tensor_mul(*aa)*g(ind1, ind2)
            t1 = t1.contract_metric(g)
            args = [t1]
            sign = 1
            for k in range(i + 2, j):
                sign = -sign
                ind2 = a[k].get_indices()[0]
                aa = a[:i] + a[i + 1:k] + a[k + 1:]
                t2 = sign*tensor_mul(*aa)*g(ind1, ind2)
                t2 = t2.contract_metric(g)
                t2 = simplify_gpgp(t2, False)
                args.append(t2)
            t3 = TensAdd(*args)
            t3 = _trace_single_line(t3)
            return t3
        else:
            a = t.split()
            t1 = _gamma_trace1(*a[i:j])
            a2 = a[:i] + a[j:]
            t2 = tensor_mul(*a2)
            t3 = t1*t2
            if not t3:
                return t3
            t3 = t3.contract_metric(g)
            return t3

    t = t.expand()
    if isinstance(t, TensAdd):
        a = [_trace_single_line1(x)*x.coeff for x in t.args]
        return TensAdd(*a)
    elif isinstance(t, (Tensor, TensMul)):
        r = t.coeff*_trace_single_line1(t)
        return r
    else:
        return trace(t)


def _gamma_trace1(*a):
    gctr = 4  # FIXME specific for d=4
    g = LorentzIndex.metric
    if not a:
        return gctr
    n = len(a)
    if n%2 == 1:
        #return TensMul.from_data(S.Zero, [], [], [])
        return S.Zero
    if n == 2:
        ind0 = a[0].get_indices()[0]
        ind1 = a[1].get_indices()[0]
        return gctr*g(ind0, ind1)
    if n == 4:
        ind0 = a[0].get_indices()[0]
        ind1 = a[1].get_indices()[0]
        ind2 = a[2].get_indices()[0]
        ind3 = a[3].get_indices()[0]

        return gctr*(g(ind0, ind1)*g(ind2, ind3) - \
           g(ind0, ind2)*g(ind1, ind3) + g(ind0, ind3)*g(ind1, ind2))


def kahane_simplify(expression):
    r"""
    This function cancels contracted elements in a product of four
    dimensional gamma matrices, resulting in an expression equal to the given
    one, without the contracted gamma matrices.

    Parameters
    ==========

    `expression`    the tensor expression containing the gamma matrices to simplify.

    Notes
    =====

    If spinor indices are given, the matrices must be given in
    the order given in the product.

    Algorithm
    =========

    The idea behind the algorithm is to use some well-known identities,
    i.e., for contractions enclosing an even number of `\gamma` matrices

    `\gamma^\mu \gamma_{a_1} \cdots \gamma_{a_{2N}} \gamma_\mu = 2 (\gamma_{a_{2N}} \gamma_{a_1} \cdots \gamma_{a_{2N-1}} + \gamma_{a_{2N-1}} \cdots \gamma_{a_1} \gamma_{a_{2N}} )`

    for an odd number of `\gamma` matrices

    `\gamma^\mu \gamma_{a_1} \cdots \gamma_{a_{2N+1}} \gamma_\mu = -2 \gamma_{a_{2N+1}} \gamma_{a_{2N}} \cdots \gamma_{a_{1}}`

    Instead of repeatedly applying these identities to cancel out all contracted indices,
    it is possible to recognize the links that would result from such an operation,
    the problem is thus reduced to a simple rearrangement of free gamma matrices.

    Examples
    ========

    When using, always remember that the original expression coefficient
    has to be handled separately

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.physics.hep.gamma_matrices import kahane_simplify
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i0, i1, i2 = tensor_indices('i0:3', LorentzIndex)
    >>> ta = G(i0)*G(-i0)
    >>> kahane_simplify(ta)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> tb = G(i0)*G(i1)*G(-i0)
    >>> kahane_simplify(tb)
    -2*GammaMatrix(i1)
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])

    If there are no contractions, the same expression is returned

    >>> tc = G(i0)*G(i1)
    >>> kahane_simplify(tc)
    GammaMatrix(i0)*GammaMatrix(i1)

    References
    ==========

    [1] Algorithm for Reducing Contracted Products of gamma Matrices,
    Joseph Kahane, Journal of Mathematical Physics, Vol. 9, No. 10, October 1968.
    """

    if isinstance(expression, Mul):
        return expression
    if isinstance(expression, TensAdd):
        return TensAdd(*[kahane_simplify(arg) for arg in expression.args])

    if isinstance(expression, Tensor):
        return expression

    assert isinstance(expression, TensMul)

    gammas = expression.args

    for gamma in gammas:
        assert gamma.component == GammaMatrix

    free = expression.free
    # spinor_free = [_ for _ in expression.free_in_args if _[1] != 0]

    # if len(spinor_free) == 2:
    #     spinor_free.sort(key=lambda x: x[2])
    #     assert spinor_free[0][1] == 1 and spinor_free[-1][1] == 2
    #     assert spinor_free[0][2] == 0
    # elif spinor_free:
    #     raise ValueError('spinor indices do not match')

    dum = []
    for dum_pair in expression.dum:
        if expression.index_types[dum_pair[0]] == LorentzIndex:
            dum.append((dum_pair[0], dum_pair[1]))

    dum = sorted(dum)

    if len(dum) == 0:  # or GammaMatrixHead:
        # no contractions in `expression`, just return it.
        return expression

    # find the `first_dum_pos`, i.e. the position of the first contracted
    # gamma matrix, Kahane's algorithm as described in his paper requires the
    # gamma matrix expression to start with a contracted gamma matrix, this is
    # a workaround which ignores possible initial free indices, and re-adds
    # them later.

    first_dum_pos = min(map(min, dum))

    # for p1, p2, a1, a2 in expression.dum_in_args:
    #     if p1 != 0 or p2 != 0:
    #         # only Lorentz indices, skip Dirac indices:
    #         continue
    #     first_dum_pos = min(p1, p2)
    #     break

    total_number = len(free) + len(dum)*2
    number_of_contractions = len(dum)

    free_pos = [None]*total_number
    for i in free:
        free_pos[i[1]] = i[0]

    # `index_is_free` is a list of booleans, to identify index position
    # and whether that index is free or dummy.
    index_is_free = [False]*total_number

    for i, indx in enumerate(free):
        index_is_free[indx[1]] = True

    # `links` is a dictionary containing the graph described in Kahane's paper,
    # to every key correspond one or two values, representing the linked indices.
    # All values in `links` are integers, negative numbers are used in the case
    # where it is necessary to insert gamma matrices between free indices, in
    # order to make Kahane's algorithm work (see paper).
    links = {i: [] for i in range(first_dum_pos, total_number)}

    # `cum_sign` is a step variable to mark the sign of every index, see paper.
    cum_sign = -1
    # `cum_sign_list` keeps storage for all `cum_sign` (every index).
    cum_sign_list = [None]*total_number
    block_free_count = 0

    # multiply `resulting_coeff` by the coefficient parameter, the rest
    # of the algorithm ignores a scalar coefficient.
    resulting_coeff = S.One

    # initialize a list of lists of indices. The outer list will contain all
    # additive tensor expressions, while the inner list will contain the
    # free indices (rearranged according to the algorithm).
    resulting_indices = [[]]

    # start to count the `connected_components`, which together with the number
    # of contractions, determines a -1 or +1 factor to be multiplied.
    connected_components = 1

    # First loop: here we fill `cum_sign_list`, and draw the links
    # among consecutive indices (they are stored in `links`). Links among
    # non-consecutive indices will be drawn later.
    for i, is_free in enumerate(index_is_free):
        # if `expression` starts with free indices, they are ignored here;
        # they are later added as they are to the beginning of all
        # `resulting_indices` list of lists of indices.
        if i < first_dum_pos:
            continue

        if is_free:
            block_free_count += 1
            # if previous index was free as well, draw an arch in `links`.
            if block_free_count > 1:
                links[i - 1].append(i)
                links[i].append(i - 1)
        else:
            # Change the sign of the index (`cum_sign`) if the number of free
            # indices preceding it is even.
            cum_sign *= 1 if (block_free_count % 2) else -1
            if block_free_count == 0 and i != first_dum_pos:
                # check if there are two consecutive dummy indices:
                # in this case create virtual indices with negative position,
                # these "virtual" indices represent the insertion of two
                # gamma^0 matrices to separate consecutive dummy indices, as
                # Kahane's algorithm requires dummy indices to be separated by
                # free indices. The product of two gamma^0 matrices is unity,
                # so the new expression being examined is the same as the
                # original one.
                if cum_sign == -1:
                    links[-1-i] = [-1-i+1]
                    links[-1-i+1] = [-1-i]
            if (i - cum_sign) in links:
                if i != first_dum_pos:
                    links[i].append(i - cum_sign)
                if block_free_count != 0:
                    if i - cum_sign < len(index_is_free):
                        if index_is_free[i - cum_sign]:
                            links[i - cum_sign].append(i)
            block_free_count = 0

        cum_sign_list[i] = cum_sign

    # The previous loop has only created links between consecutive free indices,
    # it is necessary to properly create links among dummy (contracted) indices,
    # according to the rules described in Kahane's paper. There is only one exception
    # to Kahane's rules: the negative indices, which handle the case of some
    # consecutive free indices (Kahane's paper just describes dummy indices
    # separated by free indices, hinting that free indices can be added without
    # altering the expression result).
    for i in dum:
        # get the positions of the two contracted indices:
        pos1 = i[0]
        pos2 = i[1]

        # create Kahane's upper links, i.e. the upper arcs between dummy
        # (i.e. contracted) indices:
        links[pos1].append(pos2)
        links[pos2].append(pos1)

        # create Kahane's lower links, this corresponds to the arcs below
        # the line described in the paper:

        # first we move `pos1` and `pos2` according to the sign of the indices:
        linkpos1 = pos1 + cum_sign_list[pos1]
        linkpos2 = pos2 + cum_sign_list[pos2]

        # otherwise, perform some checks before creating the lower arcs:

        # make sure we are not exceeding the total number of indices:
        if linkpos1 >= total_number:
            continue
        if linkpos2 >= total_number:
            continue

        # make sure we are not below the first dummy index in `expression`:
        if linkpos1 < first_dum_pos:
            continue
        if linkpos2 < first_dum_pos:
            continue

        # check if the previous loop created "virtual" indices between dummy
        # indices, in such a case relink `linkpos1` and `linkpos2`:
        if (-1-linkpos1) in links:
            linkpos1 = -1-linkpos1
        if (-1-linkpos2) in links:
            linkpos2 = -1-linkpos2

        # move only if not next to free index:
        if linkpos1 >= 0 and not index_is_free[linkpos1]:
            linkpos1 = pos1

        if linkpos2 >=0 and not index_is_free[linkpos2]:
            linkpos2 = pos2

        # create the lower arcs:
        if linkpos2 not in links[linkpos1]:
            links[linkpos1].append(linkpos2)
        if linkpos1 not in links[linkpos2]:
            links[linkpos2].append(linkpos1)

    # This loop starts from the `first_dum_pos` index (first dummy index)
    # walks through the graph deleting the visited indices from `links`,
    # it adds a gamma matrix for every free index in encounters, while it
    # completely ignores dummy indices and virtual indices.
    pointer = first_dum_pos
    previous_pointer = 0
    while True:
        if pointer in links:
            next_ones = links.pop(pointer)
        else:
            break

        if previous_pointer in next_ones:
            next_ones.remove(previous_pointer)

        previous_pointer = pointer

        if next_ones:
            pointer = next_ones[0]
        else:
            break

        if pointer == previous_pointer:
            break
        if pointer >=0 and free_pos[pointer] is not None:
            for ri in resulting_indices:
                ri.append(free_pos[pointer])

    # The following loop removes the remaining connected components in `links`.
    # If there are free indices inside a connected component, it gives a
    # contribution to the resulting expression given by the factor
    # `gamma_a gamma_b ... gamma_z + gamma_z ... gamma_b gamma_a`, in Kahanes's
    # paper represented as  {gamma_a, gamma_b, ... , gamma_z},
    # virtual indices are ignored. The variable `connected_components` is
    # increased by one for every connected component this loop encounters.

    # If the connected component has virtual and dummy indices only
    # (no free indices), it contributes to `resulting_indices` by a factor of two.
    # The multiplication by two is a result of the
    # factor {gamma^0, gamma^0} = 2 I, as it appears in Kahane's paper.
    # Note: curly brackets are meant as in the paper, as a generalized
    # multi-element anticommutator!

    while links:
        connected_components += 1
        pointer = min(links.keys())
        previous_pointer = pointer
        # the inner loop erases the visited indices from `links`, and it adds
        # all free indices to `prepend_indices` list, virtual indices are
        # ignored.
        prepend_indices = []
        while True:
            if pointer in links:
                next_ones = links.pop(pointer)
            else:
                break

            if previous_pointer in next_ones:
                if len(next_ones) > 1:
                    next_ones.remove(previous_pointer)

            previous_pointer = pointer

            if next_ones:
                pointer = next_ones[0]

            if pointer >= first_dum_pos and free_pos[pointer] is not None:
                prepend_indices.insert(0, free_pos[pointer])
        # if `prepend_indices` is void, it means there are no free indices
        # in the loop (and it can be shown that there must be a virtual index),
        # loops of virtual indices only contribute by a factor of two:
        if len(prepend_indices) == 0:
            resulting_coeff *= 2
        # otherwise, add the free indices in `prepend_indices` to
        # the `resulting_indices`:
        else:
            expr1 = prepend_indices
            expr2 = list(reversed(prepend_indices))
            resulting_indices = [expri + ri for ri in resulting_indices for expri in (expr1, expr2)]

    # sign correction, as described in Kahane's paper:
    resulting_coeff *= -1 if (number_of_contractions - connected_components + 1) % 2 else 1
    # power of two factor, as described in Kahane's paper:
    resulting_coeff *= 2**(number_of_contractions)

    # If `first_dum_pos` is not zero, it means that there are trailing free gamma
    # matrices in front of `expression`, so multiply by them:
    resulting_indices = [ free_pos[0:first_dum_pos] + ri for ri in resulting_indices ]

    resulting_expr = S.Zero
    for i in resulting_indices:
        temp_expr = S.One
        for j in i:
            temp_expr *= GammaMatrix(j)
        resulting_expr += temp_expr

    t = resulting_coeff * resulting_expr
    t1 = None
    if isinstance(t, TensAdd):
        t1 = t.args[0]
    elif isinstance(t, TensMul):
        t1 = t
    if t1:
        pass
    else:
        t = eye(4)*t
    return t
<|endoftext|>from collections import defaultdict

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
    cosh, sinh, tanh, coth, sech, csch, HyperbolicFunction)
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug

from sympy import SYMPY_DEBUG


# ================== Fu-like tools ===========================


def TR0(rv):
    """Simplification of rational polynomials, trying to simplify
    the expression, e.g. combine things like 3*x + 2*x, etc....
    """
    # although it would be nice to use cancel, it doesn't work
    # with noncommutatives
    return rv.normal().factor().expand()


def TR1(rv):
    """Replace sec, csc with 1/cos, 1/sin

    Examples
    ========

    >>> from sympy.simplify.fu import TR1, sec, csc
    >>> from sympy.abc import x
    >>> TR1(2*csc(x) + sec(x))
    1/cos(x) + 2/sin(x)
    """

    def f(rv):
        if isinstance(rv, sec):
            a = rv.args[0]
            return S.One/cos(a)
        elif isinstance(rv, csc):
            a = rv.args[0]
            return S.One/sin(a)
        return rv

    return bottom_up(rv, f)


def TR2(rv):
    """Replace tan and cot with sin/cos and cos/sin

    Examples
    ========

    >>> from sympy.simplify.fu import TR2
    >>> from sympy.abc import x
    >>> from sympy import tan, cot, sin, cos
    >>> TR2(tan(x))
    sin(x)/cos(x)
    >>> TR2(cot(x))
    cos(x)/sin(x)
    >>> TR2(tan(tan(x) - sin(x)/cos(x)))
    0

    """

    def f(rv):
        if isinstance(rv, tan):
            a = rv.args[0]
            return sin(a)/cos(a)
        elif isinstance(rv, cot):
            a = rv.args[0]
            return cos(a)/sin(a)
        return rv

    return bottom_up(rv, f)


def TR2i(rv, half=False):
    """Converts ratios involving sin and cos as follows::
        sin(x)/cos(x) -> tan(x)
        sin(x)/(cos(x) + 1) -> tan(x/2) if half=True

    Examples
    ========

    >>> from sympy.simplify.fu import TR2i
    >>> from sympy.abc import x, a
    >>> from sympy import sin, cos
    >>> TR2i(sin(x)/cos(x))
    tan(x)

    Powers of the numerator and denominator are also recognized

    >>> TR2i(sin(x)**2/(cos(x) + 1)**2, half=True)
    tan(x/2)**2

    The transformation does not take place unless assumptions allow
    (i.e. the base must be positive or the exponent must be an integer
    for both numerator and denominator)

    >>> TR2i(sin(x)**a/(cos(x) + 1)**a)
    sin(x)**a/(cos(x) + 1)**a

    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        n, d = rv.as_numer_denom()
        if n.is_Atom or d.is_Atom:
            return rv

        def ok(k, e):
            # initial filtering of factors
            return (
                (e.is_integer or k.is_positive) and (
                k.func in (sin, cos) or (half and
                k.is_Add and
                len(k.args) >= 2 and
                any(any(isinstance(ai, cos) or ai.is_Pow and ai.base is cos
                for ai in Mul.make_args(a)) for a in k.args))))

        n = n.as_powers_dict()
        ndone = [(k, n.pop(k)) for k in list(n.keys()) if not ok(k, n[k])]
        if not n:
            return rv

        d = d.as_powers_dict()
        ddone = [(k, d.pop(k)) for k in list(d.keys()) if not ok(k, d[k])]
        if not d:
            return rv

        # factoring if necessary

        def factorize(d, ddone):
            newk = []
            for k in d:
                if k.is_Add and len(k.args) > 1:
                    knew = factor(k) if half else factor_terms(k)
                    if knew != k:
                        newk.append((k, knew))
            if newk:
                for i, (k, knew) in enumerate(newk):
                    del d[k]
                    newk[i] = knew
                newk = Mul(*newk).as_powers_dict()
                for k in newk:
                    v = d[k] + newk[k]
                    if ok(k, v):
                        d[k] = v
                    else:
                        ddone.append((k, v))
                del newk
        factorize(n, ndone)
        factorize(d, ddone)

        # joining
        t = []
        for k in n:
            if isinstance(k, sin):
                a = cos(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0])**n[k])
                    n[k] = d[a] = None
                elif half:
                    a1 = 1 + a
                    if a1 in d and d[a1] == n[k]:
                        t.append((tan(k.args[0]/2))**n[k])
                        n[k] = d[a1] = None
            elif isinstance(k, cos):
                a = sin(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0])**-n[k])
                    n[k] = d[a] = None
            elif half and k.is_Add and k.args[0] is S.One and \
                    isinstance(k.args[1], cos):
                a = sin(k.args[1].args[0], evaluate=False)
                if a in d and d[a] == n[k] and (d[a].is_integer or \
                        a.is_positive):
                    t.append(tan(a.args[0]/2)**-n[k])
                    n[k] = d[a] = None

        if t:
            rv = Mul(*(t + [b**e for b, e in n.items() if e]))/\
                Mul(*[b**e for b, e in d.items() if e])
            rv *= Mul(*[b**e for b, e in ndone])/Mul(*[b**e for b, e in ddone])

        return rv

    return bottom_up(rv, f)


def TR3(rv):
    """Induced formula: example sin(-a) = -sin(a)

    Examples
    ========

    >>> from sympy.simplify.fu import TR3
    >>> from sympy.abc import x, y
    >>> from sympy import pi
    >>> from sympy import cos
    >>> TR3(cos(y - x*(y - x)))
    cos(x*(x - y) + y)
    >>> cos(pi/2 + x)
    -sin(x)
    >>> cos(30*pi/2 + x)
    -cos(x)

    """
    from sympy.simplify.simplify import signsimp

    # Negative argument (already automatic for funcs like sin(-x) -> -sin(x)
    # but more complicated expressions can use it, too). Also, trig angles
    # between pi/4 and pi/2 are not reduced to an angle between 0 and pi/4.
    # The following are automatically handled:
    #   Argument of type: pi/2 +/- angle
    #   Argument of type: pi +/- angle
    #   Argument of type : 2k*pi +/- angle

    def f(rv):
        if not isinstance(rv, TrigonometricFunction):
            return rv
        rv = rv.func(signsimp(rv.args[0]))
        if not isinstance(rv, TrigonometricFunction):
            return rv
        if (rv.args[0] - S.Pi/4).is_positive is (S.Pi/2 - rv.args[0]).is_positive is True:
            fmap = {cos: sin, sin: cos, tan: cot, cot: tan, sec: csc, csc: sec}
            rv = fmap[type(rv)](S.Pi/2 - rv.args[0])
        return rv

    # touch numbers iside of trig functions to let them automatically update
    rv = rv.replace(
        lambda x: isinstance(x, TrigonometricFunction),
        lambda x: x.replace(
            lambda n: n.is_number and n.is_Mul,
            lambda n: n.func(*n.args)))

    return bottom_up(rv, f)


def TR4(rv):
    """Identify values of special angles.

        a=  0   pi/6        pi/4        pi/3        pi/2
    ----------------------------------------------------
    sin(a)  0   1/2         sqrt(2)/2   sqrt(3)/2   1
    cos(a)  1   sqrt(3)/2   sqrt(2)/2   1/2         0
    tan(a)  0   sqt(3)/3    1           sqrt(3)     --

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy import cos, sin, tan, cot
    >>> for s in (0, pi/6, pi/4, pi/3, pi/2):
    ...    print('%s %s %s %s' % (cos(s), sin(s), tan(s), cot(s)))
    ...
    1 0 0 zoo
    sqrt(3)/2 1/2 sqrt(3)/3 sqrt(3)
    sqrt(2)/2 sqrt(2)/2 1 1
    1/2 sqrt(3)/2 sqrt(3) sqrt(3)/3
    0 1 zoo 0
    """
    # special values at 0, pi/6, pi/4, pi/3, pi/2 already handled
    return rv.replace(
        lambda x:
            isinstance(x, TrigonometricFunction) and
            (r:=x.args[0]/pi).is_Rational and r.q in (1, 2, 3, 4, 6),
        lambda x:
            x.func(x.args[0].func(*x.args[0].args)))


def _TR56(rv, f, g, h, max, pow):
    """Helper for TR5 and TR6 to replace f**2 with h(g**2)

    Options
    =======

    max :   controls size of exponent that can appear on f
            e.g. if max=4 then f**4 will be changed to h(g**2)**2.
    pow :   controls whether the exponent must be a perfect power of 2
            e.g. if pow=True (and max >= 6) then f**6 will not be changed
            but f**8 will be changed to h(g**2)**4

    >>> from sympy.simplify.fu import _TR56 as T
    >>> from sympy.abc import x
    >>> from sympy import sin, cos
    >>> h = lambda x: 1 - x
    >>> T(sin(x)**3, sin, cos, h, 4, False)
    (1 - cos(x)**2)*sin(x)
    >>> T(sin(x)**6, sin, cos, h, 6, False)
    (1 - cos(x)**2)**3
    >>> T(sin(x)**6, sin, cos, h, 6, True)
    sin(x)**6
    >>> T(sin(x)**8, sin, cos, h, 10, True)
    (1 - cos(x)**2)**4
    """

    def _f(rv):
        # I'm not sure if this transformation should target all even powers
        # or only those expressible as powers of 2. Also, should it only
        # make the changes in powers that appear in sums -- making an isolated
        # change is not going to allow a simplification as far as I can tell.
        if not (rv.is_Pow and rv.base.func == f):
            return rv
        if not rv.exp.is_real:
            return rv

        if (rv.exp < 0) == True:
            return rv
        if (rv.exp > max) == True:
            return rv
        if rv.exp == 1:
            return rv
        if rv.exp == 2:
            return h(g(rv.base.args[0])**2)
        else:
            if rv.exp % 2 == 1:
                e = rv.exp//2
                return f(rv.base.args[0])*h(g(rv.base.args[0])**2)**e
            elif rv.exp == 4:
                e = 2
            elif not pow:
                if rv.exp % 2:
                    return rv
                e = rv.exp//2
            else:
                p = perfect_power(rv.exp)
                if not p:
                    return rv
                e = rv.exp//2
            return h(g(rv.base.args[0])**2)**e

    return bottom_up(rv, _f)


def TR5(rv, max=4, pow=False):
    """Replacement of sin**2 with 1 - cos(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR5
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR5(sin(x)**2)
    1 - cos(x)**2
    >>> TR5(sin(x)**-2)  # unchanged
    sin(x)**(-2)
    >>> TR5(sin(x)**4)
    (1 - cos(x)**2)**2
    """
    return _TR56(rv, sin, cos, lambda x: 1 - x, max=max, pow=pow)


def TR6(rv, max=4, pow=False):
    """Replacement of cos**2 with 1 - sin(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR6
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR6(cos(x)**2)
    1 - sin(x)**2
    >>> TR6(cos(x)**-2)  #unchanged
    cos(x)**(-2)
    >>> TR6(cos(x)**4)
    (1 - sin(x)**2)**2
    """
    return _TR56(rv, cos, sin, lambda x: 1 - x, max=max, pow=pow)


def TR7(rv):
    """Lowering the degree of cos(x)**2.

    Examples
    ========

    >>> from sympy.simplify.fu import TR7
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR7(cos(x)**2)
    cos(2*x)/2 + 1/2
    >>> TR7(cos(x)**2 + 1)
    cos(2*x)/2 + 3/2

    """

    def f(rv):
        if not (rv.is_Pow and rv.base.func == cos and rv.exp == 2):
            return rv
        return (1 + cos(2*rv.base.args[0]))/2

    return bottom_up(rv, f)


def TR8(rv, first=True):
    """Converting products of ``cos`` and/or ``sin`` to a sum or
    difference of ``cos`` and or ``sin`` terms.

    Examples
    ========

    >>> from sympy.simplify.fu import TR8
    >>> from sympy import cos, sin
    >>> TR8(cos(2)*cos(3))
    cos(5)/2 + cos(1)/2
    >>> TR8(cos(2)*sin(3))
    sin(5)/2 + sin(1)/2
    >>> TR8(sin(2)*sin(3))
    -cos(5)/2 + cos(1)/2
    """

    def f(rv):
        if not (
            rv.is_Mul or
            rv.is_Pow and
            rv.base.func in (cos, sin) and
            (rv.exp.is_integer or rv.base.is_positive)):
            return rv

        if first:
            n, d = [expand_mul(i) for i in rv.as_numer_denom()]
            newn = TR8(n, first=False)
            newd = TR8(d, first=False)
            if newn != n or newd != d:
                rv = gcd_terms(newn/newd)
                if rv.is_Mul and rv.args[0].is_Rational and \
                        len(rv.args) == 2 and rv.args[1].is_Add:
                    rv = Mul(*rv.as_coeff_Mul())
            return rv

        args = {cos: [], sin: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (cos, sin):
                args[type(a)].append(a.args[0])
            elif (a.is_Pow and a.exp.is_Integer and a.exp > 0 and \
                    a.base.func in (cos, sin)):
                # XXX this is ok but pathological expression could be handled
                # more efficiently as in TRmorrie
                args[type(a.base)].extend([a.base.args[0]]*a.exp)
            else:
                args[None].append(a)
        c = args[cos]
        s = args[sin]
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv

        args = args[None]
        n = min(len(c), len(s))
        for i in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sin(a1 + a2) + sin(a1 - a2))/2)
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cos(a1 + a2) + cos(a1 - a2))/2)
        if c:
            args.append(cos(c.pop()))
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((-cos(a1 + a2) + cos(a1 - a2))/2)
        if s:
            args.append(sin(s.pop()))
        return TR8(expand_mul(Mul(*args)))

    return bottom_up(rv, f)


def TR9(rv):
    """Sum of ``cos`` or ``sin`` terms as a product of ``cos`` or ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR9
    >>> from sympy import cos, sin
    >>> TR9(cos(1) + cos(2))
    2*cos(1/2)*cos(3/2)
    >>> TR9(cos(1) + 2*sin(1) + 2*sin(2))
    cos(1) + 4*sin(3/2)*cos(1/2)

    If no change is made by TR9, no re-arrangement of the
    expression will be made. For example, though factoring
    of common term is attempted, if the factored expression
    was not changed, the original expression will be returned:

    >>> TR9(cos(3) + cos(3)*cos(2))
    cos(3) + cos(2)*cos(3)

    """

    def f(rv):
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            # cos(a)+/-cos(b) can be combined into a product of cosines and
            # sin(a)+/-sin(b) can be combined into a product of cosine and
            # sine.
            #
            # If there are more than two args, the pairs which "work" will
            # have a gcd extractable and the remaining two terms will have
            # the above structure -- all pairs must be checked to find the
            # ones that work. args that don't have a common set of symbols
            # are skipped since this doesn't lead to a simpler formula and
            # also has the arbitrariness of combining, for example, the x
            # and y term instead of the y and z term in something like
            # cos(x) + cos(y) + cos(z).

            if not rv.is_Add:
                return rv

            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new  # update in place
                            args[j] = None
                            hit = True
                            break  # go to next i
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)

                return rv

            # two-arg Add
            split = trig_split(*args)
            if not split:
                return rv
            gcd, n1, n2, a, b, iscos = split

            # application of rule if possible
            if iscos:
                if n1 == n2:
                    return gcd*n1*2*cos((a + b)/2)*cos((a - b)/2)
                if n1 < 0:
                    a, b = b, a
                return -2*gcd*sin((a + b)/2)*sin((a - b)/2)
            else:
                if n1 == n2:
                    return gcd*n1*2*sin((a + b)/2)*cos((a - b)/2)
                if n1 < 0:
                    a, b = b, a
                return 2*gcd*cos((a + b)/2)*sin((a - b)/2)

        return process_common_addends(rv, do)  # DON'T sift by free symbols

    return bottom_up(rv, f)


def TR10(rv, first=True):
    """Separate sums in ``cos`` and ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10
    >>> from sympy.abc import a, b, c
    >>> from sympy import cos, sin
    >>> TR10(cos(a + b))
    -sin(a)*sin(b) + cos(a)*cos(b)
    >>> TR10(sin(a + b))
    sin(a)*cos(b) + sin(b)*cos(a)
    >>> TR10(sin(a + b + c))
    (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
    (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    """

    def f(rv):
        if rv.func not in (cos, sin):
            return rv

        f = rv.func
        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                if f == sin:
                    return sin(a)*TR10(cos(b), first=False) + \
                        cos(a)*TR10(sin(b), first=False)
                else:
                    return cos(a)*TR10(cos(b), first=False) - \
                        sin(a)*TR10(sin(b), first=False)
            else:
                if f == sin:
                    return sin(a)*cos(b) + cos(a)*sin(b)
                else:
                    return cos(a)*cos(b) - sin(a)*sin(b)
        return rv

    return bottom_up(rv, f)


def TR10i(rv):
    """Sum of products to function of sum.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10i
    >>> from sympy import cos, sin, sqrt
    >>> from sympy.abc import x

    >>> TR10i(cos(1)*cos(3) + sin(1)*sin(3))
    cos(2)
    >>> TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3))
    cos(3) + sin(4)
    >>> TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x)
    2*sqrt(2)*x*sin(x + pi/6)

    """
    def f(rv):
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            # args which can be expressed as A*(cos(a)*cos(b)+/-sin(a)*sin(b))
            # or B*(cos(a)*sin(b)+/-cos(b)*sin(a)) can be combined into
            # A*f(a+/-b) where f is either sin or cos.
            #
            # If there are more than two args, the pairs which "work" will have
            # a gcd extractable and the remaining two terms will have the above
            # structure -- all pairs must be checked to find the ones that
            # work.

            if not rv.is_Add:
                return rv

            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new  # update in place
                            args[j] = None
                            hit = True
                            break  # go to next i
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)

                return rv

            # two-arg Add
            split = trig_split(*args, two=True)
            if not split:
                return rv
            gcd, n1, n2, a, b, same = split

            # identify and get c1 to be cos then apply rule if possible
            if same:  # coscos, sinsin
                gcd = n1*gcd
                if n1 == n2:
                    return gcd*cos(a - b)
                return gcd*cos(a + b)
            else:  #cossin, cossin
                gcd = n1*gcd
                if n1 == n2:
                    return gcd*sin(a + b)
                return gcd*sin(b - a)

        rv = process_common_addends(
            rv, do, lambda x: tuple(ordered(x.free_symbols)))

        # need to check for inducible pairs in ratio of sqrt(3):1 that
        # appeared in different lists when sorting by coefficient
        while rv.is_Add:
            byrad = defaultdict(list)
            for a in rv.args:
                hit = 0
                if a.is_Mul:
                    for ai in a.args:
                        if ai.is_Pow and ai.exp is S.Half and \
                                ai.base.is_Integer:
                            byrad[ai].append(a)
                            hit = 1
                            break
                if not hit:
                    byrad[S.One].append(a)

            # no need to check all pairs -- just check for the onees
            # that have the right ratio
            args = []
            for a in byrad:
                for b in [_ROOT3()*a, _invROOT3()]:
                    if b in byrad:
                        for i in range(len(byrad[a])):
                            if byrad[a][i] is None:
                                continue
                            for j in range(len(byrad[b])):
                                if byrad[b][j] is None:
                                    continue
                                was = Add(byrad[a][i] + byrad[b][j])
                                new = do(was)
                                if new != was:
                                    args.append(new)
                                    byrad[a][i] = None
                                    byrad[b][j] = None
                                    break
            if args:
                rv = Add(*(args + [Add(*[_f for _f in v if _f])
                    for v in byrad.values()]))
            else:
                rv = do(rv)  # final pass to resolve any new inducible pairs
                break

        return rv

    return bottom_up(rv, f)


def TR11(rv, base=None):
    """Function of double angle to product. The ``base`` argument can be used
    to indicate what is the un-doubled argument, e.g. if 3*pi/7 is the base
    then cosine and sine functions with argument 6*pi/7 will be replaced.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11
    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import x
    >>> TR11(sin(2*x))
    2*sin(x)*cos(x)
    >>> TR11(cos(2*x))
    -sin(x)**2 + cos(x)**2
    >>> TR11(sin(4*x))
    4*(-sin(x)**2 + cos(x)**2)*sin(x)*cos(x)
    >>> TR11(sin(4*x/3))
    4*(-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3)

    If the arguments are simply integers, no change is made
    unless a base is provided:

    >>> TR11(cos(2))
    cos(2)
    >>> TR11(cos(4), 2)
    -sin(2)**2 + cos(2)**2

    There is a subtle issue here in that autosimplification will convert
    some higher angles to lower angles

    >>> cos(6*pi/7) + cos(3*pi/7)
    -cos(pi/7) + cos(3*pi/7)

    The 6*pi/7 angle is now pi/7 but can be targeted with TR11 by supplying
    the 3*pi/7 base:

    >>> TR11(_, 3*pi/7)
    -sin(3*pi/7)**2 + cos(3*pi/7)**2 + cos(3*pi/7)

    """

    def f(rv):
        if rv.func not in (cos, sin):
            return rv

        if base:
            f = rv.func
            t = f(base*2)
            co = S.One
            if t.is_Mul:
                co, t = t.as_coeff_Mul()
            if t.func not in (cos, sin):
                return rv
            if rv.args[0] == t.args[0]:
                c = cos(base)
                s = sin(base)
                if f is cos:
                    return (c**2 - s**2)/co
                else:
                    return 2*c*s/co
            return rv

        elif not rv.args[0].is_Number:
            # make a change if the leading coefficient's numerator is
            # divisible by 2
            c, m = rv.args[0].as_coeff_Mul(rational=True)
            if c.p % 2 == 0:
                arg = c.p//2*m/c.q
                c = TR11(cos(arg))
                s = TR11(sin(arg))
                if rv.func == sin:
                    rv = 2*s*c
                else:
                    rv = c**2 - s**2
        return rv

    return bottom_up(rv, f)


def _TR11(rv):
    """
    Helper for TR11 to find half-arguments for sin in factors of
    num/den that appear in cos or sin factors in the den/num.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11, _TR11
    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> TR11(sin(x/3)/(cos(x/6)))
    sin(x/3)/cos(x/6)
    >>> _TR11(sin(x/3)/(cos(x/6)))
    2*sin(x/6)
    >>> TR11(sin(x/6)/(sin(x/3)))
    sin(x/6)/sin(x/3)
    >>> _TR11(sin(x/6)/(sin(x/3)))
    1/(2*cos(x/6))

    """
    def f(rv):
        if not isinstance(rv, Expr):
            return rv

        def sincos_args(flat):
            # find arguments of sin and cos that
            # appears as bases in args of flat
            # and have Integer exponents
            args = defaultdict(set)
            for fi in Mul.make_args(flat):
                b, e = fi.as_base_exp()
                if e.is_Integer and e > 0:
                    if b.func in (cos, sin):
                        args[type(b)].add(b.args[0])
            return args
        num_args, den_args = map(sincos_args, rv.as_numer_denom())
        def handle_match(rv, num_args, den_args):
            # for arg in sin args of num_args, look for arg/2
            # in den_args and pass this half-angle to TR11
            # for handling in rv
            for narg in num_args[sin]:
                half = narg/2
                if half in den_args[cos]:
                    func = cos
                elif half in den_args[sin]:
                    func = sin
                else:
                    continue
                rv = TR11(rv, half)
                den_args[func].remove(half)
            return rv
        # sin in num, sin or cos in den
        rv = handle_match(rv, num_args, den_args)
        # sin in den, sin or cos in num
        rv = handle_match(rv, den_args, num_args)
        return rv

    return bottom_up(rv, f)


def TR12(rv, first=True):
    """Separate sums in ``tan``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import tan
    >>> from sympy.simplify.fu import TR12
    >>> TR12(tan(x + y))
    (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)
    """

    def f(rv):
        if not rv.func == tan:
            return rv

        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                tb = TR12(tan(b), first=False)
            else:
                tb = tan(b)
            return (tan(a) + tb)/(1 - tan(a)*tb)
        return rv

    return bottom_up(rv, f)


def TR12i(rv):
    """Combine tan arguments as
    (tan(y) + tan(x))/(tan(x)*tan(y) - 1) -> -tan(x + y).

    Examples
    ========

    >>> from sympy.simplify.fu import TR12i
    >>> from sympy import tan
    >>> from sympy.abc import a, b, c
    >>> ta, tb, tc = [tan(i) for i in (a, b, c)]
    >>> TR12i((ta + tb)/(-ta*tb + 1))
    tan(a + b)
    >>> TR12i((ta + tb)/(ta*tb - 1))
    -tan(a + b)
    >>> TR12i((-ta - tb)/(ta*tb - 1))
    tan(a + b)
    >>> eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    >>> TR12i(eq.expand())
    -3*tan(a + b)*tan(a + c)/(2*(tan(a) + tan(b) - 1))
    """
    def f(rv):
        if not (rv.is_Add or rv.is_Mul or rv.is_Pow):
            return rv

        n, d = rv.as_numer_denom()
        if not d.args or not n.args:
            return rv

        dok = {}

        def ok(di):
            m = as_f_sign_1(di)
            if m:
                g, f, s = m
                if s is S.NegativeOne and f.is_Mul and len(f.args) == 2 and \
                        all(isinstance(fi, tan) for fi in f.args):
                    return g, f

        d_args = list(Mul.make_args(d))
        for i, di in enumerate(d_args):
            m = ok(di)
            if m:
                g, t = m
                s = Add(*[_.args[0] for _ in t.args])
                dok[s] = S.One
                d_args[i] = g
                continue
            if di.is_Add:
                di = factor(di)
                if di.is_Mul:
                    d_args.extend(di.args)
                    d_args[i] = S.One
            elif di.is_Pow and (di.exp.is_integer or di.base.is_positive):
                m = ok(di.base)
                if m:
                    g, t = m
                    s = Add(*[_.args[0] for _ in t.args])
                    dok[s] = di.exp
                    d_args[i] = g**di.exp
                else:
                    di = factor(di)
                    if di.is_Mul:
                        d_args.extend(di.args)
                        d_args[i] = S.One
        if not dok:
            return rv

        def ok(ni):
            if ni.is_Add and len(ni.args) == 2:
                a, b = ni.args
                if isinstance(a, tan) and isinstance(b, tan):
                    return a, b
        n_args = list(Mul.make_args(factor_terms(n)))
        hit = False
        for i, ni in enumerate(n_args):
            m = ok(ni)
            if not m:
                m = ok(-ni)
                if m:
                    n_args[i] = S.NegativeOne
                else:
                    if ni.is_Add:
                        ni = factor(ni)
                        if ni.is_Mul:
                            n_args.extend(ni.args)
                            n_args[i] = S.One
                        continue
                    elif ni.is_Pow and (
                            ni.exp.is_integer or ni.base.is_positive):
                        m = ok(ni.base)
                        if m:
                            n_args[i] = S.One
                        else:
                            ni = factor(ni)
                            if ni.is_Mul:
                                n_args.extend(ni.args)
                                n_args[i] = S.One
                            continue
                    else:
                        continue
            else:
                n_args[i] = S.One
            hit = True
            s = Add(*[_.args[0] for _ in m])
            ed = dok[s]
            newed = ed.extract_additively(S.One)
            if newed is not None:
                if newed:
                    dok[s] = newed
                else:
                    dok.pop(s)
            n_args[i] *= -tan(s)

        if hit:
            rv = Mul(*n_args)/Mul(*d_args)/Mul(*[(Add(*[
                tan(a) for a in i.args]) - 1)**e for i, e in dok.items()])

        return rv

    return bottom_up(rv, f)


def TR13(rv):
    """Change products of ``tan`` or ``cot``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR13
    >>> from sympy import tan, cot
    >>> TR13(tan(3)*tan(2))
    -tan(2)/tan(5) - tan(3)/tan(5) + 1
    >>> TR13(cot(3)*cot(2))
    cot(2)*cot(5) + 1 + cot(3)*cot(5)
    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        # XXX handle products of powers? or let power-reducing handle it?
        args = {tan: [], cot: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (tan, cot):
                args[type(a)].append(a.args[0])
            else:
                args[None].append(a)
        t = args[tan]
        c = args[cot]
        if len(t) < 2 and len(c) < 2:
            return rv
        args = args[None]
        while len(t) > 1:
            t1 = t.pop()
            t2 = t.pop()
            args.append(1 - (tan(t1)/tan(t1 + t2) + tan(t2)/tan(t1 + t2)))
        if t:
            args.append(tan(t.pop()))
        while len(c) > 1:
            t1 = c.pop()
            t2 = c.pop()
            args.append(1 + cot(t1)*cot(t1 + t2) + cot(t2)*cot(t1 + t2))
        if c:
            args.append(cot(c.pop()))
        return Mul(*args)

    return bottom_up(rv, f)


def TRmorrie(rv):
    """Returns cos(x)*cos(2*x)*...*cos(2**(k-1)*x) -> sin(2**k*x)/(2**k*sin(x))

    Examples
    ========

    >>> from sympy.simplify.fu import TRmorrie, TR8, TR3
    >>> from sympy.abc import x
    >>> from sympy import Mul, cos, pi
    >>> TRmorrie(cos(x)*cos(2*x))
    sin(4*x)/(4*sin(x))
    >>> TRmorrie(7*Mul(*[cos(x) for x in range(10)]))
    7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))

    Sometimes autosimplification will cause a power to be
    not recognized. e.g. in the following, cos(4*pi/7) automatically
    simplifies to -cos(3*pi/7) so only 2 of the 3 terms are
    recognized:

    >>> TRmorrie(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7))
    -sin(3*pi/7)*cos(3*pi/7)/(4*sin(pi/7))

    A touch by TR8 resolves the expression to a Rational

    >>> TR8(_)
    -1/8

    In this case, if eq is unsimplified, the answer is obtained
    directly:

    >>> eq = cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9)
    >>> TRmorrie(eq)
    1/16

    But if angles are made canonical with TR3 then the answer
    is not simplified without further work:

    >>> TR3(eq)
    sin(pi/18)*cos(pi/9)*cos(2*pi/9)/2
    >>> TRmorrie(_)
    sin(pi/18)*sin(4*pi/9)/(8*sin(pi/9))
    >>> TR8(_)
    cos(7*pi/18)/(16*sin(pi/9))
    >>> TR3(_)
    1/16

    The original expression would have resolve to 1/16 directly with TR8,
    however:

    >>> TR8(eq)
    1/16

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morrie%27s_law

    """

    def f(rv, first=True):
        if not rv.is_Mul:
            return rv
        if first:
            n, d = rv.as_numer_denom()
            return f(n, 0)/f(d, 0)

        args = defaultdict(list)
        coss = {}
        other = []
        for c in rv.args:
            b, e = c.as_base_exp()
            if e.is_Integer and isinstance(b, cos):
                co, a = b.args[0].as_coeff_Mul()
                args[a].append(co)
                coss[b] = e
            else:
                other.append(c)

        new = []
        for a in args:
            c = args[a]
            c.sort()
            while c:
                k = 0
                cc = ci = c[0]
                while cc in c:
                    k += 1
                    cc *= 2
                if k > 1:
                    newarg = sin(2**k*ci*a)/2**k/sin(ci*a)
                    # see how many times this can be taken
                    take = None
                    ccs = []
                    for i in range(k):
                        cc /= 2
                        key = cos(a*cc, evaluate=False)
                        ccs.append(cc)
                        take = min(coss[key], take or coss[key])
                    # update exponent counts
                    for i in range(k):
                        cc = ccs.pop()
                        key = cos(a*cc, evaluate=False)
                        coss[key] -= take
                        if not coss[key]:
                            c.remove(cc)
                    new.append(newarg**take)
                else:
                    b = cos(c.pop(0)*a)
                    other.append(b**coss[b])

        if new:
            rv = Mul(*(new + other + [
                cos(k*a, evaluate=False) for a in args for k in args[a]]))

        return rv

    return bottom_up(rv, f)


def TR14(rv, first=True):
    """Convert factored powers of sin and cos identities into simpler
    expressions.

    Examples
    ========

    >>> from sympy.simplify.fu import TR14
    >>> from sympy.abc import x, y
    >>> from sympy import cos, sin
    >>> TR14((cos(x) - 1)*(cos(x) + 1))
    -sin(x)**2
    >>> TR14((sin(x) - 1)*(sin(x) + 1))
    -cos(x)**2
    >>> p1 = (cos(x) + 1)*(cos(x) - 1)
    >>> p2 = (cos(y) - 1)*2*(cos(y) + 1)
    >>> p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    >>> TR14(p1*p2*p3*(x - 1))
    -18*(x - 1)*sin(x)**2*sin(y)**4

    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        if first:
            # sort them by location in numerator and denominator
            # so the code below can just deal with positive exponents
            n, d = rv.as_numer_denom()
            if d is not S.One:
                newn = TR14(n, first=False)
                newd = TR14(d, first=False)
                if newn != n or newd != d:
                    rv = newn/newd
                return rv

        other = []
        process = []
        for a in rv.args:
            if a.is_Pow:
                b, e = a.as_base_exp()
                if not (e.is_integer or b.is_positive):
                    other.append(a)
                    continue
                a = b
            else:
                e = S.One
            m = as_f_sign_1(a)
            if not m or m[1].func not in (cos, sin):
                if e is S.One:
                    other.append(a)
                else:
                    other.append(a**e)
                continue
            g, f, si = m
            process.append((g, e.is_Number, e, f, si, a))

        # sort them to get like terms next to each other
        process = list(ordered(process))

        # keep track of whether there was any change
        nother = len(other)

        # access keys
        keys = (g, t, e, f, si, a) = list(range(6))

        while process:
            A = process.pop(0)
            if process:
                B = process[0]

                if A[e].is_Number and B[e].is_Number:
                    # both exponents are numbers
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = min(A[e], B[e])

                            # reinsert any remainder
                            # the B will likely sort after A so check it first
                            if B[e] != take:
                                rem = [B[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            elif A[e] != take:
                                rem = [A[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)

                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g]*B[g]*t(A[f].args[0])**2)**take)
                            continue

                elif A[e] == B[e]:
                    # both exponents are equal symbols
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = A[e]
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g]*B[g]*t(A[f].args[0])**2)**take)
                            continue

            # either we are done or neither condition above applied
            other.append(A[a]**A[e])

        if len(other) != nother:
            rv = Mul(*other)

        return rv

    return bottom_up(rv, f)


def TR15(rv, max=4, pow=False):
    """Convert sin(x)**-2 to 1 + cot(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR15
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR15(1 - 1/sin(x)**2)
    -cot(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, sin)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, sin, cot, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    return bottom_up(rv, f)


def TR16(rv, max=4, pow=False):
    """Convert cos(x)**-2 to 1 + tan(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR16
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR16(1 - 1/cos(x)**2)
    -tan(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, cos)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, cos, tan, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    return bottom_up(rv, f)


def TR111(rv):
    """Convert f(x)**-i to g(x)**i where either ``i`` is an integer
    or the base is positive and f, g are: tan, cot; sin, csc; or cos, sec.

    Examples
    ========

    >>> from sympy.simplify.fu import TR111
    >>> from sympy.abc import x
    >>> from sympy import tan
    >>> TR111(1 - 1/tan(x)**2)
    1 - cot(x)**2

    """

    def f(rv):
        if not (
            isinstance(rv, Pow) and
            (rv.base.is_positive or rv.exp.is_integer and rv.exp.is_negative)):
            return rv

        if isinstance(rv.base, tan):
            return cot(rv.base.args[0])**-rv.exp
        elif isinstance(rv.base, sin):
            return csc(rv.base.args[0])**-rv.exp
        elif isinstance(rv.base, cos):
            return sec(rv.base.args[0])**-rv.exp
        return rv

    return bottom_up(rv, f)


def TR22(rv, max=4, pow=False):
    """Convert tan(x)**2 to sec(x)**2 - 1 and cot(x)**2 to csc(x)**2 - 1.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR22
    >>> from sympy.abc import x
    >>> from sympy import tan, cot
    >>> TR22(1 + tan(x)**2)
    sec(x)**2
    >>> TR22(1 + cot(x)**2)
    csc(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and rv.base.func in (cot, tan)):
            return rv

        rv = _TR56(rv, tan, sec, lambda x: x - 1, max=max, pow=pow)
        rv = _TR56(rv, cot, csc, lambda x: x - 1, max=max, pow=pow)
        return rv

    return bottom_up(rv, f)


def TRpower(rv):
    """Convert sin(x)**n and cos(x)**n with positive n to sums.

    Examples
    ========

    >>> from sympy.simplify.fu import TRpower
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> TRpower(sin(x)**6)
    -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 + 5/16
    >>> TRpower(sin(x)**3*cos(2*x)**4)
    (3*sin(x)/4 - sin(3*x)/4)*(cos(4*x)/2 + cos(8*x)/8 + 3/8)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Power-reduction_formulae

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, (sin, cos))):
            return rv
        b, n = rv.as_base_exp()
        x = b.args[0]
        if n.is_Integer and n.is_positive:
            if n.is_odd and isinstance(b, cos):
                rv = 2**(1-n)*Add(*[binomial(n, k)*cos((n - 2*k)*x)
                    for k in range((n + 1)/2)])
            elif n.is_odd and isinstance(b, sin):
                rv = 2**(1-n)*S.NegativeOne**((n-1)/2)*Add(*[binomial(n, k)*
                    S.NegativeOne**k*sin((n - 2*k)*x) for k in range((n + 1)/2)])
            elif n.is_even and isinstance(b, cos):
                rv = 2**(1-n)*Add(*[binomial(n, k)*cos((n - 2*k)*x)
                    for k in range(n/2)])
            elif n.is_even and isinstance(b, sin):
                rv = 2**(1-n)*S.NegativeOne**(n/2)*Add(*[binomial(n, k)*
                    S.NegativeOne**k*cos((n - 2*k)*x) for k in range(n/2)])
            if n.is_even:
                rv += 2**(-n)*binomial(n, n/2)
        return rv

    return bottom_up(rv, f)


def L(rv):
    """Return count of trigonometric functions in expression.

    Examples
    ========

    >>> from sympy.simplify.fu import L
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> L(cos(x)+sin(x))
    2
    """
    return S(rv.count(TrigonometricFunction))


# ============== end of basic Fu-like tools =====================

if SYMPY_DEBUG:
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13,
    TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22
    )= list(map(debug,
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13,
    TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22)))


# tuples are chains  --  (f, g) -> lambda x: g(f(x))
# lists are choices  --  [f, g] -> lambda x: min(f(x), g(x), key=objective)

CTR1 = [(TR5, TR0), (TR6, TR0), identity]

CTR2 = (TR11, [(TR5, TR0), (TR6, TR0), TR0])

CTR3 = [(TRmorrie, TR8, TR0), (TRmorrie, TR8, TR10i, TR0), identity]

CTR4 = [(TR4, TR10i), identity]

RL1 = (TR4, TR3, TR4, TR12, TR4, TR13, TR4, TR0)


# XXX it's a little unclear how this one is to be implemented
# see Fu paper of reference, page 7. What is the Union symbol referring to?
# The diagram shows all these as one chain of transformations, but the
# text refers to them being applied independently. Also, a break
# if L starts to increase has not been implemented.
RL2 = [
    (TR4, TR3, TR10, TR4, TR3, TR11),
    (TR5, TR7, TR11, TR4),
    (CTR3, CTR1, TR9, CTR2, TR4, TR9, TR9, CTR4),
    identity,
    ]


def fu(rv, measure=lambda x: (L(x), x.count_ops())):
    """Attempt to simplify expression by using transformation rules given
    in the algorithm by Fu et al.

    :func:`fu` will try to minimize the objective function ``measure``.
    By default this first minimizes the number of trig terms and then minimizes
    the number of total operations.

    Examples
    ========

    >>> from sympy.simplify.fu import fu
    >>> from sympy import cos, sin, tan, pi, S, sqrt
    >>> from sympy.abc import x, y, a, b

    >>> fu(sin(50)**2 + cos(50)**2 + sin(pi/6))
    3/2
    >>> fu(sqrt(6)*cos(x) + sqrt(2)*sin(x))
    2*sqrt(2)*sin(x + pi/3)

    CTR1 example

    >>> eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    >>> fu(eq)
    cos(x)**4 - 2*cos(y)**2 + 2

    CTR2 example

    >>> fu(S.Half - cos(2*x)/2)
    sin(x)**2

    CTR3 example

    >>> fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b)))
    sqrt(2)*sin(a + b + pi/4)

    CTR4 example

    >>> fu(sqrt(3)*cos(x)/2 + sin(x)/2)
    sin(x + pi/3)

    Example 1

    >>> fu(1-sin(2*x)**2/4-sin(y)**2-cos(x)**4)
    -cos(x)**2 + cos(y)**2

    Example 2

    >>> fu(cos(4*pi/9))
    sin(pi/18)
    >>> fu(cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9))
    1/16

    Example 3

    >>> fu(tan(7*pi/18)+tan(5*pi/18)-sqrt(3)*tan(5*pi/18)*tan(7*pi/18))
    -sqrt(3)

    Objective function example

    >>> fu(sin(x)/cos(x))  # default objective function
    tan(x)
    >>> fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) # maximize op count
    sin(x)/cos(x)

    References
    ==========

    .. [1] https://www.sciencedirect.com/science/article/pii/S0895717706001609
    """
    fRL1 = greedy(RL1, measure)
    fRL2 = greedy(RL2, measure)

    was = rv
    rv = sympify(rv)
    if not isinstance(rv, Expr):
        return rv.func(*[fu(a, measure=measure) for a in rv.args])
    rv = TR1(rv)
    if rv.has(tan, cot):
        rv1 = fRL1(rv)
        if (measure(rv1) < measure(rv)):
            rv = rv1
        if rv.has(tan, cot):
            rv = TR2(rv)
    if rv.has(sin, cos):
        rv1 = fRL2(rv)
        rv2 = TR8(TRmorrie(rv1))
        rv = min([was, rv, rv1, rv2], key=measure)
    return min(TR2i(rv), rv, key=measure)


def process_common_addends(rv, do, key2=None, key1=True):
    """Apply ``do`` to addends of ``rv`` that (if ``key1=True``) share at least
    a common absolute value of their coefficient and the value of ``key2`` when
    applied to the argument. If ``key1`` is False ``key2`` must be supplied and
    will be the only key applied.
    """

    # collect by absolute value of coefficient and key2
    absc = defaultdict(list)
    if key1:
        for a in rv.args:
            c, a = a.as_coeff_Mul()
            if c < 0:
                c = -c
                a = -a  # put the sign on `a`
            absc[(c, key2(a) if key2 else 1)].append(a)
    elif key2:
        for a in rv.args:
            absc[(S.One, key2(a))].append(a)
    else:
        raise ValueError('must have at least one key')

    args = []
    hit = False
    for k in absc:
        v = absc[k]
        c, _ = k
        if len(v) > 1:
            e = Add(*v, evaluate=False)
            new = do(e)
            if new != e:
                e = new
                hit = True
            args.append(c*e)
        else:
            args.append(c*v[0])
    if hit:
        rv = Add(*args)

    return rv


fufuncs = '''
    TR0 TR1 TR2 TR3 TR4 TR5 TR6 TR7 TR8 TR9 TR10 TR10i TR11
    TR12 TR13 L TR2i TRmorrie TR12i
    TR14 TR15 TR16 TR111 TR22'''.split()
FU = dict(list(zip(fufuncs, list(map(locals().get, fufuncs)))))


@cacheit
def _ROOT2():
    return sqrt(2)


@cacheit
def _ROOT3():
    return sqrt(3)


@cacheit
def _invROOT3():
    return 1/sqrt(3)


def trig_split(a, b, two=False):
    """Return the gcd, s1, s2, a1, a2, bool where

    If two is False (default) then::
        a + b = gcd*(s1*f(a1) + s2*f(a2)) where f = cos if bool else sin
    else:
        if bool, a + b was +/- cos(a1)*cos(a2) +/- sin(a1)*sin(a2) and equals
            n1*gcd*cos(a - b) if n1 == n2 else
            n1*gcd*cos(a + b)
        else a + b was +/- cos(a1)*sin(a2) +/- sin(a1)*cos(a2) and equals
            n1*gcd*sin(a + b) if n1 = n2 else
            n1*gcd*sin(b - a)

    Examples
    ========

    >>> from sympy.simplify.fu import trig_split
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, sin, sqrt

    >>> trig_split(cos(x), cos(y))
    (1, 1, 1, x, y, True)
    >>> trig_split(2*cos(x), -2*cos(y))
    (2, 1, -1, x, y, True)
    >>> trig_split(cos(x)*sin(y), cos(y)*sin(y))
    (sin(y), 1, 1, x, y, True)

    >>> trig_split(cos(x), -sqrt(3)*sin(x), two=True)
    (2, 1, -1, x, pi/6, False)
    >>> trig_split(cos(x), sin(x), two=True)
    (sqrt(2), 1, 1, x, pi/4, False)
    >>> trig_split(cos(x), -sin(x), two=True)
    (sqrt(2), 1, -1, x, pi/4, False)
    >>> trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True)
    (2*sqrt(2), 1, -1, x, pi/6, False)
    >>> trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True)
    (-2*sqrt(2), 1, 1, x, pi/3, False)
    >>> trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True)
    (sqrt(6)/3, 1, 1, x, pi/6, False)
    >>> trig_split(-sqrt(6)*cos(x)*sin(y), -sqrt(2)*sin(x)*sin(y), two=True)
    (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)

    >>> trig_split(cos(x), sin(x))
    >>> trig_split(cos(x), sin(z))
    >>> trig_split(2*cos(x), -sin(x))
    >>> trig_split(cos(x), -sqrt(3)*sin(x))
    >>> trig_split(cos(x)*cos(y), sin(x)*sin(z))
    >>> trig_split(cos(x)*cos(y), sin(x)*sin(y))
    >>> trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True)
    """
    a, b = [Factors(i) for i in (a, b)]
    ua, ub = a.normal(b)
    gcd = a.gcd(b).as_expr()
    n1 = n2 = 1
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -n1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n2 = -n2
    a, b = [i.as_expr() for i in (ua, ub)]

    def pow_cos_sin(a, two):
        """Return ``a`` as a tuple (r, c, s) such that
        ``a = (r or 1)*(c or 1)*(s or 1)``.

        Three arguments are returned (radical, c-factor, s-factor) as
        long as the conditions set by ``two`` are met; otherwise None is
        returned. If ``two`` is True there will be one or two non-None
        values in the tuple: c and s or c and r or s and r or s or c with c
        being a cosine function (if possible) else a sine, and s being a sine
        function (if possible) else oosine. If ``two`` is False then there
        will only be a c or s term in the tuple.

        ``two`` also require that either two cos and/or sin be present (with
        the condition that if the functions are the same the arguments are
        different or vice versa) or that a single cosine or a single sine
        be present with an optional radical.

        If the above conditions dictated by ``two`` are not met then None
        is returned.
        """
        c = s = None
        co = S.One
        if a.is_Mul:
            co, a = a.as_coeff_Mul()
            if len(a.args) > 2 or not two:
                return None
            if a.is_Mul:
                args = list(a.args)
            else:
                args = [a]
            a = args.pop(0)
            if isinstance(a, cos):
                c = a
            elif isinstance(a, sin):
                s = a
            elif a.is_Pow and a.exp is S.Half:  # autoeval doesn't allow -1/2
                co *= a
            else:
                return None
            if args:
                b = args[0]
                if isinstance(b, cos):
                    if c:
                        s = b
                    else:
                        c = b
                elif isinstance(b, sin):
                    if s:
                        c = b
                    else:
                        s = b
                elif b.is_Pow and b.exp is S.Half:
                    co *= b
                else:
                    return None
            return co if co is not S.One else None, c, s
        elif isinstance(a, cos):
            c = a
        elif isinstance(a, sin):
            s = a
        if c is None and s is None:
            return
        co = co if co is not S.One else None
        return co, c, s

    # get the parts
    m = pow_cos_sin(a, two)
    if m is None:
        return
    coa, ca, sa = m
    m = pow_cos_sin(b, two)
    if m is None:
        return
    cob, cb, sb = m

    # check them
    if (not ca) and cb or ca and isinstance(ca, sin):
        coa, ca, sa, cob, cb, sb = cob, cb, sb, coa, ca, sa
        n1, n2 = n2, n1
    if not two:  # need cos(x) and cos(y) or sin(x) and sin(y)
        c = ca or sa
        s = cb or sb
        if not isinstance(c, s.func):
            return None
        return gcd, n1, n2, c.args[0], s.args[0], isinstance(c, cos)
    else:
        if not coa and not cob:
            if (ca and cb and sa and sb):
                if isinstance(ca, sa.func) is not isinstance(cb, sb.func):
                    return
                args = {j.args for j in (ca, sa)}
                if not all(i.args in args for i in (cb, sb)):
                    return
                return gcd, n1, n2, ca.args[0], sa.args[0], isinstance(ca, sa.func)
        if ca and sa or cb and sb or \
            two and (ca is None and sa is None or cb is None and sb is None):
            return
        c = ca or sa
        s = cb or sb
        if c.args != s.args:
            return
        if not coa:
            coa = S.One
        if not cob:
            cob = S.One
        if coa is cob:
            gcd *= _ROOT2()
            return gcd, n1, n2, c.args[0], pi/4, False
        elif coa/cob == _ROOT3():
            gcd *= 2*cob
            return gcd, n1, n2, c.args[0], pi/3, False
        elif coa/cob == _invROOT3():
            gcd *= 2*coa
            return gcd, n1, n2, c.args[0], pi/6, False


def as_f_sign_1(e):
    """If ``e`` is a sum that can be written as ``g*(a + s)`` where
    ``s`` is ``+/-1``, return ``g``, ``a``, and ``s`` where ``a`` does
    not have a leading negative coefficient.

    Examples
    ========

    >>> from sympy.simplify.fu import as_f_sign_1
    >>> from sympy.abc import x
    >>> as_f_sign_1(x + 1)
    (1, x, 1)
    >>> as_f_sign_1(x - 1)
    (1, x, -1)
    >>> as_f_sign_1(-x + 1)
    (-1, x, -1)
    >>> as_f_sign_1(-x - 1)
    (-1, x, 1)
    >>> as_f_sign_1(2*x + 2)
    (2, x, 1)
    """
    if not e.is_Add or len(e.args) != 2:
        return
    # exact match
    a, b = e.args
    if a in (S.NegativeOne, S.One):
        g = S.One
        if b.is_Mul and b.args[0].is_Number and b.args[0] < 0:
            a, b = -a, -b
            g = -g
        return g, b, a
    # gcd match
    a, b = [Factors(i) for i in e.args]
    ua, ub = a.normal(b)
    gcd = a.gcd(b).as_expr()
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -1
        n2 = 1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n1 = 1
        n2 = -1
    else:
        n1 = n2 = 1
    a, b = [i.as_expr() for i in (ua, ub)]
    if a is S.One:
        a, b = b, a
        n1, n2 = n2, n1
    if n1 == -1:
        gcd = -gcd
        n2 = -n2

    if b is S.One:
        return gcd, a, n2


def _osborne(e, d):
    """Replace all hyperbolic functions with trig functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, HyperbolicFunction):
            return rv
        a = rv.args[0]
        a = a*d if not a.is_Add else Add._from_args([i*d for i in a.args])
        if isinstance(rv, sinh):
            return I*sin(a)
        elif isinstance(rv, cosh):
            return cos(a)
        elif isinstance(rv, tanh):
            return I*tan(a)
        elif isinstance(rv, coth):
            return cot(a)/I
        elif isinstance(rv, sech):
            return sec(a)
        elif isinstance(rv, csch):
            return csc(a)/I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)

    return bottom_up(e, f)


def _osbornei(e, d):
    """Replace all trig functions with hyperbolic functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, TrigonometricFunction):
            return rv
        const, x = rv.args[0].as_independent(d, as_Add=True)
        a = x.xreplace({d: S.One}) + const*I
        if isinstance(rv, sin):
            return sinh(a)/I
        elif isinstance(rv, cos):
            return cosh(a)
        elif isinstance(rv, tan):
            return tanh(a)/I
        elif isinstance(rv, cot):
            return coth(a)*I
        elif isinstance(rv, sec):
            return sech(a)
        elif isinstance(rv, csc):
            return csch(a)*I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)

    return bottom_up(e, f)


def hyper_as_trig(rv):
    """Return an expression containing hyperbolic functions in terms
    of trigonometric functions. Any trigonometric functions initially
    present are replaced with Dummy symbols and the function to undo
    the masking and the conversion back to hyperbolics is also returned. It
    should always be true that::

        t, f = hyper_as_trig(expr)
        expr == f(t)

    Examples
    ========

    >>> from sympy.simplify.fu import hyper_as_trig, fu
    >>> from sympy.abc import x
    >>> from sympy import cosh, sinh
    >>> eq = sinh(x)**2 + cosh(x)**2
    >>> t, f = hyper_as_trig(eq)
    >>> f(fu(t))
    cosh(2*x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """
    from sympy.simplify.simplify import signsimp
    from sympy.simplify.radsimp import collect

    # mask off trig functions
    trigs = rv.atoms(TrigonometricFunction)
    reps = [(t, Dummy()) for t in trigs]
    masked = rv.xreplace(dict(reps))

    # get inversion substitutions in place
    reps = [(v, k) for k, v in reps]

    d = Dummy()

    return _osborne(masked, d), lambda x: collect(signsimp(
        _osbornei(x, d).xreplace(dict(reps))), S.ImaginaryUnit)


def sincos_to_sum(expr):
    """Convert products and powers of sin and cos to sums.

    Explanation
    ===========

    Applied power reduction TRpower first, then expands products, and
    converts products to sums with TR8.

    Examples
    ========

    >>> from sympy.simplify.fu import sincos_to_sum
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> sincos_to_sum(16*sin(x)**3*cos(2*x)**2)
    7*sin(x) - 5*sin(3*x) + 3*sin(5*x) - sin(7*x)
    """

    if not expr.has(cos, sin):
        return expr
    else:
        return TR8(expand_mul(TRpower(expr)))
<|endoftext|>from __future__ import annotations

from typing import overload

from collections import defaultdict

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,
                        expand_func, Function, Dummy, Expr, factor_terms,
                        expand_power_exp, Eq)
from sympy.core.exprtools import factor_nc
from sympy.core.parameters import global_parameters
from sympy.core.function import (expand_log, count_ops, _mexpand,
    nfloat, expand_mul, expand)
from sympy.core.numbers import Float, I, pi, Rational, equal_valued
from sympy.core.relational import Relational
from sympy.core.rules import Transform
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympify
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk
from sympy.functions import gamma, exp, sqrt, log, exp_polar, re
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify, Abs, sign
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,
                                                  piecewise_simplify)
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import (BesselBase, besselj, besseli,
                                            besselk, bessely, jn)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Boolean
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,
                                            MatPow, MatrixSymbol)
from sympy.polys import together, cancel, factor
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq
from sympy.sets.sets import Set
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction, collect_abs
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import has_variety, sift, subsets, iterable
from sympy.utilities.misc import as_int

import mpmath


def separatevars(expr, symbols=[], dict=False, force=False):
    """
    Separates variables in an expression, if possible.  By
    default, it separates with respect to all symbols in an
    expression and collects constant coefficients that are
    independent of symbols.

    Explanation
    ===========

    If ``dict=True`` then the separated terms will be returned
    in a dictionary keyed to their corresponding symbols.
    By default, all symbols in the expression will appear as
    keys; if symbols are provided, then all those symbols will
    be used as keys, and any terms in the expression containing
    other symbols or non-symbols will be returned keyed to the
    string 'coeff'. (Passing None for symbols will return the
    expression in a dictionary keyed to 'coeff'.)

    If ``force=True``, then bases of powers will be separated regardless
    of assumptions on the symbols involved.

    Notes
    =====

    The order of the factors is determined by Mul, so that the
    separated expressions may not necessarily be grouped together.

    Although factoring is necessary to separate variables in some
    expressions, it is not necessary in all cases, so one should not
    count on the returned factors being factored.

    Examples
    ========

    >>> from sympy.abc import x, y, z, alpha
    >>> from sympy import separatevars, sin
    >>> separatevars((x*y)**y)
    (x*y)**y
    >>> separatevars((x*y)**y, force=True)
    x**y*y**y

    >>> e = 2*x**2*z*sin(y)+2*z*x**2
    >>> separatevars(e)
    2*x**2*z*(sin(y) + 1)
    >>> separatevars(e, symbols=(x, y), dict=True)
    {'coeff': 2*z, x: x**2, y: sin(y) + 1}
    >>> separatevars(e, [x, y, alpha], dict=True)
    {'coeff': 2*z, alpha: 1, x: x**2, y: sin(y) + 1}

    If the expression is not really separable, or is only partially
    separable, separatevars will do the best it can to separate it
    by using factoring.

    >>> separatevars(x + x*y - 3*x**2)
    -x*(3*x - y - 1)

    If the expression is not separable then expr is returned unchanged
    or (if dict=True) then None is returned.

    >>> eq = 2*x + y*sin(x)
    >>> separatevars(eq) == eq
    True
    >>> separatevars(2*x + y*sin(x), symbols=(x, y), dict=True) is None
    True

    """
    expr = sympify(expr)
    if dict:
        return _separatevars_dict(_separatevars(expr, force), symbols)
    else:
        return _separatevars(expr, force)


def _separatevars(expr, force):
    if isinstance(expr, Abs):
        arg = expr.args[0]
        if arg.is_Mul and not arg.is_number:
            s = separatevars(arg, dict=True, force=force)
            if s is not None:
                return Mul(*map(expr.func, s.values()))
            else:
                return expr

    if len(expr.free_symbols) < 2:
        return expr

    # don't destroy a Mul since much of the work may already be done
    if expr.is_Mul:
        args = list(expr.args)
        changed = False
        for i, a in enumerate(args):
            args[i] = separatevars(a, force)
            changed = changed or args[i] != a
        if changed:
            expr = expr.func(*args)
        return expr

    # get a Pow ready for expansion
    if expr.is_Pow and expr.base != S.Exp1:
        expr = Pow(separatevars(expr.base, force=force), expr.exp)

    # First try other expansion methods
    expr = expr.expand(mul=False, multinomial=False, force=force)

    _expr, reps = posify(expr) if force else (expr, {})
    expr = factor(_expr).subs(reps)

    if not expr.is_Add:
        return expr

    # Find any common coefficients to pull out
    args = list(expr.args)
    commonc = args[0].args_cnc(cset=True, warn=False)[0]
    for i in args[1:]:
        commonc &= i.args_cnc(cset=True, warn=False)[0]
    commonc = Mul(*commonc)
    commonc = commonc.as_coeff_Mul()[1]  # ignore constants
    commonc_set = commonc.args_cnc(cset=True, warn=False)[0]

    # remove them
    for i, a in enumerate(args):
        c, nc = a.args_cnc(cset=True, warn=False)
        c = c - commonc_set
        args[i] = Mul(*c)*Mul(*nc)
    nonsepar = Add(*args)

    if len(nonsepar.free_symbols) > 1:
        _expr = nonsepar
        _expr, reps = posify(_expr) if force else (_expr, {})
        _expr = (factor(_expr)).subs(reps)

        if not _expr.is_Add:
            nonsepar = _expr

    return commonc*nonsepar


def _separatevars_dict(expr, symbols):
    if symbols:
        if not all(t.is_Atom for t in symbols):
            raise ValueError("symbols must be Atoms.")
        symbols = list(symbols)
    elif symbols is None:
        return {'coeff': expr}
    else:
        symbols = list(expr.free_symbols)
        if not symbols:
            return None

    ret = {i: [] for i in symbols + ['coeff']}

    for i in Mul.make_args(expr):
        expsym = i.free_symbols
        intersection = set(symbols).intersection(expsym)
        if len(intersection) > 1:
            return None
        if len(intersection) == 0:
            # There are no symbols, so it is part of the coefficient
            ret['coeff'].append(i)
        else:
            ret[intersection.pop()].append(i)

    # rebuild
    for k, v in ret.items():
        ret[k] = Mul(*v)

    return ret


def posify(eq):
    """Return ``eq`` (with generic symbols made positive) and a
    dictionary containing the mapping between the old and new
    symbols.

    Explanation
    ===========

    Any symbol that has positive=None will be replaced with a positive dummy
    symbol having the same name. This replacement will allow more symbolic
    processing of expressions, especially those involving powers and
    logarithms.

    A dictionary that can be sent to subs to restore ``eq`` to its original
    symbols is also returned.

    >>> from sympy import posify, Symbol, log, solve
    >>> from sympy.abc import x
    >>> posify(x + Symbol('p', positive=True) + Symbol('n', negative=True))
    (_x + n + p, {_x: x})

    >>> eq = 1/x
    >>> log(eq).expand()
    log(1/x)
    >>> log(posify(eq)[0]).expand()
    -log(_x)
    >>> p, rep = posify(eq)
    >>> log(p).expand().subs(rep)
    -log(x)

    It is possible to apply the same transformations to an iterable
    of expressions:

    >>> eq = x**2 - 4
    >>> solve(eq, x)
    [-2, 2]
    >>> eq_x, reps = posify([eq, x]); eq_x
    [_x**2 - 4, _x]
    >>> solve(*eq_x)
    [2]
    """
    eq = sympify(eq)
    if not isinstance(eq, Basic) and iterable(eq):
        f = type(eq)
        eq = list(eq)
        syms = set()
        for e in eq:
            syms = syms.union(e.atoms(Symbol))
        reps = {}
        for s in syms:
            reps.update({v: k for k, v in posify(s)[1].items()})
        for i, e in enumerate(eq):
            eq[i] = e.subs(reps)
        return f(eq), {r: s for s, r in reps.items()}

    reps = {s: Dummy(s.name, positive=True, **s.assumptions0)
                 for s in eq.free_symbols if s.is_positive is None}
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}


def hypersimp(f, k):
    """Given combinatorial term f(k) simplify its consecutive term ratio
       i.e. f(k+1)/f(k).  The input term can be composed of functions and
       integer sequences which have equivalent representation in terms
       of gamma special function.

       Explanation
       ===========

       The algorithm performs three basic steps:

       1. Rewrite all functions in terms of gamma, if possible.

       2. Rewrite all occurrences of gamma in terms of products
          of gamma and rising factorial with integer,  absolute
          constant exponent.

       3. Perform simplification of nested fractions, powers
          and if the resulting expression is a quotient of
          polynomials, reduce their total degree.

       If f(k) is hypergeometric then as result we arrive with a
       quotient of polynomials of minimal degree. Otherwise None
       is returned.

       For more information on the implemented algorithm refer to:

       1. W. Koepf, Algorithms for m-fold Hypergeometric Summation,
          Journal of Symbolic Computation (1995) 20, 399-417
    """
    f = sympify(f)

    g = f.subs(k, k + 1) / f

    g = g.rewrite(gamma)
    if g.has(Piecewise):
        g = piecewise_fold(g)
        g = g.args[-1][0]
    g = expand_func(g)
    g = powsimp(g, deep=True, combine='exp')

    if g.is_rational_function(k):
        return simplify(g, ratio=S.Infinity)
    else:
        return None


def hypersimilar(f, g, k):
    """
    Returns True if ``f`` and ``g`` are hyper-similar.

    Explanation
    ===========

    Similarity in hypergeometric sense means that a quotient of
    f(k) and g(k) is a rational function in ``k``. This procedure
    is useful in solving recurrence relations.

    For more information see hypersimp().

    """
    f, g = list(map(sympify, (f, g)))

    h = (f/g).rewrite(gamma)
    h = h.expand(func=True, basic=False)

    return h.is_rational_function(k)


def signsimp(expr, evaluate=None):
    """Make all Add sub-expressions canonical wrt sign.

    Explanation
    ===========

    If an Add subexpression, ``a``, can have a sign extracted,
    as determined by could_extract_minus_sign, it is replaced
    with Mul(-1, a, evaluate=False). This allows signs to be
    extracted from powers and products.

    Examples
    ========

    >>> from sympy import signsimp, exp, symbols
    >>> from sympy.abc import x, y
    >>> i = symbols('i', odd=True)
    >>> n = -1 + 1/x
    >>> n/x/(-n)**2 - 1/n/x
    (-1 + 1/x)/(x*(1 - 1/x)**2) - 1/(x*(-1 + 1/x))
    >>> signsimp(_)
    0
    >>> x*n + x*-n
    x*(-1 + 1/x) + x*(1 - 1/x)
    >>> signsimp(_)
    0

    Since powers automatically handle leading signs

    >>> (-2)**i
    -2**i

    signsimp can be used to put the base of a power with an integer
    exponent into canonical form:

    >>> n**i
    (-1 + 1/x)**i

    By default, signsimp does not leave behind any hollow simplification:
    if making an Add canonical wrt sign didn't change the expression, the
    original Add is restored. If this is not desired then the keyword
    ``evaluate`` can be set to False:

    >>> e = exp(y - x)
    >>> signsimp(e) == e
    True
    >>> signsimp(e, evaluate=False)
    exp(-(x - y))

    """
    if evaluate is None:
        evaluate = global_parameters.evaluate
    expr = sympify(expr)
    if not isinstance(expr, (Expr, Relational)) or expr.is_Atom:
        return expr
    # get rid of an pre-existing unevaluation regarding sign
    e = expr.replace(lambda x: x.is_Mul and -(-x) != x, lambda x: -(-x))
    e = sub_post(sub_pre(e))
    if not isinstance(e, (Expr, Relational)) or e.is_Atom:
        return e
    if e.is_Add:
        rv = e.func(*[signsimp(a) for a in e.args])
        if not evaluate and isinstance(rv, Add
                ) and rv.could_extract_minus_sign():
            return Mul(S.NegativeOne, -rv, evaluate=False)
        return rv
    if evaluate:
        e = e.replace(lambda x: x.is_Mul and -(-x) != x, lambda x: -(-x))
    return e


@overload
def simplify(expr: Expr, **kwargs) -> Expr: ...
@overload
def simplify(expr: Boolean, **kwargs) -> Boolean: ...
@overload
def simplify(expr: Set, **kwargs) -> Set: ...
@overload
def simplify(expr: Basic, **kwargs) -> Basic: ...

def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False, doit=True, **kwargs):
    """Simplifies the given expression.

    Explanation
    ===========

    Simplification is not a well defined term and the exact strategies
    this function tries can change in the future versions of SymPy. If
    your algorithm relies on "simplification" (whatever it is), try to
    determine what you need exactly  -  is it powsimp()?, radsimp()?,
    together()?, logcombine()?, or something else? And use this particular
    function directly, because those are well defined and thus your algorithm
    will be robust.

    Nonetheless, especially for interactive use, or when you do not know
    anything about the structure of the expression, simplify() tries to apply
    intelligent heuristics to make the input expression "simpler".  For
    example:

    >>> from sympy import simplify, cos, sin
    >>> from sympy.abc import x, y
    >>> a = (x + x**2)/(x*sin(y)**2 + x*cos(y)**2)
    >>> a
    (x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)
    >>> simplify(a)
    x + 1

    Note that we could have obtained the same result by using specific
    simplification functions:

    >>> from sympy import trigsimp, cancel
    >>> trigsimp(a)
    (x**2 + x)/x
    >>> cancel(_)
    x + 1

    In some cases, applying :func:`simplify` may actually result in some more
    complicated expression. The default ``ratio=1.7`` prevents more extreme
    cases: if (result length)/(input length) > ratio, then input is returned
    unmodified.  The ``measure`` parameter lets you specify the function used
    to determine how complex an expression is.  The function should take a
    single argument as an expression and return a number such that if
    expression ``a`` is more complex than expression ``b``, then
    ``measure(a) > measure(b)``.  The default measure function is
    :func:`~.count_ops`, which returns the total number of operations in the
    expression.

    For example, if ``ratio=1``, ``simplify`` output cannot be longer
    than input.

    ::

        >>> from sympy import sqrt, simplify, count_ops, oo
        >>> root = 1/(sqrt(2)+3)

    Since ``simplify(root)`` would result in a slightly longer expression,
    root is returned unchanged instead::

       >>> simplify(root, ratio=1) == root
       True

    If ``ratio=oo``, simplify will be applied anyway::

        >>> count_ops(simplify(root, ratio=oo)) > count_ops(root)
        True

    Note that the shortest expression is not necessary the simplest, so
    setting ``ratio`` to 1 may not be a good idea.
    Heuristically, the default value ``ratio=1.7`` seems like a reasonable
    choice.

    You can easily define your own measure function based on what you feel
    should represent the "size" or "complexity" of the input expression.  Note
    that some choices, such as ``lambda expr: len(str(expr))`` may appear to be
    good metrics, but have other problems (in this case, the measure function
    may slow down simplify too much for very large expressions).  If you do not
    know what a good metric would be, the default, ``count_ops``, is a good
    one.

    For example:

    >>> from sympy import symbols, log
    >>> a, b = symbols('a b', positive=True)
    >>> g = log(a) + log(b) + log(a)*log(1/b)
    >>> h = simplify(g)
    >>> h
    log(a*b**(1 - log(a)))
    >>> count_ops(g)
    8
    >>> count_ops(h)
    5

    So you can see that ``h`` is simpler than ``g`` using the count_ops metric.
    However, we may not like how ``simplify`` (in this case, using
    ``logcombine``) has created the ``b**(log(1/a) + 1)`` term.  A simple way
    to reduce this would be to give more weight to powers as operations in
    ``count_ops``.  We can do this by using the ``visual=True`` option:

    >>> print(count_ops(g, visual=True))
    2*ADD + DIV + 4*LOG + MUL
    >>> print(count_ops(h, visual=True))
    2*LOG + MUL + POW + SUB

    >>> from sympy import Symbol, S
    >>> def my_measure(expr):
    ...     POW = Symbol('POW')
    ...     # Discourage powers by giving POW a weight of 10
    ...     count = count_ops(expr, visual=True).subs(POW, 10)
    ...     # Every other operation gets a weight of 1 (the default)
    ...     count = count.replace(Symbol, type(S.One))
    ...     return count
    >>> my_measure(g)
    8
    >>> my_measure(h)
    14
    >>> 15./8 > 1.7 # 1.7 is the default ratio
    True
    >>> simplify(g, measure=my_measure)
    -log(a)*log(b) + log(a) + log(b)

    Note that because ``simplify()`` internally tries many different
    simplification strategies and then compares them using the measure
    function, we get a completely different result that is still different
    from the input expression by doing this.

    If ``rational=True``, Floats will be recast as Rationals before simplification.
    If ``rational=None``, Floats will be recast as Rationals but the result will
    be recast as Floats. If rational=False(default) then nothing will be done
    to the Floats.

    If ``inverse=True``, it will be assumed that a composition of inverse
    functions, such as sin and asin, can be cancelled in any order.
    For example, ``asin(sin(x))`` will yield ``x`` without checking whether
    x belongs to the set where this relation is true. The default is
    False.

    Note that ``simplify()`` automatically calls ``doit()`` on the final
    expression. You can avoid this behavior by passing ``doit=False`` as
    an argument.

    Also, it should be noted that simplifying a boolean expression is not
    well defined. If the expression prefers automatic evaluation (such as
    :obj:`~.Eq()` or :obj:`~.Or()`), simplification will return ``True`` or
    ``False`` if truth value can be determined. If the expression is not
    evaluated by default (such as :obj:`~.Predicate()`), simplification will
    not reduce it and you should use :func:`~.refine` or :func:`~.ask`
    function. This inconsistency will be resolved in future version.

    See Also
    ========

    sympy.assumptions.refine.refine : Simplification using assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """

    def shorter(*choices):
        """
        Return the choice that has the fewest ops. In case of a tie,
        the expression listed first is selected.
        """
        if not has_variety(choices):
            return choices[0]
        return min(choices, key=measure)

    def done(e):
        rv = e.doit() if doit else e
        return shorter(rv, collect_abs(rv))

    expr = sympify(expr, rational=rational)
    kwargs = {
        "ratio": kwargs.get('ratio', ratio),
        "measure": kwargs.get('measure', measure),
        "rational": kwargs.get('rational', rational),
        "inverse": kwargs.get('inverse', inverse),
        "doit": kwargs.get('doit', doit)}
    # no routine for Expr needs to check for is_zero
    if isinstance(expr, Expr) and expr.is_zero:
        return S.Zero if not expr.is_Number else expr

    _eval_simplify = getattr(expr, '_eval_simplify', None)
    if _eval_simplify is not None:
        return _eval_simplify(**kwargs)

    original_expr = expr = collect_abs(signsimp(expr))

    if not isinstance(expr, Basic) or not expr.args:  # XXX: temporary hack
        return expr

    if inverse and expr.has(Function):
        expr = inversecombine(expr)
        if not expr.args:  # simplified to atomic
            return expr

    # do deep simplification
    handled = Add, Mul, Pow, ExpBase
    expr = expr.replace(
        # here, checking for x.args is not enough because Basic has
        # args but Basic does not always play well with replace, e.g.
        # when simultaneous is True found expressions will be masked
        # off with a Dummy but not all Basic objects in an expression
        # can be replaced with a Dummy
        lambda x: isinstance(x, Expr) and x.args and not isinstance(
            x, handled),
        lambda x: x.func(*[simplify(i, **kwargs) for i in x.args]),
        simultaneous=False)
    if not isinstance(expr, handled):
        return done(expr)

    if not expr.is_commutative:
        expr = nc_simplify(expr)

    # TODO: Apply different strategies, considering expression pattern:
    # is it a purely rational function? Is there any trigonometric function?...
    # See also https://github.com/sympy/sympy/pull/185.

    # rationalize Floats
    floats = False
    if rational is not False and expr.has(Float):
        floats = True
        expr = nsimplify(expr, rational=True)

    expr = _bottom_up(expr, lambda w: getattr(w, 'normal', lambda: w)())
    expr = Mul(*powsimp(expr).as_content_primitive())
    _e = cancel(expr)
    expr1 = shorter(_e, _mexpand(_e).cancel())  # issue 6829
    expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))

    if ratio is S.Infinity:
        expr = expr2
    else:
        expr = shorter(expr2, expr1, expr)
    if not isinstance(expr, Basic):  # XXX: temporary hack
        return expr

    expr = factor_terms(expr, sign=False)

    # must come before `Piecewise` since this introduces more `Piecewise` terms
    if expr.has(sign):
        expr = expr.rewrite(Abs)

    # Deal with Piecewise separately to avoid recursive growth of expressions
    if expr.has(Piecewise):
        # Fold into a single Piecewise
        expr = piecewise_fold(expr)
        # Apply doit, if doit=True
        expr = done(expr)
        # Still a Piecewise?
        if expr.has(Piecewise):
            # Fold into a single Piecewise, in case doit lead to some
            # expressions being Piecewise
            expr = piecewise_fold(expr)
            # kroneckersimp also affects Piecewise
            if expr.has(KroneckerDelta):
                expr = kroneckersimp(expr)
            # Still a Piecewise?
            if expr.has(Piecewise):
                # Do not apply doit on the segments as it has already
                # been done above, but simplify
                expr = piecewise_simplify(expr, deep=True, doit=False)
                # Still a Piecewise?
                if expr.has(Piecewise):
                    # Try factor common terms
                    expr = shorter(expr, factor_terms(expr))
                    # As all expressions have been simplified above with the
                    # complete simplify, nothing more needs to be done here
                    return expr

    # hyperexpand automatically only works on hypergeometric terms
    # Do this after the Piecewise part to avoid recursive expansion
    expr = hyperexpand(expr)

    if expr.has(KroneckerDelta):
        expr = kroneckersimp(expr)

    if expr.has(BesselBase):
        expr = besselsimp(expr)

    if expr.has(TrigonometricFunction, HyperbolicFunction):
        expr = trigsimp(expr, deep=True)

    if expr.has(log):
        expr = shorter(expand_log(expr, deep=True), logcombine(expr))

    if expr.has(CombinatorialFunction, gamma):
        # expression with gamma functions or non-integer arguments is
        # automatically passed to gammasimp
        expr = combsimp(expr)

    if expr.has(Sum):
        expr = sum_simplify(expr, **kwargs)

    if expr.has(Integral):
        expr = expr.xreplace({
            i: factor_terms(i) for i in expr.atoms(Integral)})

    if expr.has(Product):
        expr = product_simplify(expr, **kwargs)

    from sympy.physics.units import Quantity

    if expr.has(Quantity):
        from sympy.physics.units.util import quantity_simplify
        expr = quantity_simplify(expr)

    short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)
    short = shorter(short, cancel(short))
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase, exp):
        short = exptrigsimp(short)

    # get rid of hollow 2-arg Mul factorization
    hollow_mul = Transform(
        lambda x: Mul(*x.args),
        lambda x:
        x.is_Mul and
        len(x.args) == 2 and
        x.args[0].is_Number and
        x.args[1].is_Add and
        x.is_commutative)
    expr = short.xreplace(hollow_mul)

    numer, denom = expr.as_numer_denom()
    if denom.is_Add:
        n, d = fraction(radsimp(1/denom, symbolic=False, max_terms=1))
        if n is not S.One:
            expr = (numer*n).expand()/d

    if expr.could_extract_minus_sign():
        n, d = fraction(expr)
        if d != 0:
            expr = signsimp(-n/(-d))

    if measure(expr) > ratio*measure(original_expr):
        expr = original_expr

    # restore floats
    if floats and rational is None:
        expr = nfloat(expr, exponent=False)

    return done(expr)


def sum_simplify(s, **kwargs):
    """Main function for Sum simplification"""
    if not isinstance(s, Add):
        s = s.xreplace({a: sum_simplify(a, **kwargs)
            for a in s.atoms(Add) if a.has(Sum)})
    s = expand(s)
    if not isinstance(s, Add):
        return s

    terms = s.args
    s_t = [] # Sum Terms
    o_t = [] # Other Terms

    for term in terms:
        sum_terms, other = sift(Mul.make_args(term),
            lambda i: isinstance(i, Sum), binary=True)
        if not sum_terms:
            o_t.append(term)
            continue
        other = [Mul(*other)]
        s_t.append(Mul(*(other + [s._eval_simplify(**kwargs) for s in sum_terms])))

    result = Add(sum_combine(s_t), *o_t)

    return result


def sum_combine(s_t):
    """Helper function for Sum simplification

       Attempts to simplify a list of sums, by combining limits / sum function's
       returns the simplified sum
    """
    used = [False] * len(s_t)

    for method in range(2):
        for i, s_term1 in enumerate(s_t):
            if not used[i]:
                for j, s_term2 in enumerate(s_t):
                    if not used[j] and i != j:
                        temp = sum_add(s_term1, s_term2, method)
                        if isinstance(temp, (Sum, Mul)):
                            s_t[i] = temp
                            s_term1 = s_t[i]
                            used[j] = True

    result = S.Zero
    for i, s_term in enumerate(s_t):
        if not used[i]:
            result = Add(result, s_term)

    return result

def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
    """Return Sum with constant factors extracted.

    If ``limits`` is specified then ``self`` is the summand; the other
    keywords are passed to ``factor_terms``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.abc import x, y
    >>> from sympy.simplify.simplify import factor_sum
    >>> s = Sum(x*y, (x, 1, 3))
    >>> factor_sum(s)
    y*Sum(x, (x, 1, 3))
    >>> factor_sum(s.function, s.limits)
    y*Sum(x, (x, 1, 3))
    """

    # XXX deprecate in favor of direct call to factor_terms
    kwargs = {"radical": radical, "clear": clear,
        "fraction": fraction, "sign": sign}
    expr = Sum(self, *limits) if limits else self
    return factor_terms(expr, **kwargs)


def sum_add(self, other, method=0):
    """Helper function for Sum simplification"""
    #we know this is something in terms of a constant * a sum
    #so we temporarily put the constants inside for simplification
    #then simplify the result
    def __refactor(val):
        args = Mul.make_args(val)
        sumv = next(x for x in args if isinstance(x, Sum))
        constant = Mul(*[x for x in args if x != sumv])
        return Sum(constant * sumv.function, *sumv.limits)

    if isinstance(self, Mul):
        rself = __refactor(self)
    else:
        rself = self

    if isinstance(other, Mul):
        rother = __refactor(other)
    else:
        rother = other

    if type(rself) is type(rother):
        if method == 0:
            if rself.limits == rother.limits:
                return factor_sum(Sum(rself.function + rother.function, *rself.limits))
        elif method == 1:
            if simplify(rself.function - rother.function) == 0:
                if len(rself.limits) == len(rother.limits) == 1:
                    i = rself.limits[0][0]
                    x1 = rself.limits[0][1]
                    y1 = rself.limits[0][2]
                    j = rother.limits[0][0]
                    x2 = rother.limits[0][1]
                    y2 = rother.limits[0][2]

                    if i == j:
                        if x2 == y1 + 1:
                            return factor_sum(Sum(rself.function, (i, x1, y2)))
                        elif x1 == y2 + 1:
                            return factor_sum(Sum(rself.function, (i, x2, y1)))

    return Add(self, other)


def product_simplify(s, **kwargs):
    """Main function for Product simplification"""
    terms = Mul.make_args(s)
    p_t = [] # Product Terms
    o_t = [] # Other Terms

    deep = kwargs.get('deep', True)
    for term in terms:
        if isinstance(term, Product):
            if deep:
                p_t.append(Product(term.function.simplify(**kwargs),
                                   *term.limits))
            else:
                p_t.append(term)
        else:
            o_t.append(term)

    used = [False] * len(p_t)

    for method in range(2):
        for i, p_term1 in enumerate(p_t):
            if not used[i]:
                for j, p_term2 in enumerate(p_t):
                    if not used[j] and i != j:
                        tmp_prod = product_mul(p_term1, p_term2, method)
                        if isinstance(tmp_prod, Product):
                            p_t[i] = tmp_prod
                            used[j] = True

    result = Mul(*o_t)

    for i, p_term in enumerate(p_t):
        if not used[i]:
            result = Mul(result, p_term)

    return result


def product_mul(self, other, method=0):
    """Helper function for Product simplification"""
    if type(self) is type(other):
        if method == 0:
            if self.limits == other.limits:
                return Product(self.function * other.function, *self.limits)
        elif method == 1:
            if simplify(self.function - other.function) == 0:
                if len(self.limits) == len(other.limits) == 1:
                    i = self.limits[0][0]
                    x1 = self.limits[0][1]
                    y1 = self.limits[0][2]
                    j = other.limits[0][0]
                    x2 = other.limits[0][1]
                    y2 = other.limits[0][2]

                    if i == j:
                        if x2 == y1 + 1:
                            return Product(self.function, (i, x1, y2))
                        elif x1 == y2 + 1:
                            return Product(self.function, (i, x2, y1))

    return Mul(self, other)


def _nthroot_solve(p, n, prec):
    """
     helper function for ``nthroot``
     It denests ``p**Rational(1, n)`` using its minimal polynomial
    """
    from sympy.solvers import solve
    while n % 2 == 0:
        p = sqrtdenest(sqrt(p))
        n = n // 2
    if n == 1:
        return p
    pn = p**Rational(1, n)
    x = Symbol('x')
    f = _minimal_polynomial_sq(p, n, x)
    if f is None:
        return None
    sols = solve(f, x)
    for sol in sols:
        if abs(sol - pn).n() < 1./10**prec:
            sol = sqrtdenest(sol)
            if _mexpand(sol**n) == p:
                return sol


def logcombine(expr, force=False):
    """
    Takes logarithms and combines them using the following rules:

    - log(x) + log(y) == log(x*y) if both are positive
    - a*log(x) == log(x**a) if x is positive and a is real

    If ``force`` is ``True`` then the assumptions above will be assumed to hold if
    there is no assumption already in place on a quantity. For example, if
    ``a`` is imaginary or the argument negative, force will not perform a
    combination but if ``a`` is a symbol with no assumptions the change will
    take place.

    Examples
    ========

    >>> from sympy import Symbol, symbols, log, logcombine, I
    >>> from sympy.abc import a, x, y, z
    >>> logcombine(a*log(x) + log(y) - log(z))
    a*log(x) + log(y) - log(z)
    >>> logcombine(a*log(x) + log(y) - log(z), force=True)
    log(x**a*y/z)
    >>> x,y,z = symbols('x,y,z', positive=True)
    >>> a = Symbol('a', real=True)
    >>> logcombine(a*log(x) + log(y) - log(z))
    log(x**a*y/z)

    The transformation is limited to factors and/or terms that
    contain logs, so the result depends on the initial state of
    expansion:

    >>> eq = (2 + 3*I)*log(x)
    >>> logcombine(eq, force=True) == eq
    True
    >>> logcombine(eq.expand(), force=True)
    log(x**2) + I*log(x**3)

    See Also
    ========

    posify: replace all symbols with symbols having positive assumptions
    sympy.core.function.expand_log: expand the logarithms of products
        and powers; the opposite of logcombine

    """

    def f(rv):
        if not (rv.is_Add or rv.is_Mul):
            return rv

        def gooda(a):
            # bool to tell whether the leading ``a`` in ``a*log(x)``
            # could appear as log(x**a)
            return (a is not S.NegativeOne and  # -1 *could* go, but we disallow
                (a.is_extended_real or force and a.is_extended_real is not False))

        def goodlog(l):
            # bool to tell whether log ``l``'s argument can combine with others
            a = l.args[0]
            return a.is_positive or force and a.is_nonpositive is not False

        other = []
        logs = []
        log1 = defaultdict(list)
        for a in Add.make_args(rv):
            if isinstance(a, log) and goodlog(a):
                log1[()].append(([], a))
            elif not a.is_Mul:
                other.append(a)
            else:
                ot = []
                co = []
                lo = []
                for ai in a.args:
                    if ai.is_Rational and ai < 0:
                        ot.append(S.NegativeOne)
                        co.append(-ai)
                    elif isinstance(ai, log) and goodlog(ai):
                        lo.append(ai)
                    elif gooda(ai):
                        co.append(ai)
                    else:
                        ot.append(ai)
                if len(lo) > 1:
                    logs.append((ot, co, lo))
                elif lo:
                    log1[tuple(ot)].append((co, lo[0]))
                else:
                    other.append(a)

        # if there is only one log in other, put it with the
        # good logs
        if len(other) == 1 and isinstance(other[0], log):
            log1[()].append(([], other.pop()))
        # if there is only one log at each coefficient and none have
        # an exponent to place inside the log then there is nothing to do
        if not logs and all(len(log1[k]) == 1 and log1[k][0] == [] for k in log1):
            return rv

        # collapse multi-logs as far as possible in a canonical way
        # TODO: see if x*log(a)+x*log(a)*log(b) -> x*log(a)*(1+log(b))?
        # -- in this case, it's unambiguous, but if it were were a log(c) in
        # each term then it's arbitrary whether they are grouped by log(a) or
        # by log(c). So for now, just leave this alone; it's probably better to
        # let the user decide
        for o, e, l in logs:
            l = list(ordered(l))
            e = log(l.pop(0).args[0]**Mul(*e))
            while l:
                li = l.pop(0)
                e = log(li.args[0]**e)
            c, l = Mul(*o), e
            if isinstance(l, log):  # it should be, but check to be sure
                log1[(c,)].append(([], l))
            else:
                other.append(c*l)

        # logs that have the same coefficient can multiply
        for k in list(log1.keys()):
            log1[Mul(*k)] = log(logcombine(Mul(*[
                l.args[0]**Mul(*c) for c, l in log1.pop(k)]),
                force=force), evaluate=False)

        # logs that have oppositely signed coefficients can divide
        for k in ordered(list(log1.keys())):
            if k not in log1:  # already popped as -k
                continue
            if -k in log1:
                # figure out which has the minus sign; the one with
                # more op counts should be the one
                num, den = k, -k
                if num.count_ops() > den.count_ops():
                    num, den = den, num
                other.append(
                    num*log(log1.pop(num).args[0]/log1.pop(den).args[0],
                            evaluate=False))
            else:
                other.append(k*log1.pop(k))

        return Add(*other)

    return _bottom_up(expr, f)


def inversecombine(expr):
    """Simplify the composition of a function and its inverse.

    Explanation
    ===========

    No attention is paid to whether the inverse is a left inverse or a
    right inverse; thus, the result will in general not be equivalent
    to the original expression.

    Examples
    ========

    >>> from sympy.simplify.simplify import inversecombine
    >>> from sympy import asin, sin, log, exp
    >>> from sympy.abc import x
    >>> inversecombine(asin(sin(x)))
    x
    >>> inversecombine(2*log(exp(3*x)))
    6*x
    """

    def f(rv):
        if isinstance(rv, log):
            if isinstance(rv.args[0], exp) or (rv.args[0].is_Pow and rv.args[0].base == S.Exp1):
                rv = rv.args[0].exp
        elif rv.is_Function and hasattr(rv, "inverse"):
            if (len(rv.args) == 1 and len(rv.args[0].args) == 1 and
               isinstance(rv.args[0], rv.inverse(argindex=1))):
                rv = rv.args[0].args[0]
        if rv.is_Pow and rv.base == S.Exp1:
            if isinstance(rv.exp, log):
                rv = rv.exp.args[0]
        return rv

    return _bottom_up(expr, f)


def kroneckersimp(expr):
    """
    Simplify expressions with KroneckerDelta.

    The only simplification currently attempted is to identify multiplicative cancellation:

    Examples
    ========

    >>> from sympy import KroneckerDelta, kroneckersimp
    >>> from sympy.abc import i
    >>> kroneckersimp(1 + KroneckerDelta(0, i) * KroneckerDelta(1, i))
    1
    """
    def args_cancel(args1, args2):
        for i1 in range(2):
            for i2 in range(2):
                a1 = args1[i1]
                a2 = args2[i2]
                a3 = args1[(i1 + 1) % 2]
                a4 = args2[(i2 + 1) % 2]
                if Eq(a1, a2) is S.true and Eq(a3, a4) is S.false:
                    return True
        return False

    def cancel_kronecker_mul(m):
        args = m.args
        deltas = [a for a in args if isinstance(a, KroneckerDelta)]
        for delta1, delta2 in subsets(deltas, 2):
            args1 = delta1.args
            args2 = delta2.args
            if args_cancel(args1, args2):
                return S.Zero * m # In case of oo etc
        return m

    if not expr.has(KroneckerDelta):
        return expr

    if expr.has(Piecewise):
        expr = expr.rewrite(KroneckerDelta)

    newexpr = expr
    expr = None

    while newexpr != expr:
        expr = newexpr
        newexpr = expr.replace(lambda e: isinstance(e, Mul), cancel_kronecker_mul)

    return expr


def besselsimp(expr):
    """
    Simplify bessel-type functions.

    Explanation
    ===========

    This routine tries to simplify bessel-type functions. Currently it only
    works on the Bessel J and I functions, however. It works by looking at all
    such functions in turn, and eliminating factors of "I" and "-1" (actually
    their polar equivalents) in front of the argument. Then, functions of
    half-integer order are rewritten using trigonometric functions and
    functions of integer order (> 1) are rewritten using functions
    of low order.  Finally, if the expression was changed, compute
    factorization of the result with factor().

    >>> from sympy import besselj, besseli, besselsimp, polar_lift, I, S
    >>> from sympy.abc import z, nu
    >>> besselsimp(besselj(nu, z*polar_lift(-1)))
    exp(I*pi*nu)*besselj(nu, z)
    >>> besselsimp(besseli(nu, z*polar_lift(-I)))
    exp(-I*pi*nu/2)*besselj(nu, z)
    >>> besselsimp(besseli(S(-1)/2, z))
    sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    >>> besselsimp(z*besseli(0, z) + z*(besseli(2, z))/2 + besseli(1, z))
    3*z*besseli(0, z)/2
    """
    # TODO
    # - better algorithm?
    # - simplify (cos(pi*b)*besselj(b,z) - besselj(-b,z))/sin(pi*b) ...
    # - use contiguity relations?

    def replacer(fro, to, factors):
        factors = set(factors)

        def repl(nu, z):
            if factors.intersection(Mul.make_args(z)):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro, to):
        def tofunc(nu, z):
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro):
        def tofunc(nu, z):
            return exp(I*pi*nu)*fro(nu, exp_polar(-I*pi)*z)
        return tofunc

    orig_expr = expr

    ifactors = [I, exp_polar(I*pi/2), exp_polar(-I*pi/2)]
    expr = expr.replace(
        besselj, replacer(besselj,
        torewrite(besselj, besseli), ifactors))
    expr = expr.replace(
        besseli, replacer(besseli,
        torewrite(besseli, besselj), ifactors))

    minusfactors = [-1, exp_polar(I*pi)]
    expr = expr.replace(
        besselj, replacer(besselj, tominus(besselj), minusfactors))
    expr = expr.replace(
        besseli, replacer(besseli, tominus(besseli), minusfactors))

    z0 = Dummy('z')

    def expander(fro):
        def repl(nu, z):
            if (nu % 1) == S.Half:
                return simplify(trigsimp(unpolarify(
                        fro(nu, z0).rewrite(besselj).rewrite(jn).expand(
                            func=True)).subs(z0, z)))
            elif nu.is_Integer and nu > 1:
                return fro(nu, z).expand(func=True)
            return fro(nu, z)
        return repl

    expr = expr.replace(besselj, expander(besselj))
    expr = expr.replace(bessely, expander(bessely))
    expr = expr.replace(besseli, expander(besseli))
    expr = expr.replace(besselk, expander(besselk))

    def _bessel_simp_recursion(expr):

        def _use_recursion(bessel, expr):
            while True:
                bessels = expr.find(lambda x: isinstance(x, bessel))
                try:
                    for ba in sorted(bessels, key=lambda x: re(x.args[0])):
                        a, x = ba.args
                        bap1 = bessel(a+1, x)
                        bap2 = bessel(a+2, x)
                        if expr.has(bap1) and expr.has(bap2):
                            expr = expr.subs(ba, 2*(a+1)/x*bap1 - bap2)
                            break
                    else:
                        return expr
                except (ValueError, TypeError):
                    return expr
        if expr.has(besselj):
            expr = _use_recursion(besselj, expr)
        if expr.has(bessely):
            expr = _use_recursion(bessely, expr)
        return expr

    expr = _bessel_simp_recursion(expr)
    if expr != orig_expr:
        expr = expr.factor()

    return expr


def nthroot(expr, n, max_len=4, prec=15):
    """
    Compute a real nth-root of a sum of surds.

    Parameters
    ==========

    expr : sum of surds
    n : integer
    max_len : maximum number of surds passed as constants to ``nsimplify``

    Algorithm
    =========

    First ``nsimplify`` is used to get a candidate root; if it is not a
    root the minimal polynomial is computed; the answer is one of its
    roots.

    Examples
    ========

    >>> from sympy.simplify.simplify import nthroot
    >>> from sympy import sqrt
    >>> nthroot(90 + 34*sqrt(7), 3)
    sqrt(7) + 3

    """
    expr = sympify(expr)
    n = sympify(n)
    p = expr**Rational(1, n)
    if not n.is_integer:
        return p
    if not _is_sum_surds(expr):
        return p
    surds = []
    coeff_muls = [x.as_coeff_Mul() for x in expr.args]
    for x, y in coeff_muls:
        if not x.is_rational:
            return p
        if y is S.One:
            continue
        if not (y.is_Pow and y.exp == S.Half and y.base.is_integer):
            return p
        surds.append(y)
    surds.sort()
    surds = surds[:max_len]
    if expr < 0 and n % 2 == 1:
        p = (-expr)**Rational(1, n)
        a = nsimplify(p, constants=surds)
        res = a if _mexpand(a**n) == _mexpand(-expr) else p
        return -res
    a = nsimplify(p, constants=surds)
    if _mexpand(a) is not _mexpand(p) and _mexpand(a**n) == _mexpand(expr):
        return _mexpand(a)
    expr = _nthroot_solve(expr, n, prec)
    if expr is None:
        return p
    return expr


def nsimplify(expr, constants=(), tolerance=None, full=False, rational=None,
    rational_conversion='base10'):
    """
    Find a simple representation for a number or, if there are free symbols or
    if ``rational=True``, then replace Floats with their Rational equivalents. If
    no change is made and rational is not False then Floats will at least be
    converted to Rationals.

    Explanation
    ===========

    For numerical expressions, a simple formula that numerically matches the
    given numerical expression is sought (and the input should be possible
    to evalf to a precision of at least 30 digits).

    Optionally, a list of (rationally independent) constants to
    include in the formula may be given.

    A lower tolerance may be set to find less exact matches. If no tolerance
    is given then the least precise value will set the tolerance (e.g. Floats
    default to 15 digits of precision, so would be tolerance=10**-15).

    With ``full=True``, a more extensive search is performed
    (this is useful to find simpler numbers when the tolerance
    is set low).

    When converting to rational, if rational_conversion='base10' (the default), then
    convert floats to rationals using their base-10 (string) representation.
    When rational_conversion='exact' it uses the exact, base-2 representation.

    Examples
    ========

    >>> from sympy import nsimplify, sqrt, GoldenRatio, exp, I, pi
    >>> nsimplify(4/(1+sqrt(5)), [GoldenRatio])
    -2 + 2*GoldenRatio
    >>> nsimplify((1/(exp(3*pi*I/5)+1)))
    1/2 - I*sqrt(sqrt(5)/10 + 1/4)
    >>> nsimplify(I**I, [pi])
    exp(-pi/2)
    >>> nsimplify(pi, tolerance=0.01)
    22/7

    >>> nsimplify(0.333333333333333, rational=True, rational_conversion='exact')
    6004799503160655/18014398509481984
    >>> nsimplify(0.333333333333333, rational=True)
    1/3

    See Also
    ========

    sympy.core.function.nfloat

    """
    try:
        return sympify(as_int(expr))
    except (TypeError, ValueError):
        pass
    expr = sympify(expr).xreplace({
        Float('inf'): S.Infinity,
        Float('-inf'): S.NegativeInfinity,
        })
    if expr is S.Infinity or expr is S.NegativeInfinity:
        return expr
    if rational or expr.free_symbols:
        return _real_to_rational(expr, tolerance, rational_conversion)

    # SymPy's default tolerance for Rationals is 15; other numbers may have
    # lower tolerances set, so use them to pick the largest tolerance if None
    # was given
    if tolerance is None:
        tolerance = 10**-min([15] +
             [mpmath.libmp.libmpf.prec_to_dps(n._prec)
             for n in expr.atoms(Float)])
    # XXX should prec be set independent of tolerance or should it be computed
    # from tolerance?
    prec = 30
    bprec = int(prec*3.33)

    constants_dict = {}
    for constant in constants:
        constant = sympify(constant)
        v = constant.evalf(prec)
        if not v.is_Float:
            raise ValueError("constants must be real-valued")
        constants_dict[str(constant)] = v._to_mpmath(bprec)

    exprval = expr.evalf(prec, chop=True)
    re, im = exprval.as_real_imag()

    # safety check to make sure that this evaluated to a number
    if not (re.is_Number and im.is_Number):
        return expr

    def nsimplify_real(x):
        orig = mpmath.mp.dps
        xv = x._to_mpmath(bprec)
        try:
            # We'll be happy with low precision if a simple fraction
            if not (tolerance or full):
                mpmath.mp.dps = 15
                rat = mpmath.pslq([xv, 1])
                if rat is not None:
                    return Rational(-int(rat[1]), int(rat[0]))
            mpmath.mp.dps = prec
            newexpr = mpmath.identify(xv, constants=constants_dict,
                tol=tolerance, full=full)
            if not newexpr:
                raise ValueError
            if full:
                newexpr = newexpr[0]
            expr = sympify(newexpr)
            if x and not expr:  # don't let x become 0
                raise ValueError
            if expr.is_finite is False and xv not in [mpmath.inf, mpmath.ninf]:
                raise ValueError
            return expr
        finally:
            # even though there are returns above, this is executed
            # before leaving
            mpmath.mp.dps = orig
    try:
        if re:
            re = nsimplify_real(re)
        if im:
            im = nsimplify_real(im)
    except ValueError:
        if rational is None:
            return _real_to_rational(expr, rational_conversion=rational_conversion)
        return expr

    rv = re + im*S.ImaginaryUnit
    # if there was a change or rational is explicitly not wanted
    # return the value, else return the Rational representation
    if rv != expr or rational is False:
        return rv
    return _real_to_rational(expr, rational_conversion=rational_conversion)


def _real_to_rational(expr, tolerance=None, rational_conversion='base10'):
    """
    Replace all reals in expr with rationals.

    Examples
    ========

    >>> from sympy.simplify.simplify import _real_to_rational
    >>> from sympy.abc import x

    >>> _real_to_rational(.76 + .1*x**.5)
    sqrt(x)/10 + 19/25

    If rational_conversion='base10', this uses the base-10 string. If
    rational_conversion='exact', the exact, base-2 representation is used.

    >>> _real_to_rational(0.333333333333333, rational_conversion='exact')
    6004799503160655/18014398509481984
    >>> _real_to_rational(0.333333333333333)
    1/3

    """
    expr = _sympify(expr)
    inf = Float('inf')
    p = expr
    reps = {}
    reduce_num = None
    if tolerance is not None and tolerance < 1:
        reduce_num = ceiling(1/tolerance)
    for fl in p.atoms(Float):
        key = fl
        if reduce_num is not None:
            r = Rational(fl).limit_denominator(reduce_num)
        elif (tolerance is not None and tolerance >= 1 and
                fl.is_Integer is False):
            r = Rational(tolerance*round(fl/tolerance)
                ).limit_denominator(int(tolerance))
        else:
            if rational_conversion == 'exact':
                r = Rational(fl)
                reps[key] = r
                continue
            elif rational_conversion != 'base10':
                raise ValueError("rational_conversion must be 'base10' or 'exact'")

            r = nsimplify(fl, rational=False)
            # e.g. log(3).n() -> log(3) instead of a Rational
            if fl and not r:
                r = Rational(fl)
            elif not r.is_Rational:
                if fl in (inf, -inf):
                    r = S.ComplexInfinity
                elif fl < 0:
                    fl = -fl
                    d = Pow(10, int(mpmath.log(fl)/mpmath.log(10)))
                    r = -Rational(str(fl/d))*d
                elif fl > 0:
                    d = Pow(10, int(mpmath.log(fl)/mpmath.log(10)))
                    r = Rational(str(fl/d))*d
                else:
                    r = S.Zero
        reps[key] = r
    return p.subs(reps, simultaneous=True)


def clear_coefficients(expr, rhs=S.Zero):
    """Return `p, r` where `p` is the expression obtained when Rational
    additive and multiplicative coefficients of `expr` have been stripped
    away in a naive fashion (i.e. without simplification). The operations
    needed to remove the coefficients will be applied to `rhs` and returned
    as `r`.

    Examples
    ========

    >>> from sympy.simplify.simplify import clear_coefficients
    >>> from sympy.abc import x, y
    >>> from sympy import Dummy
    >>> expr = 4*y*(6*x + 3)
    >>> clear_coefficients(expr - 2)
    (y*(2*x + 1), 1/6)

    When solving 2 or more expressions like `expr = a`,
    `expr = b`, etc..., it is advantageous to provide a Dummy symbol
    for `rhs` and  simply replace it with `a`, `b`, etc... in `r`.

    >>> rhs = Dummy('rhs')
    >>> clear_coefficients(expr, rhs)
    (y*(2*x + 1), _rhs/12)
    >>> _[1].subs(rhs, 2)
    1/6
    """
    was = None
    free = expr.free_symbols
    if expr.is_Rational:
        return (S.Zero, rhs - expr)
    while expr and was != expr:
        was = expr
        m, expr = (
            expr.as_content_primitive()
            if free else
            factor_terms(expr).as_coeff_Mul(rational=True))
        rhs /= m
        c, expr = expr.as_coeff_Add(rational=True)
        rhs -= c
    expr = signsimp(expr, evaluate = False)
    if expr.could_extract_minus_sign():
        expr = -expr
        rhs = -rhs
    return expr, rhs

def nc_simplify(expr, deep=True):
    '''
    Simplify a non-commutative expression composed of multiplication
    and raising to a power by grouping repeated subterms into one power.
    Priority is given to simplifications that give the fewest number
    of arguments in the end (for example, in a*b*a*b*c*a*b*c simplifying
    to (a*b)**2*c*a*b*c gives 5 arguments while a*b*(a*b*c)**2 has 3).
    If ``expr`` is a sum of such terms, the sum of the simplified terms
    is returned.

    Keyword argument ``deep`` controls whether or not subexpressions
    nested deeper inside the main expression are simplified. See examples
    below. Setting `deep` to `False` can save time on nested expressions
    that do not need simplifying on all levels.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.simplify.simplify import nc_simplify
    >>> a, b, c = symbols("a b c", commutative=False)
    >>> nc_simplify(a*b*a*b*c*a*b*c)
    a*b*(a*b*c)**2
    >>> expr = a**2*b*a**4*b*a**4
    >>> nc_simplify(expr)
    a**2*(b*a**4)**2
    >>> nc_simplify(a*b*a*b*c**2*(a*b)**2*c**2)
    ((a*b)**2*c**2)**2
    >>> nc_simplify(a*b*a*b + 2*a*c*a**2*c*a**2*c*a)
    (a*b)**2 + 2*(a*c*a)**3
    >>> nc_simplify(b**-1*a**-1*(a*b)**2)
    a*b
    >>> nc_simplify(a**-1*b**-1*c*a)
    (b*a)**(-1)*c*a
    >>> expr = (a*b*a*b)**2*a*c*a*c
    >>> nc_simplify(expr)
    (a*b)**4*(a*c)**2
    >>> nc_simplify(expr, deep=False)
    (a*b*a*b)**2*(a*c)**2

    '''
    if isinstance(expr, MatrixExpr):
        expr = expr.doit(inv_expand=False)
        _Add, _Mul, _Pow, _Symbol = MatAdd, MatMul, MatPow, MatrixSymbol
    else:
        _Add, _Mul, _Pow, _Symbol = Add, Mul, Pow, Symbol

    # =========== Auxiliary functions ========================
    def _overlaps(args):
        # Calculate a list of lists m such that m[i][j] contains the lengths
        # of all possible overlaps between args[:i+1] and args[i+1+j:].
        # An overlap is a suffix of the prefix that matches a prefix
        # of the suffix.
        # For example, let expr=c*a*b*a*b*a*b*a*b. Then m[3][0] contains
        # the lengths of overlaps of c*a*b*a*b with a*b*a*b. The overlaps
        # are a*b*a*b, a*b and the empty word so that m[3][0]=[4,2,0].
        # All overlaps rather than only the longest one are recorded
        # because this information helps calculate other overlap lengths.
        m = [[([1, 0] if a == args[0] else [0]) for a in args[1:]]]
        for i in range(1, len(args)):
            overlaps = []
            j = 0
            for j in range(len(args) - i - 1):
                overlap = []
                for v in m[i-1][j+1]:
                    if j + i + 1 + v < len(args) and args[i] == args[j+i+1+v]:
                        overlap.append(v + 1)
                overlap += [0]
                overlaps.append(overlap)
            m.append(overlaps)
        return m

    def _reduce_inverses(_args):
        # replace consecutive negative powers by an inverse
        # of a product of positive powers, e.g. a**-1*b**-1*c
        # will simplify to (a*b)**-1*c;
        # return that new args list and the number of negative
        # powers in it (inv_tot)
        inv_tot = 0 # total number of inverses
        inverses = []
        args = []
        for arg in _args:
            if isinstance(arg, _Pow) and arg.args[1].is_extended_negative:
                inverses = [arg**-1] + inverses
                inv_tot += 1
            else:
                if len(inverses) == 1:
                    args.append(inverses[0]**-1)
                elif len(inverses) > 1:
                    args.append(_Pow(_Mul(*inverses), -1))
                    inv_tot -= len(inverses) - 1
                inverses = []
                args.append(arg)
        if inverses:
            args.append(_Pow(_Mul(*inverses), -1))
            inv_tot -= len(inverses) - 1
        return inv_tot, tuple(args)

    def get_score(s):
        # compute the number of arguments of s
        # (including in nested expressions) overall
        # but ignore exponents
        if isinstance(s, _Pow):
            return get_score(s.args[0])
        elif isinstance(s, (_Add, _Mul)):
            return sum(get_score(a) for a in s.args)
        return 1

    def compare(s, alt_s):
        # compare two possible simplifications and return a
        # "better" one
        if s != alt_s and get_score(alt_s) < get_score(s):
            return alt_s
        return s
    # ========================================================

    if not isinstance(expr, (_Add, _Mul, _Pow)) or expr.is_commutative:
        return expr
    args = expr.args[:]
    if isinstance(expr, _Pow):
        if deep:
            return _Pow(nc_simplify(args[0]), args[1]).doit()
        else:
            return expr
    elif isinstance(expr, _Add):
        return _Add(*[nc_simplify(a, deep=deep) for a in args]).doit()
    else:
        # get the non-commutative part
        c_args, args = expr.args_cnc()
        com_coeff = Mul(*c_args)
        if not equal_valued(com_coeff, 1):
            return com_coeff*nc_simplify(expr/com_coeff, deep=deep)

    inv_tot, args = _reduce_inverses(args)
    # if most arguments are negative, work with the inverse
    # of the expression, e.g. a**-1*b*a**-1*c**-1 will become
    # (c*a*b**-1*a)**-1 at the end so can work with c*a*b**-1*a
    invert = False
    if inv_tot > len(args)/2:
        invert = True
        args = [a**-1 for a in args[::-1]]

    if deep:
        args = tuple(nc_simplify(a) for a in args)

    m = _overlaps(args)

    # simps will be {subterm: end} where `end` is the ending
    # index of a sequence of repetitions of subterm;
    # this is for not wasting time with subterms that are part
    # of longer, already considered sequences
    simps = {}

    post = 1
    pre = 1

    # the simplification coefficient is the number of
    # arguments by which contracting a given sequence
    # would reduce the word; e.g. in a*b*a*b*c*a*b*c,
    # contracting a*b*a*b to (a*b)**2 removes 3 arguments
    # while a*b*c*a*b*c to (a*b*c)**2 removes 6. It's
    # better to contract the latter so simplification
    # with a maximum simplification coefficient will be chosen
    max_simp_coeff = 0
    simp = None # information about future simplification

    for i in range(1, len(args)):
        simp_coeff = 0
        l = 0 # length of a subterm
        p = 0 # the power of a subterm
        if i < len(args) - 1:
            rep = m[i][0]
        start = i # starting index of the repeated sequence
        end = i+1 # ending index of the repeated sequence
        if i == len(args)-1 or rep == [0]:
            # no subterm is repeated at this stage, at least as
            # far as the arguments are concerned - there may be
            # a repetition if powers are taken into account
            if (isinstance(args[i], _Pow) and
                            not isinstance(args[i].args[0], _Symbol)):
                subterm = args[i].args[0].args
                l = len(subterm)
                if args[i-l:i] == subterm:
                    # e.g. a*b in a*b*(a*b)**2 is not repeated
                    # in args (= [a, b, (a*b)**2]) but it
                    # can be matched here
                    p += 1
                    start -= l
                if args[i+1:i+1+l] == subterm:
                    # e.g. a*b in (a*b)**2*a*b
                    p += 1
                    end += l
            if p:
                p += args[i].args[1]
            else:
                continue
        else:
            l = rep[0] # length of the longest repeated subterm at this point
            start -= l - 1
            subterm = args[start:end]
            p = 2
            end += l

        if subterm in simps and simps[subterm] >= start:
            # the subterm is part of a sequence that
            # has already been considered
            continue

        # count how many times it's repeated
        while end < len(args):
            if l in m[end-1][0]:
                p += 1
                end += l
            elif isinstance(args[end], _Pow) and args[end].args[0].args == subterm:
                # for cases like a*b*a*b*(a*b)**2*a*b
                p += args[end].args[1]
                end += 1
            else:
                break

        # see if another match can be made, e.g.
        # for b*a**2 in b*a**2*b*a**3 or a*b in
        # a**2*b*a*b

        pre_exp = 0
        pre_arg = 1
        if start - l >= 0 and args[start-l+1:start] == subterm[1:]:
            if isinstance(subterm[0], _Pow):
                pre_arg = subterm[0].args[0]
                exp = subterm[0].args[1]
            else:
                pre_arg = subterm[0]
                exp = 1
            if isinstance(args[start-l], _Pow) and args[start-l].args[0] == pre_arg:
                pre_exp = args[start-l].args[1] - exp
                start -= l
                p += 1
            elif args[start-l] == pre_arg:
                pre_exp = 1 - exp
                start -= l
                p += 1

        post_exp = 0
        post_arg = 1
        if end + l - 1 < len(args) and args[end:end+l-1] == subterm[:-1]:
            if isinstance(subterm[-1], _Pow):
                post_arg = subterm[-1].args[0]
                exp = subterm[-1].args[1]
            else:
                post_arg = subterm[-1]
                exp = 1
            if isinstance(args[end+l-1], _Pow) and args[end+l-1].args[0] == post_arg:
                post_exp = args[end+l-1].args[1] - exp
                end += l
                p += 1
            elif args[end+l-1] == post_arg:
                post_exp = 1 - exp
                end += l
                p += 1

        # Consider a*b*a**2*b*a**2*b*a:
        # b*a**2 is explicitly repeated, but note
        # that in this case a*b*a is also repeated
        # so there are two possible simplifications:
        # a*(b*a**2)**3*a**-1 or (a*b*a)**3
        # The latter is obviously simpler.
        # But in a*b*a**2*b**2*a**2 the simplifications are
        # a*(b*a**2)**2 and (a*b*a)**3*a in which case
        # it's better to stick with the shorter subterm
        if post_exp and exp % 2 == 0 and start > 0:
            exp = exp/2
            _pre_exp = 1
            _post_exp = 1
            if isinstance(args[start-1], _Pow) and args[start-1].args[0] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = args[start-1].args[1] - exp
            elif args[start-1] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = 1 - exp
            if _pre_exp == 0 or _post_exp == 0:
                if not pre_exp:
                    start -= 1
                post_exp = _post_exp
                pre_exp = _pre_exp
                pre_arg = post_arg
                subterm = (post_arg**exp,) + subterm[:-1] + (post_arg**exp,)

        simp_coeff += end-start

        if post_exp:
            simp_coeff -= 1
        if pre_exp:
            simp_coeff -= 1

        simps[subterm] = end

        if simp_coeff > max_simp_coeff:
            max_simp_coeff = simp_coeff
            simp = (start, _Mul(*subterm), p, end, l)
            pre = pre_arg**pre_exp
            post = post_arg**post_exp

    if simp:
        subterm = _Pow(nc_simplify(simp[1], deep=deep), simp[2])
        pre = nc_simplify(_Mul(*args[:simp[0]])*pre, deep=deep)
        post = post*nc_simplify(_Mul(*args[simp[3]:]), deep=deep)
        simp = pre*subterm*post
        if pre != 1 or post != 1:
            # new simplifications may be possible but no need
            # to recurse over arguments
            simp = nc_simplify(simp, deep=False)
    else:
        simp = _Mul(*args)

    if invert:
        simp = _Pow(simp, -1)

    # see if factor_nc(expr) is simplified better
    if not isinstance(expr, MatrixExpr):
        f_expr = factor_nc(expr)
        if f_expr != expr:
            alt_simp = nc_simplify(f_expr, deep=deep)
            simp = compare(simp, alt_simp)
    else:
        simp = simp.doit(inv_expand=False)
    return simp


def dotprodsimp(expr, withsimp=False):
    """Simplification for a sum of products targeted at the kind of blowup that
    occurs during summation of products. Intended to reduce expression blowup
    during matrix multiplication or other similar operations. Only works with
    algebraic expressions and does not recurse into non.

    Parameters
    ==========

    withsimp : bool, optional
        Specifies whether a flag should be returned along with the expression
        to indicate roughly whether simplification was successful. It is used
        in ``MatrixArithmetic._eval_pow_by_recursion`` to avoid attempting to
        simplify an expression repetitively which does not simplify.
    """

    def count_ops_alg(expr):
        """Optimized count algebraic operations with no recursion into
        non-algebraic args that ``core.function.count_ops`` does. Also returns
        whether rational functions may be present according to negative
        exponents of powers or non-number fractions.

        Returns
        =======

        ops, ratfunc : int, bool
            ``ops`` is the number of algebraic operations starting at the top
            level expression (not recursing into non-alg children). ``ratfunc``
            specifies whether the expression MAY contain rational functions
            which ``cancel`` MIGHT optimize.
        """

        ops     = 0
        args    = [expr]
        ratfunc = False

        while args:
            a = args.pop()

            if not isinstance(a, Basic):
                continue

            if a.is_Rational:
                if a is not S.One: # -1/3 = NEG + DIV
                    ops += bool (a.p < 0) + bool (a.q != 1)

            elif a.is_Mul:
                if a.could_extract_minus_sign():
                    ops += 1
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a

                n, d = fraction(a)

                if n.is_Integer:
                    ops += 1 + bool (n < 0)
                    args.append(d) # won't be -Mul but could be Add

                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                        ratfunc=True

                    ops += 1
                    args.append(n) # could be -Mul

                else:
                    ops += len(a.args) - 1
                    args.extend(a.args)

            elif a.is_Add:
                laargs = len(a.args)
                negs   = 0

                for ai in a.args:
                    if ai.could_extract_minus_sign():
                        negs += 1
                        ai    = -ai
                    args.append(ai)

                ops += laargs - (negs != laargs) # -x - y = NEG + SUB

            elif a.is_Pow:
                ops += 1
                args.append(a.base)

                if not ratfunc:
                    ratfunc = a.exp.is_negative is not False

        return ops, ratfunc

    def nonalg_subs_dummies(expr, dummies):
        """Substitute dummy variables for non-algebraic expressions to avoid
        evaluation of non-algebraic terms that ``polys.polytools.cancel`` does.
        """

        if not expr.args:
            return expr

        if expr.is_Add or expr.is_Mul or expr.is_Pow:
            args = None

            for i, a in enumerate(expr.args):
                c = nonalg_subs_dummies(a, dummies)

                if c is a:
                    continue

                if args is None:
                    args = list(expr.args)

                args[i] = c

            if args is None:
                return expr

            return expr.func(*args)

        return dummies.setdefault(expr, Dummy())

    simplified = False # doesn't really mean simplified, rather "can simplify again"

    if isinstance(expr, Basic) and (expr.is_Add or expr.is_Mul or expr.is_Pow):
        expr2 = expr.expand(deep=True, modulus=None, power_base=False,
            power_exp=False, mul=True, log=False, multinomial=True, basic=False)

        if expr2 != expr:
            expr       = expr2
            simplified = True

        exprops, ratfunc = count_ops_alg(expr)

        if exprops >= 6: # empirically tested cutoff for expensive simplification
            if ratfunc:
                dummies = {}
                expr2   = nonalg_subs_dummies(expr, dummies)

                if expr2 is expr or count_ops_alg(expr2)[0] >= 6: # check again after substitution
                    expr3 = cancel(expr2)

                    if expr3 != expr2:
                        expr       = expr3.subs([(d, e) for e, d in dummies.items()])
                        simplified = True

        # very special case: x/(x-1) - 1/(x-1) -> 1
        elif (exprops == 5 and expr.is_Add and expr.args [0].is_Mul and
                expr.args [1].is_Mul and expr.args [0].args [-1].is_Pow and
                expr.args [1].args [-1].is_Pow and
                expr.args [0].args [-1].exp is S.NegativeOne and
