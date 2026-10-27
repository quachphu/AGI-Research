    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
      y_0 = a/b * x_0

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a,b
        Defines the parameters of the sinh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is ``2*pi``.

    Notes
    -----
    ``ss_diff(ss_diff(x,a,b),b,a) == x``

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'ss_diff_cache'):
            _cache.ss_diff_cache = {}
        _cache = _cache.ss_diff_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return ss_diff(tmp.real, a, b, period, _cache) + \
               1j*ss_diff(tmp.imag, a, b, period, _cache)
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    n = len(x)
    omega = _cache.get((n,a,b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k,a=a,b=b):
            if k:
                return sinh(a*k)/sinh(b*k)
            return float(a)/b
        omega = convolve.init_convolution_kernel(n,kernel)
        _cache[(n,a,b)] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,overwrite_x=overwrite_x)


def cc_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a,b : float
        Defines the parameters of the sinh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is ``2*pi``.

    Returns
    -------
    cc_diff : ndarray
        Pseudo-derivative of periodic sequence `x`.

    Notes
    -----
    ``cc_diff(cc_diff(x,a,b),b,a) == x``

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'cc_diff_cache'):
            _cache.cc_diff_cache = {}
        _cache = _cache.cc_diff_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return cc_diff(tmp.real, a, b, period, _cache) + \
               1j * cc_diff(tmp.imag, a, b, period, _cache)
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    n = len(x)
    omega = _cache.get((n,a,b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k,a=a,b=b):
            return cosh(a*k)/cosh(b*k)
        omega = convolve.init_convolution_kernel(n,kernel)
        _cache[(n,a,b)] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,overwrite_x=overwrite_x)


def shift(x, a, period=None, _cache=_cache):
    """
    Shift periodic sequence x by a: y(u) = x(u+a).

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

          y_j = exp(j*a*2*pi/period*sqrt(-1)) * x_f

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a : float
        Defines the parameters of the sinh/sinh pseudo-differential
    period : float, optional
        The period of the sequences x and y. Default period is ``2*pi``.
    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'shift_cache'):
            _cache.shift_cache = {}
        _cache = _cache.shift_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return shift(tmp.real, a, period, _cache) + 1j * shift(
            tmp.imag, a, period, _cache)
    if period is not None:
        a = a*2*pi/period
    n = len(x)
    omega = _cache.get((n,a))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel_real(k,a=a):
            return cos(a*k)

        def kernel_imag(k,a=a):
            return sin(a*k)
        omega_real = convolve.init_convolution_kernel(n,kernel_real,d=0,
                                                      zero_nyquist=0)
        omega_imag = convolve.init_convolution_kernel(n,kernel_imag,d=1,
                                                      zero_nyquist=0)
        _cache[(n,a)] = omega_real,omega_imag
    else:
        omega_real,omega_imag = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve_z(tmp,omega_real,omega_imag,
                               overwrite_x=overwrite_x)

<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'fft','ifft','fftn','ifftn','rfft','irfft',
    'fft2','ifft2'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="fftpack", module="basic",
                                   private_modules=["_basic"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
=========================================================
Legacy discrete Fourier transforms (:mod:`scipy.fftpack`)
=========================================================

.. legacy::

   New code should use :mod:`scipy.fft`.

Fast Fourier Transforms (FFTs)
==============================

.. autosummary::
   :toctree: generated/

   fft - Fast (discrete) Fourier Transform (FFT)
   ifft - Inverse FFT
   fft2 - 2-D FFT
   ifft2 - 2-D inverse FFT
   fftn - N-D FFT
   ifftn - N-D inverse FFT
   rfft - FFT of strictly real-valued sequence
   irfft - Inverse of rfft
   dct - Discrete cosine transform
   idct - Inverse discrete cosine transform
   dctn - N-D Discrete cosine transform
   idctn - N-D Inverse discrete cosine transform
   dst - Discrete sine transform
   idst - Inverse discrete sine transform
   dstn - N-D Discrete sine transform
   idstn - N-D Inverse discrete sine transform

Differential and pseudo-differential operators
==============================================

.. autosummary::
   :toctree: generated/

   diff - Differentiation and integration of periodic sequences
   tilbert - Tilbert transform:         cs_diff(x,h,h)
   itilbert - Inverse Tilbert transform: sc_diff(x,h,h)
   hilbert - Hilbert transform:         cs_diff(x,inf,inf)
   ihilbert - Inverse Hilbert transform: sc_diff(x,inf,inf)
   cs_diff - cosh/sinh pseudo-derivative of periodic sequences
   sc_diff - sinh/cosh pseudo-derivative of periodic sequences
   ss_diff - sinh/sinh pseudo-derivative of periodic sequences
   cc_diff - cosh/cosh pseudo-derivative of periodic sequences
   shift - Shift periodic sequences

Helper functions
================

.. autosummary::
   :toctree: generated/

   fftshift - Shift the zero-frequency component to the center of the spectrum
   ifftshift - The inverse of `fftshift`
   fftfreq - Return the Discrete Fourier Transform sample frequencies
   rfftfreq - DFT sample frequencies (for usage with rfft, irfft)
   next_fast_len - Find the optimal length to zero-pad an FFT for speed

Note that ``fftshift``, ``ifftshift`` and ``fftfreq`` are numpy functions
exposed by ``fftpack``; importing them from ``numpy`` should be preferred.

Convolutions (:mod:`scipy.fftpack.convolve`)
============================================

.. module:: scipy.fftpack.convolve

.. autosummary::
   :toctree: generated/

   convolve
   convolve_z
   init_convolution_kernel
   destroy_convolve_cache

"""


__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
           'fft2','ifft2',
           'diff',
           'tilbert','itilbert','hilbert','ihilbert',
           'sc_diff','cs_diff','cc_diff','ss_diff',
           'shift',
           'fftfreq', 'rfftfreq',
           'fftshift', 'ifftshift',
           'next_fast_len',
           'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'
           ]

from ._basic import *
from ._pseudo_diffs import *
from ._helper import *
from ._realtransforms import *

# Deprecated namespaces, to be removed in v2.0.0
from . import basic, helper, pseudo_diffs, realtransforms

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>import operator

import numpy as np
from numpy.fft import fftshift, ifftshift, fftfreq

import scipy.fft._pocketfft.helper as _helper

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len']


def rfftfreq(n, d=1.0):
    """DFT sample frequencies (for usage with rfft, irfft).

    The returned float array contains the frequency bins in
    cycles/unit (with zero at the start) given a window length `n` and a
    sample spacing `d`::

      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even
      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing. Default is 1.

    Returns
    -------
    out : ndarray
        The array of length `n`, containing the sample frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fftpack
    >>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> sig_fft = fftpack.rfft(sig)
    >>> n = sig_fft.size
    >>> timestep = 0.1
    >>> freq = fftpack.rfftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])

    """
    n = operator.index(n)
    if n < 0:
        raise ValueError(f"n = {n} is not valid. "
                         "n must be a nonnegative integer.")

    return (np.arange(1, n + 1, dtype=int) // 2) / float(n * d)


def next_fast_len(target):
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    On a particular machine, an FFT of prime length takes 133 ms:

    >>> from scipy import fftpack
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> min_len = 10007  # prime length is worst case for speed
    >>> a = rng.standard_normal(min_len)
    >>> b = fftpack.fft(a)

    Zero-padding to the next 5-smooth length reduces computation time to
    211 us, a speedup of 630 times:

    >>> fftpack.next_fast_len(min_len)
    10125
    >>> b = fftpack.fft(a, 10125)

    Rounding up to the next power of 2 is not optimal, taking 367 us to
    compute, 1.7 times as long as the 5-smooth size:

    >>> b = fftpack.fft(a, 16384)

    """
    # Real transforms use regular sizes so this is backwards compatible
    return _helper.good_size(target, True)


def _good_shape(x, shape, axes):
    """Ensure that shape argument is valid for scipy.fftpack

    scipy.fftpack does not support len(shape) < x.ndim when axes is not given.
    """
    if shape is not None and axes is None:
        shape = _helper._iterable_of_int(shape, 'shape')
        if len(shape) != np.ndim(x):
            raise ValueError("when given, axes and shape arguments"
                             " have to be of the same length")
    return shape
<|endoftext|>"""
Discrete Fourier Transforms - _basic.py
"""
# Created by Pearu Peterson, August,September 2002
__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
           'fft2','ifft2']

from scipy.fft import _pocketfft
from ._helper import _good_shape


def fft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)``, where

    ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

    Parameters
    ----------
    x : array_like
        Array to Fourier transform.
    n : int, optional
        Length of the Fourier transform. If ``n < x.shape[axis]``, `x` is
        truncated. If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the fft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    z : complex ndarray
        with the elements::

            [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
            [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

        where::

            y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

    See Also
    --------
    ifft : Inverse FFT
    rfft : FFT of a real sequence

    Notes
    -----
    The packing of the result is "standard": If ``A = fft(a, n)``, then
    ``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
    positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
    terms, in order of decreasingly negative frequency. So ,for an 8-point
    transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
    To rearrange the fft output so that the zero-frequency component is
    centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    Note that if ``x`` is real-valued, then ``A[j] == A[n-j].conjugate()``.
    If ``x`` is real-valued and ``n`` is even, then ``A[n/2]`` is real.

    If the data type of `x` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time. To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum. If the data is both real and
    symmetrical, the `dct` can again double the efficiency by generating
    half of the spectrum from half of the signal.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fft, ifft
    >>> x = np.arange(5)
    >>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
    True

    """
    return _pocketfft.fft(x, n, axis, None, overwrite_x)


def ifft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete inverse Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)``, where

    ``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,
        `x` is truncated. If ``n > x.shape[axis]``, `x` is zero-padded.
        The default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the ifft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    ifft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    fft : Forward FFT

    Notes
    -----
    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    If the data type of `x` is real, a "real IFFT" algorithm is automatically
    used, which roughly halves the computation time.

    Examples
    --------
    >>> from scipy.fftpack import fft, ifft
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.allclose(ifft(fft(x)), x, atol=1e-15)  # within numerical accuracy.
    True

    """
    return _pocketfft.ifft(x, n, axis, None, overwrite_x)


def rfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Discrete Fourier transform of a real sequence.

    Parameters
    ----------
    x : array_like, real-valued
        The data to transform.
    n : int, optional
        Defines the length of the Fourier transform. If `n` is not specified
        (the default) then ``n = x.shape[axis]``. If ``n < x.shape[axis]``,
        `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.
    axis : int, optional
        The axis along which the transform is applied. The default is the
        last axis.
    overwrite_x : bool, optional
        If set to true, the contents of `x` can be overwritten. Default is
        False.

    Returns
    -------
    z : real ndarray
        The returned real array contains::

          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even
          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd

        where::

          y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
          j = 0..n-1

    See Also
    --------
    fft, irfft, scipy.fft.rfft

    Notes
    -----
    Within numerical accuracy, ``y == rfft(irfft(y))``.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    To get an output with a complex datatype, consider using the newer
    function `scipy.fft.rfft`.

    Examples
    --------
    >>> from scipy.fftpack import fft, rfft
    >>> a = [9, -9, 1, 3]
    >>> fft(a)
    array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])
    >>> rfft(a)
    array([  4.,   8.,  12.,  16.])

    """
    return _pocketfft.rfft_fftpack(x, n, axis, None, overwrite_x)


def irfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return inverse discrete Fourier transform of real sequence x.

    The contents of `x` are interpreted as the output of the `rfft`
    function.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.
        If n < x.shape[axis], x is truncated.
        If n > x.shape[axis], x is zero-padded.
        The default results in n = x.shape[axis].
    axis : int, optional
        Axis along which the ifft's are computed; the default is over
        the last axis (i.e., axis=-1).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    irfft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    rfft, ifft, scipy.fft.irfft

    Notes
    -----
    The returned real array contains::

        [y(0),y(1),...,y(n-1)]

    where for n is even::

        y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0] + (-1)**(j) x[n-1])

    and for n is odd::

        y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0])

    c.c. denotes complex conjugate of preceding expression.

    For details on input parameters, see `rfft`.

    To process (conjugate-symmetric) frequency-domain data with a complex
    datatype, consider using the newer function `scipy.fft.irfft`.

    Examples
    --------
    >>> from scipy.fftpack import rfft, irfft
    >>> a = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> irfft(a)
    array([ 2.6       , -3.16405192,  1.24398433, -1.14955713,  1.46962473])
    >>> irfft(rfft(a))
    array([1., 2., 3., 4., 5.])

    """
    return _pocketfft.irfft_fftpack(x, n, axis, None, overwrite_x)


def fftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return multidimensional discrete Fourier transform.

    The returned array contains::

      y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

    where d = len(x.shape) and n = x.shape.

    Parameters
    ----------
    x : array_like
        The (N-D) array to transform.
    shape : int or array_like of ints or None, optional
        The shape of the result. If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
        The default is over all axes.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed. Default is False.

    Returns
    -------
    y : complex-valued N-D NumPy array
        The (N-D) DFT of the input array.

    See Also
    --------
    ifftn

    Notes
    -----
    If ``x`` is real-valued, then
    ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fftn, ifftn
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, fftn(ifftn(y)))
    True

    """
    shape = _good_shape(x, shape, axes)
    return _pocketfft.fftn(x, shape, axes, None, overwrite_x)


def ifftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return inverse multidimensional discrete Fourier transform.

    The sequence can be of an arbitrary type.

    The returned array contains::

      y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)

    where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.

    For description of parameters see `fftn`.

    See Also
    --------
    fftn : for detailed information.

    Examples
    --------
    >>> from scipy.fftpack import fftn, ifftn
    >>> import numpy as np
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, ifftn(fftn(y)))
    True

    """
    shape = _good_shape(x, shape, axes)
    return _pocketfft.ifftn(x, shape, axes, None, overwrite_x)


def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    """
    2-D discrete Fourier transform.

    Return the 2-D discrete Fourier transform of the 2-D argument
    `x`.

    See Also
    --------
    fftn : for detailed information.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fft2, ifft2
    >>> y = np.mgrid[:5, :5][0]
    >>> y
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> np.allclose(y, ifft2(fft2(y)))
    True
    """
    return fftn(x,shape,axes,overwrite_x)


def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    """
    2-D discrete inverse Fourier transform of real or complex sequence.

    Return inverse 2-D discrete Fourier transform of
    arbitrary type sequence x.

    See `ifft` for more information.

    See Also
    --------
    fft2, ifft

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import fft2, ifft2
    >>> y = np.mgrid[:5, :5][0]
    >>> y
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> np.allclose(y, fft2(ifft2(y)))
    True

    """
    return ifftn(x,shape,axes,overwrite_x)
<|endoftext|>import numpy as np
from warnings import warn
from ._basic import rfft, irfft
from ..special import loggamma, poch

from scipy._lib._array_api import array_namespace

__all__ = ['fht', 'ifht', 'fhtoffset']

# constants
LN_2 = np.log(2)


def fht(a, dln, mu, offset=0.0, bias=0.0):
    xp = array_namespace(a)
    a = xp.asarray(a)

    # size of transform
    n = a.shape[-1]

    # bias input array
    if bias != 0:
        # a_q(r) = a(r) (r/r_c)^{-q}
        j_c = (n-1)/2
        j = xp.arange(n, dtype=xp.float64)
        a = a * xp.exp(-bias*(j - j_c)*dln)

    # compute FHT coefficients
    u = xp.asarray(fhtcoeff(n, dln, mu, offset=offset, bias=bias))

    # transform
    A = _fhtq(a, u, xp=xp)

    # bias output array
    if bias != 0:
        # A(k) = A_q(k) (k/k_c)^{-q} (k_c r_c)^{-q}
        A *= xp.exp(-bias*((j - j_c)*dln + offset))

    return A


def ifht(A, dln, mu, offset=0.0, bias=0.0):
    xp = array_namespace(A)
    A = xp.asarray(A)

    # size of transform
    n = A.shape[-1]

    # bias input array
    if bias != 0:
        # A_q(k) = A(k) (k/k_c)^{q} (k_c r_c)^{q}
        j_c = (n-1)/2
        j = xp.arange(n, dtype=xp.float64)
        A = A * xp.exp(bias*((j - j_c)*dln + offset))

    # compute FHT coefficients
    u = xp.asarray(fhtcoeff(n, dln, mu, offset=offset, bias=bias, inverse=True))

    # transform
    a = _fhtq(A, u, inverse=True, xp=xp)

    # bias output array
    if bias != 0:
        # a(r) = a_q(r) (r/r_c)^{q}
        a /= xp.exp(-bias*(j - j_c)*dln)

    return a


def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0, inverse=False):
    """Compute the coefficient array for a fast Hankel transform."""
    lnkr, q = offset, bias

    # Hankel transform coefficients
    # u_m = (kr)^{-i 2m pi/(n dlnr)} U_mu(q + i 2m pi/(n dlnr))
    # with U_mu(x) = 2^x Gamma((mu+1+x)/2)/Gamma((mu+1-x)/2)
    xp = (mu+1+q)/2
    xm = (mu+1-q)/2
    y = np.linspace(0, np.pi*(n//2)/(n*dln), n//2+1)
    u = np.empty(n//2+1, dtype=complex)
    v = np.empty(n//2+1, dtype=complex)
    u.imag[:] = y
    u.real[:] = xm
    loggamma(u, out=v)
    u.real[:] = xp
    loggamma(u, out=u)
    y *= 2*(LN_2 - lnkr)
    u.real -= v.real
    u.real += LN_2*q
    u.imag += v.imag
    u.imag += y
    np.exp(u, out=u)

    # fix last coefficient to be real
    if n % 2 == 0:
        u.imag[-1] = 0

    # deal with special cases
    if not np.isfinite(u[0]):
        # write u_0 = 2^q Gamma(xp)/Gamma(xm) = 2^q poch(xm, xp-xm)
        # poch() handles special cases for negative integers correctly
        u[0] = 2**q * poch(xm, xp-xm)
        # the coefficient may be inf or 0, meaning the transform or the
        # inverse transform, respectively, is singular

    # check for singular transform or singular inverse transform
    if np.isinf(u[0]) and not inverse:
        warn('singular transform; consider changing the bias', stacklevel=3)
        # fix coefficient to obtain (potentially correct) transform anyway
        u = np.copy(u)
        u[0] = 0
    elif u[0] == 0 and inverse:
        warn('singular inverse transform; consider changing the bias', stacklevel=3)
        # fix coefficient to obtain (potentially correct) inverse anyway
        u = np.copy(u)
        u[0] = np.inf

    return u


def fhtoffset(dln, mu, initial=0.0, bias=0.0):
    """Return optimal offset for a fast Hankel transform.

    Returns an offset close to `initial` that fulfils the low-ringing
    condition of [1]_ for the fast Hankel transform `fht` with logarithmic
    spacing `dln`, order `mu` and bias `bias`.

    Parameters
    ----------
    dln : float
        Uniform logarithmic spacing of the transform.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    initial : float, optional
        Initial value for the offset. Returns the closest value that fulfils
        the low-ringing condition.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    offset : float
        Optimal offset of the uniform logarithmic spacing of the transform that
        fulfils a low-ringing condition.

    Examples
    --------
    >>> from scipy.fft import fhtoffset
    >>> dln = 0.1
    >>> mu = 2.0
    >>> initial = 0.5
    >>> bias = 0.0
    >>> offset = fhtoffset(dln, mu, initial, bias)
    >>> offset
    0.5454581477676637

    See Also
    --------
    fht : Definition of the fast Hankel transform.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    """

    lnkr, q = initial, bias

    xp = (mu+1+q)/2
    xm = (mu+1-q)/2
    y = np.pi/(2*dln)
    zp = loggamma(xp + 1j*y)
    zm = loggamma(xm + 1j*y)
    arg = (LN_2 - lnkr)/dln + (zp.imag + zm.imag)/np.pi
    return lnkr + (arg - np.round(arg))*dln


def _fhtq(a, u, inverse=False, *, xp=None):
    """Compute the biased fast Hankel transform.

    This is the basic FFTLog routine.
    """
    if xp is None:
        xp = np

    # size of transform
    n = a.shape[-1]

    # biased fast Hankel transform via real FFT
    A = rfft(a, axis=-1)
    if not inverse:
        # forward transform
        A *= u
    else:
        # backward transform
        A /= xp.conj(u)
    A = irfft(A, n, axis=-1)
    A = xp.flip(A, axis=-1)

    return A
<|endoftext|>from ._basic import _dispatch
from scipy._lib.uarray import Dispatchable
import numpy as np

__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']


@_dispatch
def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, orthogonalize=None):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DCT variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idctn : Inverse multidimensional DCT

    Notes
    -----
    For full details of the DCT types and normalization modes, as well as
    references, see `dct`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None, orthogonalize=None):
    """
    Return multidimensional Inverse Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are
        None, `s` is ``x.shape``; if `s` is None but `axes` is
        not None, then `s` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the IDCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized IDCT variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dctn : multidimensional DCT

    Notes
    -----
    For full details of the IDCT types and normalization modes, as well as
    references, see `idct`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `shape` is -1, the size of the corresponding dimension
        of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DST is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DST variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idstn : Inverse multidimensional DST

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see `dst`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None, orthogonalize=None):
    """
    Return multidimensional Inverse Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension of the input is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension of the input is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the IDST is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized IDST variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dstn : multidimensional DST

    Notes
    -----
    For full details of the IDST types and normalization modes, as well as
    references, see `idst`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fft import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
        orthogonalize=None):
    r"""Return the Discrete Cosine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the dct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DCT variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idct : Inverse DCT

    Notes
    -----
    For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
    MATLAB ``dct(x)``.

    .. warning:: For ``type in {1, 2, 3}``, ``norm="ortho"`` breaks the direct
                 correspondence with the direct Fourier transform. To recover
                 it you must specify ``orthogonalize=False``.

    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 1, 2 and 3 means the transform definition is
    modified to give orthogonality of the DCT matrix (see below).

    For ``norm="backward"``, there is no scaling on `dct` and the `idct` is
    scaled by ``1/N`` where ``N`` is the "logical" size of the DCT. For
    ``norm="forward"`` the ``1/N`` normalization is applied to the forward
    `dct` instead and the `idct` is unnormalized.

    There are, theoretically, 8 types of the DCT, only the first 4 types are
    implemented in SciPy.'The' DCT generally refers to DCT type 2, and 'the'
    Inverse DCT generally refers to DCT type 3.

    **Type I**

    There are several definitions of the DCT-I; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
       \frac{\pi k n}{N-1} \right)

    If ``orthogonalize=True``, ``x[0]`` and ``x[N-1]`` are multiplied by a
    scaling factor of :math:`\sqrt{2}`, and ``y[0]`` and ``y[N-1]`` are divided
    by :math:`\sqrt{2}`. When combined with ``norm="ortho"``, this makes the
    corresponding matrix of coefficients orthonormal (``O @ O.T = np.eye(N)``).

    .. note::
       The DCT-I is only supported for input size > 1.

    **Type II**

    There are several definitions of the DCT-II; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

    If ``orthogonalize=True``, ``y[0]`` is divided by :math:`\sqrt{2}` which,
    when combined with ``norm="ortho"``, makes the corresponding matrix of
    coefficients orthonormal (``O @ O.T = np.eye(N)``).

    **Type III**

    There are several definitions, we use the following (for
    ``norm="backward"``)

    .. math::

       y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

    If ``orthogonalize=True``, ``x[0]`` terms are multiplied by
    :math:`\sqrt{2}` which, when combined with ``norm="ortho"``, makes the
    corresponding matrix of coefficients orthonormal (``O @ O.T = np.eye(N)``).

    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
    to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
    the orthonormalized DCT-II.

    **Type IV**

    There are several definitions of the DCT-IV; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)

    ``orthogonalize`` has no effect here, as the DCT-IV matrix is already
    orthogonal up to a scale factor of ``2N``.

    References
    ----------
    .. [1] 'A Fast Cosine Transform in One and Two Dimensions', by J.
           Makhoul, `IEEE Transactions on acoustics, speech and signal
           processing` vol. 28(1), pp. 27-34,
           :doi:`10.1109/TASSP.1980.1163351` (1980).
    .. [2] Wikipedia, "Discrete cosine transform",
           https://en.wikipedia.org/wiki/Discrete_cosine_transform

    Examples
    --------
    The Type 1 DCT is equivalent to the FFT (though faster) for real,
    even-symmetrical inputs. The output is also real and even-symmetrical.
    Half of the FFT input is used to generate half of the FFT output:

    >>> from scipy.fft import fft, dct
    >>> import numpy as np
    >>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
    array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
    >>> dct(np.array([4., 3., 5., 10.]), 1)
    array([ 30.,  -8.,   6.,  -2.])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    """
    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized IDCT variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    idct : ndarray of real
        The transformed input array.

    See Also
    --------
    dct : Forward DCT

    Notes
    -----
    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
    MATLAB ``idct(x)``.

    .. warning:: For ``type in {1, 2, 3}``, ``norm="ortho"`` breaks the direct
                 correspondence with the inverse direct Fourier transform. To
                 recover it you must specify ``orthogonalize=False``.

    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 1, 2 and 3 means the transform definition is
    modified to give orthogonality of the IDCT matrix (see `dct` for the full
    definitions).

    'The' IDCT is the IDCT-II, which is the same as the normalized DCT-III.

    The IDCT is equivalent to a normal DCT except for the normalization and
    type. DCT type 1 and 4 are their own inverse and DCTs 2 and 3 are each
    other's inverses.

    Examples
    --------
    The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
    inputs. The output is also real and even-symmetrical. Half of the IFFT
    input is used to generate half of the IFFT output:

    >>> from scipy.fft import ifft, idct
    >>> import numpy as np
    >>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
    array([  4.,   3.,   5.,  10.,   5.,   3.])
    >>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1)
    array([  4.,   3.,   5.,  10.])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
        orthogonalize=None):
    r"""
    Return the Discrete Sine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the dst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized DST variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    dst : ndarray of reals
        The transformed input array.

    See Also
    --------
    idst : Inverse DST

    Notes
    -----
    .. warning:: For ``type in {2, 3}``, ``norm="ortho"`` breaks the direct
                 correspondence with the direct Fourier transform. To recover
                 it you must specify ``orthogonalize=False``.

    For ``norm="ortho"`` both the `dst` and `idst` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 2 and 3 means the transform definition is
    modified to give orthogonality of the DST matrix (see below).

    For ``norm="backward"``, there is no scaling on the `dst` and the `idst` is
    scaled by ``1/N`` where ``N`` is the "logical" size of the DST.

    There are, theoretically, 8 types of the DST for different combinations of
    even/odd boundary conditions and boundary off sets [1]_, only the first
    4 types are implemented in SciPy.

    **Type I**

    There are several definitions of the DST-I; we use the following for
    ``norm="backward"``. DST-I assumes the input is odd around :math:`n=-1` and
    :math:`n=N`.

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)

    Note that the DST-I is only supported for input size > 1.
    The (unnormalized) DST-I is its own inverse, up to a factor :math:`2(N+1)`.
    The orthonormalized DST-I is exactly its own inverse.

    ``orthogonalize`` has no effect here, as the DST-I matrix is already
    orthogonal up to a scale factor of ``2N``.

    **Type II**

    There are several definitions of the DST-II; we use the following for
    ``norm="backward"``. DST-II assumes the input is odd around :math:`n=-1/2` and
    :math:`n=N-1/2`; the output is odd around :math:`k=-1` and even around :math:`k=N-1`

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(2n+1)}{2N}\right)

    If ``orthogonalize=True``, ``y[-1]`` is divided :math:`\sqrt{2}` which, when
    combined with ``norm="ortho"``, makes the corresponding matrix of
    coefficients orthonormal (``O @ O.T = np.eye(N)``).

    **Type III**

    There are several definitions of the DST-III, we use the following (for
    ``norm="backward"``). DST-III assumes the input is odd around :math:`n=-1` and
    even around :math:`n=N-1`

    .. math::

        y_k = (-1)^k x_{N-1} + 2 \sum_{n=0}^{N-2} x_n \sin\left(
        \frac{\pi(2k+1)(n+1)}{2N}\right)

    If ``orthogonalize=True``, ``x[-1]`` is multiplied by :math:`\sqrt{2}`
    which, when combined with ``norm="ortho"``, makes the corresponding matrix
    of coefficients orthonormal (``O @ O.T = np.eye(N)``).

    The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up
    to a factor :math:`2N`. The orthonormalized DST-III is exactly the inverse of the
    orthonormalized DST-II.

    **Type IV**

    There are several definitions of the DST-IV, we use the following (for
    ``norm="backward"``). DST-IV assumes the input is odd around :math:`n=-0.5` and
    even around :math:`n=N-0.5`

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)

    ``orthogonalize`` has no effect here, as the DST-IV matrix is already
    orthogonal up to a scale factor of ``2N``.

    The (unnormalized) DST-IV is its own inverse, up to a factor :math:`2N`. The
    orthonormalized DST-IV is exactly its own inverse.

    References
    ----------
    .. [1] Wikipedia, "Discrete sine transform",
           https://en.wikipedia.org/wiki/Discrete_sine_transform

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, orthogonalize=None):
    """
    Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    orthogonalize : bool, optional
        Whether to use the orthogonalized IDST variant (see Notes).
        Defaults to ``True`` when ``norm="ortho"`` and ``False`` otherwise.

        .. versionadded:: 1.8.0

    Returns
    -------
    idst : ndarray of real
        The transformed input array.

    See Also
    --------
    dst : Forward DST

    Notes
    -----
    .. warning:: For ``type in {2, 3}``, ``norm="ortho"`` breaks the direct
                 correspondence with the inverse direct Fourier transform.

    For ``norm="ortho"`` both the `dst` and `idst` are scaled by the same
    overall factor in both directions. By default, the transform is also
    orthogonalized which for types 2 and 3 means the transform definition is
    modified to give orthogonality of the DST matrix (see `dst` for the full
    definitions).

    'The' IDST is the IDST-II, which is the same as the normalized DST-III.

    The IDST is equivalent to a normal DST except for the normalization and
    type. DST type 1 and 4 are their own inverse and DSTs 2 and 3 are each
    other's inverses.

    """
    return (Dispatchable(x, np.ndarray),)
<|endoftext|>"""Fast Hankel transforms using the FFTLog algorithm.

The implementation closely follows the Fortran code of Hamilton (2000).

added: 14/11/2020 Nicolas Tessore <n.tessore@ucl.ac.uk>
"""

from ._basic import _dispatch
from scipy._lib.uarray import Dispatchable
from ._fftlog_backend import fhtoffset
import numpy as np

__all__ = ['fht', 'ifht', 'fhtoffset']


@_dispatch
def fht(a, dln, mu, offset=0.0, bias=0.0):
    r'''Compute the fast Hankel transform.

    Computes the discrete Hankel transform of a logarithmically spaced periodic
    sequence using the FFTLog algorithm [1]_, [2]_.

    Parameters
    ----------
    a : array_like (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    A : array_like (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.

    See Also
    --------
    ifht : The inverse of `fht`.
    fhtoffset : Return an optimal offset for `fht`.

    Notes
    -----
    This function computes a discrete version of the Hankel transform

    .. math::

        A(k) = \int_{0}^{\infty} \! a(r) \, J_\mu(kr) \, k \, dr \;,

    where :math:`J_\mu` is the Bessel function of order :math:`\mu`.  The index
    :math:`\mu` may be any real number, positive or negative.  Note that the
    numerical Hankel transform uses an integrand of :math:`k \, dr`, while the
    mathematical Hankel transform is commonly defined using :math:`r \, dr`.

    The input array `a` is a periodic sequence of length :math:`n`, uniformly
    logarithmically spaced with spacing `dln`,

    .. math::

        a_j = a(r_j) \;, \quad
        r_j = r_c \exp[(j-j_c) \, \mathtt{dln}]

    centred about the point :math:`r_c`.  Note that the central index
    :math:`j_c = (n-1)/2` is half-integral if :math:`n` is even, so that
    :math:`r_c` falls between two input elements.  Similarly, the output
    array `A` is a periodic sequence of length :math:`n`, also uniformly
    logarithmically spaced with spacing `dln`

    .. math::

       A_j = A(k_j) \;, \quad
       k_j = k_c \exp[(j-j_c) \, \mathtt{dln}]

    centred about the point :math:`k_c`.

    The centre points :math:`r_c` and :math:`k_c` of the periodic intervals may
    be chosen arbitrarily, but it would be usual to choose the product
    :math:`k_c r_c = k_j r_{n-1-j} = k_{n-1-j} r_j` to be unity.  This can be
    changed using the `offset` parameter, which controls the logarithmic offset
    :math:`\log(k_c) = \mathtt{offset} - \log(r_c)` of the output array.
    Choosing an optimal value for `offset` may reduce ringing of the discrete
    Hankel transform.

    If the `bias` parameter is nonzero, this function computes a discrete
    version of the biased Hankel transform

    .. math::

        A(k) = \int_{0}^{\infty} \! a_q(r) \, (kr)^q \, J_\mu(kr) \, k \, dr

    where :math:`q` is the value of `bias`, and a power law bias
    :math:`a_q(r) = a(r) \, (kr)^{-q}` is applied to the input sequence.
    Biasing the transform can help approximate the continuous transform of
    :math:`a(r)` if there is a value :math:`q` such that :math:`a_q(r)` is
    close to a periodic sequence, in which case the resulting :math:`A(k)` will
    be close to the continuous transform.

    References
    ----------
    .. [1] Talman J. D., 1978, J. Comp. Phys., 29, 35
    .. [2] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    Examples
    --------

    This example is the adapted version of ``fftlogtest.f`` which is provided
    in [2]_. It evaluates the integral

    .. math::

        \int^\infty_0 r^{\mu+1} \exp(-r^2/2) J_\mu(kr) k dr
        = k^{\mu+1} \exp(-k^2/2) .

    >>> import numpy as np
    >>> from scipy import fft
    >>> import matplotlib.pyplot as plt

    Parameters for the transform.

    >>> mu = 0.0                     # Order mu of Bessel function
    >>> r = np.logspace(-7, 1, 128)  # Input evaluation points
    >>> dln = np.log(r[1]/r[0])      # Step size
    >>> offset = fft.fhtoffset(dln, initial=-6*np.log(10), mu=mu)
    >>> k = np.exp(offset)/r[::-1]   # Output evaluation points

    Define the analytical function.

    >>> def f(x, mu):
    ...     """Analytical function: x^(mu+1) exp(-x^2/2)."""
    ...     return x**(mu + 1)*np.exp(-x**2/2)

    Evaluate the function at ``r`` and compute the corresponding values at
    ``k`` using FFTLog.

    >>> a_r = f(r, mu)
    >>> fht = fft.fht(a_r, dln, mu=mu, offset=offset)

    For this example we can actually compute the analytical response (which in
    this case is the same as the input function) for comparison and compute the
    relative error.

    >>> a_k = f(k, mu)
    >>> rel_err = abs((fht-a_k)/a_k)

    Plot the result.

    >>> figargs = {'sharex': True, 'sharey': True, 'constrained_layout': True}
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), **figargs)
    >>> ax1.set_title(r'$r^{\mu+1}\ \exp(-r^2/2)$')
    >>> ax1.loglog(r, a_r, 'k', lw=2)
    >>> ax1.set_xlabel('r')
    >>> ax2.set_title(r'$k^{\mu+1} \exp(-k^2/2)$')
    >>> ax2.loglog(k, a_k, 'k', lw=2, label='Analytical')
    >>> ax2.loglog(k, fht, 'C3--', lw=2, label='FFTLog')
    >>> ax2.set_xlabel('k')
    >>> ax2.legend(loc=3, framealpha=1)
    >>> ax2.set_ylim([1e-10, 1e1])
    >>> ax2b = ax2.twinx()
    >>> ax2b.loglog(k, rel_err, 'C0', label='Rel. Error (-)')
    >>> ax2b.set_ylabel('Rel. Error (-)', color='C0')
    >>> ax2b.tick_params(axis='y', labelcolor='C0')
    >>> ax2b.legend(loc=4, framealpha=1)
    >>> ax2b.set_ylim([1e-9, 1e-3])
    >>> plt.show()

    '''
    return (Dispatchable(a, np.ndarray),)


@_dispatch
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    r"""Compute the inverse fast Hankel transform.

    Computes the discrete inverse Hankel transform of a logarithmically spaced
    periodic sequence. This is the inverse operation to `fht`.

    Parameters
    ----------
    A : array_like (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    a : array_like (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.

    See Also
    --------
    fht : Definition of the fast Hankel transform.
    fhtoffset : Return an optimal offset for `ifht`.

    Notes
    -----
    This function computes a discrete version of the Hankel transform

    .. math::

        a(r) = \int_{0}^{\infty} \! A(k) \, J_\mu(kr) \, r \, dk \;,

    where :math:`J_\mu` is the Bessel function of order :math:`\mu`.  The index
    :math:`\mu` may be any real number, positive or negative. Note that the
    numerical inverse Hankel transform uses an integrand of :math:`r \, dk`, while the
    mathematical inverse Hankel transform is commonly defined using :math:`k \, dk`.

    See `fht` for further details.
    """
    return (Dispatchable(A, np.ndarray),)
<|endoftext|>from scipy._lib._array_api import (
    array_namespace, is_numpy, xp_unsupported_param_msg, is_complex, xp_float_to_complex
)
from . import _pocketfft
import numpy as np


def _validate_fft_args(workers, plan, norm):
    if workers is not None:
        raise ValueError(xp_unsupported_param_msg("workers"))
    if plan is not None:
        raise ValueError(xp_unsupported_param_msg("plan"))
    if norm is None:
        norm = 'backward'
    return norm


# these functions expect complex input in the fft standard extension
complex_funcs = {'fft', 'ifft', 'fftn', 'ifftn', 'hfft', 'irfft', 'irfftn'}

# pocketfft is used whenever SCIPY_ARRAY_API is not set,
# or x is a NumPy array or array-like.
# When SCIPY_ARRAY_API is set, we try to use xp.fft for CuPy arrays,
# PyTorch arrays and other array API standard supporting objects.
# If xp.fft does not exist, we attempt to convert to np and back to use pocketfft.

def _execute_1D(func_str, pocketfft_func, x, n, axis, norm, overwrite_x, workers, plan):
    xp = array_namespace(x)

    if is_numpy(xp):
        x = np.asarray(x)
        return pocketfft_func(x, n=n, axis=axis, norm=norm,
                              overwrite_x=overwrite_x, workers=workers, plan=plan)

    norm = _validate_fft_args(workers, plan, norm)
    if hasattr(xp, 'fft'):
        xp_func = getattr(xp.fft, func_str)
        if func_str in complex_funcs:
            try:
                res = xp_func(x, n=n, axis=axis, norm=norm)
            except: # backends may require complex input  # noqa: E722
                x = xp_float_to_complex(x, xp)
                res = xp_func(x, n=n, axis=axis, norm=norm)
            return res
        return xp_func(x, n=n, axis=axis, norm=norm)

    x = np.asarray(x)
    y = pocketfft_func(x, n=n, axis=axis, norm=norm)
    return xp.asarray(y)


def _execute_nD(func_str, pocketfft_func, x, s, axes, norm, overwrite_x, workers, plan):
    xp = array_namespace(x)
    
    if is_numpy(xp):
        x = np.asarray(x)
        return pocketfft_func(x, s=s, axes=axes, norm=norm,
                              overwrite_x=overwrite_x, workers=workers, plan=plan)

    norm = _validate_fft_args(workers, plan, norm)
    if hasattr(xp, 'fft'):
        xp_func = getattr(xp.fft, func_str)
        if func_str in complex_funcs:
            try:
                res = xp_func(x, s=s, axes=axes, norm=norm)
            except: # backends may require complex input  # noqa: E722
                x = xp_float_to_complex(x, xp)
                res = xp_func(x, s=s, axes=axes, norm=norm)
            return res
        return xp_func(x, s=s, axes=axes, norm=norm)

    x = np.asarray(x)
    y = pocketfft_func(x, s=s, axes=axes, norm=norm)
    return xp.asarray(y)


def fft(x, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('fft', _pocketfft.fft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    return _execute_1D('ifft', _pocketfft.ifft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def rfft(x, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('rfft', _pocketfft.rfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def irfft(x, n=None, axis=-1, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('irfft', _pocketfft.irfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def hfft(x, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('hfft', _pocketfft.hfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def ihfft(x, n=None, axis=-1, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('ihfft', _pocketfft.ihfft, x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def fftn(x, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('fftn', _pocketfft.fftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)



def ifftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('ifftn', _pocketfft.ifftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def fft2(x, s=None, axes=(-2, -1), norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return fftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def ifft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return ifftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def rfftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('rfftn', _pocketfft.rfftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def rfft2(x, s=None, axes=(-2, -1), norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return rfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def irfftn(x, s=None, axes=None, norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('irfftn', _pocketfft.irfftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def irfft2(x, s=None, axes=(-2, -1), norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return irfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def _swap_direction(norm):
    if norm in (None, 'backward'):
        norm = 'forward'
    elif norm == 'forward':
        norm = 'backward'
    elif norm != 'ortho':
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho", or "forward".')
    return norm


def hfftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    xp = array_namespace(x)
    if is_numpy(xp):
        x = np.asarray(x)
        return _pocketfft.hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
    if is_complex(x, xp):
        x = xp.conj(x)
    return irfftn(x, s, axes, _swap_direction(norm),
                  overwrite_x, workers, plan=plan)


def hfft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def ihfftn(x, s=None, axes=None, norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    xp = array_namespace(x)
    if is_numpy(xp):
        x = np.asarray(x)
        return _pocketfft.ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
    return xp.conj(rfftn(x, s, axes, _swap_direction(norm),
                         overwrite_x, workers, plan=plan))

def ihfft2(x, s=None, axes=(-2, -1), norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
<|endoftext|>import numpy as np

class NumPyBackend:
    """Backend that uses numpy.fft"""
    __ua_domain__ = "numpy.scipy.fft"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        kwargs.pop("overwrite_x", None)

        fn = getattr(np.fft, method.__name__, None)
        return (NotImplemented if fn is None
                else fn(*args, **kwargs))


class EchoBackend:
    """Backend that just prints the __ua_function__ arguments"""
    __ua_domain__ = "numpy.scipy.fft"

    @staticmethod
    def __ua_function__(method, args, kwargs):
        print(method, args, kwargs, sep='\n')
<|endoftext|>from scipy._lib._array_api import array_namespace
import numpy as np
from . import _pocketfft

__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']


def _execute(pocketfft_func, x, type, s, axes, norm, 
             overwrite_x, workers, orthogonalize):
    xp = array_namespace(x)
    x = np.asarray(x)
    y = pocketfft_func(x, type, s, axes, norm,
                       overwrite_x=overwrite_x, workers=workers,
                       orthogonalize=orthogonalize)
    return xp.asarray(y)


def dctn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, *, orthogonalize=None):
    return _execute(_pocketfft.dctn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)


def idctn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, orthogonalize=None):
    return _execute(_pocketfft.idctn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)


def dstn(x, type=2, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.dstn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)


def idstn(x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, orthogonalize=None):
    return _execute(_pocketfft.idstn, x, type, s, axes, norm, 
                    overwrite_x, workers, orthogonalize)


def dct(x, type=2, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.dct, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)


def idct(x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.idct, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)


def dst(x, type=2, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.dst, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)


def idst(x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    return _execute(_pocketfft.idst, x, type, n, axis, norm, 
                    overwrite_x, workers, orthogonalize)
<|endoftext|>"""
==============================================
Discrete Fourier transforms (:mod:`scipy.fft`)
==============================================

.. currentmodule:: scipy.fft

Fast Fourier Transforms (FFTs)
==============================

.. autosummary::
   :toctree: generated/

   fft - Fast (discrete) Fourier Transform (FFT)
   ifft - Inverse FFT
   fft2 - 2-D FFT
   ifft2 - 2-D inverse FFT
   fftn - N-D FFT
   ifftn - N-D inverse FFT
   rfft - FFT of strictly real-valued sequence
   irfft - Inverse of rfft
   rfft2 - 2-D FFT of real sequence
   irfft2 - Inverse of rfft2
   rfftn - N-D FFT of real sequence
   irfftn - Inverse of rfftn
   hfft - FFT of a Hermitian sequence (real spectrum)
   ihfft - Inverse of hfft
   hfft2 - 2-D FFT of a Hermitian sequence
   ihfft2 - Inverse of hfft2
   hfftn - N-D FFT of a Hermitian sequence
   ihfftn - Inverse of hfftn

Discrete Sin and Cosine Transforms (DST and DCT)
================================================

.. autosummary::
   :toctree: generated/

   dct - Discrete cosine transform
   idct - Inverse discrete cosine transform
   dctn - N-D Discrete cosine transform
   idctn - N-D Inverse discrete cosine transform
   dst - Discrete sine transform
   idst - Inverse discrete sine transform
   dstn - N-D Discrete sine transform
   idstn - N-D Inverse discrete sine transform

Fast Hankel Transforms
======================

.. autosummary::
   :toctree: generated/

   fht - Fast Hankel transform
   ifht - Inverse of fht

Helper functions
================

.. autosummary::
   :toctree: generated/

   fftshift - Shift the zero-frequency component to the center of the spectrum
   ifftshift - The inverse of `fftshift`
   fftfreq - Return the Discrete Fourier Transform sample frequencies
   rfftfreq - DFT sample frequencies (for usage with rfft, irfft)
   fhtoffset - Compute an optimal offset for the Fast Hankel Transform
   next_fast_len - Find the optimal length to zero-pad an FFT for speed
   prev_fast_len - Find the maximum slice length that results in a fast FFT
   set_workers - Context manager to set default number of workers
   get_workers - Get the current default number of workers

Backend control
===============

.. autosummary::
   :toctree: generated/

   set_backend - Context manager to set the backend within a fixed scope
   skip_backend - Context manager to skip a backend within a fixed scope
   set_global_backend - Sets the global fft backend
   register_backend - Register a backend for permanent use

"""

from ._basic import (
    fft, ifft, fft2, ifft2, fftn, ifftn,
    rfft, irfft, rfft2, irfft2, rfftn, irfftn,
    hfft, ihfft, hfft2, ihfft2, hfftn, ihfftn)
from ._realtransforms import dct, idct, dst, idst, dctn, idctn, dstn, idstn
from ._fftlog import fht, ifht, fhtoffset
from ._helper import (
    next_fast_len, prev_fast_len, fftfreq,
    rfftfreq, fftshift, ifftshift)
from ._backend import (set_backend, skip_backend, set_global_backend,
                       register_backend)
from ._pocketfft.helper import set_workers, get_workers

__all__ = [
    'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
    'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
    'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
    'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift',
    'next_fast_len', 'prev_fast_len',
    'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
    'fht', 'ifht',
    'fhtoffset',
    'set_backend', 'skip_backend', 'set_global_backend', 'register_backend',
    'get_workers', 'set_workers']


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>from functools import update_wrapper, lru_cache
import inspect

from ._pocketfft import helper as _helper

import numpy as np
from scipy._lib._array_api import array_namespace


def next_fast_len(target, real=False):
    """Find the next fast size of input data to ``fft``, for zero-padding, etc.

    SciPy's FFT algorithms gain their speed by a recursive divide and conquer
    strategy. This relies on efficient functions for small prime factors of the
    input length. Thus, the transforms are fastest when using composites of the
    prime factors handled by the fft implementation. If there are efficient
    functions for all radices <= `n`, then the result will be a number `x`
    >= ``target`` with only prime factors < `n`. (Also known as `n`-smooth
    numbers)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output (e.g., `rfft` or `hfft`
        but not `fft`). Defaults to False.

    Returns
    -------
    out : int
        The smallest fast length greater than or equal to ``target``.

    Notes
    -----
    The result of this function may change in future as performance
    considerations change, for example, if new prime factors are added.

    Calling `fft` or `ifft` with real input data performs an ``'R2C'``
    transform internally.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 11.4 ms:

    >>> from scipy import fft
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> min_len = 93059  # prime length is worst case for speed
    >>> a = rng.standard_normal(min_len)
    >>> b = fft.fft(a)

    Zero-padding to the next regular length reduces computation time to
    1.6 ms, a speedup of 7.3 times:

    >>> fft.next_fast_len(min_len, real=True)
    93312
    >>> b = fft.fft(a, 93312)

    Rounding up to the next power of 2 is not optimal, taking 3.0 ms to
    compute; 1.9 times longer than the size given by ``next_fast_len``:

    >>> b = fft.fft(a, 131072)

    """
    pass


# Directly wrap the c-function good_size but take the docstring etc., from the
# next_fast_len function above
_sig = inspect.signature(next_fast_len)
next_fast_len = update_wrapper(lru_cache(_helper.good_size), next_fast_len)
next_fast_len.__wrapped__ = _helper.good_size
next_fast_len.__signature__ = _sig


def prev_fast_len(target, real=False):
    """Find the previous fast size of input data to ``fft``.
    Useful for discarding a minimal number of samples before FFT.

    SciPy's FFT algorithms gain their speed by a recursive divide and conquer
    strategy. This relies on efficient functions for small prime factors of the
    input length. Thus, the transforms are fastest when using composites of the
    prime factors handled by the fft implementation. If there are efficient
    functions for all radices <= `n`, then the result will be a number `x`
    <= ``target`` with only prime factors <= `n`. (Also known as `n`-smooth
    numbers)

    Parameters
    ----------
    target : int
        Maximum length to search until. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output (e.g., `rfft` or `hfft`
        but not `fft`). Defaults to False.

    Returns
    -------
    out : int
        The largest fast length less than or equal to ``target``.

    Notes
    -----
    The result of this function may change in future as performance
    considerations change, for example, if new prime factors are added.

    Calling `fft` or `ifft` with real input data performs an ``'R2C'``
    transform internally.

    In the current implementation, prev_fast_len assumes radices of
    2,3,5,7,11 for complex FFT and 2,3,5 for real FFT.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 16.2 ms:

    >>> from scipy import fft
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> max_len = 93059  # prime length is worst case for speed
    >>> a = rng.standard_normal(max_len)
    >>> b = fft.fft(a)

    Performing FFT on the maximum fast length less than max_len
    reduces the computation time to 1.5 ms, a speedup of 10.5 times:

    >>> fft.prev_fast_len(max_len, real=True)
    92160
    >>> c = fft.fft(a[:92160]) # discard last 899 samples

    """
    pass


# Directly wrap the c-function prev_good_size but take the docstring etc.,
# from the prev_fast_len function above
_sig_prev_fast_len = inspect.signature(prev_fast_len)
prev_fast_len = update_wrapper(lru_cache()(_helper.prev_good_size), prev_fast_len)
prev_fast_len.__wrapped__ = _helper.prev_good_size
prev_fast_len.__signature__ = _sig_prev_fast_len


def _init_nd_shape_and_axes(x, shape, axes):
    """Handle shape and axes arguments for N-D transforms.

    Returns the shape and axes in a standard form, taking into account negative
    values and checking for various potential errors.

    Parameters
    ----------
    x : array_like
        The input array.
    shape : int or array_like of ints or None
        The shape of the result. If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If `shape` is -1, the size of the corresponding dimension of `x` is
        used.
    axes : int or array_like of ints or None
        Axes along which the calculation is computed.
        The default is over all axes.
        Negative indices are automatically converted to their positive
        counterparts.

    Returns
    -------
    shape : tuple
        The shape of the result as a tuple of integers.
    axes : list
        Axes along which the calculation is computed, as a list of integers.

    """
    x = np.asarray(x)
    return _helper._init_nd_shape_and_axes(x, shape, axes)


def fftfreq(n, d=1.0, *, xp=None, device=None):
    """Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    xp : array_namespace, optional
        The namespace for the return array. Default is None, where NumPy is used.
    device : device, optional
        The device for the return array.
        Only valid when `xp.fft.fftfreq` implements the device parameter.
     
    Returns
    -------
    f : ndarray
        Array of length `n` containing the sample frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.fft
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = scipy.fft.fft(signal)
    >>> n = signal.size
    >>> timestep = 0.1
    >>> freq = scipy.fft.fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """
    xp = np if xp is None else xp
    # numpy does not yet support the `device` keyword
    # `xp.__name__ != 'numpy'` should be removed when numpy is compatible
    if hasattr(xp, 'fft') and xp.__name__ != 'numpy':
        return xp.fft.fftfreq(n, d=d, device=device)
    if device is not None:
        raise ValueError('device parameter is not supported for input array type')
    return np.fft.fftfreq(n, d=d)


def rfftfreq(n, d=1.0, *, xp=None, device=None):
    """Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    xp : array_namespace, optional
        The namespace for the return array. Default is None, where NumPy is used.
    device : device, optional
        The device for the return array.
        Only valid when `xp.fft.rfftfreq` implements the device parameter.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.fft
    >>> signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, -3, 4], dtype=float)
    >>> fourier = scipy.fft.rfft(signal)
    >>> n = signal.size
    >>> sample_rate = 100
    >>> freq = scipy.fft.fftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20., ..., -30., -20., -10.])
    >>> freq = scipy.fft.rfftfreq(n, d=1./sample_rate)
    >>> freq
    array([  0.,  10.,  20.,  30.,  40.,  50.])

    """
    xp = np if xp is None else xp
    # numpy does not yet support the `device` keyword
    # `xp.__name__ != 'numpy'` should be removed when numpy is compatible
    if hasattr(xp, 'fft') and xp.__name__ != 'numpy':
        return xp.fft.rfftfreq(n, d=d, device=device)
    if device is not None:
        raise ValueError('device parameter is not supported for input array type')
    return np.fft.rfftfreq(n, d=d)


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    >>> import numpy as np
    >>> freqs = np.fft.fftfreq(10, 0.1)
    >>> freqs
    array([ 0.,  1.,  2., ..., -3., -2., -1.])
    >>> np.fft.fftshift(freqs)
    array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Shift the zero-frequency component only along the second axis:

    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.fftshift(freqs, axes=(1,))
    array([[ 2.,  0.,  1.],
           [-4.,  3.,  4.],
           [-1., -3., -2.]])

    """
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.fftshift(x, axes=axes)
    x = np.asarray(x)
    y = np.fft.fftshift(x, axes=axes)
    return xp.asarray(y)


def ifftshift(x, axes=None):
    """The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    See Also
    --------
    fftshift : Shift zero-frequency component to the center of the spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    >>> freqs
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])
    >>> np.fft.ifftshift(np.fft.fftshift(freqs))
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -4.],
           [-3., -2., -1.]])

    """
    xp = array_namespace(x)
    if hasattr(xp, 'fft'):
        return xp.fft.ifftshift(x, axes=axes)
    x = np.asarray(x)
    y = np.fft.ifftshift(x, axes=axes)
    return xp.asarray(y)
<|endoftext|>import scipy._lib.uarray as ua
from . import _basic_backend
from . import _realtransforms_backend
from . import _fftlog_backend


class _ScipyBackend:
    """The default backend for fft calculations

    Notes
    -----
    We use the domain ``numpy.scipy`` rather than ``scipy`` because ``uarray``
    treats the domain as a hierarchy. This means the user can install a single
    backend for ``numpy`` and have it implement ``numpy.scipy.fft`` as well.
    """
    __ua_domain__ = "numpy.scipy.fft"

    @staticmethod
    def __ua_function__(method, args, kwargs):

        fn = getattr(_basic_backend, method.__name__, None)
        if fn is None:
            fn = getattr(_realtransforms_backend, method.__name__, None)
        if fn is None:
            fn = getattr(_fftlog_backend, method.__name__, None)
        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)


_named_backends = {
    'scipy': _ScipyBackend,
}


def _backend_from_arg(backend):
    """Maps strings to known backends and validates the backend"""

    if isinstance(backend, str):
        try:
            backend = _named_backends[backend]
        except KeyError as e:
            raise ValueError(f'Unknown backend {backend}') from e

    if backend.__ua_domain__ != 'numpy.scipy.fft':
        raise ValueError('Backend does not implement "numpy.scipy.fft"')

    return backend


def set_global_backend(backend, coerce=False, only=False, try_last=False):
    """Sets the global fft backend

    This utility method replaces the default backend for permanent use. It
    will be tried in the list of backends automatically, unless the
    ``only`` flag is set on a backend. This will be the first tried
    backend outside the :obj:`set_backend` context manager.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool
        Whether to coerce input types when trying this backend.
    only : bool
        If ``True``, no more backends will be tried if this fails.
        Implied by ``coerce=True``.
    try_last : bool
        If ``True``, the global backend is tried after registered backends.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.scipy.fft``.

    Notes
    -----
    This will overwrite the previously set global backend, which, by default, is
    the SciPy implementation.

    Examples
    --------
    We can set the global fft backend:

    >>> from scipy.fft import fft, set_global_backend
    >>> set_global_backend("scipy")  # Sets global backend (default is "scipy").
    >>> fft([1])  # Calls the global backend
    array([1.+0.j])
    """
    backend = _backend_from_arg(backend)
    ua.set_global_backend(backend, coerce=coerce, only=only, try_last=try_last)


def register_backend(backend):
    """
    Register a backend for permanent use.

    Registered backends have the lowest priority and will be tried after the
    global backend.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Raises
    ------
    ValueError: If the backend does not implement ``numpy.scipy.fft``.

    Examples
    --------
    We can register a new fft backend:

    >>> from scipy.fft import fft, register_backend, set_global_backend
    >>> class NoopBackend:  # Define an invalid Backend
    ...     __ua_domain__ = "numpy.scipy.fft"
    ...     def __ua_function__(self, func, args, kwargs):
    ...          return NotImplemented
    >>> set_global_backend(NoopBackend())  # Set the invalid backend as global
    >>> register_backend("scipy")  # Register a new backend
    # The registered backend is called because
    # the global backend returns `NotImplemented`
    >>> fft([1])
    array([1.+0.j])
    >>> set_global_backend("scipy")  # Restore global backend to default

    """
    backend = _backend_from_arg(backend)
    ua.register_backend(backend)


def set_backend(backend, coerce=False, only=False):
    """Context manager to set the backend within a fixed scope.

    Upon entering the ``with`` statement, the given backend will be added to
    the list of available backends with the highest priority. Upon exit, the
    backend is reset to the state before entering the scope.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to use.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.
    coerce : bool, optional
        Whether to allow expensive conversions for the ``x`` parameter. e.g.,
        copying a NumPy array to the GPU for a CuPy backend. Implies ``only``.
    only : bool, optional
        If only is ``True`` and this backend returns ``NotImplemented``, then a
        BackendNotImplemented error will be raised immediately. Ignoring any
        lower priority backends.

    Examples
    --------
    >>> import scipy.fft as fft
    >>> with fft.set_backend('scipy', only=True):
    ...     fft.fft([1])  # Always calls the scipy implementation
    array([1.+0.j])
    """
    backend = _backend_from_arg(backend)
    return ua.set_backend(backend, coerce=coerce, only=only)


def skip_backend(backend):
    """Context manager to skip a backend within a fixed scope.

    Within the context of a ``with`` statement, the given backend will not be
    called. This covers backends registered both locally and globally. Upon
    exit, the backend will again be considered.

    Parameters
    ----------
    backend : {object, 'scipy'}
        The backend to skip.
        Can either be a ``str`` containing the name of a known backend
        {'scipy'} or an object that implements the uarray protocol.

    Examples
    --------
    >>> import scipy.fft as fft
    >>> fft.fft([1])  # Calls default SciPy backend
    array([1.+0.j])
    >>> with fft.skip_backend('scipy'):  # We explicitly skip the SciPy backend
    ...     fft.fft([1])                 # leaving no implementation available
    Traceback (most recent call last):
        ...
    BackendNotImplementedError: No selected backends had an implementation ...
    """
    backend = _backend_from_arg(backend)
    return ua.skip_backend(backend)


set_global_backend('scipy', try_last=True)
<|endoftext|>from scipy._lib.uarray import generate_multimethod, Dispatchable
import numpy as np


def _x_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the transform input array (``x``)
    """
    if len(args) > 0:
        return (dispatchables[0],) + args[1:], kwargs
    kw = kwargs.copy()
    kw['x'] = dispatchables[0]
    return args, kw


def _dispatch(func):
    """
    Function annotation that creates a uarray multimethod from the function
    """
    return generate_multimethod(func, _x_replacer, domain="numpy.scipy.fft")


@_dispatch
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
        plan=None):
    """
    Compute the 1-D discrete Fourier Transform.

    This function computes the 1-D *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [1]_.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode. Default is "backward", meaning no normalization on
        the forward transforms and scaling by ``1/n`` on the `ifft`.
        "forward" instead applies the ``1/n`` factor on the forward transform.
        For ``norm="ortho"``, both directions are scaled by ``1/sqrt(n)``.

        .. versionadded:: 1.6.0
           ``norm={"forward", "backward"}`` options were added

    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See the notes below for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``. See below for more
        details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        if `axes` is larger than the last axis of `x`.

    See Also
    --------
    ifft : The inverse of `fft`.
    fft2 : The 2-D FFT.
    fftn : The N-D FFT.
    rfftn : The N-D FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.
    next_fast_len : Size to pad input to for most efficient transforms

    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform
    (DFT) can be calculated efficiently, by using symmetries in the calculated
    terms. The symmetry is highest when `n` is a power of 2, and the transform
    is therefore most efficient for these sizes. For poorly factorizable sizes,
    `scipy.fft` uses Bluestein's algorithm [2]_ and so is never worse than
    O(`n` log `n`). Further performance improvements may be seen by zero-padding
    the input using `next_fast_len`.

    If ``x`` is a 1d array, then the `fft` is equivalent to ::

        y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))

    The frequency term ``f=k/n`` is found at ``y[k]``. At ``y[n/2]`` we reach
    the Nyquist frequency and wrap around to the negative-frequency terms. So,
    for an 8-point transform, the frequencies of the result are
    [0, 1, 2, 3, -4, -3, -2, -1]. To rearrange the fft output so that the
    zero-frequency component is centered, like [-4, -3, -2, -1, 0, 1, 2, 3],
    use `fftshift`.

    Transforms can be done in single, double, or extended precision (long
    double) floating point. Half precision inputs will be converted to single
    precision and non-floating-point inputs will be converted to double
    precision.

    If the data type of ``x`` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time. To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum. If the data are both real and
    symmetrical, the `dct` can again double the efficiency, by generating
    half of the spectrum from half of the signal.

    When ``overwrite_x=True`` is specified, the memory referenced by ``x`` may
    be used by the implementation in any way. This may include reusing the
    memory for the result, but this is in no way guaranteed. You should not
    rely on the contents of ``x`` after the transform as this may change in
    future without warning.

    The ``workers`` argument specifies the maximum number of parallel jobs to
    split the FFT computation into. This will execute independent 1-D
    FFTs within ``x``. So, ``x`` must be at least 2-D and the
    non-transformed axes must be large enough to split into chunks. If ``x`` is
    too small, fewer jobs may be used than requested.

    References
    ----------
    .. [1] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
           machine calculation of complex Fourier series," *Math. Comput.*
           19: 297-301.
    .. [2] Bluestein, L., 1970, "A linear filtering approach to the
           computation of discrete Fourier transform". *IEEE Transactions on
           Audio and Electroacoustics.* 18 (4): 451-455.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> scipy.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])

    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part:

    >>> from scipy.fft import fft, fftfreq, fftshift
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sp = fftshift(fft(np.sin(t)))
    >>> freq = fftshift(fftfreq(t.shape[-1]))
    >>> plt.plot(freq, sp.real, freq, sp.imag)
    [<matplotlib.lines.Line2D object at 0x...>,
     <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 1-D inverse discrete Fourier Transform.

    This function computes the inverse of the 1-D *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(x)) == x`` to within numerical accuracy.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``x[0]`` should contain the zero frequency term,
    * ``x[1:n//2]`` should contain the positive-frequency terms,
    * ``x[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``x[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together. See `fft` for details.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
        See notes about padding issues.
    axis : int, optional
        Axis over which to compute the inverse DFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axes` is larger than the last axis of `x`.

    See Also
    --------
    fft : The 1-D (forward) FFT, of which `ifft` is the inverse.
    ifft2 : The 2-D inverse FFT.
    ifftn : The N-D inverse FFT.

    Notes
    -----
    If the input parameter `n` is larger than the size of the input, the input
    is padded by appending zeros at the end. Even though this is the common
    approach, it might lead to surprising results. If a different padding is
    desired, it must be performed before calling `ifft`.

    If ``x`` is a 1-D array, then the `ifft` is equivalent to ::

        y[k] = np.sum(x * np.exp(2j * np.pi * k * np.arange(n)/n)) / len(x)

    As with `fft`, `ifft` has support for all floating point types and is
    optimized for real input.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> scipy.fft.ifft([0, 4, 0, 0])
    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary

    Create and plot a band-limited signal with random phases:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> t = np.arange(400)
    >>> n = np.zeros((400,), dtype=complex)
    >>> n[40:60] = np.exp(1j*rng.uniform(0, 2*np.pi, (20,)))
    >>> s = scipy.fft.ifft(n)
    >>> plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
    [<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
    >>> plt.legend(('real', 'imaginary'))
    <matplotlib.legend.Legend object at ...>
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 1-D discrete Fourier Transform for real input.

    This function computes the 1-D *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    x : array_like
        Input array
    n : int, optional
        Number of points along transformation axis in the input to use.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        If `n` is even, the length of the transformed axis is ``(n/2)+1``.
        If `n` is odd, the length is ``(n+1)/2``.

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `a`.

    See Also
    --------
    irfft : The inverse of `rfft`.
    fft : The 1-D FFT of general (complex) input.
    fftn : The N-D FFT.
    rfft2 : The 2-D FFT of real input.
    rfftn : The N-D FFT of real input.

    Notes
    -----
    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric, i.e., the negative frequency terms are just the complex
    conjugates of the corresponding positive-frequency terms, and the
    negative-frequency terms are therefore redundant. This function does not
    compute the negative frequency terms, and the length of the transformed
    axis of the output is therefore ``n//2 + 1``.

    When ``X = rfft(x)`` and fs is the sampling frequency, ``X[0]`` contains
    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

    If `n` is even, ``A[-1]`` contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
    the largest positive frequency (fs/2*(n-1)/n), and is complex in the
    general case.

    If the input `a` contains an imaginary part, it is silently discarded.

    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.fft([0, 1, 0, 0])
    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary
    >>> scipy.fft.rfft([0, 1, 0, 0])
    array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary

    Notice how the final element of the `fft` output is the complex conjugate
    of the second element, for real input. For `rfft`, this symmetry is
    exploited to compute only the non-negative frequency terms.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Computes the inverse of `rfft`.

    This function computes the inverse of the 1-D *n*-point
    discrete Fourier Transform of real input computed by `rfft`.
    In other words, ``irfft(rfft(x), len(x)) == x`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by `rfft`, i.e., the
    real zero-frequency term followed by the complex positive frequency terms
    in order of increasing frequency. Since the discrete Fourier Transform of
    real input is Hermitian-symmetric, the negative frequency terms are taken
    to be the complex conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    x : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary. If the
        input is longer than this, it is cropped. If it is shorter than this,
        it is padded with zeros. If `n` is not given, it is taken to be
        ``2*(m-1)``, where ``m`` is the length of the input along the axis
        specified by `axis`.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified.

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `x`.

    See Also
    --------
    rfft : The 1-D FFT of real input, of which `irfft` is inverse.
    fft : The 1-D FFT.
    irfft2 : The inverse of the 2-D FFT of real input.
    irfftn : The inverse of the N-D FFT of real input.

    Notes
    -----
    Returns the real valued `n`-point inverse discrete Fourier transform
    of `x`, where `x` contains the non-negative frequency terms of a
    Hermitian-symmetric sequence. `n` is the length of the result, not the
    input.

    If you specify an `n` such that `a` must be zero-padded or truncated, the
    extra/removed values will be added/removed at high frequencies. One can
    thus resample a series to `m` points via Fourier interpolation by:
    ``a_resamp = irfft(rfft(a), m)``.

    The default value of `n` assumes an even output length. By the Hermitian
    symmetry, the last imaginary component must be 0 and so is ignored. To
    avoid losing information, the correct length of the real input *must* be
    given.

    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.ifft([1, -1j, -1, 1j])
    array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary
    >>> scipy.fft.irfft([1, -1j, -1])
    array([0.,  1.,  0.,  0.])

    Notice how the last term in the input to the ordinary `ifft` is the
    complex conjugate of the second term, and the output has zero imaginary
    part everywhere. When calling `irfft`, the negative frequencies are not
    specified, and the output array is purely real.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
    spectrum.

    Parameters
    ----------
    x : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output. For `n` output
        points, ``n//2 + 1`` input points are necessary. If the input is
        longer than this, it is cropped. If it is shorter than this, it is
        padded with zeros. If `n` is not given, it is taken to be ``2*(m-1)``,
        where ``m`` is the length of the input along the axis specified by
        `axis`.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*m - 2``, where ``m`` is the length of the transformed axis of
        the input. To get an odd number of output points, `n` must be
        specified, for instance, as ``2*m - 1`` in the typical case,

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `a`.

    See Also
    --------
    rfft : Compute the 1-D FFT for real input.
    ihfft : The inverse of `hfft`.
    hfftn : Compute the N-D FFT of a Hermitian signal.

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So, here, it's `hfft`, for
    which you must supply the length of the result if it is to be odd.
    * even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

    Examples
    --------
    >>> from scipy.fft import fft, hfft
    >>> import numpy as np
    >>> a = 2 * np.pi * np.arange(10) / 10
    >>> signal = np.cos(a) + 3j * np.sin(3 * a)
    >>> fft(signal).round(10)
    array([ -0.+0.j,   5.+0.j,  -0.+0.j,  15.-0.j,   0.+0.j,   0.+0.j,
            -0.+0.j, -15.-0.j,   0.+0.j,   5.+0.j])
    >>> hfft(signal[:6]).round(10) # Input first half of signal
    array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
    >>> hfft(signal, 10)  # Input entire signal and truncate
    array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    x : array_like
        Input array.
    n : int, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use.  If `n` is smaller than
        the length of the input, the input is cropped. If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    See Also
    --------
    hfft, irfft

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here, the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So, here, it's `hfft`, for
    which you must supply the length of the result if it is to be odd:
    * even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

    Examples
    --------
    >>> from scipy.fft import ifft, ihfft
    >>> import numpy as np
    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
    >>> ifft(spectrum)
    array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # may vary
    >>> ihfft(spectrum)
    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # may vary
    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the N-D discrete Fourier Transform.

    This function computes the N-D discrete Fourier Transform over
    any number of axes in an M-D array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    ifftn : The inverse of `fftn`, the inverse N-D FFT.
    fft : The 1-D FFT, with definitions and conventions used.
    rfftn : The N-D FFT of real input.
    fft2 : The 2-D FFT.
    fftshift : Shifts zero-frequency terms to centre of array.

    Notes
    -----
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of all axes, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.mgrid[:3, :3, :3][0]
    >>> scipy.fft.fftn(x, axes=(1, 2))
    array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[ 9.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[18.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]]])
    >>> scipy.fft.fftn(x, (2, 2), axes=(0, 1))
    array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary
            [ 0.+0.j,  0.+0.j,  0.+0.j]],
           [[-2.+0.j, -2.+0.j, -2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]]])

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
    ...                      2 * np.pi * np.arange(200) / 34)
    >>> S = np.sin(X) + np.cos(Y) + rng.uniform(0, 1, X.shape)
    >>> FS = scipy.fft.fftn(S)
    >>> plt.imshow(np.log(np.abs(scipy.fft.fftshift(FS))**2))
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D inverse discrete Fourier Transform.

    This function computes the inverse of the N-D discrete
    Fourier Transform over any number of axes in an M-D array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(x)) == x`` to within numerical accuracy.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e., it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used. See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    fftn : The forward N-D FFT, of which `ifftn` is the inverse.
    ifft : The 1-D inverse FFT.
    ifft2 : The 2-D inverse FFT.
    ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning
        of array.

    Notes
    -----
    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension. Although this is the common
    approach, it might lead to surprising results. If another form of zero
    padding is desired, it must be performed before `ifftn` is called.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.eye(4)
    >>> scipy.fft.ifftn(scipy.fft.fftn(x, axes=(0,)), axes=(1,))
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])


    Create and plot an image with band-limited frequency content:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> n = np.zeros((200,200), dtype=complex)
    >>> n[60:80, 20:40] = np.exp(1j*rng.uniform(0, 2*np.pi, (20, 20)))
    >>> im = scipy.fft.ifftn(n).real
    >>> plt.imshow(im)
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 2-D discrete Fourier Transform

    This function computes the N-D discrete Fourier Transform
    over any axes in an M-D array by means of the
    Fast Fourier Transform (FFT). By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    x : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two axes are
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    ifft2 : The inverse 2-D FFT.
    fft : The 1-D FFT.
    fftn : The N-D FFT.
    fftshift : Shifts zero-frequency terms to the center of the array.
        For 2-D input, swaps first and third quadrants, and second
        and fourth quadrants.

    Notes
    -----
    `fft2` is just `fftn` with a different default for `axes`.

    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.

    See `fftn` for details and a plotting example, and `fft` for
    definitions and conventions used.


    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.mgrid[:5, :5][0]
    >>> scipy.fft.fft2(x)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D inverse discrete Fourier Transform.

    This function computes the inverse of the 2-D discrete Fourier
    Transform over any number of axes in an M-D array by means of
    the Fast Fourier Transform (FFT). In other words, ``ifft2(fft2(x)) == x``
    to within numerical accuracy. By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e., it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two
        axes are used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    fft2 : The forward 2-D FFT, of which `ifft2` is the inverse.
    ifftn : The inverse of the N-D FFT.
    fft : The 1-D FFT.
    ifft : The 1-D inverse FFT.

    Notes
    -----
    `ifft2` is just `ifftn` with a different default for `axes`.

    See `ifftn` for details and a plotting example, and `fft` for
    definition and conventions used.

    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension. Although this is the common
    approach, it might lead to surprising results. If another form of zero
    padding is desired, it must be performed before `ifft2` is called.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = 4 * np.eye(4)
    >>> scipy.fft.ifft2(x)
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D discrete Fourier Transform for real input.

    This function computes the N-D discrete Fourier Transform over
    any number of axes in an M-D real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    irfftn : The inverse of `rfftn`, i.e., the inverse of the N-D FFT
         of real input.
    fft : The 1-D FFT, with definitions and conventions used.
    rfft : The 1-D FFT of real input.
    fftn : The N-D FFT.
    rfft2 : The 2-D FFT of real input.

    Notes
    -----
    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`. The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    See `fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.ones((2, 2, 2))
    >>> scipy.fft.rfftn(x)
    array([[[8.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    >>> scipy.fft.rfftn(x, axes=(2, 0))
    array([[[4.+0.j,  0.+0.j], # may vary
            [4.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D FFT of a real array.

    Parameters
    ----------
    x : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    See Also
    --------
    irfft2 : The inverse of the 2-D FFT of real input.
    rfft : The 1-D FFT of real input.
    rfftn : Compute the N-D discrete Fourier Transform for real
            input.

    Notes
    -----
    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Computes the inverse of `rfftn`

    This function computes the inverse of the N-D discrete
    Fourier Transform for real input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In
    other words, ``irfftn(rfftn(x), x.shape) == x`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e., as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Parameters
    ----------
    x : array_like
        Input array.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped. If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to be
        ``2*(m-1)``, where ``m`` is the length of the input along that axis.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given. In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)``, where ``m`` is the
        length of the final transformed axis of the input. To get an odd
        number of output points in the final axis, `s` must be specified.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    rfftn : The forward N-D FFT of real input,
            of which `ifftn` is the inverse.
    fft : The 1-D FFT, with definitions and conventions used.
    irfft : The inverse of the 1-D FFT of real input.
    irfft2 : The inverse of the 2-D FFT of real input.

    Notes
    -----
    See `fft` for definitions and conventions used.

    See `rfft` for definitions and conventions used for real input.

    The default value of `s` assumes an even output length in the final
    transformation axis. When performing the final complex to real
    transformation, the Hermitian symmetry requires that the last imaginary
    component along that axis must be 0 and so it is ignored. To avoid losing
    information, the correct length of the real input *must* be given.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.zeros((3, 2, 2))
    >>> x[0, 0, 0] = 3 * 2 * 2
    >>> scipy.fft.irfftn(x)
    array([[[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Computes the inverse of `rfft2`

    Parameters
    ----------
    x : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    rfft2 : The 2-D FFT of real input.
    irfft : The inverse of the 1-D FFT of real input.
    irfftn : The inverse of the N-D FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D FFT of Hermitian symmetric complex input, i.e., a
    signal with a real spectrum.

    This function computes the N-D discrete Fourier Transform for a
    Hermitian symmetric complex input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In other
    words, ``ihfftn(hfftn(x, s)) == x`` to within numerical accuracy. (``s``
    here is ``x.shape`` with ``s[-1] = x.shape[-1] * 2 - 1``, this is necessary
    for the same reason ``x.shape`` would be necessary for `irfft`.)

    Parameters
    ----------
    x : array_like
        Input array.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped. If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to be
        ``2*(m-1)`` where ``m`` is the length of the input along that axis.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given.  In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
        length of the final transformed axis of the input.  To get an odd
        number of output points in the final axis, `s` must be specified.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    ihfftn : The inverse N-D FFT with real spectrum. Inverse of `hfftn`.
    fft : The 1-D FFT, with definitions and conventions used.
    rfft : Forward FFT of real input.

    Notes
    -----
    For a 1-D signal ``x`` to have a real spectrum, it must satisfy
    the Hermitian property::

        x[i] == np.conj(x[-i]) for all i

    This generalizes into higher dimensions by reflecting over each axis in
    turn::

        x[i, j, k, ...] == np.conj(x[-i, -j, -k, ...]) for all i, j, k, ...

    This should not be confused with a Hermitian matrix, for which the
    transpose is its own conjugate::

        x[i, j] == np.conj(x[j, i]) for all i, j


    The default value of `s` assumes an even output length in the final
    transformation axis. When performing the final complex to real
    transformation, the Hermitian symmetry requires that the last imaginary
    component along that axis must be 0 and so it is ignored. To avoid losing
    information, the correct length of the real input *must* be given.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.ones((3, 2, 2))
    >>> scipy.fft.hfftn(x)
    array([[[12.,  0.],
            [ 0.,  0.]],
           [[ 0.,  0.],
            [ 0.,  0.]],
           [[ 0.,  0.],
            [ 0.,  0.]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D FFT of a Hermitian complex array.

    Parameters
    ----------
    x : array
        Input array, taken to be Hermitian complex.
    s : sequence of ints, optional
        Shape of the real output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The real result of the 2-D Hermitian complex real FFT.

    See Also
    --------
    hfftn : Compute the N-D discrete Fourier Transform for Hermitian
            complex input.

    Notes
    -----
    This is really just `hfftn` with different default behavior.
    For more details see `hfftn`.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ihfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Compute the N-D inverse discrete Fourier Transform for a real
    spectrum.

    This function computes the N-D inverse discrete Fourier Transform
    over any number of axes in an M-D real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining transforms
    are complex.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than the number of axes of `x`.

    See Also
    --------
    hfftn : The forward N-D FFT of Hermitian input.
    hfft : The 1-D FFT of Hermitian input.
    fft : The 1-D FFT, with definitions and conventions used.
    fftn : The N-D FFT.
    hfft2 : The 2-D FFT of Hermitian input.

    Notes
    -----
    The transform for real input is performed over the last transformation
    axis, as by `ihfft`, then the transform over the remaining axes is
    performed as by `ifftn`. The order of the output is the positive part of
    the Hermitian output signal, in the same format as `rfft`.

    Examples
    --------
    >>> import scipy.fft
    >>> import numpy as np
    >>> x = np.ones((2, 2, 2))
    >>> scipy.fft.ihfftn(x)
    array([[[1.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])
    >>> scipy.fft.ihfftn(x, axes=(2, 0))
    array([[[1.+0.j,  0.+0.j], # may vary
            [1.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ihfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Compute the 2-D inverse FFT of a real spectrum.

    Parameters
    ----------
    x : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real input to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    ihfftn : Compute the inverse of the N-D FFT of Hermitian input.

    Notes
    -----
    This is really `ihfftn` with different defaults.
    For more details see `ihfftn`.

    """
    return (Dispatchable(x, np.ndarray),)
<|endoftext|>from numbers import Number
import operator
import os
import threading
import contextlib

import numpy as np

from scipy._lib._util import copy_if_needed

# good_size is exposed (and used) from this import
from .pypocketfft import good_size, prev_good_size


__all__ = ['good_size', 'prev_good_size', 'set_workers', 'get_workers']

_config = threading.local()
_cpu_count = os.cpu_count()


def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int

    Parameters
    ----------
    x : value, or sequence of values, convertible to int
    name : str, optional
        Name of the argument being converted, only used in the error message

    Returns
    -------
    y : ``List[int]``
    """
    if isinstance(x, Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or "value"
        raise ValueError(f"{name} must be a scalar or iterable of integers") from e

    return x


def _init_nd_shape_and_axes(x, shape, axes):
    """Handles shape and axes arguments for nd transforms"""
    noshape = shape is None
    noaxes = axes is None

    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]

        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")

    if not noshape:
        shape = _iterable_of_int(shape, 'shape')

        if axes and len(axes) != len(shape):
            raise ValueError("when given, axes and shape arguments"
                             " have to be of the same length")
        if noaxes:
            if len(shape) > x.ndim:
                raise ValueError("shape requires more axes than are present")
            axes = range(x.ndim - len(shape), x.ndim)

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]

    if any(s < 1 for s in shape):
        raise ValueError(
            f"invalid number of data points ({shape}) specified")

    return tuple(shape), list(axes)


def _asfarray(x):
    """
    Convert to array with floating or complex dtype.

    float16 values are also promoted to float32.
    """
    if not hasattr(x, "dtype"):
        x = np.asarray(x)

    if x.dtype == np.float16:
        return np.asarray(x, np.float32)
    elif x.dtype.kind not in 'fc':
        return np.asarray(x, np.float64)

    # Require native byte order
    dtype = x.dtype.newbyteorder('=')
    # Always align input
    copy = True if not x.flags['ALIGNED'] else copy_if_needed
    return np.array(x, dtype=dtype, copy=copy)

def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    if arr is original:
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None


def _fix_shape(x, shape, axes):
    """Internal auxiliary function for _raw_fft, _raw_fftnd."""
    must_copy = False

    # Build an nd slice with the dimensions to be read from x
    index = [slice(None)]*x.ndim
    for n, ax in zip(shape, axes):
        if x.shape[ax] >= n:
            index[ax] = slice(0, n)
        else:
            index[ax] = slice(0, x.shape[ax])
            must_copy = True

    index = tuple(index)

    if not must_copy:
        return x[index], False

    s = list(x.shape)
    for n, axis in zip(shape, axes):
        s[axis] = n

    z = np.zeros(s, x.dtype)
    z[index] = x[index]
    return z, True


def _fix_shape_1d(x, n, axis):
    if n < 1:
        raise ValueError(
            f"invalid number of data points ({n}) specified")

    return _fix_shape(x, (n,), (axis,))


_NORM_MAP = {None: 0, 'backward': 0, 'ortho': 1, 'forward': 2}


def _normalization(norm, forward):
    """Returns the pypocketfft normalization mode from the norm argument"""
    try:
        inorm = _NORM_MAP[norm]
        return inorm if forward else (2 - inorm)
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm!r}, should '
            'be "backward", "ortho" or "forward"') from None


def _workers(workers):
    if workers is None:
        return getattr(_config, 'default_workers', 1)

    if workers < 0:
        if workers >= -_cpu_count:
            workers += 1 + _cpu_count
        else:
            raise ValueError(f"workers value out of range; got {workers}, must not be"
                             f" less than {-_cpu_count}")
    elif workers == 0:
        raise ValueError("workers must not be zero")

    return workers


@contextlib.contextmanager
def set_workers(workers):
    """Context manager for the default number of workers used in `scipy.fft`

    Parameters
    ----------
    workers : int
        The default number of workers to use

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import fft, signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((128, 64))
    >>> with fft.set_workers(4):
    ...     y = signal.fftconvolve(x, x)

    """
    old_workers = get_workers()
    _config.default_workers = _workers(operator.index(workers))
    try:
        yield
    finally:
        _config.default_workers = old_workers


def get_workers():
    """Returns the default number of workers within the current context

    Examples
    --------
    >>> from scipy import fft
    >>> fft.get_workers()
    1
    >>> with fft.set_workers(4):
    ...     fft.get_workers()
    4
    """
    return getattr(_config, 'default_workers', 1)
<|endoftext|>import numpy as np
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization, _workers)
import functools


def _r2r(forward, transform, x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, orthogonalize=None):
    """Forward or backward 1-D DCT/DST

    Parameters
    ----------
    forward : bool
        Transform direction (determines type and normalisation)
    transform : {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")

    out = (tmp if overwrite_x else None)

    # For complex input, transform real and imaginary components separably
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out
        transform(tmp.real, type, (axis,), norm, out.real, workers)
        transform(tmp.imag, type, (axis,), norm, out.imag, workers)
        return out

    return transform(tmp, type, (axis,), norm, out, workers, orthogonalize)


dct = functools.partial(_r2r, True, pfft.dct)
dct.__name__ = 'dct'
idct = functools.partial(_r2r, False, pfft.dct)
idct.__name__ = 'idct'

dst = functools.partial(_r2r, True, pfft.dst)
dst.__name__ = 'dst'
idst = functools.partial(_r2r, False, pfft.dst)
idst.__name__ = 'idst'


def _r2rn(forward, transform, x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, orthogonalize=None):
    """Forward or backward nd DCT/DST

    Parameters
    ----------
    forward : bool
        Transform direction (determines type and normalisation)
    transform : {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    overwrite_x = overwrite_x or _datacopied(tmp, x)

    if len(axes) == 0:
        return x

    tmp, copied = _fix_shape(tmp, shape, axes)
    overwrite_x = overwrite_x or copied

    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2

    norm = _normalization(norm, forward)
    workers = _workers(workers)
    out = (tmp if overwrite_x else None)

    # For complex input, transform real and imaginary components separably
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out
        transform(tmp.real, type, axes, norm, out.real, workers)
        transform(tmp.imag, type, axes, norm, out.imag, workers)
        return out

    return transform(tmp, type, axes, norm, out, workers, orthogonalize)


dctn = functools.partial(_r2rn, True, pfft.dct)
dctn.__name__ = 'dctn'
idctn = functools.partial(_r2rn, False, pfft.dct)
idctn.__name__ = 'idctn'

dstn = functools.partial(_r2rn, True, pfft.dst)
dstn.__name__ = 'dstn'
idstn = functools.partial(_r2rn, False, pfft.dst)
idstn.__name__ = 'idstn'
<|endoftext|>"""
Discrete Fourier Transforms - basic.py
"""
import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization,
                     _workers)

def c2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """ Return discrete Fourier transform of real or complex sequence. """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        message = f"invalid number of data points ({tmp.shape[axis]}) specified"
        raise ValueError(message)

    out = (tmp if overwrite_x and tmp.dtype.kind == 'c' else None)

    return pfft.c2c(tmp, (axis,), forward, norm, out, workers)


fft = functools.partial(c2c, True)
fft.__name__ = 'fft'
ifft = functools.partial(c2c, False)
ifft.__name__ = 'ifft'


def r2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """
    Discrete Fourier transform of a real sequence.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if not np.isrealobj(tmp):
        raise TypeError("x must be a real sequence")

    if n is not None:
        tmp, _ = _fix_shape_1d(tmp, n, axis)
    elif tmp.shape[axis] < 1:
        raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")

    # Note: overwrite_x is not utilised
    return pfft.r2c(tmp, (axis,), forward, norm, None, workers)


rfft = functools.partial(r2c, True)
rfft.__name__ = 'rfft'
ihfft = functools.partial(r2c, False)
ihfft.__name__ = 'ihfft'


def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """
    Return inverse discrete Fourier transform of real sequence x.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    # TODO: Optimize for hermitian and real?
    if np.isrealobj(tmp):
        tmp = tmp + 0.j

    # Last axis utilizes hermitian symmetry
    if n is None:
        n = (tmp.shape[axis] - 1) * 2
        if n < 1:
            raise ValueError(f"Invalid number of data points ({n}) specified")
    else:
        tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)

    # Note: overwrite_x is not utilized
    return pfft.c2r(tmp, (axis,), n, forward, norm, None, workers)


hfft = functools.partial(c2r, True)
hfft.__name__ = 'hfft'
irfft = functools.partial(c2r, False)
irfft.__name__ = 'irfft'


def hfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
          *, plan=None):
    """
    2-D discrete Fourier transform of a Hermitian sequence
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return hfftn(x, s, axes, norm, overwrite_x, workers)


def ihfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
           *, plan=None):
    """
    2-D discrete inverse Fourier transform of a Hermitian sequence
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return ihfftn(x, s, axes, norm, overwrite_x, workers)


def c2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """
    Return multidimensional discrete Fourier transform.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    workers = _workers(workers)

    if len(axes) == 0:
        return x

    tmp, copied = _fix_shape(tmp, shape, axes)
    overwrite_x = overwrite_x or copied

    norm = _normalization(norm, forward)
    out = (tmp if overwrite_x and tmp.dtype.kind == 'c' else None)

    return pfft.c2c(tmp, axes, forward, norm, out, workers)


fftn = functools.partial(c2cn, True)
fftn.__name__ = 'fftn'
ifftn = functools.partial(c2cn, False)
ifftn.__name__ = 'ifftn'

def r2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """Return multidimensional discrete Fourier transform of real input"""
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    if not np.isrealobj(tmp):
        raise TypeError("x must be a real sequence")

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    tmp, _ = _fix_shape(tmp, shape, axes)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if len(axes) == 0:
        raise ValueError("at least 1 axis must be transformed")

    # Note: overwrite_x is not utilized
    return pfft.r2c(tmp, axes, forward, norm, None, workers)


rfftn = functools.partial(r2cn, True)
rfftn.__name__ = 'rfftn'
ihfftn = functools.partial(r2cn, False)
ihfftn.__name__ = 'ihfftn'


def c2rn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """Multidimensional inverse discrete fourier transform with real output"""
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    # TODO: Optimize for hermitian and real?
    if np.isrealobj(tmp):
        tmp = tmp + 0.j

    noshape = s is None
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)

    if len(axes) == 0:
        raise ValueError("at least 1 axis must be transformed")

    shape = list(shape)
    if noshape:
        shape[-1] = (x.shape[axes[-1]] - 1) * 2

    norm = _normalization(norm, forward)
    workers = _workers(workers)

    # Last axis utilizes hermitian symmetry
    lastsize = shape[-1]
    shape[-1] = (shape[-1] // 2) + 1

    tmp, _ = tuple(_fix_shape(tmp, shape, axes))

    # Note: overwrite_x is not utilized
    return pfft.c2r(tmp, axes, lastsize, forward, norm, None, workers)


hfftn = functools.partial(c2rn, True)
hfftn.__name__ = 'hfftn'
irfftn = functools.partial(c2rn, False)
irfftn.__name__ = 'irfftn'


def r2r_fftpack(forward, x, n=None, axis=-1, norm=None, overwrite_x=False):
    """FFT of a real sequence, returning fftpack half complex format"""
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(None)

    if tmp.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")

    out = (tmp if overwrite_x else None)

    return pfft.r2r_fftpack(tmp, (axis,), forward, forward, norm, out, workers)


rfft_fftpack = functools.partial(r2r_fftpack, True)
rfft_fftpack.__name__ = 'rfft_fftpack'
irfft_fftpack = functools.partial(r2r_fftpack, False)
irfft_fftpack.__name__ = 'irfft_fftpack'
<|endoftext|>""" FFT backend using pypocketfft """

from .basic import *
from .realtransforms import *
from .helper import *

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""Implementation of an FFT-based Short-time Fourier Transform. """

# Implementation Notes for this file (as of 2023-07)
# --------------------------------------------------
# * MyPy version 1.1.1 does not seem to support decorated property methods
#   properly. Hence, applying ``@property`` to methods decorated with `@cache``
#   (as tried with the ``lower_border_end`` method) causes a mypy error when
#   accessing it as an index (e.g., ``SFT.lower_border_end[0]``).
# * Since the method `stft` and `istft` have identical names as the legacy
#   functions in the signal module, referencing them as HTML link in the
#   docstrings has to be done by an explicit `~ShortTimeFFT.stft` instead of an
#   ambiguous `stft` (The ``~`` hides the class / module name).
# * The HTML documentation currently renders each method/property on a separate
#   page without reference to the parent class. Thus, a link to `ShortTimeFFT`
#   was added to the "See Also" section of each method/property. These links
#   can be removed, when SciPy updates ``pydata-sphinx-theme`` to >= 0.13.3
#   (currently 0.9). Consult Issue 18512 and PR 16660 for further details.
#

# Provides typing union operator ``|`` in Python 3.9:
# Linter does not allow to import ``Generator`` from ``typing`` module:
from collections.abc import Generator, Callable
from functools import cache, lru_cache, partial
from typing import get_args, Literal

import numpy as np

import scipy.fft as fft_lib
from scipy.signal import detrend
from scipy.signal.windows import get_window

__all__ = ['ShortTimeFFT']


#: Allowed values for parameter `padding` of method `ShortTimeFFT.stft()`:
PAD_TYPE = Literal['zeros', 'edge', 'even', 'odd']

#: Allowed values for property `ShortTimeFFT.fft_mode`:
FFT_MODE_TYPE = Literal['twosided', 'centered', 'onesided', 'onesided2X']


def _calc_dual_canonical_window(win: np.ndarray, hop: int) -> np.ndarray:
    """Calculate canonical dual window for 1d window `win` and a time step
    of `hop` samples.

    A ``ValueError`` is raised, if the inversion fails.

    This is a separate function not a method, since it is also used in the
    class method ``ShortTimeFFT.from_dual()``.
    """
    if hop > len(win):
        raise ValueError(f"{hop=} is larger than window length of {len(win)}" +
                         " => STFT not invertible!")
    if issubclass(win.dtype.type, np.integer):
        raise ValueError("Parameter 'win' cannot be of integer type, but " +
                         f"{win.dtype=} => STFT not invertible!")
        # The calculation of `relative_resolution` does not work for ints.
        # Furthermore, `win / DD` casts the integers away, thus an implicit
        # cast is avoided, which can always cause confusion when using 32-Bit
        # floats.

    w2 = win.real**2 + win.imag**2  # win*win.conj() does not ensure w2 is real
    DD = w2.copy()
    for k_ in range(hop, len(win), hop):
        DD[k_:] += w2[:-k_]
        DD[:-k_] += w2[k_:]

    # check DD > 0:
    relative_resolution = np.finfo(win.dtype).resolution * max(DD)
    if not np.all(DD >= relative_resolution):
        raise ValueError("Short-time Fourier Transform not invertible!")

    return win / DD


# noinspection PyShadowingNames
class ShortTimeFFT:
    r"""Provide a parametrized discrete Short-time Fourier transform (stft)
    and its inverse (istft).

    .. currentmodule:: scipy.signal.ShortTimeFFT

    The `~ShortTimeFFT.stft` calculates sequential FFTs by sliding a
    window (`win`) over an input signal by `hop` increments. It can be used to
    quantify the change of the spectrum over time.

    The `~ShortTimeFFT.stft` is represented by a complex-valued matrix S[q,p]
    where the p-th column represents an FFT with the window centered at the
    time t[p] = p * `delta_t` = p * `hop` * `T` where `T` is  the sampling
    interval of the input signal. The q-th row represents the values at the
    frequency f[q] = q * `delta_f` with `delta_f` = 1 / (`mfft` * `T`) being
    the bin width of the FFT.

    The inverse STFT `~ShortTimeFFT.istft` is calculated by reversing the steps
    of the STFT: Take the IFFT of the p-th slice of S[q,p] and multiply the
    result with the so-called dual window (see `dual_win`). Shift the result by
    p * `delta_t` and add the result to previous shifted results to reconstruct
    the signal. If only the dual window is known and the STFT is invertible,
    `from_dual` can be used to instantiate this class.

    Due to the convention of time t = 0 being at the first sample of the input
    signal, the STFT values typically have negative time slots. Hence,
    negative indexes like `p_min` or `k_min` do not indicate counting
    backwards from an array's end like in standard Python indexing but being
    left of t = 0.

    More detailed information can be found in the :ref:`tutorial_stft` section
    of the :ref:`user_guide`.

    Note that all parameters of the initializer, except `scale_to` (which uses
    `scaling`) have identical named attributes.

    Parameters
    ----------
    win : np.ndarray
        The window must be a real- or complex-valued 1d array.
    hop : int
        The increment in samples, by which the window is shifted in each step.
    fs : float
        Sampling frequency of input signal and window. Its relation to the
        sampling interval `T` is ``T = 1 / fs``.
    fft_mode : 'twosided', 'centered', 'onesided', 'onesided2X'
        Mode of FFT to be used (default 'onesided').
        See property `fft_mode` for details.
    mfft: int | None
        Length of the FFT used, if a zero padded FFT is desired.
        If ``None`` (default), the length of the window `win` is used.
    dual_win : np.ndarray | None
        The dual window of `win`. If set to ``None``, it is calculated if
        needed.
    scale_to : 'magnitude', 'psd' | None
        If not ``None`` (default) the window function is scaled, so each STFT
        column represents  either a 'magnitude' or a power spectral density
        ('psd') spectrum. This parameter sets the property `scaling` to the
        same value. See method `scale_to` for details.
    phase_shift : int | None
        If set, add a linear phase `phase_shift` / `mfft` * `f` to each
        frequency `f`. The default value 0 ensures that there is no phase shift
        on the zeroth slice (in which t=0 is centered). See property
        `phase_shift` for more details.

    Examples
    --------
    The following example shows the magnitude of the STFT of a sine with
    varying frequency :math:`f_i(t)` (marked by a red dashed line in the plot):

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import ShortTimeFFT
    >>> from scipy.signal.windows import gaussian
    ...
    >>> T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
    >>> t_x = np.arange(N) * T_x  # time indexes for signal
    >>> f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
    >>> x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # the signal

    The utilized Gaussian window is 50 samples or 2.5 s long. The parameter
    ``mfft=200`` in `ShortTimeFFT` causes the spectrum to be oversampled
    by a factor of 4:

    >>> g_std = 8  # standard deviation for Gaussian window in samples
    >>> w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    >>> SFT = ShortTimeFFT(w, hop=10, fs=1/T_x, mfft=200, scale_to='magnitude')
    >>> Sx = SFT.stft(x)  # perform the STFT

    In the plot, the time extent of the signal `x` is marked by vertical dashed
    lines. Note that the SFT produces values outside the time range of `x`. The
    shaded areas on the left and the right indicate border effects caused
    by  the window slices in that area not fully being inside time range of
    `x`:

    >>> fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
    >>> t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    >>> ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, " +
    ...               rf"$\sigma_t={g_std*SFT.T}\,$s)")
    >>> ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
    ...                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
    ...         ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
    ...                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
    ...         xlim=(t_lo, t_hi))
    ...
    >>> im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
    ...                  extent=SFT.extent(N), cmap='viridis')
    >>> ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
    >>> fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")
    ...
    >>> # Shade areas where window slices stick out to the side:
    >>> for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
    ...                  (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    ...     ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    >>> for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    ...     ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    >>> ax1.legend()
    >>> fig1.tight_layout()
    >>> plt.show()

    Reconstructing the signal with the `~ShortTimeFFT.istft` is
    straightforward, but note that the length of `x1` should be specified,
    since the SFT length increases in `hop` steps:

    >>> SFT.invertible  # check if invertible
    True
    >>> x1 = SFT.istft(Sx, k1=N)
    >>> np.allclose(x, x1)
    True

    It is possible to calculate the SFT of signal parts:

    >>> N2 = SFT.nearest_k_p(N // 2)
    >>> Sx0 = SFT.stft(x[:N2])
    >>> Sx1 = SFT.stft(x[N2:])

    When assembling sequential STFT parts together, the overlap needs to be
    considered:

    >>> p0_ub = SFT.upper_border_begin(N2)[1] - SFT.p_min
    >>> p1_le = SFT.lower_border_end[1] - SFT.p_min
    >>> Sx01 = np.hstack((Sx0[:, :p0_ub],
    ...                   Sx0[:, p0_ub:] + Sx1[:, :p1_le],
    ...                   Sx1[:, p1_le:]))
    >>> np.allclose(Sx01, Sx)  # Compare with SFT of complete signal
    True

    It is also possible to calculate the `itsft` for signal parts:

    >>> y_p = SFT.istft(Sx, N//3, N//2)
    >>> np.allclose(y_p, x[N//3:N//2])
    True

    """
    # immutable attributes (only have getters but no setters):
    _win: np.ndarray  # window
    _dual_win: np.ndarray | None = None  # canonical dual window
    _hop: int  # Step of STFT in number of samples

    # mutable attributes:
    _fs: float  # sampling frequency of input signal and window
    _fft_mode: FFT_MODE_TYPE = 'onesided'  # Mode of FFT to use
    _mfft: int  # length of FFT used - defaults to len(win)
    _scaling: Literal['magnitude', 'psd'] | None = None  # Scaling of _win
    _phase_shift: int | None  # amount to shift phase of FFT in samples

    # attributes for caching calculated values:
    _fac_mag: float | None = None
    _fac_psd: float | None = None
    _lower_border_end: tuple[int, int] | None = None

    def __init__(self, win: np.ndarray, hop: int, fs: float, *,
                 fft_mode: FFT_MODE_TYPE = 'onesided',
                 mfft: int | None = None,
                 dual_win: np.ndarray | None = None,
                 scale_to: Literal['magnitude', 'psd'] | None = None,
                 phase_shift: int | None = 0):
        if not (win.ndim == 1 and win.size > 0):
            raise ValueError(f"Parameter win must be 1d, but {win.shape=}!")
        if not all(np.isfinite(win)):
            raise ValueError("Parameter win must have finite entries!")
        if not (hop >= 1 and isinstance(hop, int)):
            raise ValueError(f"Parameter {hop=} is not an integer >= 1!")
        self._win, self._hop, self.fs = win, hop, fs

        self.mfft = len(win) if mfft is None else mfft

        if dual_win is not None:
            if dual_win.shape != win.shape:
                raise ValueError(f"{dual_win.shape=} must equal {win.shape=}!")
            if not all(np.isfinite(dual_win)):
                raise ValueError("Parameter dual_win must be a finite array!")
        self._dual_win = dual_win  # needs to be set before scaling

        if scale_to is not None:  # needs to be set before fft_mode
            self.scale_to(scale_to)

        self.fft_mode, self.phase_shift = fft_mode, phase_shift

    @classmethod
    def from_dual(cls, dual_win: np.ndarray, hop: int, fs: float, *,
                  fft_mode: FFT_MODE_TYPE = 'onesided',
                  mfft: int | None = None,
                  scale_to: Literal['magnitude', 'psd'] | None = None,
                  phase_shift: int | None = 0):
        r"""Instantiate a `ShortTimeFFT` by only providing a dual window.

        If an STFT is invertible, it is possible to calculate the window `win`
        from a given dual window `dual_win`. All other parameters have the
        same meaning as in the initializer of `ShortTimeFFT`.

        As explained in the :ref:`tutorial_stft` section of the
        :ref:`user_guide`, an invertible STFT can be interpreted as series
        expansion of time-shifted and frequency modulated dual windows. E.g.,
        the series coefficient S[q,p] belongs to the term, which shifted
        `dual_win` by p * `delta_t` and multiplied it by
        exp( 2 * j * pi * t * q * `delta_f`).


        Examples
        --------
        The following example discusses decomposing a signal into time- and
        frequency-shifted Gaussians. A Gaussian with standard deviation of
        one made up of 51 samples will be used:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from scipy.signal import ShortTimeFFT
        >>> from scipy.signal.windows import gaussian
        ...
        >>> T, N = 0.1, 51
        >>> d_win = gaussian(N, std=1/T, sym=True)  # symmetric Gaussian window
        >>> t = T * (np.arange(N) - N//2)
        ...
        >>> fg1, ax1 = plt.subplots()
        >>> ax1.set_title(r"Dual Window: Gaussian with $\sigma_t=1$")
        >>> ax1.set(xlabel=f"Time $t$ in seconds ({N} samples, $T={T}$ s)",
        ...        xlim=(t[0], t[-1]), ylim=(0, 1.1*max(d_win)))
        >>> ax1.plot(t, d_win, 'C0-')

        The following plot with the overlap of 41, 11 and 2 samples show how
        the `hop` interval affects the shape of the window `win`:

        >>> fig2, axx = plt.subplots(3, 1, sharex='all')
        ...
        >>> axx[0].set_title(r"Windows for hop$\in\{10, 40, 49\}$")
        >>> for c_, h_ in enumerate([10, 40, 49]):
        ...     SFT = ShortTimeFFT.from_dual(d_win, h_, 1/T)
        ...     axx[c_].plot(t + h_ * T, SFT.win, 'k--', alpha=.3, label=None)
        ...     axx[c_].plot(t - h_ * T, SFT.win, 'k:', alpha=.3, label=None)
        ...     axx[c_].plot(t, SFT.win, f'C{c_+1}',
        ...                     label=r"$\Delta t=%0.1f\,$s" % SFT.delta_t)
        ...     axx[c_].set_ylim(0, 1.1*max(SFT.win))
        ...     axx[c_].legend(loc='center')
        >>> axx[-1].set(xlabel=f"Time $t$ in seconds ({N} samples, $T={T}$ s)",
        ...             xlim=(t[0], t[-1]))
        >>> plt.show()

        Beside the window `win` centered at t = 0 the previous (t = -`delta_t`)
        and following window (t = `delta_t`) are depicted. It can be seen that
        for small `hop` intervals, the window is compact and smooth, having a
        good time-frequency concentration in the STFT. For the large `hop`
        interval of 4.9 s, the window has small values around t = 0, which are
        not covered by the overlap of the adjacent windows, which could lead to
        numeric inaccuracies. Furthermore, the peaky shape at the beginning and
        the end of the window points to a higher bandwidth, resulting in a
        poorer time-frequency resolution of the STFT.
        Hence, the choice of the `hop` interval will be a compromise between
        a time-frequency resolution and memory requirements demanded by small
        `hop` sizes.

        See Also
        --------
        from_window: Create instance by wrapping `get_window`.
        ShortTimeFFT: Create instance using standard initializer.
        """
        win = _calc_dual_canonical_window(dual_win, hop)
        return cls(win=win, hop=hop, fs=fs, fft_mode=fft_mode, mfft=mfft,
                   dual_win=dual_win, scale_to=scale_to,
                   phase_shift=phase_shift)

    @classmethod
    def from_window(cls, win_param: str | tuple | float,
                    fs: float, nperseg: int, noverlap: int, *,
                    symmetric_win: bool = False,
                    fft_mode: FFT_MODE_TYPE = 'onesided',
                    mfft: int | None = None,
                    scale_to: Literal['magnitude', 'psd'] | None = None,
                    phase_shift: int | None = 0):
        """Instantiate `ShortTimeFFT` by using `get_window`.

        The method `get_window` is used to create a window of length
        `nperseg`. The parameter names `noverlap`, and `nperseg` are used here,
        since they more inline with other classical STFT libraries.

        Parameters
        ----------
        win_param: Union[str, tuple, float],
            Parameters passed to `get_window`. For windows with no parameters,
            it may be a string (e.g., ``'hann'``), for parametrized windows a
            tuple, (e.g., ``('gaussian', 2.)``) or a single float specifying
            the shape parameter of a kaiser window (i.e. ``4.``  and
            ``('kaiser', 4.)`` are equal. See `get_window` for more details.
        fs : float
            Sampling frequency of input signal. Its relation to the
            sampling interval `T` is ``T = 1 / fs``.
        nperseg: int
            Window length in samples, which corresponds to the `m_num`.
        noverlap: int
            Window overlap in samples. It relates to the `hop` increment by
            ``hop = npsereg - noverlap``.
        symmetric_win: bool
            If ``True`` then a symmetric window is generated, else a periodic
            window is generated (default). Though symmetric windows seem for
            most applications to be more sensible, the default of a periodic
            windows was chosen to correspond to the default of `get_window`.
        fft_mode : 'twosided', 'centered', 'onesided', 'onesided2X'
            Mode of FFT to be used (default 'onesided').
            See property `fft_mode` for details.
        mfft: int | None
            Length of the FFT used, if a zero padded FFT is desired.
            If ``None`` (default), the length of the window `win` is used.
        scale_to : 'magnitude', 'psd' | None
            If not ``None`` (default) the window function is scaled, so each
            STFT column represents  either a 'magnitude' or a power spectral
            density ('psd') spectrum. This parameter sets the property
            `scaling` to the same value. See method `scale_to` for details.
        phase_shift : int | None
            If set, add a linear phase `phase_shift` / `mfft` * `f` to each
            frequency `f`. The default value 0 ensures that there is no phase
            shift on the zeroth slice (in which t=0 is centered). See property
            `phase_shift` for more details.

        Examples
        --------
        The following instances ``SFT0`` and ``SFT1`` are equivalent:

        >>> from scipy.signal import ShortTimeFFT, get_window
        >>> nperseg = 9  # window length
        >>> w = get_window(('gaussian', 2.), nperseg)
        >>> fs = 128  # sampling frequency
        >>> hop = 3  # increment of STFT time slice
        >>> SFT0 = ShortTimeFFT(w, hop, fs=fs)
        >>> SFT1 = ShortTimeFFT.from_window(('gaussian', 2.), fs, nperseg,
        ...                                 noverlap=nperseg-hop)

        See Also
        --------
        scipy.signal.get_window: Return a window of a given length and type.
        from_dual: Create instance using dual window.
        ShortTimeFFT: Create instance using standard initializer.
        """
        win = get_window(win_param, nperseg, fftbins=not symmetric_win)
        return cls(win, hop=nperseg-noverlap, fs=fs, fft_mode=fft_mode,
                   mfft=mfft, scale_to=scale_to, phase_shift=phase_shift)

    @property
    def win(self) -> np.ndarray:
        """Window function as real- or complex-valued 1d array.

        This attribute is read only, since `dual_win` depends on it.

        See Also
        --------
        dual_win: Canonical dual window.
        m_num: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        hop: ime increment in signal samples for sliding window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._win

    @property
    def hop(self) -> int:
        """Time increment in signal samples for sliding window.

        This attribute is read only, since `dual_win` depends on it.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        m_num: Number of samples in window `win`.
        m_num_mid: Center index of window `win`.
        mfft: Length of input for the FFT used - may be larger than `m_num`.
        T: Sampling interval of input signal and of the window.
        win: Window function as real- or complex-valued 1d array.
        ShortTimeFFT: Class this property belongs to.
        """
        return self._hop

    @property
    def T(self) -> float:
        """Sampling interval of input signal and of the window.

        A ``ValueError`` is raised if it is set to a non-positive value.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
        hop: Time increment in signal samples for sliding window.
        fs: Sampling frequency (being ``1/T``)
        t: Times of STFT for an input signal with `n` samples.
        ShortTimeFFT: Class this property belongs to.
        """
        return 1 / self._fs

    @T.setter
    def T(self, v: float):
        """Sampling interval of input signal and of the window.

        A ``ValueError`` is raised if it is set to a non-positive value.
        """
        if not (v > 0):
            raise ValueError(f"Sampling interval T={v} must be positive!")
        self._fs = 1 / v

    @property
    def fs(self) -> float:
        """Sampling frequency of input signal and of the window.

        The sampling frequency is the inverse of the sampling interval `T`.
        A ``ValueError`` is raised if it is set to a non-positive value.

        See Also
        --------
        delta_t: Time increment of STFT (``hop*T``)
