Spectral functions
------------------

`cohere`
    Coherence (normalized cross spectral density)

`csd`
    Cross spectral density using Welch's average periodogram

`detrend`
    Remove the mean or best fit line from an array

`psd`
    Power spectral density using Welch's average periodogram

`specgram`
    Spectrogram (spectrum over segments of time)

`complex_spectrum`
    Return the complex-valued frequency spectrum of a signal

`magnitude_spectrum`
    Return the magnitude of the frequency spectrum of a signal

`angle_spectrum`
    Return the angle (wrapped phase) of the frequency spectrum of a signal

`phase_spectrum`
    Return the phase (unwrapped angle) of the frequency spectrum of a signal

`detrend_mean`
    Remove the mean from a line.

`detrend_linear`
    Remove the best fit line from a line.

`detrend_none`
    Return the original line.
"""

import functools
from numbers import Number

import numpy as np

from matplotlib import _api, _docstring, cbook


def window_hanning(x):
    """
    Return *x* times the Hanning (or Hann) window of len(*x*).

    See Also
    --------
    window_none : Another window algorithm.
    """
    return np.hanning(len(x))*x


def window_none(x):
    """
    No window function; simply return *x*.

    See Also
    --------
    window_hanning : Another window algorithm.
    """
    return x


def detrend(x, key=None, axis=None):
    """
    Return *x* with its trend removed.

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data.

    key : {'default', 'constant', 'mean', 'linear', 'none'} or function
        The detrending algorithm to use. 'default', 'mean', and 'constant' are
        the same as `detrend_mean`. 'linear' is the same as `detrend_linear`.
        'none' is the same as `detrend_none`. The default is 'mean'. See the
        corresponding functions for more details regarding the algorithms. Can
        also be a function that carries out the detrend operation.

    axis : int
        The axis along which to do the detrending.

    See Also
    --------
    detrend_mean : Implementation of the 'mean' algorithm.
    detrend_linear : Implementation of the 'linear' algorithm.
    detrend_none : Implementation of the 'none' algorithm.
    """
    if key is None or key in ['constant', 'mean', 'default']:
        return detrend(x, key=detrend_mean, axis=axis)
    elif key == 'linear':
        return detrend(x, key=detrend_linear, axis=axis)
    elif key == 'none':
        return detrend(x, key=detrend_none, axis=axis)
    elif callable(key):
        x = np.asarray(x)
        if axis is not None and axis + 1 > x.ndim:
            raise ValueError(f'axis(={axis}) out of bounds')
        if (axis is None and x.ndim == 0) or (not axis and x.ndim == 1):
            return key(x)
        # try to use the 'axis' argument if the function supports it,
        # otherwise use apply_along_axis to do it
        try:
            return key(x, axis=axis)
        except TypeError:
            return np.apply_along_axis(key, axis=axis, arr=x)
    else:
        raise ValueError(
            f"Unknown value for key: {key!r}, must be one of: 'default', "
            f"'constant', 'mean', 'linear', or a function")


def detrend_mean(x, axis=None):
    """
    Return *x* minus the mean(*x*).

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data
        Can have any dimensionality

    axis : int
        The axis along which to take the mean.  See `numpy.mean` for a
        description of this argument.

    See Also
    --------
    detrend_linear : Another detrend algorithm.
    detrend_none : Another detrend algorithm.
    detrend : A wrapper around all the detrend algorithms.
    """
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    return x - x.mean(axis, keepdims=True)


def detrend_none(x, axis=None):
    """
    Return *x*: no detrending.

    Parameters
    ----------
    x : any object
        An object containing the data

    axis : int
        This parameter is ignored.
        It is included for compatibility with detrend_mean

    See Also
    --------
    detrend_mean : Another detrend algorithm.
    detrend_linear : Another detrend algorithm.
    detrend : A wrapper around all the detrend algorithms.
    """
    return x


def detrend_linear(y):
    """
    Return *x* minus best fit line; 'linear' detrending.

    Parameters
    ----------
    y : 0-D or 1-D array or sequence
        Array or sequence containing the data

    See Also
    --------
    detrend_mean : Another detrend algorithm.
    detrend_none : Another detrend algorithm.
    detrend : A wrapper around all the detrend algorithms.
    """
    # This is faster than an algorithm based on linalg.lstsq.
    y = np.asarray(y)

    if y.ndim > 1:
        raise ValueError('y cannot have ndim > 1')

    # short-circuit 0-D array.
    if not y.ndim:
        return np.array(0., dtype=y.dtype)

    x = np.arange(y.size, dtype=float)

    C = np.cov(x, y, bias=1)
    b = C[0, 1]/C[0, 0]

    a = y.mean() - b*x.mean()
    return y - (b*x + a)


def _spectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None,
                     window=None, noverlap=None, pad_to=None,
                     sides=None, scale_by_freq=None, mode=None):
    """
    Private helper implementing the common parts between the psd, csd,
    spectrogram and complex, magnitude, angle, and phase spectrums.
    """
    if y is None:
        # if y is None use x for y
        same_data = True
    else:
        # The checks for if y is x are so that we can use the same function to
        # implement the core of psd(), csd(), and spectrogram() without doing
        # extra calculations.  We return the unaveraged Pxy, freqs, and t.
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    if noverlap >= NFFT:
        raise ValueError('noverlap must be less than NFFT')

    if mode is None or mode == 'default':
        mode = 'psd'
    _api.check_in_list(
        ['default', 'psd', 'complex', 'magnitude', 'angle', 'phase'],
        mode=mode)

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    _api.check_in_list(['default', 'onesided', 'twosided'], sides=sides)

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, NFFT)
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1)//2 + 1
        else:
            freqcenter = pad_to//2
        scaling_factor = 1.
    elif sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1)//2
        else:
            numFreqs = pad_to//2 + 1
        scaling_factor = 2.

    if not np.iterable(window):
        window = window(np.ones(NFFT, x.dtype))
    if len(window) != NFFT:
        raise ValueError(
            "The window length must match the data's first dimension")

    result = np.lib.stride_tricks.sliding_window_view(
        x, NFFT, axis=0)[::NFFT - noverlap].T
    result = detrend(result, detrend_func, axis=0)
    result = result * window.reshape((-1, 1))
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1/Fs)[:numFreqs]

    if not same_data:
        # if same_data is False, mode must be 'psd'
        resultY = np.lib.stride_tricks.sliding_window_view(
            y, NFFT, axis=0)[::NFFT - noverlap].T
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = resultY * window.reshape((-1, 1))
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        result = np.conj(result) * result
    elif mode == 'magnitude':
        result = np.abs(result) / window.sum()
    elif mode == 'angle' or mode == 'phase':
        # we unwrap the phase later to handle the onesided vs. twosided case
        result = np.angle(result)
    elif mode == 'complex':
        result /= window.sum()

    if mode == 'psd':

        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:

        # if we have a even number of frequencies, don't scale NFFT/2
        if not NFFT % 2:
            slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if scale_by_freq:
            result /= Fs
            # Scale the spectrum by the norm of the window to compensate for
            # windowing loss; see Bendat & Piersol Sec 11.5.2.
            result /= (window**2).sum()
        else:
            # In this case, preserve power in the segment, not amplitude
            result /= window.sum()**2

    t = np.arange(NFFT/2, len(x) - NFFT/2 + 1, NFFT - noverlap)/Fs

    if sides == 'twosided':
        # center the frequency range at zero
        freqs = np.roll(freqs, -freqcenter, axis=0)
        result = np.roll(result, -freqcenter, axis=0)
    elif not pad_to % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1

    # we unwrap the phase here to handle the onesided vs. twosided case
    if mode == 'phase':
        result = np.unwrap(result, axis=0)

    return result, freqs, t


def _single_spectrum_helper(
        mode, x, Fs=None, window=None, pad_to=None, sides=None):
    """
    Private helper implementing the commonality between the complex, magnitude,
    angle, and phase spectrums.
    """
    _api.check_in_list(['complex', 'magnitude', 'angle', 'phase'], mode=mode)

    if pad_to is None:
        pad_to = len(x)

    spec, freqs, _ = _spectral_helper(x=x, y=None, NFFT=len(x), Fs=Fs,
                                      detrend_func=detrend_none, window=window,
                                      noverlap=0, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=False,
                                      mode=mode)
    if mode != 'complex':
        spec = spec.real

    if spec.ndim == 2 and spec.shape[1] == 1:
        spec = spec[:, 0]

    return spec, freqs


# Split out these keyword docs so that they can be used elsewhere
_docstring.interpd.register(
    Spectral="""\
Fs : float, default: 2
    The sampling frequency (samples per time unit).  It is used to calculate
    the Fourier frequencies, *freqs*, in cycles per time unit.

window : callable or ndarray, default: `.window_hanning`
    A function or a vector of length *NFFT*.  To create window vectors see
    `.window_hanning`, `.window_none`, `numpy.blackman`, `numpy.hamming`,
    `numpy.bartlett`, `scipy.signal`, `scipy.signal.get_window`, etc.  If a
    function is passed as the argument, it must take a data segment as an
    argument and return the windowed version of the segment.

sides : {'default', 'onesided', 'twosided'}, optional
    Which sides of the spectrum to return. 'default' is one-sided for real
    data and two-sided for complex data. 'onesided' forces the return of a
    one-sided spectrum, while 'twosided' forces two-sided.""",

    Single_Spectrum="""\
pad_to : int, optional
    The number of points to which the data segment is padded when performing
    the FFT.  While not increasing the actual resolution of the spectrum (the
    minimum distance between resolvable peaks), this can give more points in
    the plot, allowing for more detail. This corresponds to the *n* parameter
    in the call to `~numpy.fft.fft`.  The default is None, which sets *pad_to*
    equal to the length of the input signal (i.e. no padding).""",

    PSD="""\
pad_to : int, optional
    The number of points to which the data segment is padded when performing
    the FFT.  This can be different from *NFFT*, which specifies the number
    of data points used.  While not increasing the actual resolution of the
    spectrum (the minimum distance between resolvable peaks), this can give
    more points in the plot, allowing for more detail. This corresponds to
    the *n* parameter in the call to `~numpy.fft.fft`. The default is None,
    which sets *pad_to* equal to *NFFT*

NFFT : int, default: 256
    The number of data points used in each block for the FFT.  A power 2 is
    most efficient.  This should *NOT* be used to get zero padding, or the
    scaling of the result will be incorrect; use *pad_to* for this instead.

detrend : {'none', 'mean', 'linear'} or callable, default: 'none'
    The function applied to each segment before fft-ing, designed to remove
    the mean or linear trend.  Unlike in MATLAB, where the *detrend* parameter
    is a vector, in Matplotlib it is a function.  The :mod:`~matplotlib.mlab`
    module defines `.detrend_none`, `.detrend_mean`, and `.detrend_linear`,
    but you can use a custom function as well.  You can also use a string to
    choose one of the functions: 'none' calls `.detrend_none`. 'mean' calls
    `.detrend_mean`. 'linear' calls `.detrend_linear`.

scale_by_freq : bool, default: True
    Whether the resulting density values should be scaled by the scaling
    frequency, which gives density in units of 1/Hz.  This allows for
    integration over the returned frequency values.  The default is True for
    MATLAB compatibility.""")


@_docstring.interpd
def psd(x, NFFT=None, Fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    r"""
    Compute the power spectral density.

    The power spectral density :math:`P_{xx}` by Welch's average
    periodogram method.  The vector *x* is divided into *NFFT* length
    segments.  Each segment is detrended by function *detrend* and
    windowed by function *window*.  *noverlap* gives the length of
    the overlap between segments.  The :math:`|\mathrm{fft}(i)|^2`
    of each segment :math:`i` are averaged to compute :math:`P_{xx}`.

    If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

    noverlap : int, default: 0 (no overlap)
        The number of points of overlap between segments.

    Returns
    -------
    Pxx : 1-D array
        The values for the power spectrum :math:`P_{xx}` (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *Pxx*

    References
    ----------
    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John
    Wiley & Sons (1986)

    See Also
    --------
    specgram
        `specgram` differs in the default overlap; in not returning the mean of
        the segment periodograms; and in returning the times of the segments.

    magnitude_spectrum : returns the magnitude spectrum.

    csd : returns the spectral density between two signals.
    """
    Pxx, freqs = csd(x=x, y=None, NFFT=NFFT, Fs=Fs, detrend=detrend,
                     window=window, noverlap=noverlap, pad_to=pad_to,
                     sides=sides, scale_by_freq=scale_by_freq)
    return Pxx.real, freqs


@_docstring.interpd
def csd(x, y, NFFT=None, Fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    """
    Compute the cross-spectral density.

    The cross spectral density :math:`P_{xy}` by Welch's average
    periodogram method.  The vectors *x* and *y* are divided into
    *NFFT* length segments.  Each segment is detrended by function
    *detrend* and windowed by function *window*.  *noverlap* gives
    the length of the overlap between segments.  The product of
    the direct FFTs of *x* and *y* are averaged over each segment
    to compute :math:`P_{xy}`, with a scaling to correct for power
    loss due to windowing.

    If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero
    padded to *NFFT*.

    Parameters
    ----------
    x, y : 1-D arrays or sequences
        Arrays or sequences containing the data

    %(Spectral)s

    %(PSD)s

    noverlap : int, default: 0 (no overlap)
        The number of points of overlap between segments.

    Returns
    -------
    Pxy : 1-D array
        The values for the cross spectrum :math:`P_{xy}` before scaling (real
        valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *Pxy*

    References
    ----------
    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John
    Wiley & Sons (1986)

    See Also
    --------
    psd : equivalent to setting ``y = x``.
    """
    if NFFT is None:
        NFFT = 256
    Pxy, freqs, _ = _spectral_helper(x=x, y=y, NFFT=NFFT, Fs=Fs,
                                     detrend_func=detrend, window=window,
                                     noverlap=noverlap, pad_to=pad_to,
                                     sides=sides, scale_by_freq=scale_by_freq,
                                     mode='psd')

    if Pxy.ndim == 2:
        if Pxy.shape[1] > 1:
            Pxy = Pxy.mean(axis=1)
        else:
            Pxy = Pxy[:, 0]
    return Pxy, freqs


_single_spectrum_docs = """\
Compute the {quantity} of *x*.
Data is padded to a length of *pad_to* and the windowing function *window* is
applied to the signal.

Parameters
----------
x : 1-D array or sequence
    Array or sequence containing the data

{Spectral}

{Single_Spectrum}

Returns
-------
spectrum : 1-D array
    The {quantity}.
freqs : 1-D array
    The frequencies corresponding to the elements in *spectrum*.

See Also
--------
psd
    Returns the power spectral density.
complex_spectrum
    Returns the complex-valued frequency spectrum.
magnitude_spectrum
    Returns the absolute value of the `complex_spectrum`.
angle_spectrum
    Returns the angle of the `complex_spectrum`.
phase_spectrum
    Returns the phase (unwrapped angle) of the `complex_spectrum`.
specgram
    Can return the complex spectrum of segments within the signal.
"""


complex_spectrum = functools.partial(_single_spectrum_helper, "complex")
complex_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="complex-valued frequency spectrum",
    **_docstring.interpd.params)
magnitude_spectrum = functools.partial(_single_spectrum_helper, "magnitude")
magnitude_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="magnitude (absolute value) of the frequency spectrum",
    **_docstring.interpd.params)
angle_spectrum = functools.partial(_single_spectrum_helper, "angle")
angle_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="angle of the frequency spectrum (wrapped phase spectrum)",
    **_docstring.interpd.params)
phase_spectrum = functools.partial(_single_spectrum_helper, "phase")
phase_spectrum.__doc__ = _single_spectrum_docs.format(
    quantity="phase of the frequency spectrum (unwrapped phase spectrum)",
    **_docstring.interpd.params)


@_docstring.interpd
def specgram(x, NFFT=None, Fs=None, detrend=None, window=None,
             noverlap=None, pad_to=None, sides=None, scale_by_freq=None,
             mode=None):
    """
    Compute a spectrogram.

    Compute and plot a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the spectrum of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

    Parameters
    ----------
    x : array-like
        1-D array or sequence.

    %(Spectral)s

    %(PSD)s

    noverlap : int, default: 128
        The number of points of overlap between blocks.
    mode : str, default: 'psd'
        What sort of spectrum to use:
            'psd'
                Returns the power spectral density.
            'complex'
                Returns the complex-valued frequency spectrum.
            'magnitude'
                Returns the magnitude spectrum.
            'angle'
                Returns the phase spectrum without unwrapping.
            'phase'
                Returns the phase spectrum with unwrapping.

    Returns
    -------
    spectrum : array-like
        2D array, columns are the periodograms of successive segments.

    freqs : array-like
        1-D array, frequencies corresponding to the rows in *spectrum*.

    t : array-like
        1-D array, the times corresponding to midpoints of segments
        (i.e the columns in *spectrum*).

    See Also
    --------
    psd : differs in the overlap and in the return values.
    complex_spectrum : similar, but with complex valued frequencies.
    magnitude_spectrum : similar single segment when *mode* is 'magnitude'.
    angle_spectrum : similar to single segment when *mode* is 'angle'.
    phase_spectrum : similar to single segment when *mode* is 'phase'.

    Notes
    -----
    *detrend* and *scale_by_freq* only apply when *mode* is set to 'psd'.

    """
    if noverlap is None:
        noverlap = 128  # default in _spectral_helper() is noverlap = 0
    if NFFT is None:
        NFFT = 256  # same default as in _spectral_helper()
    if len(x) <= NFFT:
        _api.warn_external("Only one segment is calculated since parameter "
                           f"NFFT (={NFFT}) >= signal length (={len(x)}).")

    spec, freqs, t = _spectral_helper(x=x, y=None, NFFT=NFFT, Fs=Fs,
                                      detrend_func=detrend, window=window,
                                      noverlap=noverlap, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=scale_by_freq,
                                      mode=mode)

    if mode != 'complex':
        spec = spec.real  # Needed since helper implements generically

    return spec, freqs, t


@_docstring.interpd
def cohere(x, y, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
           noverlap=0, pad_to=None, sides='default', scale_by_freq=None):
    r"""
    The coherence between *x* and *y*.  Coherence is the normalized
    cross spectral density:

    .. math::

        C_{xy} = \frac{|P_{xy}|^2}{P_{xx}P_{yy}}

    Parameters
    ----------
    x, y
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

    noverlap : int, default: 0 (no overlap)
        The number of points of overlap between segments.

    Returns
    -------
    Cxy : 1-D array
        The coherence vector.
    freqs : 1-D array
            The frequencies for the elements in *Cxy*.

    See Also
    --------
    :func:`psd`, :func:`csd` :
        For information about the methods used to compute :math:`P_{xy}`,
        :math:`P_{xx}` and :math:`P_{yy}`.
    """
    if len(x) < 2 * NFFT:
        raise ValueError(
            "Coherence is calculated by averaging over *NFFT* length "
            "segments.  Your signal is too short for your choice of *NFFT*.")
    Pxx, f = psd(x, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Pyy, f = psd(y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Pxy, f = csd(x, y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return Cxy, f


class GaussianKDE:
    """
    Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    dataset : array-like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2D array with shape (# of dims, # of data).
    bw_method : {'scott', 'silverman'} or float or callable, optional
        The method used to calculate the estimator bandwidth.  If a
        float, this will be used directly as `kde.factor`.  If a
        callable, it should take a `GaussianKDE` instance as only
        parameter and return a float. If None (default), 'scott' is used.

    Attributes
    ----------
    dataset : ndarray
        The dataset passed to the constructor.
    dim : int
        Number of dimensions.
    num_dp : int
        Number of datapoints.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of *dataset*, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of *covariance*.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    """

    # This implementation with minor modification was too good to pass up.
    # from scipy: https://github.com/scipy/scipy/blob/master/scipy/stats/kde.py

    def __init__(self, dataset, bw_method=None):
        self.dataset = np.atleast_2d(dataset)
        if not np.array(self.dataset).size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.dim, self.num_dp = np.array(self.dataset).shape

        if bw_method is None:
            pass
        elif cbook._str_equal(bw_method, 'scott'):
            self.covariance_factor = self.scotts_factor
        elif cbook._str_equal(bw_method, 'silverman'):
            self.covariance_factor = self.silverman_factor
        elif isinstance(bw_method, Number):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            raise ValueError("`bw_method` should be 'scott', 'silverman', a "
                             "scalar or a callable")

        # Computes the covariance matrix for each Gaussian kernel using
        # covariance_factor().

        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self.data_covariance = np.atleast_2d(
                np.cov(
                    self.dataset,
                    rowvar=1,
                    bias=False))
            self.data_inv_cov = np.linalg.inv(self.data_covariance)

        self.covariance = self.data_covariance * self.factor ** 2
        self.inv_cov = self.data_inv_cov / self.factor ** 2
        self.norm_factor = (np.sqrt(np.linalg.det(2 * np.pi * self.covariance))
                            * self.num_dp)

    def scotts_factor(self):
        return np.power(self.num_dp, -1. / (self.dim + 4))

    def silverman_factor(self):
        return np.power(
            self.num_dp * (self.dim + 2.0) / 4.0, -1. / (self.dim + 4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def evaluate(self, points):
        """
        Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different
                     than the dimensionality of the KDE.

        """
        points = np.atleast_2d(points)

        dim, num_m = np.array(points).shape
        if dim != self.dim:
            raise ValueError(f"points have dimension {dim}, dataset has "
                             f"dimension {self.dim}")

        result = np.zeros(num_m)

        if num_m >= self.num_dp:
            # there are more points than data, so loop over data
            for i in range(self.num_dp):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            # loop over points
            for i in range(num_m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)

        result = result / self.norm_factor

        return result

    __call__ = evaluate
<|endoftext|>"""
Nothing here but dictionaries for generating LinearSegmentedColormaps,
and a dictionary of these dictionaries.

Documentation for each is in pyplot.colormaps().  Please update this
with the purpose and type of your colormap if you add data for one here.
"""

from functools import partial

import numpy as np

_binary_data = {
    'red':    ((0., 1., 1.), (1., 0., 0.)),
    'green':  ((0., 1., 1.), (1., 0., 0.)),
    'blue':   ((0., 1., 1.), (1., 0., 0.))
    }

_autumn_data = {'red':   ((0., 1.0, 1.0), (1.0, 1.0, 1.0)),
                'green': ((0., 0., 0.), (1.0, 1.0, 1.0)),
                'blue':  ((0., 0., 0.), (1.0, 0., 0.))}

_bone_data = {'red':   ((0., 0., 0.),
                        (0.746032, 0.652778, 0.652778),
                        (1.0, 1.0, 1.0)),
              'green': ((0., 0., 0.),
                        (0.365079, 0.319444, 0.319444),
                        (0.746032, 0.777778, 0.777778),
                        (1.0, 1.0, 1.0)),
              'blue':  ((0., 0., 0.),
                        (0.365079, 0.444444, 0.444444),
                        (1.0, 1.0, 1.0))}

_cool_data = {'red':   ((0., 0., 0.), (1.0, 1.0, 1.0)),
              'green': ((0., 1., 1.), (1.0, 0.,  0.)),
              'blue':  ((0., 1., 1.), (1.0, 1.,  1.))}

_copper_data = {'red':   ((0., 0., 0.),
                          (0.809524, 1.000000, 1.000000),
                          (1.0, 1.0, 1.0)),
                'green': ((0., 0., 0.),
                          (1.0, 0.7812, 0.7812)),
                'blue':  ((0., 0., 0.),
                          (1.0, 0.4975, 0.4975))}

def _flag_red(x): return 0.75 * np.sin((x * 31.5 + 0.25) * np.pi) + 0.5
def _flag_green(x): return np.sin(x * 31.5 * np.pi)
def _flag_blue(x): return 0.75 * np.sin((x * 31.5 - 0.25) * np.pi) + 0.5
_flag_data = {'red': _flag_red, 'green': _flag_green, 'blue': _flag_blue}

def _prism_red(x): return 0.75 * np.sin((x * 20.9 + 0.25) * np.pi) + 0.67
def _prism_green(x): return 0.75 * np.sin((x * 20.9 - 0.25) * np.pi) + 0.33
def _prism_blue(x): return -1.1 * np.sin((x * 20.9) * np.pi)
_prism_data = {'red': _prism_red, 'green': _prism_green, 'blue': _prism_blue}

def _ch_helper(gamma, s, r, h, p0, p1, x):
    """Helper function for generating picklable cubehelix colormaps."""
    # Apply gamma factor to emphasise low or high intensity values
    xg = x ** gamma
    # Calculate amplitude and angle of deviation from the black to white
    # diagonal in the plane of constant perceived intensity.
    a = h * xg * (1 - xg) / 2
    phi = 2 * np.pi * (s / 3 + r * x)
    return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))

def cubehelix(gamma=1.0, s=0.5, r=-1.5, h=1.0):
    """
    Return custom data dictionary of (r, g, b) conversion functions, which can
    be used with `.ColormapRegistry.register`, for the cubehelix color scheme.

    Unlike most other color schemes cubehelix was designed by D.A. Green to
    be monotonically increasing in terms of perceived brightness.
    Also, when printed on a black and white postscript printer, the scheme
    results in a greyscale with monotonically increasing brightness.
    This color scheme is named cubehelix because the (r, g, b) values produced
    can be visualised as a squashed helix around the diagonal in the
    (r, g, b) color cube.

    For a unit color cube (i.e. 3D coordinates for (r, g, b) each in the
    range 0 to 1) the color scheme starts at (r, g, b) = (0, 0, 0), i.e. black,
    and finishes at (r, g, b) = (1, 1, 1), i.e. white. For some fraction *x*,
    between 0 and 1, the color is the corresponding grey value at that
    fraction along the black to white diagonal (x, x, x) plus a color
    element. This color element is calculated in a plane of constant
    perceived intensity and controlled by the following parameters.

    Parameters
    ----------
    gamma : float, default: 1
        Gamma factor emphasizing either low intensity values (gamma < 1), or
        high intensity values (gamma > 1).
    s : float, default: 0.5 (purple)
        The starting color.
    r : float, default: -1.5
        The number of r, g, b rotations in color that are made from the start
        to the end of the color scheme.  The default of -1.5 corresponds to ->
        B -> G -> R -> B.
    h : float, default: 1
        The hue, i.e. how saturated the colors are. If this parameter is zero
        then the color scheme is purely a greyscale.
    """
    return {'red': partial(_ch_helper, gamma, s, r, h, -0.14861, 1.78277),
            'green': partial(_ch_helper, gamma, s, r, h, -0.29227, -0.90649),
            'blue': partial(_ch_helper, gamma, s, r, h, 1.97294, 0.0)}

_cubehelix_data = cubehelix()

_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
_brg_data = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

# Gnuplot palette functions
def _g0(x): return 0
def _g1(x): return 0.5
def _g2(x): return 1
def _g3(x): return x
def _g4(x): return x ** 2
def _g5(x): return x ** 3
def _g6(x): return x ** 4
def _g7(x): return np.sqrt(x)
def _g8(x): return np.sqrt(np.sqrt(x))
def _g9(x): return np.sin(x * np.pi / 2)
def _g10(x): return np.cos(x * np.pi / 2)
def _g11(x): return np.abs(x - 0.5)
def _g12(x): return (2 * x - 1) ** 2
def _g13(x): return np.sin(x * np.pi)
def _g14(x): return np.abs(np.cos(x * np.pi))
def _g15(x): return np.sin(x * 2 * np.pi)
def _g16(x): return np.cos(x * 2 * np.pi)
def _g17(x): return np.abs(np.sin(x * 2 * np.pi))
def _g18(x): return np.abs(np.cos(x * 2 * np.pi))
def _g19(x): return np.abs(np.sin(x * 4 * np.pi))
def _g20(x): return np.abs(np.cos(x * 4 * np.pi))
def _g21(x): return 3 * x
def _g22(x): return 3 * x - 1
def _g23(x): return 3 * x - 2
def _g24(x): return np.abs(3 * x - 1)
def _g25(x): return np.abs(3 * x - 2)
def _g26(x): return (3 * x - 1) / 2
def _g27(x): return (3 * x - 2) / 2
def _g28(x): return np.abs((3 * x - 1) / 2)
def _g29(x): return np.abs((3 * x - 2) / 2)
def _g30(x): return x / 0.32 - 0.78125
def _g31(x): return 2 * x - 0.84
def _g32(x):
    ret = np.zeros(len(x))
    m = (x < 0.25)
    ret[m] = 4 * x[m]
    m = (x >= 0.25) & (x < 0.92)
    ret[m] = -2 * x[m] + 1.84
    m = (x >= 0.92)
    ret[m] = x[m] / 0.08 - 11.5
    return ret
def _g33(x): return np.abs(2 * x - 0.5)
def _g34(x): return 2 * x
def _g35(x): return 2 * x - 0.5
def _g36(x): return 2 * x - 1

gfunc = {i: globals()[f"_g{i}"] for i in range(37)}

_gnuplot_data = {
        'red': gfunc[7],
        'green': gfunc[5],
        'blue': gfunc[15],
}

_gnuplot2_data = {
        'red': gfunc[30],
        'green': gfunc[31],
        'blue': gfunc[32],
}

_ocean_data = {
        'red': gfunc[23],
        'green': gfunc[28],
        'blue': gfunc[3],
}

_afmhot_data = {
        'red': gfunc[34],
        'green': gfunc[35],
        'blue': gfunc[36],
}

_rainbow_data = {
        'red': gfunc[33],
        'green': gfunc[13],
        'blue': gfunc[10],
}

_seismic_data = (
        (0.0, 0.0, 0.3), (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0), (1.0, 0.0, 0.0),
        (0.5, 0.0, 0.0))

_terrain_data = (
        (0.00, (0.2, 0.2, 0.6)),
        (0.15, (0.0, 0.6, 1.0)),
        (0.25, (0.0, 0.8, 0.4)),
        (0.50, (1.0, 1.0, 0.6)),
        (0.75, (0.5, 0.36, 0.33)),
        (1.00, (1.0, 1.0, 1.0)))

_gray_data = {'red':   ((0., 0, 0), (1., 1, 1)),
              'green': ((0., 0, 0), (1., 1, 1)),
              'blue':  ((0., 0, 0), (1., 1, 1))}

_hot_data = {'red':   ((0., 0.0416, 0.0416),
                       (0.365079, 1.000000, 1.000000),
                       (1.0, 1.0, 1.0)),
             'green': ((0., 0., 0.),
                       (0.365079, 0.000000, 0.000000),
                       (0.746032, 1.000000, 1.000000),
                       (1.0, 1.0, 1.0)),
             'blue':  ((0., 0., 0.),
                       (0.746032, 0.000000, 0.000000),
                       (1.0, 1.0, 1.0))}

_hsv_data = {'red':   ((0., 1., 1.),
                       (0.158730, 1.000000, 1.000000),
                       (0.174603, 0.968750, 0.968750),
                       (0.333333, 0.031250, 0.031250),
                       (0.349206, 0.000000, 0.000000),
                       (0.666667, 0.000000, 0.000000),
                       (0.682540, 0.031250, 0.031250),
                       (0.841270, 0.968750, 0.968750),
                       (0.857143, 1.000000, 1.000000),
                       (1.0, 1.0, 1.0)),
             'green': ((0., 0., 0.),
                       (0.158730, 0.937500, 0.937500),
                       (0.174603, 1.000000, 1.000000),
                       (0.507937, 1.000000, 1.000000),
                       (0.666667, 0.062500, 0.062500),
                       (0.682540, 0.000000, 0.000000),
                       (1.0, 0., 0.)),
             'blue':  ((0., 0., 0.),
                       (0.333333, 0.000000, 0.000000),
                       (0.349206, 0.062500, 0.062500),
                       (0.507937, 1.000000, 1.000000),
                       (0.841270, 1.000000, 1.000000),
                       (0.857143, 0.937500, 0.937500),
                       (1.0, 0.09375, 0.09375))}

_jet_data = {'red':   ((0.00, 0, 0),
                       (0.35, 0, 0),
                       (0.66, 1, 1),
                       (0.89, 1, 1),
                       (1.00, 0.5, 0.5)),
             'green': ((0.000, 0, 0),
                       (0.125, 0, 0),
                       (0.375, 1, 1),
                       (0.640, 1, 1),
                       (0.910, 0, 0),
                       (1.000, 0, 0)),
             'blue':  ((0.00, 0.5, 0.5),
                       (0.11, 1, 1),
                       (0.34, 1, 1),
                       (0.65, 0, 0),
                       (1.00, 0, 0))}

_pink_data = {'red':   ((0., 0.1178, 0.1178), (0.015873, 0.195857, 0.195857),
                        (0.031746, 0.250661, 0.250661),
                        (0.047619, 0.295468, 0.295468),
                        (0.063492, 0.334324, 0.334324),
                        (0.079365, 0.369112, 0.369112),
                        (0.095238, 0.400892, 0.400892),
                        (0.111111, 0.430331, 0.430331),
                        (0.126984, 0.457882, 0.457882),
                        (0.142857, 0.483867, 0.483867),
                        (0.158730, 0.508525, 0.508525),
                        (0.174603, 0.532042, 0.532042),
                        (0.190476, 0.554563, 0.554563),
                        (0.206349, 0.576204, 0.576204),
                        (0.222222, 0.597061, 0.597061),
                        (0.238095, 0.617213, 0.617213),
                        (0.253968, 0.636729, 0.636729),
                        (0.269841, 0.655663, 0.655663),
                        (0.285714, 0.674066, 0.674066),
                        (0.301587, 0.691980, 0.691980),
                        (0.317460, 0.709441, 0.709441),
                        (0.333333, 0.726483, 0.726483),
                        (0.349206, 0.743134, 0.743134),
                        (0.365079, 0.759421, 0.759421),
                        (0.380952, 0.766356, 0.766356),
                        (0.396825, 0.773229, 0.773229),
                        (0.412698, 0.780042, 0.780042),
                        (0.428571, 0.786796, 0.786796),
                        (0.444444, 0.793492, 0.793492),
                        (0.460317, 0.800132, 0.800132),
                        (0.476190, 0.806718, 0.806718),
                        (0.492063, 0.813250, 0.813250),
                        (0.507937, 0.819730, 0.819730),
                        (0.523810, 0.826160, 0.826160),
                        (0.539683, 0.832539, 0.832539),
                        (0.555556, 0.838870, 0.838870),
                        (0.571429, 0.845154, 0.845154),
                        (0.587302, 0.851392, 0.851392),
                        (0.603175, 0.857584, 0.857584),
                        (0.619048, 0.863731, 0.863731),
                        (0.634921, 0.869835, 0.869835),
                        (0.650794, 0.875897, 0.875897),
                        (0.666667, 0.881917, 0.881917),
                        (0.682540, 0.887896, 0.887896),
                        (0.698413, 0.893835, 0.893835),
                        (0.714286, 0.899735, 0.899735),
                        (0.730159, 0.905597, 0.905597),
                        (0.746032, 0.911421, 0.911421),
                        (0.761905, 0.917208, 0.917208),
                        (0.777778, 0.922958, 0.922958),
                        (0.793651, 0.928673, 0.928673),
                        (0.809524, 0.934353, 0.934353),
                        (0.825397, 0.939999, 0.939999),
                        (0.841270, 0.945611, 0.945611),
                        (0.857143, 0.951190, 0.951190),
                        (0.873016, 0.956736, 0.956736),
                        (0.888889, 0.962250, 0.962250),
                        (0.904762, 0.967733, 0.967733),
                        (0.920635, 0.973185, 0.973185),
                        (0.936508, 0.978607, 0.978607),
                        (0.952381, 0.983999, 0.983999),
                        (0.968254, 0.989361, 0.989361),
                        (0.984127, 0.994695, 0.994695), (1.0, 1.0, 1.0)),
              'green': ((0., 0., 0.), (0.015873, 0.102869, 0.102869),
                        (0.031746, 0.145479, 0.145479),
                        (0.047619, 0.178174, 0.178174),
                        (0.063492, 0.205738, 0.205738),
                        (0.079365, 0.230022, 0.230022),
                        (0.095238, 0.251976, 0.251976),
                        (0.111111, 0.272166, 0.272166),
                        (0.126984, 0.290957, 0.290957),
                        (0.142857, 0.308607, 0.308607),
                        (0.158730, 0.325300, 0.325300),
                        (0.174603, 0.341178, 0.341178),
                        (0.190476, 0.356348, 0.356348),
                        (0.206349, 0.370899, 0.370899),
                        (0.222222, 0.384900, 0.384900),
                        (0.238095, 0.398410, 0.398410),
                        (0.253968, 0.411476, 0.411476),
                        (0.269841, 0.424139, 0.424139),
                        (0.285714, 0.436436, 0.436436),
                        (0.301587, 0.448395, 0.448395),
                        (0.317460, 0.460044, 0.460044),
                        (0.333333, 0.471405, 0.471405),
                        (0.349206, 0.482498, 0.482498),
                        (0.365079, 0.493342, 0.493342),
                        (0.380952, 0.517549, 0.517549),
                        (0.396825, 0.540674, 0.540674),
                        (0.412698, 0.562849, 0.562849),
                        (0.428571, 0.584183, 0.584183),
                        (0.444444, 0.604765, 0.604765),
                        (0.460317, 0.624669, 0.624669),
                        (0.476190, 0.643958, 0.643958),
                        (0.492063, 0.662687, 0.662687),
                        (0.507937, 0.680900, 0.680900),
                        (0.523810, 0.698638, 0.698638),
                        (0.539683, 0.715937, 0.715937),
                        (0.555556, 0.732828, 0.732828),
                        (0.571429, 0.749338, 0.749338),
                        (0.587302, 0.765493, 0.765493),
                        (0.603175, 0.781313, 0.781313),
                        (0.619048, 0.796819, 0.796819),
                        (0.634921, 0.812029, 0.812029),
                        (0.650794, 0.826960, 0.826960),
                        (0.666667, 0.841625, 0.841625),
                        (0.682540, 0.856040, 0.856040),
                        (0.698413, 0.870216, 0.870216),
                        (0.714286, 0.884164, 0.884164),
                        (0.730159, 0.897896, 0.897896),
                        (0.746032, 0.911421, 0.911421),
                        (0.761905, 0.917208, 0.917208),
                        (0.777778, 0.922958, 0.922958),
                        (0.793651, 0.928673, 0.928673),
                        (0.809524, 0.934353, 0.934353),
                        (0.825397, 0.939999, 0.939999),
                        (0.841270, 0.945611, 0.945611),
                        (0.857143, 0.951190, 0.951190),
                        (0.873016, 0.956736, 0.956736),
                        (0.888889, 0.962250, 0.962250),
                        (0.904762, 0.967733, 0.967733),
                        (0.920635, 0.973185, 0.973185),
                        (0.936508, 0.978607, 0.978607),
                        (0.952381, 0.983999, 0.983999),
                        (0.968254, 0.989361, 0.989361),
                        (0.984127, 0.994695, 0.994695), (1.0, 1.0, 1.0)),
              'blue':  ((0., 0., 0.), (0.015873, 0.102869, 0.102869),
                        (0.031746, 0.145479, 0.145479),
                        (0.047619, 0.178174, 0.178174),
                        (0.063492, 0.205738, 0.205738),
                        (0.079365, 0.230022, 0.230022),
                        (0.095238, 0.251976, 0.251976),
                        (0.111111, 0.272166, 0.272166),
                        (0.126984, 0.290957, 0.290957),
                        (0.142857, 0.308607, 0.308607),
                        (0.158730, 0.325300, 0.325300),
                        (0.174603, 0.341178, 0.341178),
                        (0.190476, 0.356348, 0.356348),
                        (0.206349, 0.370899, 0.370899),
                        (0.222222, 0.384900, 0.384900),
                        (0.238095, 0.398410, 0.398410),
                        (0.253968, 0.411476, 0.411476),
                        (0.269841, 0.424139, 0.424139),
                        (0.285714, 0.436436, 0.436436),
                        (0.301587, 0.448395, 0.448395),
                        (0.317460, 0.460044, 0.460044),
                        (0.333333, 0.471405, 0.471405),
                        (0.349206, 0.482498, 0.482498),
                        (0.365079, 0.493342, 0.493342),
                        (0.380952, 0.503953, 0.503953),
                        (0.396825, 0.514344, 0.514344),
                        (0.412698, 0.524531, 0.524531),
                        (0.428571, 0.534522, 0.534522),
                        (0.444444, 0.544331, 0.544331),
                        (0.460317, 0.553966, 0.553966),
                        (0.476190, 0.563436, 0.563436),
                        (0.492063, 0.572750, 0.572750),
                        (0.507937, 0.581914, 0.581914),
                        (0.523810, 0.590937, 0.590937),
                        (0.539683, 0.599824, 0.599824),
                        (0.555556, 0.608581, 0.608581),
                        (0.571429, 0.617213, 0.617213),
                        (0.587302, 0.625727, 0.625727),
                        (0.603175, 0.634126, 0.634126),
                        (0.619048, 0.642416, 0.642416),
                        (0.634921, 0.650600, 0.650600),
                        (0.650794, 0.658682, 0.658682),
                        (0.666667, 0.666667, 0.666667),
                        (0.682540, 0.674556, 0.674556),
                        (0.698413, 0.682355, 0.682355),
                        (0.714286, 0.690066, 0.690066),
                        (0.730159, 0.697691, 0.697691),
                        (0.746032, 0.705234, 0.705234),
                        (0.761905, 0.727166, 0.727166),
                        (0.777778, 0.748455, 0.748455),
                        (0.793651, 0.769156, 0.769156),
                        (0.809524, 0.789314, 0.789314),
                        (0.825397, 0.808969, 0.808969),
                        (0.841270, 0.828159, 0.828159),
                        (0.857143, 0.846913, 0.846913),
                        (0.873016, 0.865261, 0.865261),
                        (0.888889, 0.883229, 0.883229),
                        (0.904762, 0.900837, 0.900837),
                        (0.920635, 0.918109, 0.918109),
                        (0.936508, 0.935061, 0.935061),
                        (0.952381, 0.951711, 0.951711),
                        (0.968254, 0.968075, 0.968075),
                        (0.984127, 0.984167, 0.984167), (1.0, 1.0, 1.0))}

_spring_data = {'red':   ((0., 1., 1.), (1.0, 1.0, 1.0)),
                'green': ((0., 0., 0.), (1.0, 1.0, 1.0)),
                'blue':  ((0., 1., 1.), (1.0, 0.0, 0.0))}


_summer_data = {'red':   ((0., 0., 0.), (1.0, 1.0, 1.0)),
                'green': ((0., 0.5, 0.5), (1.0, 1.0, 1.0)),
                'blue':  ((0., 0.4, 0.4), (1.0, 0.4, 0.4))}


_winter_data = {'red':   ((0., 0., 0.), (1.0, 0.0, 0.0)),
                'green': ((0., 0., 0.), (1.0, 1.0, 1.0)),
                'blue':  ((0., 1., 1.), (1.0, 0.5, 0.5))}

_nipy_spectral_data = {
    'red': [
        (0.0, 0.0, 0.0), (0.05, 0.4667, 0.4667),
        (0.10, 0.5333, 0.5333), (0.15, 0.0, 0.0),
        (0.20, 0.0, 0.0), (0.25, 0.0, 0.0),
        (0.30, 0.0, 0.0), (0.35, 0.0, 0.0),
        (0.40, 0.0, 0.0), (0.45, 0.0, 0.0),
        (0.50, 0.0, 0.0), (0.55, 0.0, 0.0),
        (0.60, 0.0, 0.0), (0.65, 0.7333, 0.7333),
        (0.70, 0.9333, 0.9333), (0.75, 1.0, 1.0),
        (0.80, 1.0, 1.0), (0.85, 1.0, 1.0),
        (0.90, 0.8667, 0.8667), (0.95, 0.80, 0.80),
        (1.0, 0.80, 0.80),
    ],
    'green': [
        (0.0, 0.0, 0.0), (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0), (0.15, 0.0, 0.0),
        (0.20, 0.0, 0.0), (0.25, 0.4667, 0.4667),
        (0.30, 0.6000, 0.6000), (0.35, 0.6667, 0.6667),
        (0.40, 0.6667, 0.6667), (0.45, 0.6000, 0.6000),
        (0.50, 0.7333, 0.7333), (0.55, 0.8667, 0.8667),
        (0.60, 1.0, 1.0), (0.65, 1.0, 1.0),
        (0.70, 0.9333, 0.9333), (0.75, 0.8000, 0.8000),
        (0.80, 0.6000, 0.6000), (0.85, 0.0, 0.0),
        (0.90, 0.0, 0.0), (0.95, 0.0, 0.0),
        (1.0, 0.80, 0.80),
    ],
    'blue': [
        (0.0, 0.0, 0.0), (0.05, 0.5333, 0.5333),
        (0.10, 0.6000, 0.6000), (0.15, 0.6667, 0.6667),
        (0.20, 0.8667, 0.8667), (0.25, 0.8667, 0.8667),
        (0.30, 0.8667, 0.8667), (0.35, 0.6667, 0.6667),
        (0.40, 0.5333, 0.5333), (0.45, 0.0, 0.0),
        (0.5, 0.0, 0.0), (0.55, 0.0, 0.0),
        (0.60, 0.0, 0.0), (0.65, 0.0, 0.0),
        (0.70, 0.0, 0.0), (0.75, 0.0, 0.0),
        (0.80, 0.0, 0.0), (0.85, 0.0, 0.0),
        (0.90, 0.0, 0.0), (0.95, 0.0, 0.0),
        (1.0, 0.80, 0.80),
    ],
}


# 34 colormaps based on color specifications and designs
# developed by Cynthia Brewer (https://colorbrewer2.org/).
# The ColorBrewer palettes have been included under the terms
# of an Apache-stype license (for details, see the file
# LICENSE_COLORBREWER in the license directory of the matplotlib
# source distribution).

# RGB values taken from Brewer's Excel sheet, divided by 255

_Blues_data = (
    (0.96862745098039216,  0.98431372549019602,  1.0                ),
    (0.87058823529411766,  0.92156862745098034,  0.96862745098039216),
    (0.77647058823529413,  0.85882352941176465,  0.93725490196078431),
    (0.61960784313725492,  0.792156862745098  ,  0.88235294117647056),
    (0.41960784313725491,  0.68235294117647061,  0.83921568627450982),
    (0.25882352941176473,  0.5725490196078431 ,  0.77647058823529413),
    (0.12941176470588237,  0.44313725490196076,  0.70980392156862748),
    (0.03137254901960784,  0.31764705882352939,  0.61176470588235299),
    (0.03137254901960784,  0.18823529411764706,  0.41960784313725491)
    )

_BrBG_data = (
    (0.32941176470588235,  0.18823529411764706,  0.0196078431372549 ),
    (0.5490196078431373 ,  0.31764705882352939,  0.0392156862745098 ),
    (0.74901960784313726,  0.50588235294117645,  0.17647058823529413),
    (0.87450980392156863,  0.76078431372549016,  0.49019607843137253),
    (0.96470588235294119,  0.90980392156862744,  0.76470588235294112),
    (0.96078431372549022,  0.96078431372549022,  0.96078431372549022),
    (0.7803921568627451 ,  0.91764705882352937,  0.89803921568627454),
    (0.50196078431372548,  0.80392156862745101,  0.75686274509803919),
    (0.20784313725490197,  0.59215686274509804,  0.5607843137254902 ),
    (0.00392156862745098,  0.4                ,  0.36862745098039218),
    (0.0                ,  0.23529411764705882,  0.18823529411764706)
    )

_BuGn_data = (
    (0.96862745098039216,  0.9882352941176471 ,  0.99215686274509807),
    (0.89803921568627454,  0.96078431372549022,  0.97647058823529409),
    (0.8                ,  0.92549019607843142,  0.90196078431372551),
    (0.6                ,  0.84705882352941175,  0.78823529411764703),
    (0.4                ,  0.76078431372549016,  0.64313725490196083),
    (0.25490196078431371,  0.68235294117647061,  0.46274509803921571),
    (0.13725490196078433,  0.54509803921568623,  0.27058823529411763),
    (0.0                ,  0.42745098039215684,  0.17254901960784313),
    (0.0                ,  0.26666666666666666,  0.10588235294117647)
    )

_BuPu_data = (
    (0.96862745098039216,  0.9882352941176471 ,  0.99215686274509807),
    (0.8784313725490196 ,  0.92549019607843142,  0.95686274509803926),
    (0.74901960784313726,  0.82745098039215681,  0.90196078431372551),
    (0.61960784313725492,  0.73725490196078436,  0.85490196078431369),
    (0.5490196078431373 ,  0.58823529411764708,  0.77647058823529413),
    (0.5490196078431373 ,  0.41960784313725491,  0.69411764705882351),
    (0.53333333333333333,  0.25490196078431371,  0.61568627450980395),
    (0.50588235294117645,  0.05882352941176471,  0.48627450980392156),
    (0.30196078431372547,  0.0                ,  0.29411764705882354)
    )

_GnBu_data = (
    (0.96862745098039216,  0.9882352941176471 ,  0.94117647058823528),
    (0.8784313725490196 ,  0.95294117647058818,  0.85882352941176465),
    (0.8                ,  0.92156862745098034,  0.77254901960784317),
    (0.6588235294117647 ,  0.8666666666666667 ,  0.70980392156862748),
    (0.4823529411764706 ,  0.8                ,  0.7686274509803922 ),
    (0.30588235294117649,  0.70196078431372544,  0.82745098039215681),
    (0.16862745098039217,  0.5490196078431373 ,  0.74509803921568629),
    (0.03137254901960784,  0.40784313725490196,  0.67450980392156867),
    (0.03137254901960784,  0.25098039215686274,  0.50588235294117645)
    )

_Greens_data = (
    (0.96862745098039216,  0.9882352941176471 ,  0.96078431372549022),
    (0.89803921568627454,  0.96078431372549022,  0.8784313725490196 ),
    (0.7803921568627451 ,  0.9137254901960784 ,  0.75294117647058822),
    (0.63137254901960782,  0.85098039215686272,  0.60784313725490191),
    (0.45490196078431372,  0.7686274509803922 ,  0.46274509803921571),
    (0.25490196078431371,  0.6705882352941176 ,  0.36470588235294116),
    (0.13725490196078433,  0.54509803921568623,  0.27058823529411763),
    (0.0                ,  0.42745098039215684,  0.17254901960784313),
    (0.0                ,  0.26666666666666666,  0.10588235294117647)
    )

_Greys_data = (
    (1.0                ,  1.0                ,  1.0                ),
    (0.94117647058823528,  0.94117647058823528,  0.94117647058823528),
    (0.85098039215686272,  0.85098039215686272,  0.85098039215686272),
    (0.74117647058823533,  0.74117647058823533,  0.74117647058823533),
    (0.58823529411764708,  0.58823529411764708,  0.58823529411764708),
    (0.45098039215686275,  0.45098039215686275,  0.45098039215686275),
    (0.32156862745098042,  0.32156862745098042,  0.32156862745098042),
    (0.14509803921568629,  0.14509803921568629,  0.14509803921568629),
    (0.0                ,  0.0                ,  0.0                )
    )

_Oranges_data = (
    (1.0                ,  0.96078431372549022,  0.92156862745098034),
    (0.99607843137254903,  0.90196078431372551,  0.80784313725490198),
    (0.99215686274509807,  0.81568627450980391,  0.63529411764705879),
    (0.99215686274509807,  0.68235294117647061,  0.41960784313725491),
    (0.99215686274509807,  0.55294117647058827,  0.23529411764705882),
    (0.94509803921568625,  0.41176470588235292,  0.07450980392156863),
    (0.85098039215686272,  0.28235294117647058,  0.00392156862745098),
    (0.65098039215686276,  0.21176470588235294,  0.01176470588235294),
    (0.49803921568627452,  0.15294117647058825,  0.01568627450980392)
    )

_OrRd_data = (
    (1.0                ,  0.96862745098039216,  0.92549019607843142),
    (0.99607843137254903,  0.90980392156862744,  0.78431372549019607),
    (0.99215686274509807,  0.83137254901960789,  0.61960784313725492),
    (0.99215686274509807,  0.73333333333333328,  0.51764705882352946),
    (0.9882352941176471 ,  0.55294117647058827,  0.34901960784313724),
    (0.93725490196078431,  0.396078431372549  ,  0.28235294117647058),
    (0.84313725490196079,  0.18823529411764706,  0.12156862745098039),
    (0.70196078431372544,  0.0                ,  0.0                ),
    (0.49803921568627452,  0.0                ,  0.0                )
    )

_PiYG_data = (
    (0.55686274509803924,  0.00392156862745098,  0.32156862745098042),
    (0.77254901960784317,  0.10588235294117647,  0.49019607843137253),
    (0.87058823529411766,  0.46666666666666667,  0.68235294117647061),
    (0.94509803921568625,  0.71372549019607845,  0.85490196078431369),
    (0.99215686274509807,  0.8784313725490196 ,  0.93725490196078431),
    (0.96862745098039216,  0.96862745098039216,  0.96862745098039216),
    (0.90196078431372551,  0.96078431372549022,  0.81568627450980391),
    (0.72156862745098038,  0.88235294117647056,  0.52549019607843139),
    (0.49803921568627452,  0.73725490196078436,  0.25490196078431371),
    (0.30196078431372547,  0.5725490196078431 ,  0.12941176470588237),
    (0.15294117647058825,  0.39215686274509803,  0.09803921568627451)
    )

_PRGn_data = (
    (0.25098039215686274,  0.0                ,  0.29411764705882354),
    (0.46274509803921571,  0.16470588235294117,  0.51372549019607838),
    (0.6                ,  0.4392156862745098 ,  0.6705882352941176 ),
    (0.76078431372549016,  0.6470588235294118 ,  0.81176470588235294),
    (0.90588235294117647,  0.83137254901960789,  0.90980392156862744),
    (0.96862745098039216,  0.96862745098039216,  0.96862745098039216),
    (0.85098039215686272,  0.94117647058823528,  0.82745098039215681),
    (0.65098039215686276,  0.85882352941176465,  0.62745098039215685),
    (0.35294117647058826,  0.68235294117647061,  0.38039215686274508),
    (0.10588235294117647,  0.47058823529411764,  0.21568627450980393),
    (0.0                ,  0.26666666666666666,  0.10588235294117647)
    )

_PuBu_data = (
    (1.0                ,  0.96862745098039216,  0.98431372549019602),
    (0.92549019607843142,  0.90588235294117647,  0.94901960784313721),
    (0.81568627450980391,  0.81960784313725488,  0.90196078431372551),
    (0.65098039215686276,  0.74117647058823533,  0.85882352941176465),
    (0.45490196078431372,  0.66274509803921566,  0.81176470588235294),
    (0.21176470588235294,  0.56470588235294117,  0.75294117647058822),
    (0.0196078431372549 ,  0.4392156862745098 ,  0.69019607843137254),
    (0.01568627450980392,  0.35294117647058826,  0.55294117647058827),
    (0.00784313725490196,  0.2196078431372549 ,  0.34509803921568627)
    )

_PuBuGn_data = (
    (1.0                ,  0.96862745098039216,  0.98431372549019602),
    (0.92549019607843142,  0.88627450980392153,  0.94117647058823528),
    (0.81568627450980391,  0.81960784313725488,  0.90196078431372551),
    (0.65098039215686276,  0.74117647058823533,  0.85882352941176465),
    (0.40392156862745099,  0.66274509803921566,  0.81176470588235294),
    (0.21176470588235294,  0.56470588235294117,  0.75294117647058822),
    (0.00784313725490196,  0.50588235294117645,  0.54117647058823526),
    (0.00392156862745098,  0.42352941176470588,  0.34901960784313724),
    (0.00392156862745098,  0.27450980392156865,  0.21176470588235294)
    )

_PuOr_data = (
    (0.49803921568627452,  0.23137254901960785,  0.03137254901960784),
    (0.70196078431372544,  0.34509803921568627,  0.02352941176470588),
    (0.8784313725490196 ,  0.50980392156862742,  0.07843137254901961),
    (0.99215686274509807,  0.72156862745098038,  0.38823529411764707),
    (0.99607843137254903,  0.8784313725490196 ,  0.71372549019607845),
    (0.96862745098039216,  0.96862745098039216,  0.96862745098039216),
    (0.84705882352941175,  0.85490196078431369,  0.92156862745098034),
    (0.69803921568627447,  0.6705882352941176 ,  0.82352941176470584),
    (0.50196078431372548,  0.45098039215686275,  0.67450980392156867),
    (0.32941176470588235,  0.15294117647058825,  0.53333333333333333),
    (0.17647058823529413,  0.0                ,  0.29411764705882354)
    )

_PuRd_data = (
    (0.96862745098039216,  0.95686274509803926,  0.97647058823529409),
    (0.90588235294117647,  0.88235294117647056,  0.93725490196078431),
    (0.83137254901960789,  0.72549019607843135,  0.85490196078431369),
    (0.78823529411764703,  0.58039215686274515,  0.7803921568627451 ),
    (0.87450980392156863,  0.396078431372549  ,  0.69019607843137254),
    (0.90588235294117647,  0.16078431372549021,  0.54117647058823526),
    (0.80784313725490198,  0.07058823529411765,  0.33725490196078434),
    (0.59607843137254901,  0.0                ,  0.2627450980392157 ),
    (0.40392156862745099,  0.0                ,  0.12156862745098039)
    )

_Purples_data = (
    (0.9882352941176471 ,  0.98431372549019602,  0.99215686274509807),
    (0.93725490196078431,  0.92941176470588238,  0.96078431372549022),
    (0.85490196078431369,  0.85490196078431369,  0.92156862745098034),
    (0.73725490196078436,  0.74117647058823533,  0.86274509803921573),
    (0.61960784313725492,  0.60392156862745094,  0.78431372549019607),
    (0.50196078431372548,  0.49019607843137253,  0.72941176470588232),
    (0.41568627450980394,  0.31764705882352939,  0.63921568627450975),
    (0.32941176470588235,  0.15294117647058825,  0.5607843137254902 ),
    (0.24705882352941178,  0.0                ,  0.49019607843137253)
    )

_RdBu_data = (
    (0.40392156862745099,  0.0                ,  0.12156862745098039),
    (0.69803921568627447,  0.09411764705882353,  0.16862745098039217),
    (0.83921568627450982,  0.37647058823529411,  0.30196078431372547),
    (0.95686274509803926,  0.6470588235294118 ,  0.50980392156862742),
    (0.99215686274509807,  0.85882352941176465,  0.7803921568627451 ),
    (0.96862745098039216,  0.96862745098039216,  0.96862745098039216),
    (0.81960784313725488,  0.89803921568627454,  0.94117647058823528),
    (0.5725490196078431 ,  0.77254901960784317,  0.87058823529411766),
    (0.2627450980392157 ,  0.57647058823529407,  0.76470588235294112),
    (0.12941176470588237,  0.4                ,  0.67450980392156867),
    (0.0196078431372549 ,  0.18823529411764706,  0.38039215686274508)
    )

_RdGy_data = (
    (0.40392156862745099,  0.0                ,  0.12156862745098039),
    (0.69803921568627447,  0.09411764705882353,  0.16862745098039217),
    (0.83921568627450982,  0.37647058823529411,  0.30196078431372547),
    (0.95686274509803926,  0.6470588235294118 ,  0.50980392156862742),
    (0.99215686274509807,  0.85882352941176465,  0.7803921568627451 ),
    (1.0                ,  1.0                ,  1.0                ),
    (0.8784313725490196 ,  0.8784313725490196 ,  0.8784313725490196 ),
    (0.72941176470588232,  0.72941176470588232,  0.72941176470588232),
    (0.52941176470588236,  0.52941176470588236,  0.52941176470588236),
    (0.30196078431372547,  0.30196078431372547,  0.30196078431372547),
    (0.10196078431372549,  0.10196078431372549,  0.10196078431372549)
    )

_RdPu_data = (
    (1.0                ,  0.96862745098039216,  0.95294117647058818),
    (0.99215686274509807,  0.8784313725490196 ,  0.86666666666666667),
    (0.9882352941176471 ,  0.77254901960784317,  0.75294117647058822),
    (0.98039215686274506,  0.62352941176470589,  0.70980392156862748),
    (0.96862745098039216,  0.40784313725490196,  0.63137254901960782),
    (0.86666666666666667,  0.20392156862745098,  0.59215686274509804),
    (0.68235294117647061,  0.00392156862745098,  0.49411764705882355),
    (0.47843137254901963,  0.00392156862745098,  0.46666666666666667),
    (0.28627450980392155,  0.0                ,  0.41568627450980394)
    )

_RdYlBu_data = (
    (0.6470588235294118 , 0.0                 , 0.14901960784313725),
    (0.84313725490196079, 0.18823529411764706 , 0.15294117647058825),
    (0.95686274509803926, 0.42745098039215684 , 0.2627450980392157 ),
    (0.99215686274509807, 0.68235294117647061 , 0.38039215686274508),
    (0.99607843137254903, 0.8784313725490196  , 0.56470588235294117),
    (1.0                , 1.0                 , 0.74901960784313726),
    (0.8784313725490196 , 0.95294117647058818 , 0.97254901960784312),
    (0.6705882352941176 , 0.85098039215686272 , 0.9137254901960784 ),
    (0.45490196078431372, 0.67843137254901964 , 0.81960784313725488),
    (0.27058823529411763, 0.45882352941176469 , 0.70588235294117652),
    (0.19215686274509805, 0.21176470588235294 , 0.58431372549019611)
    )

_RdYlGn_data = (
    (0.6470588235294118 , 0.0                 , 0.14901960784313725),
    (0.84313725490196079, 0.18823529411764706 , 0.15294117647058825),
    (0.95686274509803926, 0.42745098039215684 , 0.2627450980392157 ),
    (0.99215686274509807, 0.68235294117647061 , 0.38039215686274508),
    (0.99607843137254903, 0.8784313725490196  , 0.54509803921568623),
    (1.0                , 1.0                 , 0.74901960784313726),
    (0.85098039215686272, 0.93725490196078431 , 0.54509803921568623),
    (0.65098039215686276, 0.85098039215686272 , 0.41568627450980394),
    (0.4                , 0.74117647058823533 , 0.38823529411764707),
    (0.10196078431372549, 0.59607843137254901 , 0.31372549019607843),
    (0.0                , 0.40784313725490196 , 0.21568627450980393)
    )

_Reds_data = (
    (1.0                , 0.96078431372549022 , 0.94117647058823528),
    (0.99607843137254903, 0.8784313725490196  , 0.82352941176470584),
    (0.9882352941176471 , 0.73333333333333328 , 0.63137254901960782),
    (0.9882352941176471 , 0.5725490196078431  , 0.44705882352941179),
    (0.98431372549019602, 0.41568627450980394 , 0.29019607843137257),
    (0.93725490196078431, 0.23137254901960785 , 0.17254901960784313),
    (0.79607843137254897, 0.094117647058823528, 0.11372549019607843),
    (0.6470588235294118 , 0.058823529411764705, 0.08235294117647058),
    (0.40392156862745099, 0.0                 , 0.05098039215686274)
    )

_Spectral_data = (
    (0.61960784313725492, 0.003921568627450980, 0.25882352941176473),
    (0.83529411764705885, 0.24313725490196078 , 0.30980392156862746),
    (0.95686274509803926, 0.42745098039215684 , 0.2627450980392157 ),
    (0.99215686274509807, 0.68235294117647061 , 0.38039215686274508),
    (0.99607843137254903, 0.8784313725490196  , 0.54509803921568623),
    (1.0                , 1.0                 , 0.74901960784313726),
    (0.90196078431372551, 0.96078431372549022 , 0.59607843137254901),
    (0.6705882352941176 , 0.8666666666666667  , 0.64313725490196083),
    (0.4                , 0.76078431372549016 , 0.6470588235294118 ),
    (0.19607843137254902, 0.53333333333333333 , 0.74117647058823533),
    (0.36862745098039218, 0.30980392156862746 , 0.63529411764705879)
    )

_YlGn_data = (
    (1.0                , 1.0                 , 0.89803921568627454),
    (0.96862745098039216, 0.9882352941176471  , 0.72549019607843135),
    (0.85098039215686272, 0.94117647058823528 , 0.63921568627450975),
    (0.67843137254901964, 0.8666666666666667  , 0.55686274509803924),
    (0.47058823529411764, 0.77647058823529413 , 0.47450980392156861),
    (0.25490196078431371, 0.6705882352941176  , 0.36470588235294116),
    (0.13725490196078433, 0.51764705882352946 , 0.2627450980392157 ),
    (0.0                , 0.40784313725490196 , 0.21568627450980393),
    (0.0                , 0.27058823529411763 , 0.16078431372549021)
    )

_YlGnBu_data = (
    (1.0                , 1.0                 , 0.85098039215686272),
    (0.92941176470588238, 0.97254901960784312 , 0.69411764705882351),
    (0.7803921568627451 , 0.9137254901960784  , 0.70588235294117652),
    (0.49803921568627452, 0.80392156862745101 , 0.73333333333333328),
    (0.25490196078431371, 0.71372549019607845 , 0.7686274509803922 ),
    (0.11372549019607843, 0.56862745098039214 , 0.75294117647058822),
    (0.13333333333333333, 0.36862745098039218 , 0.6588235294117647 ),
    (0.14509803921568629, 0.20392156862745098 , 0.58039215686274515),
    (0.03137254901960784, 0.11372549019607843 , 0.34509803921568627)
    )

_YlOrBr_data = (
    (1.0                , 1.0                 , 0.89803921568627454),
    (1.0                , 0.96862745098039216 , 0.73725490196078436),
    (0.99607843137254903, 0.8901960784313725  , 0.56862745098039214),
    (0.99607843137254903, 0.7686274509803922  , 0.30980392156862746),
    (0.99607843137254903, 0.6                 , 0.16078431372549021),
    (0.92549019607843142, 0.4392156862745098  , 0.07843137254901961),
    (0.8                , 0.29803921568627451 , 0.00784313725490196),
    (0.6                , 0.20392156862745098 , 0.01568627450980392),
    (0.4                , 0.14509803921568629 , 0.02352941176470588)
    )

_YlOrRd_data = (
    (1.0                , 1.0                 , 0.8                ),
    (1.0                , 0.92941176470588238 , 0.62745098039215685),
    (0.99607843137254903, 0.85098039215686272 , 0.46274509803921571),
    (0.99607843137254903, 0.69803921568627447 , 0.29803921568627451),
    (0.99215686274509807, 0.55294117647058827 , 0.23529411764705882),
    (0.9882352941176471 , 0.30588235294117649 , 0.16470588235294117),
    (0.8901960784313725 , 0.10196078431372549 , 0.10980392156862745),
    (0.74117647058823533, 0.0                 , 0.14901960784313725),
    (0.50196078431372548, 0.0                 , 0.14901960784313725)
    )


# ColorBrewer's qualitative maps, implemented using ListedColormap
# for use with mpl.colors.NoNorm

_Accent_data = (
    (0.49803921568627452, 0.78823529411764703, 0.49803921568627452),
    (0.74509803921568629, 0.68235294117647061, 0.83137254901960789),
    (0.99215686274509807, 0.75294117647058822, 0.52549019607843139),
    (1.0,                 1.0,                 0.6                ),
    (0.2196078431372549,  0.42352941176470588, 0.69019607843137254),
    (0.94117647058823528, 0.00784313725490196, 0.49803921568627452),
    (0.74901960784313726, 0.35686274509803922, 0.09019607843137254),
    (0.4,                 0.4,                 0.4                ),
    )

_Dark2_data = (
    (0.10588235294117647, 0.61960784313725492, 0.46666666666666667),
    (0.85098039215686272, 0.37254901960784315, 0.00784313725490196),
    (0.45882352941176469, 0.4392156862745098,  0.70196078431372544),
    (0.90588235294117647, 0.16078431372549021, 0.54117647058823526),
    (0.4,                 0.65098039215686276, 0.11764705882352941),
    (0.90196078431372551, 0.6705882352941176,  0.00784313725490196),
    (0.65098039215686276, 0.46274509803921571, 0.11372549019607843),
    (0.4,                 0.4,                 0.4                ),
    )

_Paired_data = (
    (0.65098039215686276, 0.80784313725490198, 0.8901960784313725 ),
    (0.12156862745098039, 0.47058823529411764, 0.70588235294117652),
    (0.69803921568627447, 0.87450980392156863, 0.54117647058823526),
    (0.2,                 0.62745098039215685, 0.17254901960784313),
    (0.98431372549019602, 0.60392156862745094, 0.6                ),
    (0.8901960784313725,  0.10196078431372549, 0.10980392156862745),
    (0.99215686274509807, 0.74901960784313726, 0.43529411764705883),
    (1.0,                 0.49803921568627452, 0.0                ),
    (0.792156862745098,   0.69803921568627447, 0.83921568627450982),
    (0.41568627450980394, 0.23921568627450981, 0.60392156862745094),
    (1.0,                 1.0,                 0.6                ),
    (0.69411764705882351, 0.34901960784313724, 0.15686274509803921),
    )

_Pastel1_data = (
    (0.98431372549019602, 0.70588235294117652, 0.68235294117647061),
    (0.70196078431372544, 0.80392156862745101, 0.8901960784313725 ),
    (0.8,                 0.92156862745098034, 0.77254901960784317),
    (0.87058823529411766, 0.79607843137254897, 0.89411764705882357),
    (0.99607843137254903, 0.85098039215686272, 0.65098039215686276),
    (1.0,                 1.0,                 0.8                ),
    (0.89803921568627454, 0.84705882352941175, 0.74117647058823533),
    (0.99215686274509807, 0.85490196078431369, 0.92549019607843142),
    (0.94901960784313721, 0.94901960784313721, 0.94901960784313721),
    )

_Pastel2_data = (
    (0.70196078431372544, 0.88627450980392153, 0.80392156862745101),
    (0.99215686274509807, 0.80392156862745101, 0.67450980392156867),
    (0.79607843137254897, 0.83529411764705885, 0.90980392156862744),
    (0.95686274509803926, 0.792156862745098,   0.89411764705882357),
    (0.90196078431372551, 0.96078431372549022, 0.78823529411764703),
    (1.0,                 0.94901960784313721, 0.68235294117647061),
    (0.94509803921568625, 0.88627450980392153, 0.8                ),
    (0.8,                 0.8,                 0.8                ),
    )

_Set1_data = (
    (0.89411764705882357, 0.10196078431372549, 0.10980392156862745),
    (0.21568627450980393, 0.49411764705882355, 0.72156862745098038),
    (0.30196078431372547, 0.68627450980392157, 0.29019607843137257),
    (0.59607843137254901, 0.30588235294117649, 0.63921568627450975),
    (1.0,                 0.49803921568627452, 0.0                ),
    (1.0,                 1.0,                 0.2                ),
    (0.65098039215686276, 0.33725490196078434, 0.15686274509803921),
    (0.96862745098039216, 0.50588235294117645, 0.74901960784313726),
    (0.6,                 0.6,                 0.6),
    )

_Set2_data = (
    (0.4,                 0.76078431372549016, 0.6470588235294118 ),
    (0.9882352941176471,  0.55294117647058827, 0.3843137254901961 ),
    (0.55294117647058827, 0.62745098039215685, 0.79607843137254897),
    (0.90588235294117647, 0.54117647058823526, 0.76470588235294112),
    (0.65098039215686276, 0.84705882352941175, 0.32941176470588235),
    (1.0,                 0.85098039215686272, 0.18431372549019609),
    (0.89803921568627454, 0.7686274509803922,  0.58039215686274515),
    (0.70196078431372544, 0.70196078431372544, 0.70196078431372544),
    )

_Set3_data = (
    (0.55294117647058827, 0.82745098039215681, 0.7803921568627451 ),
    (1.0,                 1.0,                 0.70196078431372544),
    (0.74509803921568629, 0.72941176470588232, 0.85490196078431369),
    (0.98431372549019602, 0.50196078431372548, 0.44705882352941179),
    (0.50196078431372548, 0.69411764705882351, 0.82745098039215681),
    (0.99215686274509807, 0.70588235294117652, 0.3843137254901961 ),
    (0.70196078431372544, 0.87058823529411766, 0.41176470588235292),
    (0.9882352941176471,  0.80392156862745101, 0.89803921568627454),
    (0.85098039215686272, 0.85098039215686272, 0.85098039215686272),
    (0.73725490196078436, 0.50196078431372548, 0.74117647058823533),
    (0.8,                 0.92156862745098034, 0.77254901960784317),
    (1.0,                 0.92941176470588238, 0.43529411764705883),
    )


# The next 7 palettes are from the Yorick scientific visualization package,
# an evolution of the GIST package, both by David H. Munro.
# They are released under a BSD-like license (see LICENSE_YORICK in
# the license directory of the matplotlib source distribution).
#
# Most palette functions have been reduced to simple function descriptions
# by Reinier Heeres, since the rgb components were mostly straight lines.
# gist_earth_data and gist_ncar_data were simplified by a script and some
# manual effort.

_gist_earth_data = {
    'red': (
        (0.0, 0.0, 0.0000),
        (0.2824, 0.1882, 0.1882),
        (0.4588, 0.2714, 0.2714),
        (0.5490, 0.4719, 0.4719),
        (0.6980, 0.7176, 0.7176),
        (0.7882, 0.7553, 0.7553),
        (1.0000, 0.9922, 0.9922),
    ),
    'green': (
        (0.0, 0.0, 0.0000),
        (0.0275, 0.0000, 0.0000),
        (0.1098, 0.1893, 0.1893),
        (0.1647, 0.3035, 0.3035),
        (0.2078, 0.3841, 0.3841),
        (0.2824, 0.5020, 0.5020),
        (0.5216, 0.6397, 0.6397),
        (0.6980, 0.7171, 0.7171),
        (0.7882, 0.6392, 0.6392),
        (0.7922, 0.6413, 0.6413),
        (0.8000, 0.6447, 0.6447),
        (0.8078, 0.6481, 0.6481),
        (0.8157, 0.6549, 0.6549),
        (0.8667, 0.6991, 0.6991),
        (0.8745, 0.7103, 0.7103),
        (0.8824, 0.7216, 0.7216),
        (0.8902, 0.7323, 0.7323),
        (0.8980, 0.7430, 0.7430),
        (0.9412, 0.8275, 0.8275),
        (0.9569, 0.8635, 0.8635),
        (0.9647, 0.8816, 0.8816),
        (0.9961, 0.9733, 0.9733),
        (1.0000, 0.9843, 0.9843),
    ),
    'blue': (
        (0.0, 0.0, 0.0000),
        (0.0039, 0.1684, 0.1684),
        (0.0078, 0.2212, 0.2212),
        (0.0275, 0.4329, 0.4329),
        (0.0314, 0.4549, 0.4549),
        (0.2824, 0.5004, 0.5004),
        (0.4667, 0.2748, 0.2748),
        (0.5451, 0.3205, 0.3205),
        (0.7843, 0.3961, 0.3961),
        (0.8941, 0.6651, 0.6651),
        (1.0000, 0.9843, 0.9843),
    )
}

_gist_gray_data = {
        'red': gfunc[3],
        'green': gfunc[3],
        'blue': gfunc[3],
}

def _gist_heat_red(x): return 1.5 * x
def _gist_heat_green(x): return 2 * x - 1
def _gist_heat_blue(x): return 4 * x - 3
_gist_heat_data = {
    'red': _gist_heat_red, 'green': _gist_heat_green, 'blue': _gist_heat_blue}

_gist_ncar_data = {
    'red': (
        (0.0, 0.0, 0.0000),
        (0.3098, 0.0000, 0.0000),
        (0.3725, 0.3993, 0.3993),
        (0.4235, 0.5003, 0.5003),
        (0.5333, 1.0000, 1.0000),
        (0.7922, 1.0000, 1.0000),
        (0.8471, 0.6218, 0.6218),
        (0.8980, 0.9235, 0.9235),
        (1.0000, 0.9961, 0.9961),
    ),
    'green': (
        (0.0, 0.0, 0.0000),
        (0.0510, 0.3722, 0.3722),
        (0.1059, 0.0000, 0.0000),
        (0.1569, 0.7202, 0.7202),
        (0.1608, 0.7537, 0.7537),
        (0.1647, 0.7752, 0.7752),
        (0.2157, 1.0000, 1.0000),
        (0.2588, 0.9804, 0.9804),
        (0.2706, 0.9804, 0.9804),
        (0.3176, 1.0000, 1.0000),
        (0.3686, 0.8081, 0.8081),
        (0.4275, 1.0000, 1.0000),
        (0.5216, 1.0000, 1.0000),
        (0.6314, 0.7292, 0.7292),
        (0.6863, 0.2796, 0.2796),
        (0.7451, 0.0000, 0.0000),
        (0.7922, 0.0000, 0.0000),
        (0.8431, 0.1753, 0.1753),
        (0.8980, 0.5000, 0.5000),
        (1.0000, 0.9725, 0.9725),
    ),
    'blue': (
        (0.0, 0.5020, 0.5020),
        (0.0510, 0.0222, 0.0222),
        (0.1098, 1.0000, 1.0000),
        (0.2039, 1.0000, 1.0000),
        (0.2627, 0.6145, 0.6145),
        (0.3216, 0.0000, 0.0000),
        (0.4157, 0.0000, 0.0000),
        (0.4745, 0.2342, 0.2342),
        (0.5333, 0.0000, 0.0000),
        (0.5804, 0.0000, 0.0000),
        (0.6314, 0.0549, 0.0549),
        (0.6902, 0.0000, 0.0000),
        (0.7373, 0.0000, 0.0000),
        (0.7922, 0.9738, 0.9738),
        (0.8000, 1.0000, 1.0000),
        (0.8431, 1.0000, 1.0000),
        (0.8980, 0.9341, 0.9341),
        (1.0000, 0.9961, 0.9961),
    )
}

_gist_rainbow_data = (
        (0.000, (1.00, 0.00, 0.16)),
        (0.030, (1.00, 0.00, 0.00)),
        (0.215, (1.00, 1.00, 0.00)),
        (0.400, (0.00, 1.00, 0.00)),
        (0.586, (0.00, 1.00, 1.00)),
        (0.770, (0.00, 0.00, 1.00)),
        (0.954, (1.00, 0.00, 1.00)),
        (1.000, (1.00, 0.00, 0.75))
)

_gist_stern_data = {
        'red': (
            (0.000, 0.000, 0.000), (0.0547, 1.000, 1.000),
            (0.250, 0.027, 0.250),  # (0.2500, 0.250, 0.250),
            (1.000, 1.000, 1.000)),
        'green': ((0, 0, 0), (1, 1, 1)),
        'blue': (
            (0.000, 0.000, 0.000), (0.500, 1.000, 1.000),
            (0.735, 0.000, 0.000), (1.000, 1.000, 1.000))
}

def _gist_yarg(x): return 1 - x
_gist_yarg_data = {'red': _gist_yarg, 'green': _gist_yarg, 'blue': _gist_yarg}

# This bipolar colormap was generated from CoolWarmFloat33.csv of
# "Diverging Color Maps for Scientific Visualization" by Kenneth Moreland.
# <http://www.kennethmoreland.com/color-maps/>
_coolwarm_data = {
    'red': [
        (0.0, 0.2298057, 0.2298057),
        (0.03125, 0.26623388, 0.26623388),
        (0.0625, 0.30386891, 0.30386891),
        (0.09375, 0.342804478, 0.342804478),
        (0.125, 0.38301334, 0.38301334),
        (0.15625, 0.424369608, 0.424369608),
        (0.1875, 0.46666708, 0.46666708),
        (0.21875, 0.509635204, 0.509635204),
        (0.25, 0.552953156, 0.552953156),
        (0.28125, 0.596262162, 0.596262162),
        (0.3125, 0.639176211, 0.639176211),
        (0.34375, 0.681291281, 0.681291281),
        (0.375, 0.722193294, 0.722193294),
        (0.40625, 0.761464949, 0.761464949),
        (0.4375, 0.798691636, 0.798691636),
        (0.46875, 0.833466556, 0.833466556),
        (0.5, 0.865395197, 0.865395197),
        (0.53125, 0.897787179, 0.897787179),
        (0.5625, 0.924127593, 0.924127593),
        (0.59375, 0.944468518, 0.944468518),
        (0.625, 0.958852946, 0.958852946),
        (0.65625, 0.96732803, 0.96732803),
        (0.6875, 0.969954137, 0.969954137),
        (0.71875, 0.966811177, 0.966811177),
        (0.75, 0.958003065, 0.958003065),
        (0.78125, 0.943660866, 0.943660866),
        (0.8125, 0.923944917, 0.923944917),
        (0.84375, 0.89904617, 0.89904617),
        (0.875, 0.869186849, 0.869186849),
        (0.90625, 0.834620542, 0.834620542),
        (0.9375, 0.795631745, 0.795631745),
        (0.96875, 0.752534934, 0.752534934),
        (1.0, 0.705673158, 0.705673158)],
    'green': [
        (0.0, 0.298717966, 0.298717966),
        (0.03125, 0.353094838, 0.353094838),
        (0.0625, 0.406535296, 0.406535296),
        (0.09375, 0.458757618, 0.458757618),
        (0.125, 0.50941904, 0.50941904),
        (0.15625, 0.558148092, 0.558148092),
        (0.1875, 0.604562568, 0.604562568),
        (0.21875, 0.648280772, 0.648280772),
        (0.25, 0.688929332, 0.688929332),
        (0.28125, 0.726149107, 0.726149107),
        (0.3125, 0.759599947, 0.759599947),
        (0.34375, 0.788964712, 0.788964712),
        (0.375, 0.813952739, 0.813952739),
        (0.40625, 0.834302879, 0.834302879),
        (0.4375, 0.849786142, 0.849786142),
        (0.46875, 0.860207984, 0.860207984),
        (0.5, 0.86541021, 0.86541021),
        (0.53125, 0.848937047, 0.848937047),
        (0.5625, 0.827384882, 0.827384882),
        (0.59375, 0.800927443, 0.800927443),
        (0.625, 0.769767752, 0.769767752),
        (0.65625, 0.734132809, 0.734132809),
        (0.6875, 0.694266682, 0.694266682),
        (0.71875, 0.650421156, 0.650421156),
        (0.75, 0.602842431, 0.602842431),
        (0.78125, 0.551750968, 0.551750968),
        (0.8125, 0.49730856, 0.49730856),
        (0.84375, 0.439559467, 0.439559467),
        (0.875, 0.378313092, 0.378313092),
        (0.90625, 0.312874446, 0.312874446),
        (0.9375, 0.24128379, 0.24128379),
        (0.96875, 0.157246067, 0.157246067),
        (1.0, 0.01555616, 0.01555616)],
    'blue': [
        (0.0, 0.753683153, 0.753683153),
        (0.03125, 0.801466763, 0.801466763),
        (0.0625, 0.84495867, 0.84495867),
        (0.09375, 0.883725899, 0.883725899),
        (0.125, 0.917387822, 0.917387822),
        (0.15625, 0.945619588, 0.945619588),
        (0.1875, 0.968154911, 0.968154911),
        (0.21875, 0.98478814, 0.98478814),
        (0.25, 0.995375608, 0.995375608),
        (0.28125, 0.999836203, 0.999836203),
        (0.3125, 0.998151185, 0.998151185),
        (0.34375, 0.990363227, 0.990363227),
        (0.375, 0.976574709, 0.976574709),
        (0.40625, 0.956945269, 0.956945269),
        (0.4375, 0.931688648, 0.931688648),
        (0.46875, 0.901068838, 0.901068838),
        (0.5, 0.865395561, 0.865395561),
        (0.53125, 0.820880546, 0.820880546),
        (0.5625, 0.774508472, 0.774508472),
        (0.59375, 0.726736146, 0.726736146),
        (0.625, 0.678007945, 0.678007945),
        (0.65625, 0.628751763, 0.628751763),
        (0.6875, 0.579375448, 0.579375448),
        (0.71875, 0.530263762, 0.530263762),
        (0.75, 0.481775914, 0.481775914),
        (0.78125, 0.434243684, 0.434243684),
        (0.8125, 0.387970225, 0.387970225),
        (0.84375, 0.343229596, 0.343229596),
        (0.875, 0.300267182, 0.300267182),
        (0.90625, 0.259301199, 0.259301199),
        (0.9375, 0.220525627, 0.220525627),
        (0.96875, 0.184115123, 0.184115123),
        (1.0, 0.150232812, 0.150232812)]
    }

# Implementation of Carey Rappaport's CMRmap.
# See `A Color Map for Effective Black-and-White Rendering of Color-Scale
# Images' by Carey Rappaport
# https://www.mathworks.com/matlabcentral/fileexchange/2662-cmrmap-m
_CMRmap_data = {'red':    ((0.000, 0.00, 0.00),
                           (0.125, 0.15, 0.15),
                           (0.250, 0.30, 0.30),
                           (0.375, 0.60, 0.60),
                           (0.500, 1.00, 1.00),
                           (0.625, 0.90, 0.90),
                           (0.750, 0.90, 0.90),
                           (0.875, 0.90, 0.90),
                           (1.000, 1.00, 1.00)),
                'green':  ((0.000, 0.00, 0.00),
                           (0.125, 0.15, 0.15),
                           (0.250, 0.15, 0.15),
                           (0.375, 0.20, 0.20),
                           (0.500, 0.25, 0.25),
                           (0.625, 0.50, 0.50),
                           (0.750, 0.75, 0.75),
                           (0.875, 0.90, 0.90),
                           (1.000, 1.00, 1.00)),
                'blue':   ((0.000, 0.00, 0.00),
                           (0.125, 0.50, 0.50),
                           (0.250, 0.75, 0.75),
                           (0.375, 0.50, 0.50),
                           (0.500, 0.15, 0.15),
                           (0.625, 0.00, 0.00),
                           (0.750, 0.10, 0.10),
                           (0.875, 0.50, 0.50),
                           (1.000, 1.00, 1.00))}


# An MIT licensed, colorblind-friendly heatmap from Wistia:
#   https://github.com/wistia/heatmap-palette
#   https://wistia.com/learn/culture/heatmaps-for-colorblindness
#
# >>> import matplotlib.colors as c
# >>> colors = ["#e4ff7a", "#ffe81a", "#ffbd00", "#ffa000", "#fc7f00"]
# >>> cm = c.LinearSegmentedColormap.from_list('wistia', colors)
# >>> _wistia_data = cm._segmentdata
# >>> del _wistia_data['alpha']
#
_wistia_data = {
    'red': [(0.0, 0.8941176470588236, 0.8941176470588236),
            (0.25, 1.0, 1.0),
            (0.5, 1.0, 1.0),
            (0.75, 1.0, 1.0),
            (1.0, 0.9882352941176471, 0.9882352941176471)],
    'green': [(0.0, 1.0, 1.0),
              (0.25, 0.9098039215686274, 0.9098039215686274),
              (0.5, 0.7411764705882353, 0.7411764705882353),
              (0.75, 0.6274509803921569, 0.6274509803921569),
              (1.0, 0.4980392156862745, 0.4980392156862745)],
    'blue': [(0.0, 0.47843137254901963, 0.47843137254901963),
             (0.25, 0.10196078431372549, 0.10196078431372549),
             (0.5, 0.0, 0.0),
             (0.75, 0.0, 0.0),
             (1.0, 0.0, 0.0)],
}


# Categorical palettes from Vega:
# https://github.com/vega/vega/wiki/Scales
# (divided by 255)
#

_tab10_data = (
    (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
    (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
    (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
    (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728
    (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd
    (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b
    (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2
    (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f
    (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22
    (0.09019607843137255, 0.7450980392156863,  0.8117647058823529),    # 17becf
)

_tab20_data = (
    (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
    (0.6823529411764706,  0.7803921568627451,  0.9098039215686274  ),  # aec7e8
    (1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
    (1.0,                 0.7333333333333333,  0.47058823529411764 ),  # ffbb78
    (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
    (0.596078431372549,   0.8745098039215686,  0.5411764705882353  ),  # 98df8a
    (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728
    (1.0,                 0.596078431372549,   0.5882352941176471  ),  # ff9896
    (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd
    (0.7725490196078432,  0.6901960784313725,  0.8352941176470589  ),  # c5b0d5
    (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b
    (0.7686274509803922,  0.611764705882353,   0.5803921568627451  ),  # c49c94
    (0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2
    (0.9686274509803922,  0.7137254901960784,  0.8235294117647058  ),  # f7b6d2
    (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f
    (0.7803921568627451,  0.7803921568627451,  0.7803921568627451  ),  # c7c7c7
    (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22
    (0.8588235294117647,  0.8588235294117647,  0.5529411764705883  ),  # dbdb8d
    (0.09019607843137255, 0.7450980392156863,  0.8117647058823529  ),  # 17becf
    (0.6196078431372549,  0.8549019607843137,  0.8980392156862745),    # 9edae5
)

_tab20b_data = (
    (0.2235294117647059,  0.23137254901960785, 0.4745098039215686 ),  # 393b79
    (0.3215686274509804,  0.32941176470588235, 0.6392156862745098 ),  # 5254a3
    (0.4196078431372549,  0.43137254901960786, 0.8117647058823529 ),  # 6b6ecf
    (0.611764705882353,   0.6196078431372549,  0.8705882352941177 ),  # 9c9ede
    (0.38823529411764707, 0.4745098039215686,  0.2235294117647059 ),  # 637939
    (0.5490196078431373,  0.6352941176470588,  0.3215686274509804 ),  # 8ca252
    (0.7098039215686275,  0.8117647058823529,  0.4196078431372549 ),  # b5cf6b
    (0.807843137254902,   0.8588235294117647,  0.611764705882353  ),  # cedb9c
    (0.5490196078431373,  0.42745098039215684, 0.19215686274509805),  # 8c6d31
    (0.7411764705882353,  0.6196078431372549,  0.2235294117647059 ),  # bd9e39
    (0.9058823529411765,  0.7294117647058823,  0.3215686274509804 ),  # e7ba52
    (0.9058823529411765,  0.796078431372549,   0.5803921568627451 ),  # e7cb94
    (0.5176470588235295,  0.23529411764705882, 0.2235294117647059 ),  # 843c39
    (0.6784313725490196,  0.28627450980392155, 0.2901960784313726 ),  # ad494a
    (0.8392156862745098,  0.3803921568627451,  0.4196078431372549 ),  # d6616b
    (0.9058823529411765,  0.5882352941176471,  0.611764705882353  ),  # e7969c
    (0.4823529411764706,  0.2549019607843137,  0.45098039215686275),  # 7b4173
    (0.6470588235294118,  0.3176470588235294,  0.5803921568627451 ),  # a55194
    (0.807843137254902,   0.42745098039215684, 0.7411764705882353 ),  # ce6dbd
    (0.8705882352941177,  0.6196078431372549,  0.8392156862745098 ),  # de9ed6
)

_tab20c_data = (
    (0.19215686274509805, 0.5098039215686274,  0.7411764705882353  ),  # 3182bd
    (0.4196078431372549,  0.6823529411764706,  0.8392156862745098  ),  # 6baed6
    (0.6196078431372549,  0.792156862745098,   0.8823529411764706  ),  # 9ecae1
    (0.7764705882352941,  0.8588235294117647,  0.9372549019607843  ),  # c6dbef
    (0.9019607843137255,  0.3333333333333333,  0.050980392156862744),  # e6550d
    (0.9921568627450981,  0.5529411764705883,  0.23529411764705882 ),  # fd8d3c
    (0.9921568627450981,  0.6823529411764706,  0.4196078431372549  ),  # fdae6b
    (0.9921568627450981,  0.8156862745098039,  0.6352941176470588  ),  # fdd0a2
    (0.19215686274509805, 0.6392156862745098,  0.32941176470588235 ),  # 31a354
    (0.4549019607843137,  0.7686274509803922,  0.4627450980392157  ),  # 74c476
    (0.6313725490196078,  0.8509803921568627,  0.6078431372549019  ),  # a1d99b
    (0.7803921568627451,  0.9137254901960784,  0.7529411764705882  ),  # c7e9c0
    (0.4588235294117647,  0.4196078431372549,  0.6941176470588235  ),  # 756bb1
    (0.6196078431372549,  0.6039215686274509,  0.7843137254901961  ),  # 9e9ac8
    (0.7372549019607844,  0.7411764705882353,  0.8627450980392157  ),  # bcbddc
    (0.8549019607843137,  0.8549019607843137,  0.9215686274509803  ),  # dadaeb
    (0.38823529411764707, 0.38823529411764707, 0.38823529411764707 ),  # 636363
    (0.5882352941176471,  0.5882352941176471,  0.5882352941176471  ),  # 969696
    (0.7411764705882353,  0.7411764705882353,  0.7411764705882353  ),  # bdbdbd
    (0.8509803921568627,  0.8509803921568627,  0.8509803921568627  ),  # d9d9d9
)


_petroff10_data = (
    (0.24705882352941178, 0.5647058823529412,  0.8549019607843137),    # 3f90da
    (1.0,                 0.6627450980392157,  0.054901960784313725),  # ffa90e
    (0.7411764705882353,  0.12156862745098039, 0.00392156862745098),   # bd1f01
    (0.5803921568627451,  0.6431372549019608,  0.6352941176470588),    # 94a4a2
    (0.5137254901960784,  0.17647058823529413, 0.7137254901960784),    # 832db6
    (0.6627450980392157,  0.4196078431372549,  0.34901960784313724),   # a96b59
    (0.9058823529411765,  0.38823529411764707, 0.0),                   # e76300
    (0.7254901960784313,  0.6745098039215687,  0.4392156862745098),    # b9ac70
    (0.44313725490196076, 0.4588235294117647,  0.5058823529411764),    # 717581
    (0.5725490196078431,  0.8549019607843137,  0.8666666666666667),    # 92dadd
)


datad = {
    'Blues': _Blues_data,
    'BrBG': _BrBG_data,
    'BuGn': _BuGn_data,
    'BuPu': _BuPu_data,
    'CMRmap': _CMRmap_data,
    'GnBu': _GnBu_data,
    'Greens': _Greens_data,
    'Greys': _Greys_data,
    'OrRd': _OrRd_data,
    'Oranges': _Oranges_data,
    'PRGn': _PRGn_data,
    'PiYG': _PiYG_data,
    'PuBu': _PuBu_data,
    'PuBuGn': _PuBuGn_data,
    'PuOr': _PuOr_data,
    'PuRd': _PuRd_data,
    'Purples': _Purples_data,
    'RdBu': _RdBu_data,
    'RdGy': _RdGy_data,
    'RdPu': _RdPu_data,
    'RdYlBu': _RdYlBu_data,
    'RdYlGn': _RdYlGn_data,
    'Reds': _Reds_data,
    'Spectral': _Spectral_data,
    'Wistia': _wistia_data,
    'YlGn': _YlGn_data,
    'YlGnBu': _YlGnBu_data,
    'YlOrBr': _YlOrBr_data,
    'YlOrRd': _YlOrRd_data,
    'afmhot': _afmhot_data,
    'autumn': _autumn_data,
    'binary': _binary_data,
    'bone': _bone_data,
    'brg': _brg_data,
    'bwr': _bwr_data,
    'cool': _cool_data,
    'coolwarm': _coolwarm_data,
    'copper': _copper_data,
    'cubehelix': _cubehelix_data,
    'flag': _flag_data,
    'gist_earth': _gist_earth_data,
    'gist_gray': _gist_gray_data,
    'gist_heat': _gist_heat_data,
    'gist_ncar': _gist_ncar_data,
    'gist_rainbow': _gist_rainbow_data,
    'gist_stern': _gist_stern_data,
    'gist_yarg': _gist_yarg_data,
    'gnuplot': _gnuplot_data,
    'gnuplot2': _gnuplot2_data,
    'gray': _gray_data,
    'hot': _hot_data,
    'hsv': _hsv_data,
    'jet': _jet_data,
    'nipy_spectral': _nipy_spectral_data,
    'ocean': _ocean_data,
    'pink': _pink_data,
    'prism': _prism_data,
    'rainbow': _rainbow_data,
    'seismic': _seismic_data,
    'spring': _spring_data,
    'summer': _summer_data,
    'terrain': _terrain_data,
    'winter': _winter_data,
    # Qualitative
    'Accent': {'listed': _Accent_data},
    'Dark2': {'listed': _Dark2_data},
    'Paired': {'listed': _Paired_data},
    'Pastel1': {'listed': _Pastel1_data},
    'Pastel2': {'listed': _Pastel2_data},
    'Set1': {'listed': _Set1_data},
    'Set2': {'listed': _Set2_data},
    'Set3': {'listed': _Set3_data},
    'tab10': {'listed': _tab10_data},
    'tab20': {'listed': _tab20_data},
    'tab20b': {'listed': _tab20b_data},
    'tab20c': {'listed': _tab20c_data},
}
<|endoftext|>r"""
A module for dealing with the polylines used throughout Matplotlib.

The primary class for polyline handling in Matplotlib is `Path`.  Almost all
vector drawing makes use of `Path`\s somewhere in the drawing pipeline.

Whilst a `Path` instance itself cannot be drawn, some `.Artist` subclasses,
such as `.PathPatch` and `.PathCollection`, can be used for convenient `Path`
visualisation.
"""

import copy
from functools import lru_cache
from weakref import WeakValueDictionary

import numpy as np

import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment


class Path:
    """
    A series of possibly disconnected, possibly closed, line and curve
    segments.

    The underlying storage is made up of two parallel numpy arrays:

    - *vertices*: an (N, 2) float array of vertices
    - *codes*: an N-length `numpy.uint8` array of path codes, or None

    These two arrays always have the same length in the first
    dimension.  For example, to represent a cubic curve, you must
    provide three vertices and three `CURVE4` codes.

    The code types are:

    - `STOP`   :  1 vertex (ignored)
        A marker for the end of the entire path (currently not required and
        ignored)

    - `MOVETO` :  1 vertex
        Pick up the pen and move to the given vertex.

    - `LINETO` :  1 vertex
        Draw a line from the current position to the given vertex.

    - `CURVE3` :  1 control point, 1 endpoint
        Draw a quadratic Bézier curve from the current position, with the given
        control point, to the given end point.

    - `CURVE4` :  2 control points, 1 endpoint
        Draw a cubic Bézier curve from the current position, with the given
        control points, to the given end point.

    - `CLOSEPOLY` : 1 vertex (ignored)
        Draw a line segment to the start point of the current polyline.

    If *codes* is None, it is interpreted as a `MOVETO` followed by a series
    of `LINETO`.

    Users of Path objects should not access the vertices and codes arrays
    directly.  Instead, they should use `iter_segments` or `cleaned` to get the
    vertex/code pairs.  This helps, in particular, to consistently handle the
    case of *codes* being None.

    Some behavior of Path objects can be controlled by rcParams. See the
    rcParams whose keys start with 'path.'.

    .. note::

        The vertices and codes arrays should be treated as
        immutable -- there are a number of optimizations and assumptions
        made up front in the constructor that will not change when the
        data changes.
    """

    code_type = np.uint8

    # Path codes
    STOP = code_type(0)         # 1 vertex
    MOVETO = code_type(1)       # 1 vertex
    LINETO = code_type(2)       # 1 vertex
    CURVE3 = code_type(3)       # 2 vertices
    CURVE4 = code_type(4)       # 3 vertices
    CLOSEPOLY = code_type(79)   # 1 vertex

    #: A dictionary mapping Path codes to the number of vertices that the
    #: code expects.
    NUM_VERTICES_FOR_CODE = {STOP: 1,
                             MOVETO: 1,
                             LINETO: 1,
                             CURVE3: 2,
                             CURVE4: 3,
                             CLOSEPOLY: 1}

    def __init__(self, vertices, codes=None, _interpolation_steps=1,
                 closed=False, readonly=False):
        """
        Create a new path with the given vertices and codes.

        Parameters
        ----------
        vertices : (N, 2) array-like
            The path vertices, as an array, masked array or sequence of pairs.
            Masked values, if any, will be converted to NaNs, which are then
            handled correctly by the Agg PathIterator and other consumers of
            path data, such as :meth:`iter_segments`.
        codes : array-like or None, optional
            N-length array of integers representing the codes of the path.
            If not None, codes must be the same length as vertices.
            If None, *vertices* will be treated as a series of line segments.
        _interpolation_steps : int, optional
            Used as a hint to certain projections, such as Polar, that this
            path should be linearly interpolated immediately before drawing.
            This attribute is primarily an implementation detail and is not
            intended for public use.
        closed : bool, optional
            If *codes* is None and closed is True, vertices will be treated as
            line segments of a closed polygon.  Note that the last vertex will
            then be ignored (as the corresponding code will be set to
            `CLOSEPOLY`).
        readonly : bool, optional
            Makes the path behave in an immutable way and sets the vertices
            and codes as read-only arrays.
        """
        vertices = _to_unmasked_float_array(vertices)
        _api.check_shape((None, 2), vertices=vertices)

        if codes is not None and len(vertices):
            codes = np.asarray(codes, self.code_type)
            if codes.ndim != 1 or len(codes) != len(vertices):
                raise ValueError("'codes' must be a 1D list or array with the "
                                 "same length of 'vertices'. "
                                 f"Your vertices have shape {vertices.shape} "
                                 f"but your codes have shape {codes.shape}")
            if len(codes) and codes[0] != self.MOVETO:
                raise ValueError("The first element of 'code' must be equal "
                                 f"to 'MOVETO' ({self.MOVETO}).  "
                                 f"Your first code is {codes[0]}")
        elif closed and len(vertices):
            codes = np.empty(len(vertices), dtype=self.code_type)
            codes[0] = self.MOVETO
            codes[1:-1] = self.LINETO
            codes[-1] = self.CLOSEPOLY

        self._vertices = vertices
        self._codes = codes
        self._interpolation_steps = _interpolation_steps
        self._update_values()

        if readonly:
            self._vertices.flags.writeable = False
            if self._codes is not None:
                self._codes.flags.writeable = False
            self._readonly = True
        else:
            self._readonly = False

    @classmethod
    def _fast_from_codes_and_verts(cls, verts, codes, internals_from=None):
        """
        Create a Path instance without the expense of calling the constructor.

        Parameters
        ----------
        verts : array-like
        codes : array
        internals_from : Path or None
            If not None, another `Path` from which the attributes
            ``should_simplify``, ``simplify_threshold``, and
            ``interpolation_steps`` will be copied.  Note that ``readonly`` is
            never copied, and always set to ``False`` by this constructor.
        """
        pth = cls.__new__(cls)
        pth._vertices = _to_unmasked_float_array(verts)
        pth._codes = codes
        pth._readonly = False
        if internals_from is not None:
            pth._should_simplify = internals_from._should_simplify
            pth._simplify_threshold = internals_from._simplify_threshold
            pth._interpolation_steps = internals_from._interpolation_steps
        else:
            pth._should_simplify = True
            pth._simplify_threshold = mpl.rcParams['path.simplify_threshold']
            pth._interpolation_steps = 1
        return pth

    @classmethod
    def _create_closed(cls, vertices):
        """
        Create a closed polygonal path going through *vertices*.

        Unlike ``Path(..., closed=True)``, *vertices* should **not** end with
        an entry for the CLOSEPATH; this entry is added by `._create_closed`.
        """
        v = _to_unmasked_float_array(vertices)
        return cls(np.concatenate([v, v[:1]]), closed=True)

    def _update_values(self):
        self._simplify_threshold = mpl.rcParams['path.simplify_threshold']
        self._should_simplify = (
            self._simplify_threshold > 0 and
            mpl.rcParams['path.simplify'] and
            len(self._vertices) >= 128 and
            (self._codes is None or np.all(self._codes <= Path.LINETO))
        )

    @property
    def vertices(self):
        """The vertices of the `Path` as an (N, 2) array."""
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        if self._readonly:
            raise AttributeError("Can't set vertices on a readonly Path")
        self._vertices = vertices
        self._update_values()

    @property
    def codes(self):
        """
        The list of codes in the `Path` as a 1D array.

        Each code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4` or
        `CLOSEPOLY`.  For codes that correspond to more than one vertex
        (`CURVE3` and `CURVE4`), that code will be repeated so that the length
        of `vertices` and `codes` is always the same.
        """
        return self._codes

    @codes.setter
    def codes(self, codes):
        if self._readonly:
            raise AttributeError("Can't set codes on a readonly Path")
        self._codes = codes
        self._update_values()

    @property
    def simplify_threshold(self):
        """
        The fraction of a pixel difference below which vertices will
        be simplified out.
        """
        return self._simplify_threshold

    @simplify_threshold.setter
    def simplify_threshold(self, threshold):
        self._simplify_threshold = threshold

    @property
    def should_simplify(self):
        """
        `True` if the vertices array should be simplified.
        """
        return self._should_simplify

    @should_simplify.setter
    def should_simplify(self, should_simplify):
        self._should_simplify = should_simplify

    @property
    def readonly(self):
        """
        `True` if the `Path` is read-only.
        """
        return self._readonly

    def copy(self):
        """
        Return a shallow copy of the `Path`, which will share the
        vertices and codes with the source `Path`.
        """
        return copy.copy(self)

    def __deepcopy__(self, memo):
        """
        Return a deepcopy of the `Path`.  The `Path` will not be
        readonly, even if the source `Path` is.
        """
        # Deepcopying arrays (vertices, codes) strips the writeable=False flag.
        cls = type(self)
        memo[id(self)] = p = cls.__new__(cls)

        for k, v in self.__dict__.items():
            setattr(p, k, copy.deepcopy(v, memo))

        p._readonly = False
        return p

    def deepcopy(self, memo=None):
        """
        Return a deep copy of the `Path`.  The `Path` will not be readonly,
        even if the source `Path` is.

        Parameters
        ----------
        memo : dict, optional
            A dictionary to use for memoizing, passed to `copy.deepcopy`.

        Returns
        -------
        Path
            A deep copy of the `Path`, but not readonly.
        """
        return copy.deepcopy(self, memo)

    @classmethod
    def make_compound_path_from_polys(cls, XY):
        """
        Make a compound `Path` object to draw a number of polygons with equal
        numbers of sides.

        .. plot:: gallery/misc/histogram_path.py

        Parameters
        ----------
        XY : (numpolys, numsides, 2) array
        """
        # for each poly: 1 for the MOVETO, (numsides-1) for the LINETO, 1 for
        # the CLOSEPOLY; the vert for the closepoly is ignored but we still
        # need it to keep the codes aligned with the vertices
        numpolys, numsides, two = XY.shape
        if two != 2:
            raise ValueError("The third dimension of 'XY' must be 2")
        stride = numsides + 1
        nverts = numpolys * stride
        verts = np.zeros((nverts, 2))
        codes = np.full(nverts, cls.LINETO, dtype=cls.code_type)
        codes[0::stride] = cls.MOVETO
        codes[numsides::stride] = cls.CLOSEPOLY
        for i in range(numsides):
            verts[i::stride] = XY[:, i]
        return cls(verts, codes)

    @classmethod
    def make_compound_path(cls, *args):
        r"""
        Concatenate a list of `Path`\s into a single `Path`, removing all `STOP`\s.
        """
        if not args:
            return Path(np.empty([0, 2], dtype=np.float32))
        vertices = np.concatenate([path.vertices for path in args])
        codes = np.empty(len(vertices), dtype=cls.code_type)
        i = 0
        for path in args:
            size = len(path.vertices)
            if path.codes is None:
                if size:
                    codes[i] = cls.MOVETO
                    codes[i+1:i+size] = cls.LINETO
            else:
                codes[i:i+size] = path.codes
            i += size
        not_stop_mask = codes != cls.STOP  # Remove STOPs, as internal STOPs are a bug.
        return cls(vertices[not_stop_mask], codes[not_stop_mask])

    def __repr__(self):
        return f"Path({self.vertices!r}, {self.codes!r})"

    def __len__(self):
        return len(self.vertices)

    def iter_segments(self, transform=None, remove_nans=True, clip=None,
                      snap=False, stroke_width=1.0, simplify=None,
                      curves=True, sketch=None):
        """
        Iterate over all curve segments in the path.

        Each iteration returns a pair ``(vertices, code)``, where ``vertices``
        is a sequence of 1-3 coordinate pairs, and ``code`` is a `Path` code.

        Additionally, this method can provide a number of standard cleanups and
        conversions to the path.

        Parameters
        ----------
        transform : None or :class:`~matplotlib.transforms.Transform`
            If not None, the given affine transformation will be applied to the
            path.
        remove_nans : bool, optional
            Whether to remove all NaNs from the path and skip over them using
            MOVETO commands.
        clip : None or (float, float, float, float), optional
            If not None, must be a four-tuple (x1, y1, x2, y2)
            defining a rectangle in which to clip the path.
        snap : None or bool, optional
            If True, snap all nodes to pixels; if False, don't snap them.
            If None, snap if the path contains only segments
            parallel to the x or y axes, and no more than 1024 of them.
        stroke_width : float, optional
            The width of the stroke being drawn (used for path snapping).
        simplify : None or bool, optional
            Whether to simplify the path by removing vertices
            that do not affect its appearance.  If None, use the
            :attr:`should_simplify` attribute.  See also :rc:`path.simplify`
            and :rc:`path.simplify_threshold`.
        curves : bool, optional
            If True, curve segments will be returned as curve segments.
            If False, all curves will be converted to line segments.
        sketch : None or sequence, optional
            If not None, must be a 3-tuple of the form
            (scale, length, randomness), representing the sketch parameters.
        """
        if not len(self):
            return

        cleaned = self.cleaned(transform=transform,
                               remove_nans=remove_nans, clip=clip,
                               snap=snap, stroke_width=stroke_width,
                               simplify=simplify, curves=curves,
                               sketch=sketch)

        # Cache these object lookups for performance in the loop.
        NUM_VERTICES_FOR_CODE = self.NUM_VERTICES_FOR_CODE
        STOP = self.STOP

        vertices = iter(cleaned.vertices)
        codes = iter(cleaned.codes)
        for curr_vertices, code in zip(vertices, codes):
            if code == STOP:
                break
            extra_vertices = NUM_VERTICES_FOR_CODE[code] - 1
            if extra_vertices:
                for i in range(extra_vertices):
                    next(codes)
                    curr_vertices = np.append(curr_vertices, next(vertices))
            yield curr_vertices, code

    def iter_bezier(self, **kwargs):
        """
        Iterate over each Bézier curve (lines included) in a `Path`.

        Parameters
        ----------
        **kwargs
            Forwarded to `.iter_segments`.

        Yields
        ------
        B : `~matplotlib.bezier.BezierSegment`
            The Bézier curves that make up the current path. Note in particular
            that freestanding points are Bézier curves of order 0, and lines
            are Bézier curves of order 1 (with two control points).
        code : `~matplotlib.path.Path.code_type`
            The code describing what kind of curve is being returned.
            `MOVETO`, `LINETO`, `CURVE3`, and `CURVE4` correspond to
            Bézier curves with 1, 2, 3, and 4 control points (respectively).
            `CLOSEPOLY` is a `LINETO` with the control points correctly
            chosen based on the start/end points of the current stroke.
        """
        first_vert = None
        prev_vert = None
        for verts, code in self.iter_segments(**kwargs):
            if first_vert is None:
                if code != Path.MOVETO:
                    raise ValueError("Malformed path, must start with MOVETO.")
            if code == Path.MOVETO:  # a point is like "CURVE1"
                first_vert = verts
                yield BezierSegment(np.array([first_vert])), code
            elif code == Path.LINETO:  # "CURVE2"
                yield BezierSegment(np.array([prev_vert, verts])), code
            elif code == Path.CURVE3:
                yield BezierSegment(np.array([prev_vert, verts[:2],
                                              verts[2:]])), code
            elif code == Path.CURVE4:
                yield BezierSegment(np.array([prev_vert, verts[:2],
                                              verts[2:4], verts[4:]])), code
            elif code == Path.CLOSEPOLY:
                yield BezierSegment(np.array([prev_vert, first_vert])), code
            elif code == Path.STOP:
                return
            else:
                raise ValueError(f"Invalid Path.code_type: {code}")
            prev_vert = verts[-2:]

    def _iter_connected_components(self):
        """Return subpaths split at MOVETOs."""
        if self.codes is None:
            yield self
        else:
            idxs = np.append((self.codes == Path.MOVETO).nonzero()[0], len(self.codes))
            for sl in map(slice, idxs, idxs[1:]):
                yield Path._fast_from_codes_and_verts(
                    self.vertices[sl], self.codes[sl], self)

    def cleaned(self, transform=None, remove_nans=False, clip=None,
                *, simplify=False, curves=False,
                stroke_width=1.0, snap=False, sketch=None):
        """
        Return a new `Path` with vertices and codes cleaned according to the
        parameters.

        See Also
        --------
        Path.iter_segments : for details of the keyword arguments.
        """
        vertices, codes = _path.cleanup_path(
            self, transform, remove_nans, clip, snap, stroke_width, simplify,
            curves, sketch)
        pth = Path._fast_from_codes_and_verts(vertices, codes, self)
        if not simplify:
            pth._should_simplify = False
        return pth

    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        See Also
        --------
        matplotlib.transforms.TransformedPath
            A specialized path class that will cache the transformed result and
            automatically update when the transform changes.
        """
        return Path(transform.transform(self.vertices), self.codes,
                    self._interpolation_steps)

    def contains_point(self, point, transform=None, radius=0.0):
        """
        Return whether the area enclosed by the path contains the given point.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *point* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *point*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *point*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        bool

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
        if transform is not None:
            transform = transform.frozen()
        # `point_in_path` does not handle nonlinear transforms, so we
        # transform the path ourselves.  If *transform* is affine, letting
        # `point_in_path` handle the transform avoids allocating an extra
        # buffer.
        if transform and not transform.is_affine:
            self = transform.transform_path(self)
            transform = None
        return _path.point_in_path(point[0], point[1], radius, self, transform)

    def contains_points(self, points, transform=None, radius=0.0):
        """
        Return whether the area enclosed by the path contains the given points.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        points : (N, 2) array
            The points to check. Columns contain x and y values.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *points* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *points*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *points*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
        if transform is not None:
            transform = transform.frozen()
        result = _path.points_in_path(points, radius, self, transform)
        return result.astype('bool')

    def contains_path(self, path, transform=None):
        """
        Return whether this (closed) path completely contains the given path.

        If *transform* is not ``None``, the path will be transformed before
        checking for containment.
        """
        if transform is not None:
            transform = transform.frozen()
        return _path.path_in_path(self, None, path, transform)

    def get_extents(self, transform=None, **kwargs):
        """
        Get Bbox of the path.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`, optional
            Transform to apply to path before computing extents, if any.
        **kwargs
            Forwarded to `.iter_bezier`.

        Returns
        -------
        matplotlib.transforms.Bbox
            The extents of the path Bbox([[xmin, ymin], [xmax, ymax]])
        """
        from .transforms import Bbox
        if transform is not None:
            self = transform.transform_path(self)
        if self.codes is None:
            xys = self.vertices
        elif len(np.intersect1d(self.codes, [Path.CURVE3, Path.CURVE4])) == 0:
            # Optimization for the straight line case.
            # Instead of iterating through each curve, consider
            # each line segment's end-points
            # (recall that STOP and CLOSEPOLY vertices are ignored)
            xys = self.vertices[np.isin(self.codes,
                                        [Path.MOVETO, Path.LINETO])]
        else:
            xys = []
            for curve, code in self.iter_bezier(**kwargs):
                # places where the derivative is zero can be extrema
                _, dzeros = curve.axis_aligned_extrema()
                # as can the ends of the curve
                xys.append(curve([0, *dzeros, 1]))
            xys = np.concatenate(xys)
        if len(xys):
            return Bbox([xys.min(axis=0), xys.max(axis=0)])
        else:
            return Bbox.null()

    def intersects_path(self, other, filled=True):
        """
        Return whether if this path intersects another given path.

        If *filled* is True, then this also returns True if one path completely
        encloses the other (i.e., the paths are treated as filled).
        """
        return _path.path_intersects_path(self, other, filled)

    def intersects_bbox(self, bbox, filled=True):
        """
        Return whether this path intersects a given `~.transforms.Bbox`.

        If *filled* is True, then this also returns True if the path completely
        encloses the `.Bbox` (i.e., the path is treated as filled).

        The bounding box is always considered filled.
        """
        return _path.path_intersects_rectangle(
            self, bbox.x0, bbox.y0, bbox.x1, bbox.y1, filled)

    def interpolated(self, steps):
        """
        Return a new path with each segment divided into *steps* parts.

        Codes other than `LINETO`, `MOVETO`, and `CLOSEPOLY` are not handled correctly.

        Parameters
        ----------
        steps : int
            The number of segments in the new path for each in the original.

        Returns
        -------
        Path
            The interpolated path.
        """
        if steps == 1 or len(self) == 0:
            return self

        if self.codes is not None and self.MOVETO in self.codes[1:]:
            return self.make_compound_path(
                *(p.interpolated(steps) for p in self._iter_connected_components()))

        if self.codes is not None and self.CLOSEPOLY in self.codes and not np.all(
                self.vertices[self.codes == self.CLOSEPOLY] == self.vertices[0]):
            vertices = self.vertices.copy()
            vertices[self.codes == self.CLOSEPOLY] = vertices[0]
        else:
            vertices = self.vertices

        vertices = simple_linear_interpolation(vertices, steps)
        codes = self.codes
        if codes is not None:
            new_codes = np.full((len(codes) - 1) * steps + 1, Path.LINETO,
                                dtype=self.code_type)
            new_codes[0::steps] = codes
        else:
            new_codes = None
        return Path(vertices, new_codes)

    def to_polygons(self, transform=None, width=0, height=0, closed_only=True):
        """
        Convert this path to a list of polygons or polylines.  Each
        polygon/polyline is an (N, 2) array of vertices.  In other words,
        each polygon has no `MOVETO` instructions or curves.  This
        is useful for displaying in backends that do not support
        compound paths or Bézier curves.

        If *width* and *height* are both non-zero then the lines will
        be simplified so that vertices outside of (0, 0), (width,
        height) will be clipped.

        The resulting polygons will be simplified if the
        :attr:`Path.should_simplify` attribute of the path is `True`.

        If *closed_only* is `True` (default), only closed polygons,
        with the last point being the same as the first point, will be
        returned.  Any unclosed polylines in the path will be
        explicitly closed.  If *closed_only* is `False`, any unclosed
        polygons in the path will be returned as unclosed polygons,
        and the closed polygons will be returned explicitly closed by
        setting the last point to the same as the first point.
        """
        if len(self.vertices) == 0:
            return []

        if transform is not None:
            transform = transform.frozen()

        if self.codes is None and (width == 0 or height == 0):
            vertices = self.vertices
            if closed_only:
                if len(vertices) < 3:
                    return []
                elif np.any(vertices[0] != vertices[-1]):
                    vertices = [*vertices, vertices[0]]

            if transform is None:
                return [vertices]
            else:
                return [transform.transform(vertices)]

        # Deal with the case where there are curves and/or multiple
        # subpaths (using extension code)
        return _path.convert_path_to_polygons(
            self, transform, width, height, closed_only)

    _unit_rectangle = None

    @classmethod
    def unit_rectangle(cls):
        """
        Return a `Path` instance of the unit rectangle from (0, 0) to (1, 1).
        """
        if cls._unit_rectangle is None:
            cls._unit_rectangle = cls([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                                      closed=True, readonly=True)
        return cls._unit_rectangle

    _unit_regular_polygons = WeakValueDictionary()

    @classmethod
    def unit_regular_polygon(cls, numVertices):
        """
        Return a :class:`Path` instance for a unit regular polygon with the
        given *numVertices* such that the circumscribing circle has radius 1.0,
        centered at (0, 0).
        """
        if numVertices <= 16:
            path = cls._unit_regular_polygons.get(numVertices)
        else:
            path = None
        if path is None:
            theta = ((2 * np.pi / numVertices) * np.arange(numVertices + 1)
                     # This initial rotation is to make sure the polygon always
                     # "points-up".
                     + np.pi / 2)
            verts = np.column_stack((np.cos(theta), np.sin(theta)))
            path = cls(verts, closed=True, readonly=True)
            if numVertices <= 16:
                cls._unit_regular_polygons[numVertices] = path
        return path

    _unit_regular_stars = WeakValueDictionary()

    @classmethod
    def unit_regular_star(cls, numVertices, innerCircle=0.5):
        """
        Return a :class:`Path` for a unit regular star with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
        if numVertices <= 16:
            path = cls._unit_regular_stars.get((numVertices, innerCircle))
        else:
            path = None
        if path is None:
            ns2 = numVertices * 2
            theta = (2*np.pi/ns2 * np.arange(ns2 + 1))
            # This initial rotation is to make sure the polygon always
            # "points-up"
            theta += np.pi / 2.0
            r = np.ones(ns2 + 1)
            r[1::2] = innerCircle
            verts = (r * np.vstack((np.cos(theta), np.sin(theta)))).T
            path = cls(verts, closed=True, readonly=True)
            if numVertices <= 16:
                cls._unit_regular_stars[(numVertices, innerCircle)] = path
        return path

    @classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        Return a :class:`Path` for a unit regular asterisk with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
        return cls.unit_regular_star(numVertices, 0.0)

    _unit_circle = None

    @classmethod
    def unit_circle(cls):
        """
        Return the readonly :class:`Path` of the unit circle.

        For most cases, :func:`Path.circle` will be what you want.
        """
        if cls._unit_circle is None:
            cls._unit_circle = cls.circle(center=(0, 0), radius=1,
                                          readonly=True)
        return cls._unit_circle

    @classmethod
    def circle(cls, center=(0., 0.), radius=1., readonly=False):
        """
        Return a `Path` representing a circle of a given radius and center.

        Parameters
        ----------
        center : (float, float), default: (0, 0)
            The center of the circle.
        radius : float, default: 1
            The radius of the circle.
        readonly : bool
            Whether the created path should have the "readonly" argument
            set when creating the Path instance.

        Notes
        -----
        The circle is approximated using 8 cubic Bézier curves, as described in

          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
          Bezier Cubic Splines <https://www.tinaja.com/glib/ellipse4.pdf>`_.
        """
        MAGIC = 0.2652031
        SQRTHALF = np.sqrt(0.5)
        MAGIC45 = SQRTHALF * MAGIC

        vertices = np.array([[0.0, -1.0],

                             [MAGIC, -1.0],
                             [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
                             [SQRTHALF, -SQRTHALF],

                             [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
                             [1.0, -MAGIC],
                             [1.0, 0.0],

                             [1.0, MAGIC],
                             [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
                             [SQRTHALF, SQRTHALF],

                             [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
                             [MAGIC, 1.0],
                             [0.0, 1.0],

                             [-MAGIC, 1.0],
                             [-SQRTHALF+MAGIC45, SQRTHALF+MAGIC45],
                             [-SQRTHALF, SQRTHALF],

                             [-SQRTHALF-MAGIC45, SQRTHALF-MAGIC45],
                             [-1.0, MAGIC],
                             [-1.0, 0.0],

                             [-1.0, -MAGIC],
                             [-SQRTHALF-MAGIC45, -SQRTHALF+MAGIC45],
                             [-SQRTHALF, -SQRTHALF],

                             [-SQRTHALF+MAGIC45, -SQRTHALF-MAGIC45],
                             [-MAGIC, -1.0],
                             [0.0, -1.0],

                             [0.0, -1.0]],
                            dtype=float)

        codes = [cls.CURVE4] * 26
        codes[0] = cls.MOVETO
        codes[-1] = cls.CLOSEPOLY
        return Path(vertices * radius + center, codes, readonly=readonly)

    _unit_circle_righthalf = None

    @classmethod
    def unit_circle_righthalf(cls):
        """
        Return a `Path` of the right half of a unit circle.

        See `Path.circle` for the reference on the approximation used.
        """
        if cls._unit_circle_righthalf is None:
            MAGIC = 0.2652031
            SQRTHALF = np.sqrt(0.5)
            MAGIC45 = SQRTHALF * MAGIC

            vertices = np.array(
                [[0.0, -1.0],

                 [MAGIC, -1.0],
                 [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
                 [SQRTHALF, -SQRTHALF],

                 [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
                 [1.0, -MAGIC],
                 [1.0, 0.0],

                 [1.0, MAGIC],
                 [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
                 [SQRTHALF, SQRTHALF],

                 [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
                 [MAGIC, 1.0],
                 [0.0, 1.0],

                 [0.0, -1.0]],

                float)

            codes = np.full(14, cls.CURVE4, dtype=cls.code_type)
            codes[0] = cls.MOVETO
            codes[-1] = cls.CLOSEPOLY

            cls._unit_circle_righthalf = cls(vertices, codes, readonly=True)
        return cls._unit_circle_righthalf

    @classmethod
    def arc(cls, theta1, theta2, n=None, is_wedge=False):
        """
        Return a `Path` for the unit circle arc from angles *theta1* to
        *theta2* (in degrees).

        *theta2* is unwrapped to produce the shortest arc within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
        *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

           Masionobe, L.  2003.  `Drawing an elliptical arc using
           polylines, quadratic or cubic Bezier curves
           <https://web.archive.org/web/20190318044212/http://www.spaceroots.org/documents/ellipse/index.html>`_.
        """
        halfpi = np.pi * 0.5

        eta1 = theta1
        eta2 = theta2 - 360 * np.floor((theta2 - theta1) / 360)
        # Ensure 2pi range is not flattened to 0 due to floating-point errors,
        # but don't try to expand existing 0 range.
        if theta2 != theta1 and eta2 <= eta1:
            eta2 += 360
        eta1, eta2 = np.deg2rad([eta1, eta2])

        # number of curve segments to make
        if n is None:
            n = int(2 ** np.ceil((eta2 - eta1) / halfpi))
        if n < 1:
            raise ValueError("n must be >= 1 or None")

        deta = (eta2 - eta1) / n
        t = np.tan(0.5 * deta)
        alpha = np.sin(deta) * (np.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0

        steps = np.linspace(eta1, eta2, n + 1, True)
        cos_eta = np.cos(steps)
        sin_eta = np.sin(steps)

        xA = cos_eta[:-1]
        yA = sin_eta[:-1]
        xA_dot = -yA
        yA_dot = xA

        xB = cos_eta[1:]
        yB = sin_eta[1:]
        xB_dot = -yB
        yB_dot = xB

        if is_wedge:
            length = n * 3 + 4
            vertices = np.zeros((length, 2), float)
            codes = np.full(length, cls.CURVE4, dtype=cls.code_type)
            vertices[1] = [xA[0], yA[0]]
            codes[0:2] = [cls.MOVETO, cls.LINETO]
            codes[-2:] = [cls.LINETO, cls.CLOSEPOLY]
            vertex_offset = 2
            end = length - 2
        else:
            length = n * 3 + 1
            vertices = np.empty((length, 2), float)
            codes = np.full(length, cls.CURVE4, dtype=cls.code_type)
            vertices[0] = [xA[0], yA[0]]
            codes[0] = cls.MOVETO
            vertex_offset = 1
            end = length

        vertices[vertex_offset:end:3, 0] = xA + alpha * xA_dot
        vertices[vertex_offset:end:3, 1] = yA + alpha * yA_dot
        vertices[vertex_offset+1:end:3, 0] = xB - alpha * xB_dot
        vertices[vertex_offset+1:end:3, 1] = yB - alpha * yB_dot
        vertices[vertex_offset+2:end:3, 0] = xB
        vertices[vertex_offset+2:end:3, 1] = yB

        return cls(vertices, codes, readonly=True)

    @classmethod
    def wedge(cls, theta1, theta2, n=None):
        """
        Return a `Path` for the unit circle wedge from angles *theta1* to
        *theta2* (in degrees).

        *theta2* is unwrapped to produce the shortest wedge within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the wedge will be from *theta1*
        to *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

        See `Path.arc` for the reference on the approximation used.
        """
        return cls.arc(theta1, theta2, n, True)

    @staticmethod
    @lru_cache(8)
    def hatch(hatchpattern, density=6):
        """
        Given a hatch specifier, *hatchpattern*, generates a `Path` that
        can be used in a repeated hatching pattern.  *density* is the
        number of lines per unit square.
        """
        from matplotlib.hatch import get_path
        return (get_path(hatchpattern, density)
                if hatchpattern is not None else None)

    def clip_to_bbox(self, bbox, inside=True):
        """
        Clip the path to the given bounding box.

        The path must be made up of one or more closed polygons.  This
        algorithm will not behave correctly for unclosed paths.

        If *inside* is `True`, clip to the inside of the box, otherwise
        to the outside of the box.
        """
        verts = _path.clip_path_to_rect(self, bbox, inside)
        paths = [Path(poly) for poly in verts]
        return self.make_compound_path(*paths)


def get_path_collection_extents(
        master_transform, paths, transforms, offsets, offset_transform):
    r"""
    Get bounding box of a `.PathCollection`\s internal objects.

    That is, given a sequence of `Path`\s, `.Transform`\s objects, and offsets, as found
    in a `.PathCollection`, return the bounding box that encapsulates all of them.

    Parameters
    ----------
    master_transform : `~matplotlib.transforms.Transform`
        Global transformation applied to all paths.
    paths : list of `Path`
    transforms : list of `~matplotlib.transforms.Affine2DBase`
        If non-empty, this overrides *master_transform*.
    offsets : (N, 2) array-like
    offset_transform : `~matplotlib.transforms.Affine2DBase`
        Transform applied to the offsets before offsetting the path.

    Notes
    -----
    The way that *paths*, *transforms* and *offsets* are combined follows the same
    method as for collections: each is iterated over independently, so if you have 3
    paths (A, B, C), 2 transforms (α, β) and 1 offset (O), their combinations are as
    follows:

    - (A, α, O)
    - (B, β, O)
    - (C, α, O)
    """
    from .transforms import Bbox
    if len(paths) == 0:
        raise ValueError("No paths provided")
    if len(offsets) == 0:
        raise ValueError("No offsets provided")
    extents, minpos = _path.get_path_collection_extents(
        master_transform, paths, np.atleast_3d(transforms),
        offsets, offset_transform)
    return Bbox.from_extents(*extents, minpos=minpos)
<|endoftext|>"""
Manage figures for the pyplot interface.
"""

import atexit
from collections import OrderedDict


class Gcf:
    """
    Singleton to maintain the relation between figures and their managers, and
    keep track of and "active" figure and manager.

    The canvas of a figure created through pyplot is associated with a figure
    manager, which handles the interaction between the figure and the backend.
    pyplot keeps track of figure managers using an identifier, the "figure
    number" or "manager number" (which can actually be any hashable value);
    this number is available as the :attr:`number` attribute of the manager.

    This class is never instantiated; it consists of an `OrderedDict` mapping
    figure/manager numbers to managers, and a set of class methods that
    manipulate this `OrderedDict`.

    Attributes
    ----------
    figs : OrderedDict
        `OrderedDict` mapping numbers to managers; the active manager is at the
        end.
    """

    figs = OrderedDict()

    @classmethod
    def get_fig_manager(cls, num):
        """
        If manager number *num* exists, make it the active one and return it;
        otherwise return *None*.
        """
        manager = cls.figs.get(num, None)
        if manager is not None:
            cls.set_active(manager)
        return manager

    @classmethod
    def destroy(cls, num):
        """
        Destroy manager *num* -- either a manager instance or a manager number.

        In the interactive backends, this is bound to the window "destroy" and
        "delete" events.

        It is recommended to pass a manager instance, to avoid confusion when
        two managers share the same number.
        """
        if all(hasattr(num, attr) for attr in ["num", "destroy"]):
            manager = num
            if cls.figs.get(manager.num) is manager:
                cls.figs.pop(manager.num)
        else:
            try:
                manager = cls.figs.pop(num)
            except KeyError:
                return
        if hasattr(manager, "_cidgcf"):
            manager.canvas.mpl_disconnect(manager._cidgcf)
        manager.destroy()

    @classmethod
    def destroy_fig(cls, fig):
        """Destroy figure *fig*."""
        num = next((manager.num for manager in cls.figs.values()
                    if manager.canvas.figure == fig), None)
        if num is not None:
            cls.destroy(num)

    @classmethod
    def destroy_all(cls):
        """Destroy all figures."""
        for manager in list(cls.figs.values()):
            manager.canvas.mpl_disconnect(manager._cidgcf)
            manager.destroy()
        cls.figs.clear()

    @classmethod
    def has_fignum(cls, num):
        """Return whether figure number *num* exists."""
        return num in cls.figs

    @classmethod
    def get_all_fig_managers(cls):
        """Return a list of figure managers."""
        return list(cls.figs.values())

    @classmethod
    def get_num_fig_managers(cls):
        """Return the number of figures being managed."""
        return len(cls.figs)

    @classmethod
    def get_active(cls):
        """Return the active manager, or *None* if there is no manager."""
        return next(reversed(cls.figs.values())) if cls.figs else None

    @classmethod
    def _set_new_active_manager(cls, manager):
        """Adopt *manager* into pyplot and make it the active manager."""
        if not hasattr(manager, "_cidgcf"):
            manager._cidgcf = manager.canvas.mpl_connect(
                "button_press_event", lambda event: cls.set_active(manager))
        fig = manager.canvas.figure
        fig._number = manager.num
        label = fig.get_label()
        if label:
            manager.set_window_title(label)
        cls.set_active(manager)

    @classmethod
    def set_active(cls, manager):
        """Make *manager* the active manager."""
        cls.figs[manager.num] = manager
        cls.figs.move_to_end(manager.num)

    @classmethod
    def draw_all(cls, force=False):
        """
        Redraw all stale managed figures, or, if *force* is True, all managed
        figures.
        """
        for manager in cls.get_all_fig_managers():
            if force or manager.canvas.figure.stale:
                manager.canvas.draw_idle()


atexit.register(Gcf.destroy_all)
<|endoftext|>def blocking_input_loop(figure, event_names, timeout, handler):
    """
    Run *figure*'s event loop while listening to interactive events.

    The events listed in *event_names* are passed to *handler*.

    This function is used to implement `.Figure.waitforbuttonpress`,
    `.Figure.ginput`, and `.Axes.clabel`.

    Parameters
    ----------
    figure : `~matplotlib.figure.Figure`
    event_names : list of str
        The names of the events passed to *handler*.
    timeout : float
        If positive, the event loop is stopped after *timeout* seconds.
    handler : Callable[[Event], Any]
        Function called for each event; it can force an early exit of the event
        loop by calling ``canvas.stop_event_loop()``.
    """
    if figure.canvas.manager:
        figure.show()  # Ensure that the figure is shown if we are managing it.
    # Connect the events to the on_event function call.
    cids = [figure.canvas.mpl_connect(name, handler) for name in event_names]
    try:
        figure.canvas.start_event_loop(timeout)  # Start event loop.
    finally:  # Run even on exception like ctrl-c.
        # Disconnect the callbacks.
        for cid in cids:
            figure.canvas.mpl_disconnect(cid)
<|endoftext|>r"""
Patches are `.Artist`\s with a face color and an edge color.
"""

import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D

import numpy as np

import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
               lines as mlines, transforms)
from .bezier import (
    NonIntersectingPathException, get_cos_sin, get_intersection,
    get_parallels, inside_circle, make_wedged_bezier2,
    split_bezier_intersecting_with_closedpath, split_path_inout)
from .path import Path
from ._enums import JoinStyle, CapStyle


@_docstring.interpd
@_api.define_aliases({
    "antialiased": ["aa"],
    "edgecolor": ["ec"],
    "facecolor": ["fc"],
    "linestyle": ["ls"],
    "linewidth": ["lw"],
})
class Patch(artist.Artist):
    """
    A patch is a 2D artist with a face color and an edge color.

    If any of *edgecolor*, *facecolor*, *linewidth*, or *antialiased*
    are *None*, they default to their rc params setting.
    """
    zorder = 1

    # Whether to draw an edge by default.  Set on a
    # subclass-by-subclass basis.
    _edge_default = False

    def __init__(self, *,
                 edgecolor=None,
                 facecolor=None,
                 color=None,
                 linewidth=None,
                 linestyle=None,
                 antialiased=None,
                 hatch=None,
                 fill=True,
                 capstyle=None,
                 joinstyle=None,
                 **kwargs):
        """
        The following kwarg properties are supported

        %(Patch:kwdoc)s
        """
        super().__init__()

        if linestyle is None:
            linestyle = "solid"
        if capstyle is None:
            capstyle = CapStyle.butt
        if joinstyle is None:
            joinstyle = JoinStyle.miter

        self._hatch_color = colors.to_rgba(mpl.rcParams['hatch.color'])
        self._hatch_linewidth = mpl.rcParams['hatch.linewidth']
        self._fill = bool(fill)  # needed for set_facecolor call
        if color is not None:
            if edgecolor is not None or facecolor is not None:
                _api.warn_external(
                    "Setting the 'color' property will override "
                    "the edgecolor or facecolor properties.")
            self.set_color(color)
        else:
            self.set_edgecolor(edgecolor)
            self.set_facecolor(facecolor)

        self._linewidth = 0
        self._unscaled_dash_pattern = (0, None)  # offset, dash
        self._dash_pattern = (0, None)  # offset, dash (scaled by linewidth)

        self.set_linestyle(linestyle)
        self.set_linewidth(linewidth)
        self.set_antialiased(antialiased)
        self.set_hatch(hatch)
        self.set_capstyle(capstyle)
        self.set_joinstyle(joinstyle)

        if len(kwargs):
            self._internal_update(kwargs)

    def get_verts(self):
        """
        Return a copy of the vertices used in this patch.

        If the patch contains Bézier curves, the curves will be interpolated by
        line segments.  To access the curves as curves, use `get_path`.
        """
        trans = self.get_transform()
        path = self.get_path()
        polygons = path.to_polygons(trans)
        if len(polygons):
            return polygons[0]
        return []

    def _process_radius(self, radius):
        if radius is not None:
            return radius
        if isinstance(self._picker, Number):
            _radius = self._picker
        else:
            if self.get_edgecolor()[3] == 0:
                _radius = 0
            else:
                _radius = self.get_linewidth()
        return _radius

    def contains(self, mouseevent, radius=None):
        """
        Test whether the mouse event occurred in the patch.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`
            Where the user clicked.

        radius : float, optional
            Additional margin on the patch in target coordinates of
            `.Patch.get_transform`. See `.Path.contains_point` for further
            details.

            If `None`, the default value depends on the state of the object:

            - If `.Artist.get_picker` is a number, the default
              is that value.  This is so that picking works as expected.
            - Otherwise if the edge color has a non-zero alpha, the default
              is half of the linewidth.  This is so that all the colored
              pixels are "in" the patch.
            - Finally, if the edge has 0 alpha, the default is 0.  This is
              so that patches without a stroked edge do not have points
              outside of the filled region report as "in" due to an
              invisible edge.


        Returns
        -------
        (bool, empty dict)
        """
        if self._different_canvas(mouseevent):
            return False, {}
        radius = self._process_radius(radius)
        codes = self.get_path().codes
        if codes is not None:
            vertices = self.get_path().vertices
            # if the current path is concatenated by multiple sub paths.
            # get the indexes of the starting code(MOVETO) of all sub paths
            idxs, = np.where(codes == Path.MOVETO)
            # Don't split before the first MOVETO.
            idxs = idxs[1:]
            subpaths = map(
                Path, np.split(vertices, idxs), np.split(codes, idxs))
        else:
            subpaths = [self.get_path()]
        inside = any(
            subpath.contains_point(
                (mouseevent.x, mouseevent.y), self.get_transform(), radius)
            for subpath in subpaths)
        return inside, {}

    def contains_point(self, point, radius=None):
        """
        Return whether the given point is inside the patch.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check, in target coordinates of
            ``.Patch.get_transform()``. These are display coordinates for patches
            that are added to a figure or Axes.
        radius : float, optional
            Additional margin on the patch in target coordinates of
            `.Patch.get_transform`. See `.Path.contains_point` for further
            details.

            If `None`, the default value depends on the state of the object:

            - If `.Artist.get_picker` is a number, the default
              is that value.  This is so that picking works as expected.
            - Otherwise if the edge color has a non-zero alpha, the default
              is half of the linewidth.  This is so that all the colored
              pixels are "in" the patch.
            - Finally, if the edge has 0 alpha, the default is 0.  This is
              so that patches without a stroked edge do not have points
              outside of the filled region report as "in" due to an
              invisible edge.

        Returns
        -------
        bool

        Notes
        -----
        The proper use of this method depends on the transform of the patch.
        Isolated patches do not have a transform. In this case, the patch
        creation coordinates and the point coordinates match. The following
        example checks that the center of a circle is within the circle

        >>> center = 0, 0
        >>> c = Circle(center, radius=1)
        >>> c.contains_point(center)
        True

        The convention of checking against the transformed patch stems from
        the fact that this method is predominantly used to check if display
        coordinates (e.g. from mouse events) are within the patch. If you want
        to do the above check with data coordinates, you have to properly
        transform them first:

        >>> center = 0, 0
        >>> c = Circle(center, radius=3)
        >>> plt.gca().add_patch(c)
        >>> transformed_interior_point = c.get_data_transform().transform((0, 2))
        >>> c.contains_point(transformed_interior_point)
        True

        """
        radius = self._process_radius(radius)
        return self.get_path().contains_point(point,
                                              self.get_transform(),
                                              radius)

    def contains_points(self, points, radius=None):
        """
        Return whether the given points are inside the patch.

        Parameters
        ----------
        points : (N, 2) array
            The points to check, in target coordinates of
            ``self.get_transform()``. These are display coordinates for patches
            that are added to a figure or Axes. Columns contain x and y values.
        radius : float, optional
            Additional margin on the patch in target coordinates of
            `.Patch.get_transform`. See `.Path.contains_point` for further
            details.

            If `None`, the default value depends on the state of the object:

            - If `.Artist.get_picker` is a number, the default
              is that value.  This is so that picking works as expected.
            - Otherwise if the edge color has a non-zero alpha, the default
              is half of the linewidth.  This is so that all the colored
              pixels are "in" the patch.
            - Finally, if the edge has 0 alpha, the default is 0.  This is
              so that patches without a stroked edge do not have points
              outside of the filled region report as "in" due to an
              invisible edge.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The proper use of this method depends on the transform of the patch.
        See the notes on `.Patch.contains_point`.
        """
        radius = self._process_radius(radius)
        return self.get_path().contains_points(points,
                                               self.get_transform(),
                                               radius)

    def update_from(self, other):
        # docstring inherited.
        super().update_from(other)
        # For some properties we don't need or don't want to go through the
        # getters/setters, so we just copy them directly.
        self._edgecolor = other._edgecolor
        self._facecolor = other._facecolor
        self._original_edgecolor = other._original_edgecolor
        self._original_facecolor = other._original_facecolor
        self._fill = other._fill
        self._hatch = other._hatch
        self._hatch_color = other._hatch_color
        self._unscaled_dash_pattern = other._unscaled_dash_pattern
        self.set_linewidth(other._linewidth)  # also sets scaled dashes
        self.set_transform(other.get_data_transform())
        # If the transform of other needs further initialization, then it will
        # be the case for this artist too.
        self._transformSet = other.is_transform_set()

    def get_extents(self):
        """
        Return the `Patch`'s axis-aligned extents as a `~.transforms.Bbox`.
        """
        return self.get_path().get_extents(self.get_transform())

    def get_transform(self):
        """Return the `~.transforms.Transform` applied to the `Patch`."""
        return self.get_patch_transform() + artist.Artist.get_transform(self)

    def get_data_transform(self):
        """
        Return the `~.transforms.Transform` mapping data coordinates to
        physical coordinates.
        """
        return artist.Artist.get_transform(self)

    def get_patch_transform(self):
        """
        Return the `~.transforms.Transform` instance mapping patch coordinates
        to data coordinates.

        For example, one may define a patch of a circle which represents a
        radius of 5 by providing coordinates for a unit circle, and a
        transform which scales the coordinates (the patch coordinate) by 5.
        """
        return transforms.IdentityTransform()

    def get_antialiased(self):
        """Return whether antialiasing is used for drawing."""
        return self._antialiased

    def get_edgecolor(self):
        """Return the edge color."""
        return self._edgecolor

    def get_facecolor(self):
        """Return the face color."""
        return self._facecolor

    def get_linewidth(self):
        """Return the line width in points."""
        return self._linewidth

    def get_linestyle(self):
        """Return the linestyle."""
        return self._linestyle

    def set_antialiased(self, aa):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        aa : bool or None
        """
        if aa is None:
            aa = mpl.rcParams['patch.antialiased']
        self._antialiased = aa
        self.stale = True

    def _set_edgecolor(self, color):
        set_hatch_color = True
        if color is None:
            if (mpl.rcParams['patch.force_edgecolor'] or
                    not self._fill or self._edge_default):
                color = mpl.rcParams['patch.edgecolor']
            else:
                color = 'none'
                set_hatch_color = False

        self._edgecolor = colors.to_rgba(color, self._alpha)
        if set_hatch_color:
            self._hatch_color = self._edgecolor
        self.stale = True

    def set_edgecolor(self, color):
        """
        Set the patch edge color.

        Parameters
        ----------
        color : :mpltype:`color` or None
        """
        self._original_edgecolor = color
        self._set_edgecolor(color)

    def _set_facecolor(self, color):
        if color is None:
            color = mpl.rcParams['patch.facecolor']
        alpha = self._alpha if self._fill else 0
        self._facecolor = colors.to_rgba(color, alpha)
        self.stale = True

    def set_facecolor(self, color):
        """
        Set the patch face color.

        Parameters
        ----------
        color : :mpltype:`color` or None
        """
        self._original_facecolor = color
        self._set_facecolor(color)

    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.

        Parameters
        ----------
        c : :mpltype:`color`

        See Also
        --------
        Patch.set_facecolor, Patch.set_edgecolor
            For setting the edge or face color individually.
        """
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def set_alpha(self, alpha):
        # docstring inherited
        super().set_alpha(alpha)
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)
        # stale is already True

    def set_linewidth(self, w):
        """
        Set the patch linewidth in points.

        Parameters
        ----------
        w : float or None
        """
        if w is None:
            w = mpl.rcParams['patch.linewidth']
        self._linewidth = float(w)
        self._dash_pattern = mlines._scale_dashes(
            *self._unscaled_dash_pattern, w)
        self.stale = True

    def set_linestyle(self, ls):
        """
        Set the patch linestyle.

        ==========================================  =================
        linestyle                                   description
        ==========================================  =================
        ``'-'`` or ``'solid'``                      solid line
        ``'--'`` or  ``'dashed'``                   dashed line
        ``'-.'`` or  ``'dashdot'``                  dash-dotted line
        ``':'`` or ``'dotted'``                     dotted line
        ``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
        ==========================================  =================

        Alternatively a dash tuple of the following form can be provided::

            (offset, onoffseq)

        where ``onoffseq`` is an even length tuple of on and off ink in points.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            The line style.
        """
        if ls is None:
            ls = "solid"
        if ls in [' ', '', 'none']:
            ls = 'None'
        self._linestyle = ls
        self._unscaled_dash_pattern = mlines._get_dash_pattern(ls)
        self._dash_pattern = mlines._scale_dashes(
            *self._unscaled_dash_pattern, self._linewidth)
        self.stale = True

    def set_fill(self, b):
        """
        Set whether to fill the patch.

        Parameters
        ----------
        b : bool
        """
        self._fill = bool(b)
        self._set_facecolor(self._original_facecolor)
        self._set_edgecolor(self._original_edgecolor)
        self.stale = True

    def get_fill(self):
        """Return whether the patch is filled."""
        return self._fill

    # Make fill a property so as to preserve the long-standing
    # but somewhat inconsistent behavior in which fill was an
    # attribute.
    fill = property(get_fill, set_fill)

    @_docstring.interpd
    def set_capstyle(self, s):
        """
        Set the `.CapStyle`.

        The default capstyle is 'round' for `.FancyArrowPatch` and 'butt' for
        all other patches.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
        cs = CapStyle(s)
        self._capstyle = cs
        self.stale = True

    def get_capstyle(self):
        """Return the capstyle."""
        return self._capstyle.name

    @_docstring.interpd
    def set_joinstyle(self, s):
        """
        Set the `.JoinStyle`.

        The default joinstyle is 'round' for `.FancyArrowPatch` and 'miter' for
        all other patches.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
        js = JoinStyle(s)
        self._joinstyle = js
        self.stale = True

    def get_joinstyle(self):
        """Return the joinstyle."""
        return self._joinstyle.name

    def set_hatch(self, hatch):
        r"""
        Set the hatching pattern.

        *hatch* can be one of::

          /   - diagonal hatching
          \   - back diagonal
          |   - vertical
          -   - horizontal
          +   - crossed
          x   - crossed diagonal
          o   - small circle
          O   - large circle
          .   - dots
          *   - stars

        Letters can be combined, in which case all the specified
        hatchings are done.  If same letter repeats, it increases the
        density of hatching of that pattern.

        Parameters
        ----------
        hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
        """
        # Use validate_hatch(list) after deprecation.
        mhatch._validate_hatch_pattern(hatch)
        self._hatch = hatch
        self.stale = True

    def get_hatch(self):
        """Return the hatching pattern."""
        return self._hatch

    def set_hatch_linewidth(self, lw):
        """Set the hatch linewidth."""
        self._hatch_linewidth = lw

    def get_hatch_linewidth(self):
        """Return the hatch linewidth."""
        return self._hatch_linewidth

    def _draw_paths_with_artist_properties(
            self, renderer, draw_path_args_list):
        """
        ``draw()`` helper factored out for sharing with `FancyArrowPatch`.

        Configure *renderer* and the associated graphics context *gc*
        from the artist properties, then repeatedly call
        ``renderer.draw_path(gc, *draw_path_args)`` for each tuple
        *draw_path_args* in *draw_path_args_list*.
        """

        renderer.open_group('patch', self.get_gid())
        gc = renderer.new_gc()

        gc.set_foreground(self._edgecolor, isRGBA=True)

        lw = self._linewidth
        if self._edgecolor[3] == 0 or self._linestyle == 'None':
            lw = 0
        gc.set_linewidth(lw)
        gc.set_dashes(*self._dash_pattern)
        gc.set_capstyle(self._capstyle)
        gc.set_joinstyle(self._joinstyle)

        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_url(self._url)
        gc.set_snap(self.get_snap())

        gc.set_alpha(self._alpha)

        if self._hatch:
            gc.set_hatch(self._hatch)
            gc.set_hatch_color(self._hatch_color)
            gc.set_hatch_linewidth(self._hatch_linewidth)

        if self.get_sketch_params() is not None:
            gc.set_sketch_params(*self.get_sketch_params())

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        for draw_path_args in draw_path_args_list:
            renderer.draw_path(gc, *draw_path_args)

        gc.restore()
        renderer.close_group('patch')
        self.stale = False

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if not self.get_visible():
            return
        path = self.get_path()
        transform = self.get_transform()
        tpath = transform.transform_path_non_affine(path)
        affine = transform.get_affine()
        self._draw_paths_with_artist_properties(
            renderer,
            [(tpath, affine,
              # Work around a bug in the PDF and SVG renderers, which
              # do not draw the hatches if the facecolor is fully
              # transparent, but do if it is None.
              self._facecolor if self._facecolor[3] else None)])

    def get_path(self):
        """Return the path of this patch."""
        raise NotImplementedError('Derived must override')

    def get_window_extent(self, renderer=None):
        return self.get_path().get_extents(self.get_transform())

    def _convert_xy_units(self, xy):
        """Convert x and y units for a tuple (x, y)."""
        x = self.convert_xunits(xy[0])
        y = self.convert_yunits(xy[1])
        return x, y


class Shadow(Patch):
    def __str__(self):
        return f"Shadow({self.patch})"

    @_docstring.interpd
    def __init__(self, patch, ox, oy, *, shade=0.7, **kwargs):
        """
        Create a shadow of the given *patch*.

        By default, the shadow will have the same face color as the *patch*,
        but darkened. The darkness can be controlled by *shade*.

        Parameters
        ----------
        patch : `~matplotlib.patches.Patch`
            The patch to create the shadow for.
        ox, oy : float
            The shift of the shadow in data coordinates, scaled by a factor
            of dpi/72.
        shade : float, default: 0.7
            How the darkness of the shadow relates to the original color. If 1, the
            shadow is black, if 0, the shadow has the same color as the *patch*.

            .. versionadded:: 3.8

        **kwargs
            Properties of the shadow patch. Supported keys are:

            %(Patch:kwdoc)s
        """
        super().__init__()
        self.patch = patch
        self._ox, self._oy = ox, oy
        self._shadow_transform = transforms.Affine2D()

        self.update_from(self.patch)
        if not 0 <= shade <= 1:
            raise ValueError("shade must be between 0 and 1.")
        color = (1 - shade) * np.asarray(colors.to_rgb(self.patch.get_facecolor()))
        self.update({'facecolor': color, 'edgecolor': color, 'alpha': 0.5,
                     # Place shadow patch directly behind the inherited patch.
                     'zorder': np.nextafter(self.patch.zorder, -np.inf),
                     **kwargs})

    def _update_transform(self, renderer):
        ox = renderer.points_to_pixels(self._ox)
        oy = renderer.points_to_pixels(self._oy)
        self._shadow_transform.clear().translate(ox, oy)

    def get_path(self):
        return self.patch.get_path()

    def get_patch_transform(self):
        return self.patch.get_patch_transform() + self._shadow_transform

    def draw(self, renderer):
        self._update_transform(renderer)
        super().draw(renderer)


class Rectangle(Patch):
    """
    A rectangle defined via an anchor point *xy* and its *width* and *height*.

    The rectangle extends from ``xy[0]`` to ``xy[0] + width`` in x-direction
    and from ``xy[1]`` to ``xy[1] + height`` in y-direction. ::

      :                +------------------+
      :                |                  |
      :              height               |
      :                |                  |
      :               (xy)---- width -----+

    One may picture *xy* as the bottom left corner, but which corner *xy* is
    actually depends on the direction of the axis and the sign of *width*
    and *height*; e.g. *xy* would be the bottom right corner if the x-axis
    was inverted or if *width* was negative.
    """

    def __str__(self):
        pars = self._x0, self._y0, self._width, self._height, self.angle
        fmt = "Rectangle(xy=(%g, %g), width=%g, height=%g, angle=%g)"
        return fmt % pars

    @_docstring.interpd
    def __init__(self, xy, width, height, *,
                 angle=0.0, rotation_point='xy', **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about the rotation point.
        rotation_point : {'xy', 'center', (number, number)}, default: 'xy'
            If ``'xy'``, rotate around the anchor point. If ``'center'`` rotate
            around the center. If 2-tuple of number, rotate around this
            coordinate.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Patch` properties
            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        self._x0 = xy[0]
        self._y0 = xy[1]
        self._width = width
        self._height = height
        self.angle = float(angle)
        self.rotation_point = rotation_point
        # Required for RectangleSelector with axes aspect ratio != 1
        # The patch is defined in data coordinates and when changing the
        # selector with square modifier and not in data coordinates, we need
        # to correct for the aspect ratio difference between the data and
        # display coordinate systems. Its value is typically provide by
        # Axes._get_aspect_ratio()
        self._aspect_ratio_correction = 1.0
        self._convert_units()  # Validate the inputs.

    def get_path(self):
        """Return the vertices of the rectangle."""
        return Path.unit_rectangle()

    def _convert_units(self):
        """Convert bounds of the rectangle."""
        x0 = self.convert_xunits(self._x0)
        y0 = self.convert_yunits(self._y0)
        x1 = self.convert_xunits(self._x0 + self._width)
        y1 = self.convert_yunits(self._y0 + self._height)
        return x0, y0, x1, y1

    def get_patch_transform(self):
        # Note: This cannot be called until after this has been added to
        # an Axes, otherwise unit conversion will fail. This makes it very
        # important to call the accessor method and not directly access the
        # transformation member variable.
        bbox = self.get_bbox()
        if self.rotation_point == 'center':
            width, height = bbox.x1 - bbox.x0, bbox.y1 - bbox.y0
            rotation_point = bbox.x0 + width / 2., bbox.y0 + height / 2.
        elif self.rotation_point == 'xy':
            rotation_point = bbox.x0, bbox.y0
        else:
            rotation_point = self.rotation_point
        return transforms.BboxTransformTo(bbox) \
                + transforms.Affine2D() \
                .translate(-rotation_point[0], -rotation_point[1]) \
                .scale(1, self._aspect_ratio_correction) \
                .rotate_deg(self.angle) \
                .scale(1, 1 / self._aspect_ratio_correction) \
                .translate(*rotation_point)

    @property
    def rotation_point(self):
        """The rotation point of the patch."""
        return self._rotation_point

    @rotation_point.setter
    def rotation_point(self, value):
        if value in ['center', 'xy'] or (
                isinstance(value, tuple) and len(value) == 2 and
                isinstance(value[0], Real) and isinstance(value[1], Real)
                ):
            self._rotation_point = value
        else:
            raise ValueError("`rotation_point` must be one of "
                             "{'xy', 'center', (number, number)}.")

    def get_x(self):
        """Return the left coordinate of the rectangle."""
        return self._x0

    def get_y(self):
        """Return the bottom coordinate of the rectangle."""
        return self._y0

    def get_xy(self):
        """Return the left and bottom coords of the rectangle as a tuple."""
        return self._x0, self._y0

    def get_corners(self):
        """
        Return the corners of the rectangle, moving anti-clockwise from
        (x0, y0).
        """
        return self.get_patch_transform().transform(
            [(0, 0), (1, 0), (1, 1), (0, 1)])

    def get_center(self):
        """Return the centre of the rectangle."""
        return self.get_patch_transform().transform((0.5, 0.5))

    def get_width(self):
        """Return the width of the rectangle."""
        return self._width

    def get_height(self):
        """Return the height of the rectangle."""
        return self._height

    def get_angle(self):
        """Get the rotation angle in degrees."""
        return self.angle

    def set_x(self, x):
        """Set the left coordinate of the rectangle."""
        self._x0 = x
        self.stale = True

    def set_y(self, y):
        """Set the bottom coordinate of the rectangle."""
        self._y0 = y
        self.stale = True

    def set_angle(self, angle):
        """
        Set the rotation angle in degrees.

        The rotation is performed anti-clockwise around *xy*.
        """
        self.angle = angle
        self.stale = True

    def set_xy(self, xy):
        """
        Set the left and bottom coordinates of the rectangle.

        Parameters
        ----------
        xy : (float, float)
        """
        self._x0, self._y0 = xy
        self.stale = True

    def set_width(self, w):
        """Set the width of the rectangle."""
        self._width = w
        self.stale = True

    def set_height(self, h):
        """Set the height of the rectangle."""
        self._height = h
        self.stale = True

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle as *left*, *bottom*, *width*, *height*.

        The values may be passed as separate parameters or as a tuple::

            set_bounds(left, bottom, width, height)
            set_bounds((left, bottom, width, height))

        .. ACCEPTS: (left, bottom, width, height)
        """
        if len(args) == 1:
            l, b, w, h = args[0]
        else:
            l, b, w, h = args
        self._x0 = l
        self._y0 = b
        self._width = w
        self._height = h
        self.stale = True

    def get_bbox(self):
        """Return the `.Bbox`."""
        return transforms.Bbox.from_extents(*self._convert_units())

    xy = property(get_xy, set_xy)


class RegularPolygon(Patch):
    """A regular polygon patch."""

    def __str__(self):
        s = "RegularPolygon((%g, %g), %d, radius=%g, orientation=%g)"
        return s % (self.xy[0], self.xy[1], self.numvertices, self.radius,
                    self.orientation)

    @_docstring.interpd
    def __init__(self, xy, numVertices, *,
                 radius=5, orientation=0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            The center position.

        numVertices : int
            The number of vertices.

        radius : float
            The distance from the center to each of the vertices.

        orientation : float
            The polygon rotation angle (in radians).

        **kwargs
            `Patch` properties:

            %(Patch:kwdoc)s
        """
        self.xy = xy
        self.numvertices = numVertices
        self.orientation = orientation
        self.radius = radius
        self._path = Path.unit_regular_polygon(numVertices)
        self._patch_transform = transforms.Affine2D()
        super().__init__(**kwargs)

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        return self._patch_transform.clear() \
            .scale(self.radius) \
            .rotate(self.orientation) \
            .translate(*self.xy)


class PathPatch(Patch):
    """A general polycurve path patch."""

    _edge_default = True

    def __str__(self):
        s = "PathPatch%d((%g, %g) ...)"
        return s % (len(self._path.vertices), *tuple(self._path.vertices[0]))

    @_docstring.interpd
    def __init__(self, path, **kwargs):
        """
        *path* is a `.Path` object.

        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        self._path = path

    def get_path(self):
        return self._path

    def set_path(self, path):
        self._path = path


class StepPatch(PathPatch):
    """
    A path patch describing a stepwise constant function.

    By default, the path is not closed and starts and stops at
    baseline value.
    """

    _edge_default = False

    @_docstring.interpd
    def __init__(self, values, edges, *,
                 orientation='vertical', baseline=0, **kwargs):
        """
        Parameters
        ----------
        values : array-like
            The step heights.

        edges : array-like
            The edge positions, with ``len(edges) == len(vals) + 1``,
            between which the curve takes on vals values.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            The direction of the steps. Vertical means that *values* are
            along the y-axis, and edges are along the x-axis.

        baseline : float, array-like or None, default: 0
            The bottom value of the bounding edges or when
            ``fill=True``, position of lower edge. If *fill* is
            True or an array is passed to *baseline*, a closed
            path is drawn.

        **kwargs
            `Patch` properties:

            %(Patch:kwdoc)s
        """
        self.orientation = orientation
        self._edges = np.asarray(edges)
        self._values = np.asarray(values)
        self._baseline = np.asarray(baseline) if baseline is not None else None
        self._update_path()
        super().__init__(self._path, **kwargs)

    def _update_path(self):
        if np.isnan(np.sum(self._edges)):
            raise ValueError('Nan values in "edges" are disallowed')
        if self._edges.size - 1 != self._values.size:
            raise ValueError('Size mismatch between "values" and "edges". '
                             "Expected `len(values) + 1 == len(edges)`, but "
                             f"`len(values) = {self._values.size}` and "
                             f"`len(edges) = {self._edges.size}`.")
        # Initializing with empty arrays allows supporting empty stairs.
        verts, codes = [np.empty((0, 2))], [np.empty(0, dtype=Path.code_type)]

        _nan_mask = np.isnan(self._values)
        if self._baseline is not None:
            _nan_mask |= np.isnan(self._baseline)
        for idx0, idx1 in cbook.contiguous_regions(~_nan_mask):
            x = np.repeat(self._edges[idx0:idx1+1], 2)
            y = np.repeat(self._values[idx0:idx1], 2)
            if self._baseline is None:
                y = np.concatenate([y[:1], y, y[-1:]])
            elif self._baseline.ndim == 0:  # single baseline value
                y = np.concatenate([[self._baseline], y, [self._baseline]])
            elif self._baseline.ndim == 1:  # baseline array
                base = np.repeat(self._baseline[idx0:idx1], 2)[::-1]
                x = np.concatenate([x, x[::-1]])
                y = np.concatenate([base[-1:], y, base[:1],
                                    base[:1], base, base[-1:]])
            else:  # no baseline
                raise ValueError('Invalid `baseline` specified')
            if self.orientation == 'vertical':
                xy = np.column_stack([x, y])
            else:
                xy = np.column_stack([y, x])
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO]*(len(xy)-1))
        self._path = Path(np.concatenate(verts), np.concatenate(codes))

    def get_data(self):
        """Get `.StepPatch` values, edges and baseline as namedtuple."""
        StairData = namedtuple('StairData', 'values edges baseline')
        return StairData(self._values, self._edges, self._baseline)

    def set_data(self, values=None, edges=None, baseline=None):
        """
        Set `.StepPatch` values, edges and baseline.

        Parameters
        ----------
        values : 1D array-like or None
            Will not update values, if passing None
        edges : 1D array-like, optional
        baseline : float, 1D array-like or None
        """
        if values is None and edges is None and baseline is None:
            raise ValueError("Must set *values*, *edges* or *baseline*.")
        if values is not None:
            self._values = np.asarray(values)
        if edges is not None:
            self._edges = np.asarray(edges)
        if baseline is not None:
            self._baseline = np.asarray(baseline)
        self._update_path()
        self.stale = True


class Polygon(Patch):
    """A general polygon patch."""

    def __str__(self):
        if len(self._path.vertices):
            s = "Polygon%d((%g, %g) ...)"
            return s % (len(self._path.vertices), *self._path.vertices[0])
        else:
            return "Polygon0()"

    @_docstring.interpd
    def __init__(self, xy, *, closed=True, **kwargs):
        """
        Parameters
        ----------
        xy : (N, 2) array

        closed : bool, default: True
            Whether the polygon is closed (i.e., has identical start and end
            points).

        **kwargs
            %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        self._closed = closed
        self.set_xy(xy)

    def get_path(self):
        """Get the `.Path` of the polygon."""
        return self._path

    def get_closed(self):
        """Return whether the polygon is closed."""
        return self._closed

    def set_closed(self, closed):
        """
        Set whether the polygon is closed.

        Parameters
        ----------
        closed : bool
            True if the polygon is closed
        """
        if self._closed == bool(closed):
            return
        self._closed = bool(closed)
        self.set_xy(self.get_xy())
        self.stale = True

    def get_xy(self):
        """
        Get the vertices of the path.

        Returns
        -------
        (N, 2) array
            The coordinates of the vertices.
        """
        return self._path.vertices

    def set_xy(self, xy):
        """
        Set the vertices of the polygon.

        Parameters
        ----------
        xy : (N, 2) array-like
            The coordinates of the vertices.

        Notes
        -----
        Unlike `.Path`, we do not ignore the last input vertex. If the
        polygon is meant to be closed, and the last point of the polygon is not
        equal to the first, we assume that the user has not explicitly passed a
        ``CLOSEPOLY`` vertex, and add it ourselves.
        """
        xy = np.asarray(xy)
        nverts, _ = xy.shape
        if self._closed:
            # if the first and last vertex are the "same", then we assume that
            # the user explicitly passed the CLOSEPOLY vertex. Otherwise, we
            # have to append one since the last vertex will be "ignored" by
            # Path
            if nverts == 1 or nverts > 1 and (xy[0] != xy[-1]).any():
                xy = np.concatenate([xy, [xy[0]]])
        else:
            # if we aren't closed, and the last vertex matches the first, then
            # we assume we have an unnecessary CLOSEPOLY vertex and remove it
            if nverts > 2 and (xy[0] == xy[-1]).all():
                xy = xy[:-1]
        self._path = Path(xy, closed=self._closed)
        self.stale = True

    xy = property(get_xy, set_xy,
                  doc='The vertices of the path as a (N, 2) array.')


class Wedge(Patch):
    """Wedge shaped patch."""

    def __str__(self):
        pars = (self.center[0], self.center[1], self.r,
                self.theta1, self.theta2, self.width)
        fmt = "Wedge(center=(%g, %g), r=%g, theta1=%g, theta2=%g, width=%s)"
        return fmt % pars

    @_docstring.interpd
    def __init__(self, center, r, theta1, theta2, *, width=None, **kwargs):
        """
        A wedge centered at *x*, *y* center with radius *r* that
        sweeps *theta1* to *theta2* (in degrees).  If *width* is given,
        then a partial wedge is drawn from inner radius *r* - *width*
        to outer radius *r*.

        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        super().__init__(**kwargs)
        self.center = center
        self.r, self.width = r, width
        self.theta1, self.theta2 = theta1, theta2
        self._patch_transform = transforms.IdentityTransform()
        self._recompute_path()

    def _recompute_path(self):
        # Inner and outer rings are connected unless the annulus is complete
        if abs((self.theta2 - self.theta1) - 360) <= 1e-12:
            theta1, theta2 = 0, 360
            connector = Path.MOVETO
        else:
            theta1, theta2 = self.theta1, self.theta2
            connector = Path.LINETO

        # Form the outer ring
        arc = Path.arc(theta1, theta2)

        if self.width is not None:
            # Partial annulus needs to draw the outer ring
            # followed by a reversed and scaled inner ring
            v1 = arc.vertices
            v2 = arc.vertices[::-1] * (self.r - self.width) / self.r
            v = np.concatenate([v1, v2, [(0, 0)]])
            c = [*arc.codes, connector, *arc.codes[1:], Path.CLOSEPOLY]
        else:
            # Wedge doesn't need an inner ring
            v = np.concatenate([arc.vertices, [(0, 0), (0, 0)]])
            c = [*arc.codes, connector, Path.CLOSEPOLY]

        # Shift and scale the wedge to the final location.
        self._path = Path(v * self.r + self.center, c)

    def set_center(self, center):
        self._path = None
        self.center = center
        self.stale = True

    def set_radius(self, radius):
        self._path = None
        self.r = radius
        self.stale = True

    def set_theta1(self, theta1):
        self._path = None
        self.theta1 = theta1
        self.stale = True

    def set_theta2(self, theta2):
        self._path = None
        self.theta2 = theta2
        self.stale = True

    def set_width(self, width):
        self._path = None
        self.width = width
        self.stale = True

    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path


# COVERAGE NOTE: Not used internally or from examples
class Arrow(Patch):
    """An arrow patch."""

    def __str__(self):
        return "Arrow()"

    _path = Path._create_closed([
        [0.0, 0.1], [0.0, -0.1], [0.8, -0.1], [0.8, -0.3], [1.0, 0.0],
        [0.8, 0.3], [0.8, 0.1]])

    @_docstring.interpd
    def __init__(self, x, y, dx, dy, *, width=1.0, **kwargs):
        """
        Draws an arrow from (*x*, *y*) to (*x* + *dx*, *y* + *dy*).
        The width of the arrow is scaled by *width*.

        Parameters
        ----------
        x : float
            x coordinate of the arrow tail.
        y : float
            y coordinate of the arrow tail.
        dx : float
            Arrow length in the x direction.
        dy : float
            Arrow length in the y direction.
        width : float, default: 1
            Scale factor for the width of the arrow. With a default value of 1,
            the tail width is 0.2 and head width is 0.6.
        **kwargs
            Keyword arguments control the `Patch` properties:

            %(Patch:kwdoc)s

        See Also
        --------
        FancyArrow
            Patch that allows independent control of the head and tail
            properties.
        """
        super().__init__(**kwargs)
        self.set_data(x, y, dx, dy, width)

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        return self._patch_transform

    def set_data(self, x=None, y=None, dx=None, dy=None, width=None):
        """
        Set `.Arrow` x, y, dx, dy and width.
        Values left as None will not be updated.

        Parameters
        ----------
        x, y : float or None, default: None
            The x and y coordinates of the arrow base.

        dx, dy : float or None, default: None
            The length of the arrow along x and y direction.

        width : float or None, default: None
            Width of full arrow tail.
        """
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y
        if dx is not None:
            self._dx = dx
        if dy is not None:
            self._dy = dy
        if width is not None:
            self._width = width
        self._patch_transform = (
            transforms.Affine2D()
            .scale(np.hypot(self._dx, self._dy), self._width)
            .rotate(np.arctan2(self._dy, self._dx))
            .translate(self._x, self._y)
            .frozen())


class FancyArrow(Polygon):
    """
    Like Arrow, but lets you set head width and head height independently.
    """

    _edge_default = True

    def __str__(self):
        return "FancyArrow()"

    @_docstring.interpd
    def __init__(self, x, y, dx, dy, *,
                 width=0.001, length_includes_head=False, head_width=None,
                 head_length=None, shape='full', overhang=0,
                 head_starts_at_zero=False, **kwargs):
        """
        Parameters
        ----------
        x, y : float
            The x and y coordinates of the arrow base.

        dx, dy : float
            The length of the arrow along x and y direction.

        width : float, default: 0.001
            Width of full arrow tail.

        length_includes_head : bool, default: False
            True if head is to be counted in calculating the length.

        head_width : float or None, default: 3*width
            Total width of the full arrow head.

        head_length : float or None, default: 1.5*head_width
            Length of arrow head.

        shape : {'full', 'left', 'right'}, default: 'full'
            Draw the left-half, right-half, or full arrow.

        overhang : float, default: 0
            Fraction that the arrow is swept back (0 overhang means
            triangular shape). Can be negative or greater than one.

        head_starts_at_zero : bool, default: False
            If True, the head starts being drawn at coordinate 0
            instead of ending at coordinate 0.

        **kwargs
            `.Patch` properties:

            %(Patch:kwdoc)s
        """
        self._x = x
        self._y = y
        self._dx = dx
        self._dy = dy
        self._width = width
        self._length_includes_head = length_includes_head
        self._head_width = head_width
        self._head_length = head_length
        self._shape = shape
        self._overhang = overhang
        self._head_starts_at_zero = head_starts_at_zero
        self._make_verts()
        super().__init__(self.verts, closed=True, **kwargs)

    def set_data(self, *, x=None, y=None, dx=None, dy=None, width=None,
                 head_width=None, head_length=None):
        """
        Set `.FancyArrow` x, y, dx, dy, width, head_with, and head_length.
        Values left as None will not be updated.

        Parameters
        ----------
        x, y : float or None, default: None
            The x and y coordinates of the arrow base.

        dx, dy : float or None, default: None
            The length of the arrow along x and y direction.

        width : float or None, default: None
            Width of full arrow tail.

        head_width : float or None, default: None
            Total width of the full arrow head.

        head_length : float or None, default: None
            Length of arrow head.
        """
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y
        if dx is not None:
            self._dx = dx
        if dy is not None:
            self._dy = dy
        if width is not None:
            self._width = width
        if head_width is not None:
            self._head_width = head_width
        if head_length is not None:
            self._head_length = head_length
        self._make_verts()
        self.set_xy(self.verts)

    def _make_verts(self):
        if self._head_width is None:
            head_width = 3 * self._width
        else:
            head_width = self._head_width
        if self._head_length is None:
            head_length = 1.5 * head_width
        else:
            head_length = self._head_length

        distance = np.hypot(self._dx, self._dy)

        if self._length_includes_head:
            length = distance
        else:
            length = distance + head_length
        if np.size(length) == 0:
            self.verts = np.empty([0, 2])  # display nothing if empty
        else:
            # start by drawing horizontal arrow, point at (0, 0)
            hw, hl = head_width, head_length
            hs, lw = self._overhang, self._width
            left_half_arrow = np.array([
                [0.0, 0.0],                 # tip
                [-hl, -hw / 2],             # leftmost
                [-hl * (1 - hs), -lw / 2],  # meets stem
                [-length, -lw / 2],         # bottom left
                [-length, 0],
            ])
            # if we're not including the head, shift up by head length
            if not self._length_includes_head:
                left_half_arrow += [head_length, 0]
            # if the head starts at 0, shift up by another head length
            if self._head_starts_at_zero:
                left_half_arrow += [head_length / 2, 0]
            # figure out the shape, and complete accordingly
            if self._shape == 'left':
                coords = left_half_arrow
            else:
                right_half_arrow = left_half_arrow * [1, -1]
                if self._shape == 'right':
                    coords = right_half_arrow
                elif self._shape == 'full':
                    # The half-arrows contain the midpoint of the stem,
                    # which we can omit from the full arrow. Including it
                    # twice caused a problem with xpdf.
                    coords = np.concatenate([left_half_arrow[:-1],
                                             right_half_arrow[-2::-1]])
                else:
                    raise ValueError(f"Got unknown shape: {self._shape!r}")
            if distance != 0:
                cx = self._dx / distance
                sx = self._dy / distance
            else:
                # Account for division by zero
                cx, sx = 0, 1
            M = [[cx, sx], [-sx, cx]]
            self.verts = np.dot(coords, M) + [
                self._x + self._dx,
                self._y + self._dy,
            ]


_docstring.interpd.register(
    FancyArrow="\n".join(
        (inspect.getdoc(FancyArrow.__init__) or "").splitlines()[2:]))


class CirclePolygon(RegularPolygon):
    """A polygon-approximation of a circle patch."""

    def __str__(self):
        s = "CirclePolygon((%g, %g), radius=%g, resolution=%d)"
        return s % (self.xy[0], self.xy[1], self.radius, self.numvertices)

    @_docstring.interpd
    def __init__(self, xy, radius=5, *,
                 resolution=20,  # the number of vertices
                 ** kwargs):
        """
        Create a circle at *xy* = (*x*, *y*) with given *radius*.

        This circle is approximated by a regular polygon with *resolution*
        sides.  For a smoother circle drawn with splines, see `Circle`.

        Valid keyword arguments are:

        %(Patch:kwdoc)s
        """
        super().__init__(
            xy, resolution, radius=radius, orientation=0, **kwargs)


class Ellipse(Patch):
    """A scale-free ellipse."""

    def __str__(self):
        pars = (self._center[0], self._center[1],
                self.width, self.height, self.angle)
        fmt = "Ellipse(xy=(%s, %s), width=%s, height=%s, angle=%s)"
        return fmt % pars

    @_docstring.interpd
    def __init__(self, xy, width, height, *, angle=0, **kwargs):
        """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of ellipse centre.
        width : float
