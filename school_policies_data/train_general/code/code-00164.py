                # Sxx has one additional dimension for time strides
                if axis < 0:
                    axis -= 1
                Sxx = np.unwrap(Sxx, axis=axis)

        # mode =='complex' is same as `stft`, doesn't need modification

    return freqs, time, Sxx


def check_COLA(window, nperseg, noverlap, tol=1e-10):
    r"""Check whether the Constant OverLap Add (COLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies COLA within `tol`,
        `False` otherwise

    See Also
    --------
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, it is sufficient that the signal windowing obeys the constraint of
    "Constant OverLap Add" (COLA). This ensures that every point in the input
    data is equally weighted, thereby avoiding aliasing and allowing full
    reconstruction.

    Some examples of windows that satisfy COLA:
        - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...
        - Bartlett window at overlap of 1/2, 3/4, 5/6, ...
        - Hann window at 1/2, 2/3, 3/4, ...
        - Any Blackman family window at 2/3 overlap
        - Any window with ``noverlap = nperseg-1``

    A very comprehensive list of other windows may be found in [2]_,
    wherein the COLA condition is satisfied when the "Amplitude
    Flatness" is unity.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    Examples
    --------
    >>> from scipy import signal

    Confirm COLA condition for rectangular window of 75% (3/4) overlap:

    >>> signal.check_COLA(signal.windows.boxcar(100), 100, 75)
    True

    COLA is not true for 25% (1/4) overlap, though:

    >>> signal.check_COLA(signal.windows.boxcar(100), 100, 25)
    False

    "Symmetrical" Hann window (for filter design) is not COLA:

    >>> signal.check_COLA(signal.windows.hann(120, sym=True), 120, 60)
    False

    "Periodic" or "DFT-even" Hann window (for FFT analysis) is COLA for
    overlap of 1/2, 2/3, 3/4, etc.:

    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 60)
    True

    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 80)
    True

    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 90)
    True

    """
    nperseg = int(nperseg)

    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    noverlap = int(noverlap)

    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    step = nperseg - noverlap
    binsums = sum(win[ii*step:(ii+1)*step] for ii in range(nperseg//step))

    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]

    deviation = binsums - np.median(binsums)
    return np.max(np.abs(deviation)) < tol


def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    r"""Check whether the Nonzero Overlap Add (NOLA) constraint is met.

    Parameters
    ----------
    window : str or tuple or array_like
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg.
    nperseg : int
        Length of each segment.
    noverlap : int
        Number of points to overlap between segments.
    tol : float, optional
        The allowed variance of a bin's weighted sum from the median bin
        sum.

    Returns
    -------
    verdict : bool
        `True` if chosen combination satisfies the NOLA constraint within
        `tol`, `False` otherwise

    See Also
    --------
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met
    stft: Short Time Fourier Transform
    istft: Inverse Short Time Fourier Transform

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    for all :math:`n`, where :math:`w` is the window function, :math:`t` is the
    frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -
    `noverlap`).

    This ensures that the normalization factors in the denominator of the
    overlap-add inversion equation are not zero. Only very pathological windows
    will fail the NOLA constraint.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
           Publishing, 2011,ISBN 978-0-9745607-3-1.
    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
           spectral density estimation by the Discrete Fourier transform
           (DFT), including a comprehensive list of window functions and
           some new at-top windows", 2002,
           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal

    Confirm NOLA condition for rectangular window of 75% (3/4) overlap:

    >>> signal.check_NOLA(signal.windows.boxcar(100), 100, 75)
    True

    NOLA is also true for 25% (1/4) overlap:

    >>> signal.check_NOLA(signal.windows.boxcar(100), 100, 25)
    True

    "Symmetrical" Hann window (for filter design) is also NOLA:

    >>> signal.check_NOLA(signal.windows.hann(120, sym=True), 120, 60)
    True

    As long as there is overlap, it takes quite a pathological window to fail
    NOLA:

    >>> w = np.ones(64, dtype="float")
    >>> w[::2] = 0
    >>> signal.check_NOLA(w, 64, 32)
    False

    If there is not enough overlap, a window with zeros at the ends will not
    work:

    >>> signal.check_NOLA(signal.windows.hann(64), 64, 0)
    False
    >>> signal.check_NOLA(signal.windows.hann(64), 64, 1)
    False
    >>> signal.check_NOLA(signal.windows.hann(64), 64, 2)
    True

    """
    nperseg = int(nperseg)

    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')

    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg')
    if noverlap < 0:
        raise ValueError('noverlap must be a nonnegative integer')
    noverlap = int(noverlap)

    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')

    step = nperseg - noverlap
    binsums = sum(win[ii*step:(ii+1)*step]**2 for ii in range(nperseg//step))

    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]**2

    return np.min(binsums) > tol


def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True,
         axis=-1, scaling='spectrum'):
    r"""Compute the Short Time Fourier Transform (legacy function).

    STFTs can be used as a way of quantifying the change of a
    nonstationary signal's frequency and phase content over time.

    .. legacy:: function

        `ShortTimeFFT` is a newer STFT / ISTFT implementation with more
        features. A :ref:`comparison <tutorial_stft_legacy_stft>` between the
        implementations can be found in the :ref:`tutorial_stft` section of the
        :ref:`user_guide`.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`. When
        specified, the COLA constraint must be met (see Notes below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        Axis along which the STFT is computed; the default is over the
        last axis (i.e. ``axis=-1``).
    scaling: {'spectrum', 'psd'}
        The default 'spectrum' scaling allows each frequency line of `Zxx` to
        be interpreted as a magnitude spectrum. The 'psd' option scales each
        line to a power spectral density - it allows to calculate the signal's
        energy by numerically integrating over ``abs(Zxx)**2``.

        .. versionadded:: 1.9.0

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of `x`. By default, the last axis of `Zxx` corresponds
        to the segment times.

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    ShortTimeFFT: Newer STFT/ISTFT implementation providing more features.
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    welch: Power spectral density by Welch's method.
    spectrogram: Spectrogram by Welch's method.
    csd: Cross spectral density by Welch's method.
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "Nonzero
    OverLap Add" (NOLA), and the input signal must have complete
    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %
    (nperseg-noverlap) == 0``). The `padded` argument may be used to
    accomplish this.

    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop
    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index
    :math:`t` is given by

    .. math:: x_{t}[n]=x[n]w[n-tH]

    The overlap-add (OLA) reconstruction equation is given by

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    The NOLA constraint ensures that every normalization term that appears
    in the denominator of the OLA reconstruction equation is nonzero. Whether a
    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can
    be tested with `check_NOLA`.


    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> mod = 500*np.cos(2*np.pi*0.25*time)
    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    >>> noise = rng.normal(scale=np.sqrt(noise_power),
    ...                    size=time.shape)
    >>> noise *= np.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the STFT's magnitude.

    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()

    Compare the energy of the signal `x` with the energy of its STFT:

    >>> E_x = sum(x**2) / fs  # Energy of x
    >>> # Calculate a two-sided STFT with PSD scaling:
    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000, return_onesided=False,
    ...                         scaling='psd')
    >>> # Integrate numerically over abs(Zxx)**2:
    >>> df, dt = f[1] - f[0], t[1] - t[0]
    >>> E_Zxx = sum(np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df) * dt
    >>> # The energy is the same, but the numerical errors are quite large:
    >>> np.isclose(E_x, E_Zxx, rtol=1e-2)
    True

    """
    if scaling == 'psd':
        scaling = 'density'
    elif scaling != 'spectrum':
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    freqs, time, Zxx = _spectral_helper(x, x, fs, window, nperseg, noverlap,
                                        nfft, detrend, return_onesided,
                                        scaling=scaling, axis=axis,
                                        mode='stft', boundary=boundary,
                                        padded=padded)

    return freqs, time, Zxx


def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2,
          scaling='spectrum'):
    r"""Perform the inverse Short Time Fourier transform (legacy function).

    .. legacy:: function

        `ShortTimeFFT` is a newer STFT / ISTFT implementation with more
        features. A :ref:`comparison <tutorial_stft_legacy_stft>` between the
        implementations can be found in the :ref:`tutorial_stft` section of the
        :ref:`user_guide`.

    Parameters
    ----------
    Zxx : array_like
        STFT of the signal to be reconstructed. If a purely real array
        is passed, it will be cast to a complex data type.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window. Must match the window used to generate the
        STFT for faithful inversion.
    nperseg : int, optional
        Number of data points corresponding to each STFT segment. This
        parameter must be specified if the number of data points per
        segment is odd, or if the STFT was padded via ``nfft >
        nperseg``. If `None`, the value depends on the shape of
        `Zxx` and `input_onesided`. If `input_onesided` is `True`,
        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,
        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`, half
        of the segment length. Defaults to `None`. When specified, the
        COLA constraint must be met (see Notes below), and should match
        the parameter used to generate the STFT. Defaults to `None`.
    nfft : int, optional
        Number of FFT points corresponding to each STFT segment. This
        parameter must be specified if the STFT was padded via ``nfft >
        nperseg``. If `None`, the default values are the same as for
        `nperseg`, detailed above, with one exception: if
        `input_onesided` is True and
        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on
        that value. This case allows the proper inversion of an
        odd-length unpadded STFT using ``nfft=None``. Defaults to
        `None`.
    input_onesided : bool, optional
        If `True`, interpret the input array as one-sided FFTs, such
        as is returned by `stft` with ``return_onesided=True`` and
        `numpy.fft.rfft`. If `False`, interpret the input as a a
        two-sided FFT. Defaults to `True`.
    boundary : bool, optional
        Specifies whether the input signal was extended at its
        boundaries by supplying a non-`None` ``boundary`` argument to
        `stft`. Defaults to `True`.
    time_axis : int, optional
        Where the time segments of the STFT is located; the default is
        the last axis (i.e. ``axis=-1``).
    freq_axis : int, optional
        Where the frequency axis of the STFT is located; the default is
        the penultimate axis (i.e. ``axis=-2``).
    scaling: {'spectrum', 'psd'}
        The default 'spectrum' scaling allows each frequency line of `Zxx` to
        be interpreted as a magnitude spectrum. The 'psd' option scales each
        line to a power spectral density - it allows to calculate the signal's
        energy by numerically integrating over ``abs(Zxx)**2``.

    Returns
    -------
    t : ndarray
        Array of output data times.
    x : ndarray
        iSTFT of `Zxx`.

    See Also
    --------
    stft: Short Time Fourier Transform
    ShortTimeFFT: Newer STFT/ISTFT implementation providing more features.
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT with
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    This ensures that the normalization factors that appear in the denominator
    of the overlap-add reconstruction equation

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    are not zero. The NOLA constraint can be checked with the `check_NOLA`
    function.

    An STFT which has been modified (via masking or otherwise) is not
    guaranteed to correspond to a exactly realizible signal. This
    function implements the iSTFT via the least-squares estimation
    algorithm detailed in [2]_, which produces a signal that minimizes
    the mean squared error between the STFT of the returned signal and
    the modified STFT.


    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by
    0.001 V**2/Hz of white noise sampled at 1024 Hz.

    >>> fs = 1024
    >>> N = 10*fs
    >>> nperseg = 512
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> carrier = amp * np.sin(2*np.pi*50*time)
    >>> noise = rng.normal(scale=np.sqrt(noise_power),
    ...                    size=time.shape)
    >>> x = carrier + noise

    Compute the STFT, and plot its magnitude

    >>> f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
    >>> plt.figure()
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.ylim([f[1], f[-1]])
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.yscale('log')
    >>> plt.show()

    Zero the components that are 10% or less of the carrier magnitude,
    then convert back to a time series via inverse STFT

    >>> Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
    >>> _, xrec = signal.istft(Zxx, fs)

    Compare the cleaned signal with the original and true carrier signals.

    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([2, 2.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    Note that the cleaned signal does not start as abruptly as the original,
    since some of the coefficients of the transient were also removed:

    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([0, 0.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    """
    # Make sure input is an ndarray of appropriate complex dtype
    Zxx = np.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    if Zxx.ndim < 2:
        raise ValueError('Input stft must be at least 2d!')

    if freq_axis == time_axis:
        raise ValueError('Must specify differing time and frequency axes!')

    nseg = Zxx.shape[time_axis]

    if input_onesided:
        # Assume even segment length
        n_default = 2*(Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    # Check windowing parameters
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            # Odd nperseg, no FFT padding
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Rearrange axes if necessary
    if time_axis != Zxx.ndim-1 or freq_axis != Zxx.ndim-2:
        # Turn negative indices to positive for the call to transpose
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        Zxx = np.transpose(Zxx, zouter+[freq_axis, time_axis])

    # Get window as array
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError(f'window must have length of {nperseg}')

    ifunc = sp_fft.irfft if input_onesided else sp_fft.ifft
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # Initialize output and normalization arrays
    outputlength = nperseg + (nseg-1)*nstep
    x = np.zeros(list(Zxx.shape[:-2])+[outputlength], dtype=xsubs.dtype)
    norm = np.zeros(outputlength, dtype=xsubs.dtype)

    if np.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)

    if scaling == 'spectrum':
        xsubs *= win.sum()
    elif scaling == 'psd':
        xsubs *= np.sqrt(fs * sum(win**2))
    else:
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    # Construct the output from the ifft segments
    # This loop could perhaps be vectorized/strided somehow...
    for ii in range(nseg):
        # Window the ifft
        x[..., ii*nstep:ii*nstep+nperseg] += xsubs[..., ii] * win
        norm[..., ii*nstep:ii*nstep+nperseg] += win**2

    # Remove extension points
    if boundary:
        x = x[..., nperseg//2:-(nperseg//2)]
        norm = norm[..., nperseg//2:-(nperseg//2)]

    # Divide out normalization where non-tiny
    if np.sum(norm > 1e-10) != len(norm):
        warnings.warn(
            "NOLA condition failed, STFT may not be invertible."
            + (" Possibly due to missing boundary" if not boundary else ""),
            stacklevel=2
        )
    x /= np.where(norm > 1e-10, norm, 1.0)

    if input_onesided:
        x = x.real

    # Put axes back
    if x.ndim > 1:
        if time_axis != Zxx.ndim-1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = np.moveaxis(x, -1, time_axis)

    time = np.arange(x.shape[0])/float(fs)
    return time, x


def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1):
    r"""
    Estimate the magnitude squared coherence estimate, Cxy, of
    discrete-time signals X and Y using Welch's method.

    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power
    spectral density estimates of X and Y, and `Pxy` is the cross
    spectral density estimate of X and Y.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    y : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` and `y` time series. Defaults
        to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    axis : int, optional
        Axis along which the coherence is computed for both inputs; the
        default is over the last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Cxy : ndarray
        Magnitude squared coherence of x and y.

    See Also
    --------
    periodogram: Simple, optionally modified periodogram
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data
    welch: Power spectral density by Welch's method.
    csd: Cross spectral density by Welch's method.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
           Signals" Prentice Hall, 2005

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate two test signals with some common features.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 20
    >>> freq = 1234.0
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> b, a = signal.butter(2, 0.25, 'low')
    >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> y = signal.lfilter(b, a, x)
    >>> x += amp*np.sin(2*np.pi*freq*time)
    >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

    Compute and plot the coherence.

    >>> f, Cxy = signal.coherence(x, y, fs, nperseg=1024)
    >>> plt.semilogy(f, Cxy)
    >>> plt.xlabel('frequency [Hz]')
    >>> plt.ylabel('Coherence')
    >>> plt.show()

    """
    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft, detrend=detrend,
                       axis=axis)
    _, Pyy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)

    Cxy = np.abs(Pxy)**2 / Pxx / Pyy

    return freqs, Cxy


def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    if mode not in ['psd', 'stft']:
        raise ValueError(f"Unknown value for mode {mode}, must be one of: "
                         "{'psd', 'stft'}")

    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError(f"Unknown boundary option '{boundary}', "
                         f"must be one of: {list(boundary_funcs.keys())}")

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.moveaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
            if not same_data and y.ndim > 1:
                y = np.moveaxis(y, axis, -1)

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg//2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg//2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e. make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return _signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.moveaxis(d, -1, axis)
            d = detrend(d)
            return np.moveaxis(d, axis, -1)
    else:
        detrend_func = detrend

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError(f'Unknown scaling: {scaling!r}')

    if mode == 'stft':
        scale = np.sqrt(scale)

    if return_onesided:
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to return_onesided=False',
                          stacklevel=3)
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False',
                                  stacklevel=3)
    else:
        sides = 'twosided'

    if sides == 'twosided':
        freqs = sp_fft.fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        freqs = sp_fft.rfftfreq(nfft, 1/fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                               sides)
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        result = np.conjugate(result) * result

    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    if boundary is not None:
        time -= (nperseg/2) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != 'stft':
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = np.moveaxis(result, -1, axis)

    return freqs, time, result


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Created sliding window view of array
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        result = np.lib.stride_tricks.sliding_window_view(
            x, window_shape=nperseg, axis=-1, writeable=True
        )
        result = result[..., 0::step, :]

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == 'twosided':
        func = sp_fft.fft
    else:
        result = result.real
        func = sp_fft.rfft
    result = func(result, n=nfft)

    return result


def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn(f'nperseg = {nperseg:d} is greater than input length '
                          f' = {input_length:d}, using nperseg = {input_length:d}',
                          stacklevel=3)
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg


def _median_bias(n):
    """
    Returns the bias of the median of a set of periodograms relative to
    the mean.

    See Appendix B from [1]_ for details.

    Parameters
    ----------
    n : int
        Numbers of periodograms being averaged.

    Returns
    -------
    bias : float
        Calculated bias.

    References
    ----------
    .. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.
           "FINDCHIRP: an algorithm for detection of gravitational waves from
           inspiraling compact binaries", Physical Review D 85, 2012,
           :arxiv:`gr-qc/0509116`
    """
    ii_2 = 2 * np.arange(1., (n-1) // 2 + 1)
    return 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)
<|endoftext|># This program is public domain
# Authors: Paul Kienzle, Nadav Horesh
"""
Chirp z-transform.

We provide two interfaces to the chirp z-transform: an object interface
which precalculates part of the transform and can be applied efficiently
to many different data sets, and a functional interface which is applied
only to the given data set.

Transforms
----------

CZT : callable (x, axis=-1) -> array
   Define a chirp z-transform that can be applied to different signals.
ZoomFFT : callable (x, axis=-1) -> array
   Define a Fourier transform on a range of frequencies.

Functions
---------

czt : array
   Compute the chirp z-transform for a signal.
zoom_fft : array
   Compute the Fourier transform on a range of frequencies.
"""

import cmath
import numbers
import numpy as np
from numpy import pi, arange
from scipy.fft import fft, ifft, next_fast_len

__all__ = ['czt', 'zoom_fft', 'CZT', 'ZoomFFT', 'czt_points']


def _validate_sizes(n, m):
    if n < 1 or not isinstance(n, numbers.Integral):
        raise ValueError('Invalid number of CZT data '
                         f'points ({n}) specified. '
                         'n must be positive and integer type.')

    if m is None:
        m = n
    elif m < 1 or not isinstance(m, numbers.Integral):
        raise ValueError('Invalid number of CZT output '
                         f'points ({m}) specified. '
                         'm must be positive and integer type.')

    return m


def czt_points(m, w=None, a=1+0j):
    """
    Return the points at which the chirp z-transform is computed.

    Parameters
    ----------
    m : int
        The number of points desired.
    w : complex, optional
        The ratio between points in each step.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    out : ndarray
        The points in the Z plane at which `CZT` samples the z-transform,
        when called with arguments `m`, `w`, and `a`, as complex numbers.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    czt : Convenience function for quickly calculating CZT.

    Examples
    --------
    Plot the points of a 16-point FFT:

    >>> import numpy as np
    >>> from scipy.signal import czt_points
    >>> points = czt_points(16)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(points.real, points.imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()

    and a 91-point logarithmic spiral that crosses the unit circle:

    >>> m, w, a = 91, 0.995*np.exp(-1j*np.pi*.05), 0.8*np.exp(1j*np.pi/6)
    >>> points = czt_points(m, w, a)
    >>> plt.plot(points.real, points.imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()
    """
    m = _validate_sizes(1, m)

    k = arange(m)

    a = 1.0 * a  # at least float

    if w is None:
        # Nothing specified, default to FFT
        return a * np.exp(2j * pi * k / m)
    else:
        # w specified
        w = 1.0 * w  # at least float
        return a * w**-k


class CZT:
    """
    Create a callable chirp z-transform function.

    Transform to compute the frequency response around a spiral.
    Objects of this class are callables which can compute the
    chirp z-transform on their inputs.  This object precalculates the constant
    chirps used in the given transform.

    Parameters
    ----------
    n : int
        The size of the signal.
    m : int, optional
        The number of output points desired.  Default is `n`.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    f : CZT
        Callable object ``f(x, axis=-1)`` for computing the chirp z-transform
        on `x`.

    See Also
    --------
    czt : Convenience function for quickly calculating CZT.
    ZoomFFT : Class that creates a callable partial FFT function.

    Notes
    -----
    The defaults are chosen such that ``f(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, m)`` is equivalent to
    ``fft.fft(x, m)``.

    If `w` does not lie on the unit circle, then the transform will be
    around a spiral with exponentially-increasing radius.  Regardless,
    angle will increase linearly.

    For transforms that do lie on the unit circle, accuracy is better when
    using `ZoomFFT`, since any numerical error in `w` is
    accumulated for long data lengths, drifting away from the unit circle.

    The chirp z-transform can be faster than an equivalent FFT with
    zero padding.  Try it with your own array sizes to see.

    However, the chirp z-transform is considerably less precise than the
    equivalent zero-padded FFT.

    As this CZT is implemented using the Bluestein algorithm, it can compute
    large prime-length Fourier transforms in O(N log N) time, rather than the
    O(N**2) time required by the direct DFT calculation.  (`scipy.fft` also
    uses Bluestein's algorithm'.)

    (The name "chirp z-transform" comes from the use of a chirp in the
    Bluestein algorithm.  It does not decompose signals into chirps, like
    other transforms with "chirp" in the name.)

    References
    ----------
    .. [1] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).
    .. [2] Rabiner, Schafer, and Rader, "The chirp z-transform algorithm and
           its application," Bell Syst. Tech. J. 48, 1249-1292 (1969).

    Examples
    --------
    Compute multiple prime-length FFTs:

    >>> from scipy.signal import CZT
    >>> import numpy as np
    >>> a = np.random.rand(7)
    >>> b = np.random.rand(7)
    >>> c = np.random.rand(7)
    >>> czt_7 = CZT(n=7)
    >>> A = czt_7(a)
    >>> B = czt_7(b)
    >>> C = czt_7(c)

    Display the points at which the FFT is calculated:

    >>> czt_7.points()
    array([ 1.00000000+0.j        ,  0.62348980+0.78183148j,
           -0.22252093+0.97492791j, -0.90096887+0.43388374j,
           -0.90096887-0.43388374j, -0.22252093-0.97492791j,
            0.62348980-0.78183148j])
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(czt_7.points().real, czt_7.points().imag, 'o')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal')
    >>> plt.show()
    """

    def __init__(self, n, m=None, w=None, a=1+0j):
        m = _validate_sizes(n, m)

        k = arange(max(m, n), dtype=np.min_scalar_type(-max(m, n)**2))

        if w is None:
            # Nothing specified, default to FFT-like
            w = cmath.exp(-2j*pi/m)
            wk2 = np.exp(-(1j * pi * ((k**2) % (2*m))) / m)
        else:
            # w specified
            wk2 = w**(k**2/2.)

        a = 1.0 * a  # at least float

        self.w, self.a = w, a
        self.m, self.n = m, n

        nfft = next_fast_len(n + m - 1)
        self._Awk2 = a**-k[:n] * wk2[:n]
        self._nfft = nfft
        self._Fwk2 = fft(1/np.hstack((wk2[n-1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n-1, n+m-1)

    def __call__(self, x, *, axis=-1):
        """
        Calculate the chirp z-transform of a signal.

        Parameters
        ----------
        x : array
            The signal to transform.
        axis : int, optional
            Axis over which to compute the FFT. If not given, the last axis is
            used.

        Returns
        -------
        out : ndarray
            An array of the same dimensions as `x`, but with the length of the
            transformed axis set to `m`.
        """
        x = np.asarray(x)
        if x.shape[axis] != self.n:
            raise ValueError(f"CZT defined for length {self.n}, not "
                             f"{x.shape[axis]}")
        # Calculate transpose coordinates, to allow operation on any given axis
        trnsp = np.arange(x.ndim)
        trnsp[[axis, -1]] = [-1, axis]
        x = x.transpose(*trnsp)
        y = ifft(self._Fwk2 * fft(x*self._Awk2, self._nfft))
        y = y[..., self._yidx] * self._wk2
        return y.transpose(*trnsp)

    def points(self):
        """
        Return the points at which the chirp z-transform is computed.
        """
        return czt_points(self.m, self.w, self.a)


class ZoomFFT(CZT):
    """
    Create a callable zoom FFT transform function.

    This is a specialization of the chirp z-transform (`CZT`) for a set of
    equally-spaced frequencies around the unit circle, used to calculate a
    section of the FFT more efficiently than calculating the entire FFT and
    truncating.

    Parameters
    ----------
    n : int
        The size of the signal.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    m : int, optional
        The number of points to evaluate.  Default is `n`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    f : ZoomFFT
        Callable object ``f(x, axis=-1)`` for computing the zoom FFT on `x`.

    See Also
    --------
    zoom_fft : Convenience function for calculating a zoom FFT.

    Notes
    -----
    The defaults are chosen such that ``f(x, 2)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``f(x, 2, m)`` is equivalent to
    ``fft.fft(x, m)``.

    Sampling frequency is 1/dt, the time step between samples in the
    signal `x`.  The unit circle corresponds to frequencies from 0 up
    to the sampling frequency.  The default sampling frequency of 2
    means that `f1`, `f2` values up to the Nyquist frequency are in the
    range [0, 1). For `f1`, `f2` values expressed in radians, a sampling
    frequency of 2*pi should be used.

    Remember that a zoom FFT can only interpolate the points of the existing
    FFT.  It cannot help to resolve two separate nearby frequencies.
    Frequency resolution can only be increased by increasing acquisition
    time.

    These functions are implemented using Bluestein's algorithm (as is
    `scipy.fft`). [2]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 29 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf
    .. [2] Leo I. Bluestein, "A linear filtering approach to the computation
           of the discrete Fourier transform," Northeast Electronics Research
           and Engineering Meeting Record 10, 218-219 (1968).

    Examples
    --------
    To plot the transform results use something like the following:

    >>> import numpy as np
    >>> from scipy.signal import ZoomFFT
    >>> t = np.linspace(0, 1, 1021)
    >>> x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
    >>> f1, f2 = 5, 27
    >>> transform = ZoomFFT(len(x), [f1, f2], len(x), fs=1021)
    >>> X = transform(x)
    >>> f = np.linspace(f1, f2, len(x))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, 20*np.log10(np.abs(X)))
    >>> plt.show()
    """

    def __init__(self, n, fn, m=None, *, fs=2, endpoint=False):
        m = _validate_sizes(n, m)

        k = arange(max(m, n), dtype=np.min_scalar_type(-max(m, n)**2))

        if np.size(fn) == 2:
            f1, f2 = fn
        elif np.size(fn) == 1:
            f1, f2 = 0.0, fn
        else:
            raise ValueError('fn must be a scalar or 2-length sequence')

        self.f1, self.f2, self.fs = f1, f2, fs

        if endpoint:
            scale = ((f2 - f1) * m) / (fs * (m - 1))
        else:
            scale = (f2 - f1) / fs
        a = cmath.exp(2j * pi * f1/fs)
        wk2 = np.exp(-(1j * pi * scale * k**2) / m)

        self.w = cmath.exp(-2j*pi/m * scale)
        self.a = a
        self.m, self.n = m, n

        ak = np.exp(-2j * pi * f1/fs * k[:n])
        self._Awk2 = ak * wk2[:n]

        nfft = next_fast_len(n + m - 1)
        self._nfft = nfft
        self._Fwk2 = fft(1/np.hstack((wk2[n-1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n-1, n+m-1)


def czt(x, m=None, w=None, a=1+0j, *, axis=-1):
    """
    Compute the frequency response around a spiral in the Z plane.

    Parameters
    ----------
    x : array
        The signal to transform.
    m : int, optional
        The number of output points desired.  Default is the length of the
        input data.
    w : complex, optional
        The ratio between points in each step.  This must be precise or the
        accumulated error will degrade the tail of the output sequence.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        An array of the same dimensions as `x`, but with the length of the
        transformed axis set to `m`.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    zoom_fft : Convenience function for partial FFT calculations.

    Notes
    -----
    The defaults are chosen such that ``signal.czt(x)`` is equivalent to
    ``fft.fft(x)`` and, if ``m > len(x)``, that ``signal.czt(x, m)`` is
    equivalent to ``fft.fft(x, m)``.

    If the transform needs to be repeated, use `CZT` to construct a
    specialized transform function which can be reused without
    recomputing constants.

    An example application is in system identification, repeatedly evaluating
    small slices of the z-transform of a system, around where a pole is
    expected to exist, to refine the estimate of the pole's true location. [1]_

    References
    ----------
    .. [1] Steve Alan Shilling, "A study of the chirp z-transform and its
           applications", pg 20 (1970)
           https://krex.k-state.edu/dspace/bitstream/handle/2097/7844/LD2668R41972S43.pdf

    Examples
    --------
    Generate a sinusoid:

    >>> import numpy as np
    >>> f1, f2, fs = 8, 10, 200  # Hz
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> x = np.sin(2*np.pi*t*f2)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, x)
    >>> plt.axis([0, 1, -1.1, 1.1])
    >>> plt.show()

    Its discrete Fourier transform has all of its energy in a single frequency
    bin:

    >>> from scipy.fft import rfft, rfftfreq
    >>> from scipy.signal import czt, czt_points
    >>> plt.plot(rfftfreq(fs, 1/fs), abs(rfft(x)))
    >>> plt.margins(0, 0.1)
    >>> plt.show()

    However, if the sinusoid is logarithmically-decaying:

    >>> x = np.exp(-t*f1) * np.sin(2*np.pi*t*f2)
    >>> plt.plot(t, x)
    >>> plt.axis([0, 1, -1.1, 1.1])
    >>> plt.show()

    the DFT will have spectral leakage:

    >>> plt.plot(rfftfreq(fs, 1/fs), abs(rfft(x)))
    >>> plt.margins(0, 0.1)
    >>> plt.show()

    While the DFT always samples the z-transform around the unit circle, the
    chirp z-transform allows us to sample the Z-transform along any
    logarithmic spiral, such as a circle with radius smaller than unity:

    >>> M = fs // 2  # Just positive frequencies, like rfft
    >>> a = np.exp(-f1/fs)  # Starting point of the circle, radius < 1
    >>> w = np.exp(-1j*np.pi/M)  # "Step size" of circle
    >>> points = czt_points(M + 1, w, a)  # M + 1 to include Nyquist
    >>> plt.plot(points.real, points.imag, '.')
    >>> plt.gca().add_patch(plt.Circle((0,0), radius=1, fill=False, alpha=.3))
    >>> plt.axis('equal'); plt.axis([-1.05, 1.05, -0.05, 1.05])
    >>> plt.show()

    With the correct radius, this transforms the decaying sinusoid (and others
    with the same decay rate) without spectral leakage:

    >>> z_vals = czt(x, M + 1, w, a)  # Include Nyquist for comparison to rfft
    >>> freqs = np.angle(points)*fs/(2*np.pi)  # angle = omega, radius = sigma
    >>> plt.plot(freqs, abs(z_vals))
    >>> plt.margins(0, 0.1)
    >>> plt.show()
    """
    x = np.asarray(x)
    transform = CZT(x.shape[axis], m=m, w=w, a=a)
    return transform(x, axis=axis)


def zoom_fft(x, fn, m=None, *, fs=2, endpoint=False, axis=-1):
    """
    Compute the DFT of `x` only for frequencies in range `fn`.

    Parameters
    ----------
    x : array
        The signal to transform.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    m : int, optional
        The number of points to evaluate.  The default is the length of `x`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        The transformed signal.  The Fourier transform will be calculated
        at the points f1, f1+df, f1+2df, ..., f2, where df=(f2-f1)/m.

    See Also
    --------
    ZoomFFT : Class that creates a callable partial FFT function.

    Notes
    -----
    The defaults are chosen such that ``signal.zoom_fft(x, 2)`` is equivalent
    to ``fft.fft(x)`` and, if ``m > len(x)``, that ``signal.zoom_fft(x, 2, m)``
    is equivalent to ``fft.fft(x, m)``.

    To graph the magnitude of the resulting transform, use::

        plot(linspace(f1, f2, m, endpoint=False), abs(zoom_fft(x, [f1, f2], m)))

    If the transform needs to be repeated, use `ZoomFFT` to construct
    a specialized transform function which can be reused without
    recomputing constants.

    Examples
    --------
    To plot the transform results use something like the following:

    >>> import numpy as np
    >>> from scipy.signal import zoom_fft
    >>> t = np.linspace(0, 1, 1021)
    >>> x = np.cos(2*np.pi*15*t) + np.sin(2*np.pi*17*t)
    >>> f1, f2 = 5, 27
    >>> X = zoom_fft(x, [f1, f2], len(x), fs=1021)
    >>> f = np.linspace(f1, f2, len(x))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(f, 20*np.log10(np.abs(X)))
    >>> plt.show()
    """
    x = np.asarray(x)
    transform = ZoomFFT(x.shape[axis], fn, m=m, fs=fs, endpoint=endpoint)
    return transform(x, axis=axis)
<|endoftext|>import numpy as np
from scipy.signal import convolve


def _ricker(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    # Determine output type
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode='same')
    return output
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.signal` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk',
    'cont2discrete', 'tf2zpk', 'zpk2tf', 'normalize'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="signal", module="lti_conversion",
                                   private_modules=["_lti_conversion"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in the future
# versions of SciPy. Use the `scipy.signal` namespace for importing the
# functions included below.

import warnings

from . import _spline

__all__ = ['sepfir2d']  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(
            f"scipy.signal.spline is deprecated and has no attribute {name}. "
            "Try looking in scipy.signal instead.")

    warnings.warn(f"Please use `{name}` from the `scipy.signal` namespace, "
                  "the `scipy.signal.spline` namespace is deprecated.",
                  category=DeprecationWarning, stacklevel=2)
    return getattr(_spline, name)
<|endoftext|>from numpy import (asarray, pi, zeros_like,
                   array, arctan2, tan, ones, arange, floor,
                   r_, atleast_1d, sqrt, exp, greater, cos, add, sin,
                   moveaxis, abs, arctan, complex64, float32)
import numpy as np

from scipy._lib._util import normalize_axis_index

# From splinemodule.c
from ._spline import sepfir2d, symiirorder1_ic, symiirorder2_ic_fwd, symiirorder2_ic_bwd
from ._signaltools import lfilter, sosfilt, lfiltic
from ._arraytools import axis_slice, axis_reverse

from scipy.interpolate import BSpline


__all__ = ['spline_filter', 'gauss_spline',
           'cspline1d', 'qspline1d', 'qspline2d', 'cspline2d',
           'cspline1d_eval', 'qspline1d_eval', 'symiirorder1', 'symiirorder2']


def spline_filter(Iin, lmbda=5.0):
    """Smoothing spline (cubic) filtering of a rank-2 array.

    Filter an input data set, `Iin`, using a (cubic) smoothing spline of
    fall-off `lmbda`.

    Parameters
    ----------
    Iin : array_like
        input data set
    lmbda : float, optional
        spline smoothing fall-off value, default is `5.0`.

    Returns
    -------
    res : ndarray
        filtered input data

    Examples
    --------
    We can filter an multi dimensional signal (ex: 2D image) using cubic
    B-spline filter:

    >>> import numpy as np
    >>> from scipy.signal import spline_filter
    >>> import matplotlib.pyplot as plt
    >>> orig_img = np.eye(20)  # create an image
    >>> orig_img[10, :] = 1.0
    >>> sp_filter = spline_filter(orig_img, lmbda=0.1)
    >>> f, ax = plt.subplots(1, 2, sharex=True)
    >>> for ind, data in enumerate([[orig_img, "original image"],
    ...                             [sp_filter, "spline filter"]]):
    ...     ax[ind].imshow(data[0], cmap='gray_r')
    ...     ax[ind].set_title(data[1])
    >>> plt.tight_layout()
    >>> plt.show()

    """
    if Iin.dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
        raise TypeError(f"Invalid data type for Iin: {Iin.dtype = }")

    # XXX: note that complex-valued computations are done in single precision
    # this is historic, and the root reason is unclear,
    # see https://github.com/scipy/scipy/issues/9209
    # Attempting to work in complex double precision leads to symiirorder1
    # failing to converge for the boundary conditions.
    intype = Iin.dtype
    hcol = array([1.0, 4.0, 1.0], np.float32) / 6.0
    if intype == np.complex128:
        Iin = Iin.astype(np.complex64)

    ck = cspline2d(Iin, lmbda)
    out = sepfir2d(ck, hcol, hcol)
    out = out.astype(intype)
    return out


_splinefunc_cache = {}


def gauss_spline(x, n):
    r"""Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    x : array_like
        a knot vector
    n : int
        The order of the spline. Must be non-negative, i.e., n >= 0

    Returns
    -------
    res : ndarray
        B-spline basis function values approximated by a zero-mean Gaussian
        function.

    Notes
    -----
    The B-spline basis function can be approximated well by a zero-mean
    Gaussian function with standard-deviation equal to :math:`\sigma=(n+1)/12`
    for large `n` :

    .. math::  \frac{1}{\sqrt {2\pi\sigma^2}}exp(-\frac{x^2}{2\sigma})

    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines. In:
       Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
    .. [2] http://folk.uio.no/inf3330/scripting/doc/python/SciPy/tutorial/old/node24.html

    Examples
    --------
    We can calculate B-Spline basis functions approximated by a gaussian
    distribution:

    >>> import numpy as np
    >>> from scipy.signal import gauss_spline
    >>> knots = np.array([-1.0, 0.0, -1.0])
    >>> gauss_spline(knots, 3)
    array([0.15418033, 0.6909883, 0.15418033])  # may vary

    """
    x = asarray(x)
    signsq = (n + 1) / 12.0
    return 1 / sqrt(2 * pi * signsq) * exp(-x ** 2 / 2 / signsq)


def _cubic(x):
    x = asarray(x, dtype=float)
    b = BSpline.basis_element([-2, -1, 0, 1, 2], extrapolate=False)
    out = b(x)
    out[(x < -2) | (x > 2)] = 0
    return out


def _quadratic(x):
    x = abs(asarray(x, dtype=float))
    b = BSpline.basis_element([-1.5, -0.5, 0.5, 1.5], extrapolate=False)
    out = b(x)
    out[(x < -1.5) | (x > 1.5)] = 0
    return out


def _coeff_smooth(lam):
    xi = 1 - 96 * lam + 24 * lam * sqrt(3 + 144 * lam)
    omeg = arctan2(sqrt(144 * lam - 1), sqrt(xi))
    rho = (24 * lam - 1 - sqrt(xi)) / (24 * lam)
    rho = rho * sqrt((48 * lam + 24 * lam * sqrt(3 + 144 * lam)) / xi)
    return rho, omeg


def _hc(k, cs, rho, omega):
    return (cs / sin(omega) * (rho ** k) * sin(omega * (k + 1)) *
            greater(k, -1))


def _hs(k, cs, rho, omega):
    c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
          (1 - 2 * rho * rho * cos(2 * omega) + rho ** 4))
    gamma = (1 - rho * rho) / (1 + rho * rho) / tan(omega)
    ak = abs(k)
    return c0 * rho ** ak * (cos(omega * ak) + gamma * sin(omega * ak))


def _cubic_smooth_coeff(signal, lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cos(omega) + rho * rho
    K = len(signal)
    k = arange(K)

    zi_2 = (_hc(0, cs, rho, omega) * signal[0] +
            add.reduce(_hc(k + 1, cs, rho, omega) * signal))
    zi_1 = (_hc(0, cs, rho, omega) * signal[0] +
            _hc(1, cs, rho, omega) * signal[1] +
            add.reduce(_hc(k + 2, cs, rho, omega) * signal))

    # Forward filter:
    # for n in range(2, K):
    #     yp[n] = (cs * signal[n] + 2 * rho * cos(omega) * yp[n - 1] -
    #              rho * rho * yp[n - 2])
    zi = lfiltic(cs, r_[1, -2 * rho * cos(omega), rho * rho], r_[zi_1, zi_2])
    zi = zi.reshape(1, -1)

    sos = r_[cs, 0, 0, 1, -2 * rho * cos(omega), rho * rho]
    sos = sos.reshape(1, -1)

    yp, _ = sosfilt(sos, signal[2:], zi=zi)
    yp = r_[zi_2, zi_1, yp]

    # Reverse filter:
    # for n in range(K - 3, -1, -1):
    #     y[n] = (cs * yp[n] + 2 * rho * cos(omega) * y[n + 1] -
    #             rho * rho * y[n + 2])

    zi_2 = add.reduce((_hs(k, cs, rho, omega) +
                       _hs(k + 1, cs, rho, omega)) * signal[::-1])
    zi_1 = add.reduce((_hs(k - 1, cs, rho, omega) +
                       _hs(k + 2, cs, rho, omega)) * signal[::-1])

    zi = lfiltic(cs, r_[1, -2 * rho * cos(omega), rho * rho], r_[zi_1, zi_2])
    zi = zi.reshape(1, -1)
    y, _ = sosfilt(sos, yp[-3::-1], zi=zi)
    y = r_[y[::-1], zi_1, zi_2]
    return y


def _cubic_coeff(signal):
    zi = -2 + sqrt(3)
    K = len(signal)
    powers = zi ** arange(K)

    if K == 1:
        yplus = signal[0] + zi * add.reduce(powers * signal)
        output = zi / (zi - 1) * yplus
        return atleast_1d(output)

    # Forward filter:
    # yplus[0] = signal[0] + zi * add.reduce(powers * signal)
    # for k in range(1, K):
    #     yplus[k] = signal[k] + zi * yplus[k - 1]

    state = lfiltic(1, r_[1, -zi], atleast_1d(add.reduce(powers * signal)))

    b = ones(1)
    a = r_[1, -zi]
    yplus, _ = lfilter(b, a, signal, zi=state)

    # Reverse filter:
    # output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    # for k in range(K - 2, -1, -1):
    #     output[k] = zi * (output[k + 1] - yplus[k])
    out_last = zi / (zi - 1) * yplus[K - 1]
    state = lfiltic(-zi, r_[1, -zi], atleast_1d(out_last))

    b = asarray([-zi])
    output, _ = lfilter(b, a, yplus[-2::-1], zi=state)
    output = r_[output[::-1], out_last]
    return output * 6.0


def _quadratic_coeff(signal):
    zi = -3 + 2 * sqrt(2.0)
    K = len(signal)
    powers = zi ** arange(K)

    if K == 1:
        yplus = signal[0] + zi * add.reduce(powers * signal)
        output = zi / (zi - 1) * yplus
        return atleast_1d(output)

    # Forward filter:
    # yplus[0] = signal[0] + zi * add.reduce(powers * signal)
    # for k in range(1, K):
    #     yplus[k] = signal[k] + zi * yplus[k - 1]

    state = lfiltic(1, r_[1, -zi], atleast_1d(add.reduce(powers * signal)))

    b = ones(1)
    a = r_[1, -zi]
    yplus, _ = lfilter(b, a, signal, zi=state)

    # Reverse filter:
    # output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    # for k in range(K - 2, -1, -1):
    #     output[k] = zi * (output[k + 1] - yplus[k])
    out_last = zi / (zi - 1) * yplus[K - 1]
    state = lfiltic(-zi, r_[1, -zi], atleast_1d(out_last))

    b = asarray([-zi])
    output, _ = lfilter(b, a, yplus[-2::-1], zi=state)
    output = r_[output[::-1], out_last]
    return output * 8.0


def compute_root_from_lambda(lamb):
    tmp = sqrt(3 + 144 * lamb)
    xi = 1 - 96 * lamb + 24 * lamb * tmp
    omega = arctan(sqrt((144 * lamb - 1.0) / xi))
    tmp2 = sqrt(xi)
    r = ((24 * lamb - 1 - tmp2) / (24 * lamb) *
         sqrt(48*lamb + 24 * lamb * tmp) / tmp2)
    return r, omega


def cspline1d(signal, lamb=0.0):
    """
    Compute cubic spline coefficients for rank-1 array.

    Find the cubic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.

    Returns
    -------
    c : ndarray
        Cubic spline coefficients.

    See Also
    --------
    cspline1d_eval : Evaluate a cubic spline at the new set of points.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        return _cubic_coeff(signal)


def qspline1d(signal, lamb=0.0):
    """Compute quadratic spline coefficients for rank-1 array.

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient (must be zero for now).

    Returns
    -------
    c : ndarray
        Quadratic spline coefficients.

    See Also
    --------
    qspline1d_eval : Evaluate a quadratic spline at the new set of points.

    Notes
    -----
    Find the quadratic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions. To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 6.0, 1.0]/ 8.0 .

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a quadratic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import qspline1d, qspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = qspline1d_eval(qspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    if lamb != 0.0:
        raise ValueError("Smoothing quadratic splines not supported yet.")
    else:
        return _quadratic_coeff(signal)


def collapse_2d(x, axis):
    x = moveaxis(x, axis, -1)
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    if not x.flags.c_contiguous:
        x = x.copy()
    return x, x_shape


def symiirorder_nd(func, input, *args, axis=-1, **kwargs):
    axis = normalize_axis_index(axis, input.ndim)
    input_shape = input.shape
    input_ndim = input.ndim
    if input.ndim > 1:
        input, input_shape = collapse_2d(input, axis)

    out = func(input, *args, **kwargs)

    if input_ndim > 1:
        out = out.reshape(input_shape)
        out = moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()
    return out


def qspline2d(signal, lamb=0.0, precision=-1.0):
    """
    Coefficients for 2-D quadratic (2nd order) B-spline.

    Return the second-order B-spline coefficients over a regularly spaced
    input grid for the two-dimensional input image.

    Parameters
    ----------
    input : ndarray
        The input signal.
    lamb : float
        Specifies the amount of smoothing in the transfer function.
    precision : float
        Specifies the precision for computing the infinite sum needed to apply
        mirror-symmetric boundary conditions.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    if precision < 0.0 or precision >= 1.0:
        if signal.dtype in [float32, complex64]:
            precision = 1e-3
        else:
            precision = 1e-6

    if lamb > 0:
        raise ValueError('lambda must be negative or zero')

    # normal quadratic spline
    r = -3 + 2 * sqrt(2.0)
    c0 = -r * 8.0
    z1 = r

    out = symiirorder_nd(symiirorder1, signal, c0, z1, precision, axis=-1)
    out = symiirorder_nd(symiirorder1, out, c0, z1, precision, axis=0)
    return out


def cspline2d(signal, lamb=0.0, precision=-1.0):
    """
    Coefficients for 2-D cubic (3rd order) B-spline.

    Return the third-order B-spline coefficients over a regularly spaced
    input grid for the two-dimensional input image.

    Parameters
    ----------
    input : ndarray
        The input signal.
    lamb : float
        Specifies the amount of smoothing in the transfer function.
    precision : float
        Specifies the precision for computing the infinite sum needed to apply
        mirror-symmetric boundary conditions.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    if precision < 0.0 or precision >= 1.0:
        if signal.dtype in [float32, complex64]:
            precision = 1e-3
        else:
            precision = 1e-6

    if lamb <= 1 / 144.0:
        # Normal cubic spline
        r = -2 + sqrt(3.0)
        out = symiirorder_nd(
            symiirorder1, signal, -r * 6.0, r, precision=precision, axis=-1)
        out = symiirorder_nd(
            symiirorder1, out, -r * 6.0, r, precision=precision, axis=0)
        return out

    r, omega = compute_root_from_lambda(lamb)
    out = symiirorder_nd(symiirorder2, signal, r, omega,
                         precision=precision, axis=-1)
    out = symiirorder_nd(symiirorder2, out, r, omega,
                         precision=precision, axis=0)
    return out


def cspline1d_eval(cj, newx, dx=1.0, x0=0):
    """Evaluate a cubic spline at the new set of points.

    `dx` is the old sample-spacing while `x0` was the old origin. In
    other-words the old-sample points (knot-points) for which the `cj`
    represent spline coefficients were at equally-spaced points of:

      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)

    Edges are handled using mirror-symmetric boundary conditions.

    Parameters
    ----------
    cj : ndarray
        cublic spline coefficients
    newx : ndarray
        New set of points.
    dx : float, optional
        Old sample-spacing, the default value is 1.0.
    x0 : int, optional
        Old origin, the default value is 0.

    Returns
    -------
    res : ndarray
        Evaluated a cubic spline points.

    See Also
    --------
    cspline1d : Compute cubic spline coefficients for rank-1 array.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a cubic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import cspline1d, cspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = cspline1d_eval(cspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    newx = (asarray(newx) - x0) / float(dx)
    res = zeros_like(newx, dtype=cj.dtype)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > (N - 1)
    cond3 = ~(cond1 | cond2)
    # handle general mirror-symmetry
    res[cond1] = cspline1d_eval(cj, -newx[cond1])
    res[cond2] = cspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = zeros_like(newx, dtype=cj.dtype)
    jlower = floor(newx - 2).astype(int) + 1
    for i in range(4):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1)  # handle edge cases
        result += cj[indj] * _cubic(newx - thisj)
    res[cond3] = result
    return res


def qspline1d_eval(cj, newx, dx=1.0, x0=0):
    """Evaluate a quadratic spline at the new set of points.

    Parameters
    ----------
    cj : ndarray
        Quadratic spline coefficients
    newx : ndarray
        New set of points.
    dx : float, optional
        Old sample-spacing, the default value is 1.0.
    x0 : int, optional
        Old origin, the default value is 0.

    Returns
    -------
    res : ndarray
        Evaluated a quadratic spline points.

    See Also
    --------
    qspline1d : Compute quadratic spline coefficients for rank-1 array.

    Notes
    -----
    `dx` is the old sample-spacing while `x0` was the old origin. In
    other-words the old-sample points (knot-points) for which the `cj`
    represent spline coefficients were at equally-spaced points of::

      oldx = x0 + j*dx  j=0...N-1, with N=len(cj)

    Edges are handled using mirror-symmetric boundary conditions.

    Examples
    --------
    We can filter a signal to reduce and smooth out high-frequency noise with
    a quadratic spline:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import qspline1d, qspline1d_eval
    >>> rng = np.random.default_rng()
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> sig += rng.standard_normal(len(sig))*0.05  # add noise
    >>> time = np.linspace(0, len(sig))
    >>> filtered = qspline1d_eval(qspline1d(sig), time)
    >>> plt.plot(sig, label="signal")
    >>> plt.plot(time, filtered, label="filtered")
    >>> plt.legend()
    >>> plt.show()

    """
    newx = (asarray(newx) - x0) / dx
    res = zeros_like(newx)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > (N - 1)
    cond3 = ~(cond1 | cond2)
    # handle general mirror-symmetry
    res[cond1] = qspline1d_eval(cj, -newx[cond1])
    res[cond2] = qspline1d_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = zeros_like(newx)
    jlower = floor(newx - 1.5).astype(int) + 1
    for i in range(3):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1)  # handle edge cases
        result += cj[indj] * _quadratic(newx - thisj)
    res[cond3] = result
    return res


def symiirorder1(signal, c0, z1, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of first-order sections.

    The second section uses a reversed sequence.  This implements a system with
    the following transfer function and mirror-symmetric boundary conditions::

                           c0
           H(z) = ---------------------
                   (1-z1/z) (1 - z1 z)

    The resulting signal will have mirror symmetric boundary conditions
    as well.

    Parameters
    ----------
    signal : ndarray
        The input signal. If 2D, then the filter will be applied in a batched
        fashion across the last axis.
    c0, z1 : scalar
        Parameters in the transfer function.
    precision :
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    if np.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')

    if signal.ndim > 2:
        raise ValueError('Input must be 1D or 2D')

    squeeze_dim = False
    if signal.ndim == 1:
        signal = signal[None, :]
        squeeze_dim = True

    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.promote_types(signal.dtype, np.float32))

    y0 = symiirorder1_ic(signal, z1, precision)

    # Apply first the system 1 / (1 - z1 * z^-1)
    b = np.ones(1, dtype=signal.dtype)
    a = np.r_[1, -z1]
    a = a.astype(signal.dtype)

    # Compute the initial state for lfilter.
    zii = y0 * z1

    y1, _ = lfilter(b, a, axis_slice(signal, 1), zi=zii)
    y1 = np.c_[y0, y1]

    # Compute backward symmetric condition and apply the system
    # c0 / (1 - z1 * z)
    b = np.asarray([c0], dtype=signal.dtype)
    out_last = -c0 / (z1 - 1.0) * axis_slice(y1, -1)

    # Compute the initial state for lfilter.
    zii = out_last * z1

    # Apply the system c0 / (1 - z1 * z) by reversing the output of the previous stage
    out, _ = lfilter(b, a, axis_slice(y1, -2, step=-1), zi=zii)
    out = np.c_[axis_reverse(out), out_last]

    if squeeze_dim:
        out = out[0]

    return out


def symiirorder2(input, r, omega, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of second-order sections.

    The second section uses a reversed sequence.  This implements the following
    transfer function::

                                  cs^2
         H(z) = ---------------------------------------
                (1 - a2/z - a3/z^2) (1 - a2 z - a3 z^2 )

    where::

          a2 = 2 * r * cos(omega)
          a3 = - r ** 2
          cs = 1 - 2 * r * cos(omega) + r ** 2

    Parameters
    ----------
    input : ndarray
        The input signal.
    r, omega : float
        Parameters in the transfer function.
    precision : float
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')

    if input.ndim > 2:
        raise ValueError('Input must be 1D or 2D')

    if not input.flags.c_contiguous:
        input = input.copy()

    squeeze_dim = False
    if input.ndim == 1:
        input = input[None, :]
        squeeze_dim = True

    if np.issubdtype(input.dtype, np.integer):
        input = input.astype(np.promote_types(input.dtype, np.float32))

    rsq = r * r
    a2 = 2 * r * np.cos(omega)
    a3 = -rsq
    cs = np.atleast_1d(1 - 2 * r * np.cos(omega) + rsq)
    sos = np.atleast_2d(np.r_[cs, 0, 0, 1, -a2, -a3]).astype(input.dtype)

    # Find the starting (forward) conditions.
    ic_fwd = symiirorder2_ic_fwd(input, r, omega, precision)

    # Apply first the system cs / (1 - a2 * z^-1 - a3 * z^-2)
    # Compute the initial conditions in the form expected by sosfilt
    # coef = np.asarray([[a3, a2], [0, a3]], dtype=input.dtype)
    coef = np.r_[a3, a2, 0, a3].reshape(2, 2).astype(input.dtype)
    zi = np.matmul(coef, ic_fwd[:, :, None])[:, :, 0]

    y_fwd, _ = sosfilt(sos, axis_slice(input, 2), zi=zi[None])
    y_fwd = np.c_[ic_fwd, y_fwd]

    # Then compute the symmetric backward starting conditions
    ic_bwd = symiirorder2_ic_bwd(input, r, omega, precision)

    # Apply the system cs / (1 - a2 * z^1 - a3 * z^2)
    # Compute the initial conditions in the form expected by sosfilt
    zi = np.matmul(coef, ic_bwd[:, :, None])[:, :, 0]
    y, _ = sosfilt(sos, axis_slice(y_fwd, -3, step=-1), zi=zi[None])
    out = np.c_[axis_reverse(y), axis_reverse(ic_bwd)]

    if squeeze_dim:
        out = out[0]

    return out
<|endoftext|>"""
ltisys -- a collection of classes and functions for modeling linear
time invariant systems.
"""
#
# Author: Travis Oliphant 2001
#
# Feb 2010: Warren Weckesser
#   Rewrote lsim2 and added impulse2.
# Apr 2011: Jeffrey Armstrong <jeff@approximatrix.com>
#   Added dlsim, dstep, dimpulse, cont2discrete
# Aug 2013: Juan Luis Cano
#   Rewrote abcd_normalize.
# Jan 2015: Irvin Probst irvin DOT probst AT ensta-bretagne DOT fr
#   Added pole placement
# Mar 2015: Clancy Rowley
#   Rewrote lsim
# May 2015: Felix Berkenkamp
#   Split lti class into subclasses
#   Merged discrete systems and added dlti

import warnings

# np.linalg.qr fails on some tests with LinAlgError: zgeqrf returns -7
# use scipy's qr until this is solved

from scipy.linalg import qr as s_qr
from scipy import linalg
from scipy.interpolate import make_interp_spline
from ._filter_design import (tf2zpk, zpk2tf, normalize, freqs, freqz, freqs_zpk,
                            freqz_zpk)
from ._lti_conversion import (tf2ss, abcd_normalize, ss2tf, zpk2ss, ss2zpk,
                             cont2discrete, _atleast_2d_or_none)

import numpy as np
from numpy import (real, atleast_1d, squeeze, asarray, zeros,
                   dot, transpose, ones, linspace)
import copy

__all__ = ['lti', 'dlti', 'TransferFunction', 'ZerosPolesGain', 'StateSpace',
           'lsim', 'impulse', 'step', 'bode',
           'freqresp', 'place_poles', 'dlsim', 'dstep', 'dimpulse',
           'dfreqresp', 'dbode']


class LinearTimeInvariant:
    def __new__(cls, *system, **kwargs):
        """Create a new object, don't allow direct instances."""
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LinearTimeInvariant class is not '
                                      'meant to be used directly, use `lti` '
                                      'or `dlti` instead.')
        return super().__new__(cls)

    def __init__(self):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        super().__init__()

        self.inputs = None
        self.outputs = None
        self._dt = None

    @property
    def dt(self):
        """Return the sampling time of the system, `None` for `lti` systems."""
        return self._dt

    @property
    def _dt_dict(self):
        if self.dt is None:
            return {}
        else:
            return {'dt': self.dt}

    @property
    def zeros(self):
        """Zeros of the system."""
        return self.to_zpk().zeros

    @property
    def poles(self):
        """Poles of the system."""
        return self.to_zpk().poles

    def _as_ss(self):
        """Convert to `StateSpace` system, without copying.

        Returns
        -------
        sys: StateSpace
            The `StateSpace` system. If the class is already an instance of
            `StateSpace` then this instance is returned.
        """
        if isinstance(self, StateSpace):
            return self
        else:
            return self.to_ss()

    def _as_zpk(self):
        """Convert to `ZerosPolesGain` system, without copying.

        Returns
        -------
        sys: ZerosPolesGain
            The `ZerosPolesGain` system. If the class is already an instance of
            `ZerosPolesGain` then this instance is returned.
        """
        if isinstance(self, ZerosPolesGain):
            return self
        else:
            return self.to_zpk()

    def _as_tf(self):
        """Convert to `TransferFunction` system, without copying.

        Returns
        -------
        sys: ZerosPolesGain
            The `TransferFunction` system. If the class is already an instance of
            `TransferFunction` then this instance is returned.
        """
        if isinstance(self, TransferFunction):
            return self
        else:
            return self.to_tf()


class lti(LinearTimeInvariant):
    r"""
    Continuous-time linear time invariant system base class.

    Parameters
    ----------
    *system : arguments
        The `lti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        continuous-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.

    See Also
    --------
    ZerosPolesGain, StateSpace, TransferFunction, dlti

    Notes
    -----
    `lti` instances do not exist directly. Instead, `lti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``s^2 + 3s + 5`` would be represented as ``[1, 3,
    5]``).

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies. It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    Examples
    --------
    >>> from scipy import signal

    >>> signal.lti(1, 2, 3, 4)
    StateSpaceContinuous(
    array([[1]]),
    array([[2]]),
    array([[3]]),
    array([[4]]),
    dt: None
    )

    Construct the transfer function
    :math:`H(s) = \frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> signal.lti([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    Construct the transfer function :math:`H(s) = \frac{3s + 4}{1s + 2}`:

    >>> signal.lti([3, 4], [1, 2])
    TransferFunctionContinuous(
    array([3., 4.]),
    array([1., 2.]),
    dt: None
    )

    """
    def __new__(cls, *system):
        """Create an instance of the appropriate subclass."""
        if cls is lti:
            N = len(system)
            if N == 2:
                return TransferFunctionContinuous.__new__(
                    TransferFunctionContinuous, *system)
            elif N == 3:
                return ZerosPolesGainContinuous.__new__(
                    ZerosPolesGainContinuous, *system)
            elif N == 4:
                return StateSpaceContinuous.__new__(StateSpaceContinuous,
                                                    *system)
            else:
                raise ValueError("`system` needs to be an instance of `lti` "
                                 "or have 2, 3 or 4 arguments.")
        # __new__ was called from a subclass, let it call its own functions
        return super().__new__(cls)

    def __init__(self, *system):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        super().__init__(*system)

    def impulse(self, X0=None, T=None, N=None):
        """
        Return the impulse response of a continuous-time system.
        See `impulse` for details.
        """
        return impulse(self, X0=X0, T=T, N=N)

    def step(self, X0=None, T=None, N=None):
        """
        Return the step response of a continuous-time system.
        See `step` for details.
        """
        return step(self, X0=X0, T=T, N=N)

    def output(self, U, T, X0=None):
        """
        Return the response of a continuous-time system to input `U`.
        See `lsim` for details.
        """
        return lsim(self, U, T, X0=X0)

    def bode(self, w=None, n=100):
        """
        Calculate Bode magnitude and phase data of a continuous-time system.

        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude
        [dB] and phase [deg]. See `bode` for details.

        Examples
        --------
        >>> from scipy import signal
        >>> import matplotlib.pyplot as plt

        >>> sys = signal.TransferFunction([1], [1, 1])
        >>> w, mag, phase = sys.bode()

        >>> plt.figure()
        >>> plt.semilogx(w, mag)    # Bode magnitude plot
        >>> plt.figure()
        >>> plt.semilogx(w, phase)  # Bode phase plot
        >>> plt.show()

        """
        return bode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000):
        """
        Calculate the frequency response of a continuous-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `freqresp` for details.
        """
        return freqresp(self, w=w, n=n)

    def to_discrete(self, dt, method='zoh', alpha=None):
        """Return a discretized version of the current system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti`
        """
        raise NotImplementedError('to_discrete is not implemented for this '
                                  'system class.')


class dlti(LinearTimeInvariant):
    r"""
    Discrete-time linear time invariant system base class.

    Parameters
    ----------
    *system: arguments
        The `dlti` class can be instantiated with either 2, 3 or 4 arguments.
        The following gives the number of arguments and the corresponding
        discrete-time subclass that is created:

            * 2: `TransferFunction`:  (numerator, denominator)
            * 3: `ZerosPolesGain`: (zeros, poles, gain)
            * 4: `StateSpace`:  (A, B, C, D)

        Each argument can be an array or a sequence.
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to ``True``
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    ZerosPolesGain, StateSpace, TransferFunction, lti

    Notes
    -----
    `dlti` instances do not exist directly. Instead, `dlti` creates an instance
    of one of its subclasses: `StateSpace`, `TransferFunction` or
    `ZerosPolesGain`.

    Changing the value of properties that are not directly part of the current
    system representation (such as the `zeros` of a `StateSpace` system) is
    very inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    If (numerator, denominator) is passed in for ``*system``, coefficients for
    both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as ``[1, 3,
    5]``).

    .. versionadded:: 0.18.0

    Examples
    --------
    >>> from scipy import signal

    >>> signal.dlti(1, 2, 3, 4)
    StateSpaceDiscrete(
    array([[1]]),
    array([[2]]),
    array([[3]]),
    array([[4]]),
    dt: True
    )

    >>> signal.dlti(1, 2, 3, 4, dt=0.1)
    StateSpaceDiscrete(
    array([[1]]),
    array([[2]]),
    array([[3]]),
    array([[4]]),
    dt: 0.1
    )

    Construct the transfer function
    :math:`H(z) = \frac{5(z - 1)(z - 2)}{(z - 3)(z - 4)}` with a sampling time
    of 0.1 seconds:

    >>> signal.dlti([1, 2], [3, 4], 5, dt=0.1)
    ZerosPolesGainDiscrete(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: 0.1
    )

    Construct the transfer function :math:`H(z) = \frac{3z + 4}{1z + 2}` with
    a sampling time of 0.1 seconds:

    >>> signal.dlti([3, 4], [1, 2], dt=0.1)
    TransferFunctionDiscrete(
    array([3., 4.]),
    array([1., 2.]),
    dt: 0.1
    )

    """
    def __new__(cls, *system, **kwargs):
        """Create an instance of the appropriate subclass."""
        if cls is dlti:
            N = len(system)
            if N == 2:
                return TransferFunctionDiscrete.__new__(
                    TransferFunctionDiscrete, *system, **kwargs)
            elif N == 3:
                return ZerosPolesGainDiscrete.__new__(ZerosPolesGainDiscrete,
                                                      *system, **kwargs)
            elif N == 4:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete, *system,
                                                  **kwargs)
            else:
                raise ValueError("`system` needs to be an instance of `dlti` "
                                 "or have 2, 3 or 4 arguments.")
        # __new__ was called from a subclass, let it call its own functions
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """
        Initialize the `lti` baseclass.

        The heavy lifting is done by the subclasses.
        """
        dt = kwargs.pop('dt', True)
        super().__init__(*system, **kwargs)

        self.dt = dt

    @property
    def dt(self):
        """Return the sampling time of the system."""
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    def impulse(self, x0=None, t=None, n=None):
        """
        Return the impulse response of the discrete-time `dlti` system.
        See `dimpulse` for details.
        """
        return dimpulse(self, x0=x0, t=t, n=n)

    def step(self, x0=None, t=None, n=None):
        """
        Return the step response of the discrete-time `dlti` system.
        See `dstep` for details.
        """
        return dstep(self, x0=x0, t=t, n=n)

    def output(self, u, t, x0=None):
        """
        Return the response of the discrete-time system to input `u`.
        See `dlsim` for details.
        """
        return dlsim(self, u, t, x0=x0)

    def bode(self, w=None, n=100):
        r"""
        Calculate Bode magnitude and phase data of a discrete-time system.

        Returns a 3-tuple containing arrays of frequencies [rad/s], magnitude
        [dB] and phase [deg]. See `dbode` for details.

        Examples
        --------
        >>> from scipy import signal
        >>> import matplotlib.pyplot as plt

        Construct the transfer function :math:`H(z) = \frac{1}{z^2 + 2z + 3}`
        with sampling time 0.5s:

        >>> sys = signal.TransferFunction([1], [1, 2, 3], dt=0.5)

        Equivalent: signal.dbode(sys)

        >>> w, mag, phase = sys.bode()

        >>> plt.figure()
        >>> plt.semilogx(w, mag)    # Bode magnitude plot
        >>> plt.figure()
        >>> plt.semilogx(w, phase)  # Bode phase plot
        >>> plt.show()

        """
        return dbode(self, w=w, n=n)

    def freqresp(self, w=None, n=10000, whole=False):
        """
        Calculate the frequency response of a discrete-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `dfreqresp` for details.

        """
        return dfreqresp(self, w=w, n=n, whole=whole)


class TransferFunction(LinearTimeInvariant):
    r"""Linear Time Invariant system class in transfer function form.

    Represents the system as the continuous-time transfer function
    :math:`H(s)=\sum_{i=0}^N b[N-i] s^i / \sum_{j=0}^M a[M-j] s^j` or the
    discrete-time transfer function
    :math:`H(z)=\sum_{i=0}^N b[N-i] z^i / \sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    `TransferFunction` systems inherit additional
    functionality from the `lti`, respectively the `dlti` classes, depending on
    which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    ZerosPolesGain, StateSpace, lti, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` or ``z^2 + 3z + 5`` would be
    represented as ``[1, 3, 5]``)

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}`:

    >>> from scipy import signal

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    >>> signal.TransferFunction(num, den)
    TransferFunctionContinuous(
    array([1., 3., 3.]),
    array([1., 2., 1.]),
    dt: None
    )

    Construct the transfer function
    :math:`H(z) = \frac{z^2 + 3z + 3}{z^2 + 2z + 1}` with a sampling time of
    0.1 seconds:

    >>> signal.TransferFunction(num, den, dt=0.1)
    TransferFunctionDiscrete(
    array([1., 3., 3.]),
    array([1., 2., 1.]),
    dt: 0.1
    )

    """
    def __new__(cls, *system, **kwargs):
        """Handle object conversion if input is an instance of lti."""
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_tf()

        # Choose whether to inherit from `lti` or from `dlti`
        if cls is TransferFunction:
            if kwargs.get('dt') is None:
                return TransferFunctionContinuous.__new__(
                    TransferFunctionContinuous,
                    *system,
                    **kwargs)
            else:
                return TransferFunctionDiscrete.__new__(
                    TransferFunctionDiscrete,
                    *system,
                    **kwargs)

        # No special conversion needed
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the state space LTI system."""
        # Conversion of lti instances is handled in __new__
        if isinstance(system[0], LinearTimeInvariant):
            return

        # Remove system arguments, not needed by parents anymore
        super().__init__(**kwargs)

        self._num = None
        self._den = None

        self.num, self.den = normalize(*system)

    def __repr__(self):
        """Return representation of the system's transfer function"""
        return (
            f'{self.__class__.__name__}(\n'
            f'{repr(self.num)},\n'
            f'{repr(self.den)},\n'
            f'dt: {repr(self.dt)}\n)'
        )

    @property
    def num(self):
        """Numerator of the `TransferFunction` system."""
        return self._num

    @num.setter
    def num(self, num):
        self._num = atleast_1d(num)

        # Update dimensions
        if len(self.num.shape) > 1:
            self.outputs, self.inputs = self.num.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def den(self):
        """Denominator of the `TransferFunction` system."""
        return self._den

    @den.setter
    def den(self, den):
        self._den = atleast_1d(den)

    def _copy(self, system):
        """
        Copy the parameters of another `TransferFunction` object

        Parameters
        ----------
        system : `TransferFunction`
            The `StateSpace` system that is to be copied

        """
        self.num = system.num
        self.den = system.den

    def to_tf(self):
        """
        Return a copy of the current `TransferFunction` system.

        Returns
        -------
        sys : instance of `TransferFunction`
            The current system (copy)

        """
        return copy.deepcopy(self)

    def to_zpk(self):
        """
        Convert system representation to `ZerosPolesGain`.

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
        return ZerosPolesGain(*tf2zpk(self.num, self.den),
                              **self._dt_dict)

    def to_ss(self):
        """
        Convert system representation to `StateSpace`.

        Returns
        -------
        sys : instance of `StateSpace`
            State space model of the current system

        """
        return StateSpace(*tf2ss(self.num, self.den),
                          **self._dt_dict)

    @staticmethod
    def _z_to_zinv(num, den):
        """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.
        """
        diff = len(num) - len(den)
        if diff > 0:
            den = np.hstack((np.zeros(diff), den))
        elif diff < 0:
            num = np.hstack((np.zeros(-diff), num))
        return num, den

    @staticmethod
    def _zinv_to_z(num, den):
        """Change a transfer function from the variable `z` to `z**-1`.

        Parameters
        ----------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of ascending degree of 'z**-1'.
            That is, ``5 + 3 z**-1 + 2 z**-2`` is presented as ``[5, 3, 2]``.

        Returns
        -------
        num, den: 1d array_like
            Sequences representing the coefficients of the numerator and
            denominator polynomials, in order of descending degree of 'z'.
            That is, ``5z**2 + 3z + 2`` is presented as ``[5, 3, 2]``.
        """
        diff = len(num) - len(den)
        if diff > 0:
            den = np.hstack((den, np.zeros(diff)))
        elif diff < 0:
            num = np.hstack((num, np.zeros(-diff)))
        return num, den


class TransferFunctionContinuous(TransferFunction, lti):
    r"""
    Continuous-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(s)=\sum_{i=0}^N b[N-i] s^i / \sum_{j=0}^M a[M-j] s^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Continuous-time `TransferFunction` systems inherit additional
    functionality from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)

    See Also
    --------
    ZerosPolesGain, StateSpace, lti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g. ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``)

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}`:

    >>> from scipy import signal

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    >>> signal.TransferFunction(num, den)
    TransferFunctionContinuous(
    array([ 1.,  3.,  3.]),
    array([ 1.,  2.,  1.]),
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `TransferFunction` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        """
        return TransferFunction(*cont2discrete((self.num, self.den),
                                               dt,
                                               method=method,
                                               alpha=alpha)[:-1],
                                dt=dt)


class TransferFunctionDiscrete(TransferFunction, dlti):
    r"""
    Discrete-time Linear Time Invariant system in transfer function form.

    Represents the system as the transfer function
    :math:`H(z)=\sum_{i=0}^N b[N-i] z^i / \sum_{j=0}^M a[M-j] z^j`, where
    :math:`b` are elements of the numerator `num`, :math:`a` are elements of
    the denominator `den`, and ``N == len(b) - 1``, ``M == len(a) - 1``.
    Discrete-time `TransferFunction` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system: arguments
        The `TransferFunction` class can be instantiated with 1 or 2
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 2: array_like: (numerator, denominator)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    ZerosPolesGain, StateSpace, dlti
    tf2ss, tf2zpk, tf2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `TransferFunction` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.

    If (numerator, denominator) is passed in for ``*system``, coefficients
    for both the numerator and denominator should be specified in descending
    exponent order (e.g., ``z^2 + 3z + 5`` would be represented as
    ``[1, 3, 5]``).

    Examples
    --------
    Construct the transfer function
    :math:`H(z) = \frac{z^2 + 3z + 3}{z^2 + 2z + 1}` with a sampling time of
    0.5 seconds:

    >>> from scipy import signal

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    >>> signal.TransferFunction(num, den, dt=0.5)
    TransferFunctionDiscrete(
    array([ 1.,  3.,  3.]),
    array([ 1.,  2.,  1.]),
    dt: 0.5
    )

    """
    pass


class ZerosPolesGain(LinearTimeInvariant):
    r"""
    Linear Time Invariant system class in zeros, poles, gain form.

    Represents the system as the continuous- or discrete-time transfer function
    :math:`H(s)=k \prod_i (s - z[i]) / \prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    `ZerosPolesGain` systems inherit additional functionality from the `lti`,
    respectively the `dlti` classes, depending on which system representation
    is used.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.


    See Also
    --------
    TransferFunction, StateSpace, lti, dlti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    Construct the transfer function
    :math:`H(z) = \frac{5(z - 1)(z - 2)}{(z - 3)(z - 4)}` with a sampling time
    of 0.1 seconds:

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5, dt=0.1)
    ZerosPolesGainDiscrete(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: 0.1
    )

    """
    def __new__(cls, *system, **kwargs):
        """Handle object conversion if input is an instance of `lti`"""
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_zpk()

        # Choose whether to inherit from `lti` or from `dlti`
        if cls is ZerosPolesGain:
            if kwargs.get('dt') is None:
                return ZerosPolesGainContinuous.__new__(
                    ZerosPolesGainContinuous,
                    *system,
                    **kwargs)
            else:
                return ZerosPolesGainDiscrete.__new__(
                    ZerosPolesGainDiscrete,
                    *system,
                    **kwargs
                    )

        # No special conversion needed
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the zeros, poles, gain system."""
        # Conversion of lti instances is handled in __new__
        if isinstance(system[0], LinearTimeInvariant):
            return

        super().__init__(**kwargs)

        self._zeros = None
        self._poles = None
        self._gain = None

        self.zeros, self.poles, self.gain = system

    def __repr__(self):
        """Return representation of the `ZerosPolesGain` system."""
        return (
            f'{self.__class__.__name__}(\n'
            f'{repr(self.zeros)},\n'
            f'{repr(self.poles)},\n'
            f'{repr(self.gain)},\n'
            f'dt: {repr(self.dt)}\n)'
        )

    @property
    def zeros(self):
        """Zeros of the `ZerosPolesGain` system."""
        return self._zeros

    @zeros.setter
    def zeros(self, zeros):
        self._zeros = atleast_1d(zeros)

        # Update dimensions
        if len(self.zeros.shape) > 1:
            self.outputs, self.inputs = self.zeros.shape
        else:
            self.outputs = 1
            self.inputs = 1

    @property
    def poles(self):
        """Poles of the `ZerosPolesGain` system."""
        return self._poles

    @poles.setter
    def poles(self, poles):
        self._poles = atleast_1d(poles)

    @property
    def gain(self):
        """Gain of the `ZerosPolesGain` system."""
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    def _copy(self, system):
        """
        Copy the parameters of another `ZerosPolesGain` system.

        Parameters
        ----------
        system : instance of `ZerosPolesGain`
            The zeros, poles gain system that is to be copied

        """
        self.poles = system.poles
        self.zeros = system.zeros
        self.gain = system.gain

    def to_tf(self):
        """
        Convert system representation to `TransferFunction`.

        Returns
        -------
        sys : instance of `TransferFunction`
            Transfer function of the current system

        """
        return TransferFunction(*zpk2tf(self.zeros, self.poles, self.gain),
                                **self._dt_dict)

    def to_zpk(self):
        """
        Return a copy of the current 'ZerosPolesGain' system.

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            The current system (copy)

        """
        return copy.deepcopy(self)

    def to_ss(self):
        """
        Convert system representation to `StateSpace`.

        Returns
        -------
        sys : instance of `StateSpace`
            State space model of the current system

        """
        return StateSpace(*zpk2ss(self.zeros, self.poles, self.gain),
                          **self._dt_dict)


class ZerosPolesGainContinuous(ZerosPolesGain, lti):
    r"""
    Continuous-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the continuous time transfer function
    :math:`H(s)=k \prod_i (s - z[i]) / \prod_j (s - p[j])`, where :math:`k` is
    the `gain`, :math:`z` are the `zeros` and :math:`p` are the `poles`.
    Continuous-time `ZerosPolesGain` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)

    See Also
    --------
    TransferFunction, StateSpace, lti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s)=\frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `ZerosPolesGain` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `ZerosPolesGain`
        """
        return ZerosPolesGain(
            *cont2discrete((self.zeros, self.poles, self.gain),
                           dt,
                           method=method,
                           alpha=alpha)[:-1],
            dt=dt)


class ZerosPolesGainDiscrete(ZerosPolesGain, dlti):
    r"""
    Discrete-time Linear Time Invariant system in zeros, poles, gain form.

    Represents the system as the discrete-time transfer function
    :math:`H(z)=k \prod_i (z - q[i]) / \prod_j (z - p[j])`, where :math:`k` is
    the `gain`, :math:`q` are the `zeros` and :math:`p` are the `poles`.
    Discrete-time `ZerosPolesGain` systems inherit additional functionality
    from the `dlti` class.

    Parameters
    ----------
    *system : arguments
        The `ZerosPolesGain` class can be instantiated with 1 or 3
        arguments. The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 3: array_like: (zeros, poles, gain)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    TransferFunction, StateSpace, dlti
    zpk2ss, zpk2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `ZerosPolesGain` system representation (such as the `A`, `B`, `C`, `D`
    state-space matrices) is very inefficient and may lead to numerical
    inaccuracies.  It is better to convert to the specific system
    representation first. For example, call ``sys = sys.to_ss()`` before
    accessing/changing the A, B, C, D system matrices.

    Examples
    --------
    Construct the transfer function
    :math:`H(s) = \frac{5(s - 1)(s - 2)}{(s - 3)(s - 4)}`:

    >>> from scipy import signal

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5)
    ZerosPolesGainContinuous(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: None
    )

    Construct the transfer function
    :math:`H(z) = \frac{5(z - 1)(z - 2)}{(z - 3)(z - 4)}` with a sampling time
    of 0.1 seconds:

    >>> signal.ZerosPolesGain([1, 2], [3, 4], 5, dt=0.1)
    ZerosPolesGainDiscrete(
    array([1, 2]),
    array([3, 4]),
    5,
    dt: 0.1
    )

    """
    pass


class StateSpace(LinearTimeInvariant):
    r"""
    Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\dot{x} = A x + B u` or the discrete-time difference
    equation :math:`x[k+1] = A x[k] + B u[k]`. `StateSpace` systems
    inherit additional functionality from the `lti`, respectively the `dlti`
    classes, depending on which system representation is used.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 4 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` or `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `None`
        (continuous-time). Must be specified as a keyword argument, for
        example, ``dt=0.1``.

    See Also
    --------
    TransferFunction, ZerosPolesGain, lti, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    Examples
    --------
    >>> from scipy import signal
    >>> import numpy as np
    >>> a = np.array([[0, 1], [0, 0]])
    >>> b = np.array([[0], [1]])
    >>> c = np.array([[1, 0]])
    >>> d = np.array([[0]])

    >>> sys = signal.StateSpace(a, b, c, d)
    >>> print(sys)
    StateSpaceContinuous(
    array([[0, 1],
           [0, 0]]),
    array([[0],
           [1]]),
    array([[1, 0]]),
    array([[0]]),
    dt: None
    )

    >>> sys.to_discrete(0.1)
    StateSpaceDiscrete(
    array([[1. , 0.1],
           [0. , 1. ]]),
    array([[0.005],
           [0.1  ]]),
    array([[1, 0]]),
    array([[0]]),
    dt: 0.1
    )

    >>> a = np.array([[1, 0.1], [0, 1]])
    >>> b = np.array([[0.005], [0.1]])

    >>> signal.StateSpace(a, b, c, d, dt=0.1)
    StateSpaceDiscrete(
    array([[1. , 0.1],
           [0. , 1. ]]),
    array([[0.005],
           [0.1  ]]),
    array([[1, 0]]),
    array([[0]]),
    dt: 0.1
    )

    """

    # Override NumPy binary operations and ufuncs
    __array_priority__ = 100.0
    __array_ufunc__ = None

    def __new__(cls, *system, **kwargs):
        """Create new StateSpace object and settle inheritance."""
        # Handle object conversion if input is an instance of `lti`
        if len(system) == 1 and isinstance(system[0], LinearTimeInvariant):
            return system[0].to_ss()

        # Choose whether to inherit from `lti` or from `dlti`
        if cls is StateSpace:
            if kwargs.get('dt') is None:
                return StateSpaceContinuous.__new__(StateSpaceContinuous,
                                                    *system, **kwargs)
            else:
                return StateSpaceDiscrete.__new__(StateSpaceDiscrete,
                                                  *system, **kwargs)

        # No special conversion needed
        return super().__new__(cls)

    def __init__(self, *system, **kwargs):
        """Initialize the state space lti/dlti system."""
        # Conversion of lti instances is handled in __new__
        if isinstance(system[0], LinearTimeInvariant):
            return

        # Remove system arguments, not needed by parents anymore
        super().__init__(**kwargs)

        self._A = None
        self._B = None
        self._C = None
        self._D = None

        self.A, self.B, self.C, self.D = abcd_normalize(*system)

    def __repr__(self):
        """Return representation of the `StateSpace` system."""
        return (
            f'{self.__class__.__name__}(\n'
            f'{repr(self.A)},\n'
            f'{repr(self.B)},\n'
            f'{repr(self.C)},\n'
            f'{repr(self.D)},\n'
            f'dt: {repr(self.dt)}\n)'
        )

    def _check_binop_other(self, other):
        return isinstance(other, (StateSpace, np.ndarray, float, complex,
                                  np.number, int))

    def __mul__(self, other):
        """
        Post-multiply another system or a scalar

        Handles multiplication of systems in the sense of a frequency domain
        multiplication. That means, given two systems E1(s) and E2(s), their
        multiplication, H(s) = E1(s) * E2(s), means that applying H(s) to U(s)
        is equivalent to first applying E2(s), and then E1(s).

        Notes
        -----
        For SISO systems the order of system application does not matter.
        However, for MIMO systems, where the two systems are matrices, the
        order above ensures standard Matrix multiplication rules apply.
        """
        if not self._check_binop_other(other):
            return NotImplemented

        if isinstance(other, StateSpace):
            # Disallow mix of discrete and continuous systems.
            if type(other) is not type(self):
                return NotImplemented

            if self.dt != other.dt:
                raise TypeError('Cannot multiply systems with different `dt`.')

            n1 = self.A.shape[0]
            n2 = other.A.shape[0]

            # Interconnection of systems
            # x1' = A1 x1 + B1 u1
            # y1  = C1 x1 + D1 u1
            # x2' = A2 x2 + B2 y1
            # y2  = C2 x2 + D2 y1
            #
            # Plugging in with u1 = y2 yields
            # [x1']   [A1 B1*C2 ] [x1]   [B1*D2]
            # [x2'] = [0  A2    ] [x2] + [B2   ] u2
            #                    [x1]
            #  y2   = [C1 D1*C2] [x2] + D1*D2 u2
            a = np.vstack((np.hstack((self.A, np.dot(self.B, other.C))),
                           np.hstack((zeros((n2, n1)), other.A))))
            b = np.vstack((np.dot(self.B, other.D), other.B))
            c = np.hstack((self.C, np.dot(self.D, other.C)))
            d = np.dot(self.D, other.D)
        else:
            # Assume that other is a scalar / matrix
            # For post multiplication the input gets scaled
            a = self.A
            b = np.dot(self.B, other)
            c = self.C
            d = np.dot(self.D, other)

        common_dtype = np.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(np.asarray(a, dtype=common_dtype),
                          np.asarray(b, dtype=common_dtype),
                          np.asarray(c, dtype=common_dtype),
                          np.asarray(d, dtype=common_dtype),
                          **self._dt_dict)

    def __rmul__(self, other):
        """Pre-multiply a scalar or matrix (but not StateSpace)"""
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented

        # For pre-multiplication only the output gets scaled
        a = self.A
        b = self.B
        c = np.dot(other, self.C)
        d = np.dot(other, self.D)

        common_dtype = np.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(np.asarray(a, dtype=common_dtype),
                          np.asarray(b, dtype=common_dtype),
                          np.asarray(c, dtype=common_dtype),
                          np.asarray(d, dtype=common_dtype),
                          **self._dt_dict)

    def __neg__(self):
        """Negate the system (equivalent to pre-multiplying by -1)."""
        return StateSpace(self.A, self.B, -self.C, -self.D, **self._dt_dict)

    def __add__(self, other):
        """
        Adds two systems in the sense of frequency domain addition.
        """
        if not self._check_binop_other(other):
            return NotImplemented

        if isinstance(other, StateSpace):
            # Disallow mix of discrete and continuous systems.
            if type(other) is not type(self):
                raise TypeError(f'Cannot add {type(self)} and {type(other)}')

            if self.dt != other.dt:
                raise TypeError('Cannot add systems with different `dt`.')
            # Interconnection of systems
            # x1' = A1 x1 + B1 u
            # y1  = C1 x1 + D1 u
            # x2' = A2 x2 + B2 u
            # y2  = C2 x2 + D2 u
            # y   = y1 + y2
            #
            # Plugging in yields
            # [x1']   [A1 0 ] [x1]   [B1]
            # [x2'] = [0  A2] [x2] + [B2] u
            #                 [x1]
            #  y    = [C1 C2] [x2] + [D1 + D2] u
            a = linalg.block_diag(self.A, other.A)
            b = np.vstack((self.B, other.B))
            c = np.hstack((self.C, other.C))
            d = self.D + other.D
        else:
            other = np.atleast_2d(other)
            if self.D.shape == other.shape:
                # A scalar/matrix is really just a static system (A=0, B=0, C=0)
                a = self.A
                b = self.B
                c = self.C
                d = self.D + other
            else:
                raise ValueError("Cannot add systems with incompatible "
                                 f"dimensions ({self.D.shape} and {other.shape})")

        common_dtype = np.result_type(a.dtype, b.dtype, c.dtype, d.dtype)
        return StateSpace(np.asarray(a, dtype=common_dtype),
                          np.asarray(b, dtype=common_dtype),
                          np.asarray(c, dtype=common_dtype),
                          np.asarray(d, dtype=common_dtype),
                          **self._dt_dict)

    def __sub__(self, other):
        if not self._check_binop_other(other):
            return NotImplemented

        return self.__add__(-other)

    def __radd__(self, other):
        if not self._check_binop_other(other):
            return NotImplemented

        return self.__add__(other)

    def __rsub__(self, other):
        if not self._check_binop_other(other):
            return NotImplemented

        return (-self).__add__(other)

    def __truediv__(self, other):
        """
        Divide by a scalar
        """
        # Division by non-StateSpace scalars
        if not self._check_binop_other(other) or isinstance(other, StateSpace):
            return NotImplemented

        if isinstance(other, np.ndarray) and other.ndim > 0:
            # It's ambiguous what this means, so disallow it
            raise ValueError("Cannot divide StateSpace by non-scalar numpy arrays")

        return self.__mul__(1/other)

    @property
    def A(self):
        """State matrix of the `StateSpace` system."""
        return self._A

    @A.setter
    def A(self, A):
        self._A = _atleast_2d_or_none(A)

    @property
    def B(self):
        """Input matrix of the `StateSpace` system."""
        return self._B

    @B.setter
    def B(self, B):
        self._B = _atleast_2d_or_none(B)
        self.inputs = self.B.shape[-1]

    @property
    def C(self):
        """Output matrix of the `StateSpace` system."""
        return self._C

    @C.setter
    def C(self, C):
        self._C = _atleast_2d_or_none(C)
        self.outputs = self.C.shape[0]

    @property
    def D(self):
        """Feedthrough matrix of the `StateSpace` system."""
        return self._D

    @D.setter
    def D(self, D):
        self._D = _atleast_2d_or_none(D)

    def _copy(self, system):
        """
        Copy the parameters of another `StateSpace` system.

        Parameters
        ----------
        system : instance of `StateSpace`
            The state-space system that is to be copied

        """
        self.A = system.A
        self.B = system.B
        self.C = system.C
        self.D = system.D

    def to_tf(self, **kwargs):
        """
        Convert system representation to `TransferFunction`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `TransferFunction`
            Transfer function of the current system

        """
        return TransferFunction(*ss2tf(self._A, self._B, self._C, self._D,
                                       **kwargs), **self._dt_dict)

    def to_zpk(self, **kwargs):
        """
        Convert system representation to `ZerosPolesGain`.

        Parameters
        ----------
        kwargs : dict, optional
            Additional keywords passed to `ss2zpk`

        Returns
        -------
        sys : instance of `ZerosPolesGain`
            Zeros, poles, gain representation of the current system

        """
        return ZerosPolesGain(*ss2zpk(self._A, self._B, self._C, self._D,
                                      **kwargs), **self._dt_dict)

    def to_ss(self):
        """
        Return a copy of the current `StateSpace` system.

        Returns
        -------
        sys : instance of `StateSpace`
            The current system (copy)

        """
        return copy.deepcopy(self)


class StateSpaceContinuous(StateSpace, lti):
    r"""
    Continuous-time Linear Time Invariant system in state-space form.

    Represents the system as the continuous-time, first order differential
    equation :math:`\dot{x} = A x + B u`.
    Continuous-time `StateSpace` systems inherit additional functionality
    from the `lti` class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `lti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)

    See Also
    --------
    TransferFunction, ZerosPolesGain, lti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal

    >>> a = np.array([[0, 1], [0, 0]])
    >>> b = np.array([[0], [1]])
    >>> c = np.array([[1, 0]])
    >>> d = np.array([[0]])

    >>> sys = signal.StateSpace(a, b, c, d)
    >>> print(sys)
    StateSpaceContinuous(
    array([[0, 1],
           [0, 0]]),
    array([[0],
           [1]]),
    array([[1, 0]]),
    array([[0]]),
    dt: None
    )

    """

    def to_discrete(self, dt, method='zoh', alpha=None):
        """
        Returns the discretized `StateSpace` system.

        Parameters: See `cont2discrete` for details.

        Returns
        -------
        sys: instance of `dlti` and `StateSpace`
        """
        return StateSpace(*cont2discrete((self.A, self.B, self.C, self.D),
                                         dt,
                                         method=method,
                                         alpha=alpha)[:-1],
                          dt=dt)


class StateSpaceDiscrete(StateSpace, dlti):
    r"""
    Discrete-time Linear Time Invariant system in state-space form.

    Represents the system as the discrete-time difference equation
    :math:`x[k+1] = A x[k] + B u[k]`.
    `StateSpace` systems inherit additional functionality from the `dlti`
    class.

    Parameters
    ----------
    *system: arguments
        The `StateSpace` class can be instantiated with 1 or 3 arguments.
        The following gives the number of input arguments and their
        interpretation:

            * 1: `dlti` system: (`StateSpace`, `TransferFunction` or
              `ZerosPolesGain`)
            * 4: array_like: (A, B, C, D)
    dt: float, optional
        Sampling time [s] of the discrete-time systems. Defaults to `True`
        (unspecified sampling time). Must be specified as a keyword argument,
        for example, ``dt=0.1``.

    See Also
    --------
    TransferFunction, ZerosPolesGain, dlti
    ss2zpk, ss2tf, zpk2sos

    Notes
    -----
    Changing the value of properties that are not part of the
    `StateSpace` system representation (such as `zeros` or `poles`) is very
    inefficient and may lead to numerical inaccuracies.  It is better to
    convert to the specific system representation first. For example, call
    ``sys = sys.to_zpk()`` before accessing/changing the zeros, poles or gain.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal

    >>> a = np.array([[1, 0.1], [0, 1]])
    >>> b = np.array([[0.005], [0.1]])
    >>> c = np.array([[1, 0]])
    >>> d = np.array([[0]])

    >>> signal.StateSpace(a, b, c, d, dt=0.1)
    StateSpaceDiscrete(
    array([[ 1. ,  0.1],
           [ 0. ,  1. ]]),
    array([[ 0.005],
           [ 0.1  ]]),
    array([[1, 0]]),
    array([[0]]),
    dt: 0.1
    )

    """
    pass


def lsim(system, U, T, X0=None, interp=True):
    """
    Simulate output of a continuous-time linear system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 1: (instance of `lti`)
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)

    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U = 0 or None, a zero input is used.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.  Must be nonnegative, increasing, and equally spaced.
    X0 : array_like, optional
        The initial conditions on the state vector (zero by default).
    interp : bool, optional
        Whether to use linear (True, the default) or zero-order-hold (False)
        interpolation for the input array.

    Returns
    -------
    T : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time evolution of the state vector.

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    We'll use `lsim` to simulate an analog Bessel filter applied to
    a signal.

    >>> import numpy as np
    >>> from scipy.signal import bessel, lsim
    >>> import matplotlib.pyplot as plt

    Create a low-pass Bessel filter with a cutoff of 12 Hz.

    >>> b, a = bessel(N=5, Wn=2*np.pi*12, btype='lowpass', analog=True)

    Generate data to which the filter is applied.

    >>> t = np.linspace(0, 1.25, 500, endpoint=False)

    The input signal is the sum of three sinusoidal curves, with
    frequencies 4 Hz, 40 Hz, and 80 Hz.  The filter should mostly
    eliminate the 40 Hz and 80 Hz components, leaving just the 4 Hz signal.

    >>> u = (np.cos(2*np.pi*4*t) + 0.6*np.sin(2*np.pi*40*t) +
    ...      0.5*np.cos(2*np.pi*80*t))

    Simulate the filter with `lsim`.

    >>> tout, yout, xout = lsim((b, a), U=u, T=t)

    Plot the result.

    >>> plt.plot(t, u, 'r', alpha=0.5, linewidth=1, label='input')
    >>> plt.plot(tout, yout, 'k', linewidth=1.5, label='output')
    >>> plt.legend(loc='best', shadow=True, framealpha=1)
    >>> plt.grid(alpha=0.3)
    >>> plt.xlabel('t')
    >>> plt.show()

    In a second example, we simulate a double integrator ``y'' = u``, with
    a constant input ``u = 1``.  We'll use the state space representation
    of the integrator.

    >>> from scipy.signal import lti
    >>> A = np.array([[0.0, 1.0], [0.0, 0.0]])
    >>> B = np.array([[0.0], [1.0]])
    >>> C = np.array([[1.0, 0.0]])
    >>> D = 0.0
    >>> system = lti(A, B, C, D)

    `t` and `u` define the time and input signal for the system to
    be simulated.

    >>> t = np.linspace(0, 5, num=50)
    >>> u = np.ones_like(t)

    Compute the simulation, and then plot `y`.  As expected, the plot shows
    the curve ``y = 0.5*t**2``.

    >>> tout, y, x = lsim(system, u, t)
    >>> plt.plot(t, y)
    >>> plt.grid(alpha=0.3)
    >>> plt.xlabel('t')
    >>> plt.show()

    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_ss()
    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    A, B, C, D = map(np.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    n_steps = T.size
    if X0 is None:
        X0 = zeros(n_states, sys.A.dtype)
    xout = np.empty((n_steps, n_states), sys.A.dtype)

    if T[0] == 0:
        xout[0] = X0
    elif T[0] > 0:
        # step forward to initial time, with zero input
        xout[0] = dot(X0, linalg.expm(transpose(A) * T[0]))
    else:
        raise ValueError("Initial time must be nonnegative")

    no_input = (U is None or
                (isinstance(U, (int, float)) and U == 0.) or
                not np.any(U))

    if n_steps == 1:
        yout = squeeze(xout @ C.T)
        if not no_input:
            yout += squeeze(U @ D.T)
        return T, yout, squeeze(xout)

    dt = T[1] - T[0]
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Time steps are not equally spaced.")

    if no_input:
        # Zero input: just use matrix exponential
        # take transpose because state is a row vector
        expAT_dt = linalg.expm(A.T * dt)
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ expAT_dt
        yout = squeeze(xout @ C.T)
        return T, yout, squeeze(xout)

    # Nonzero input
    U = atleast_1d(U)
    if U.ndim == 1:
        U = U[:, np.newaxis]

    if U.shape[0] != n_steps:
        raise ValueError("U must have the same number of rows "
                         "as elements in T.")

    if U.shape[1] != n_inputs:
        raise ValueError("System does not define that many inputs.")

    if not interp:
        # Zero-order hold
        # Algorithm: to integrate from time 0 to time dt, we solve
        #   xdot = A x + B u,  x(0) = x0
        #   udot = 0,          u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
        #   [ u(dt) ] = exp [  0     0    ] [ u0 ]
        M = np.vstack([np.hstack([A * dt, B * dt]),
                       np.zeros((n_inputs, n_states + n_inputs))])
        # transpose everything because the state and input are row vectors
        expMT = linalg.expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ Ad + U[i-1] @ Bd
    else:
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time dt, with linear
        # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / dt,   u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        M = np.vstack([np.hstack([A * dt, B * dt,
                                  np.zeros((n_states, n_inputs))]),
                       np.hstack([np.zeros((n_inputs, n_states + n_inputs)),
                                  np.identity(n_inputs)]),
                       np.zeros((n_inputs, n_states + 2 * n_inputs))])
        expMT = linalg.expm(M.T)
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states+n_inputs:, :n_states]
        Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in range(1, n_steps):
            xout[i] = xout[i-1] @ Ad + U[i-1] @ Bd0 + U[i] @ Bd1

    yout = squeeze(xout @ C.T) + squeeze(U @ D.T)
    return T, yout, squeeze(xout)


def _default_response_times(A, n):
    """Compute a reasonable set of time samples for the response time.

    This function is used by `impulse` and `step`  to compute the response time
    when the `T` argument to the function is None.

    Parameters
    ----------
    A : array_like
        The system matrix, which is square.
    n : int
        The number of time samples to generate.

    Returns
    -------
    t : ndarray
        The 1-D array of length `n` of time samples at which the response
        is to be computed.
    """
    # Create a reasonable time interval.
    # TODO: This could use some more work.
    # For example, what is expected when the system is unstable?
    vals = linalg.eigvals(A)
    r = min(abs(real(vals)))
    if r == 0.0:
        r = 1.0
    tc = 1.0 / r
    t = linspace(0.0, 7 * tc, n)
    return t


def impulse(system, X0=None, T=None, N=None):
    """Impulse response of continuous-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector.  Defaults to zero.
    T : array_like, optional
        Time points.  Computed if not given.
    N : int, optional
        The number of time points to compute (if `T` is not given).

    Returns
    -------
    T : ndarray
        A 1-D array of time points.
    yout : ndarray
        A 1-D array containing the impulse response of the system (except for
        singularities at zero).

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    Compute the impulse response of a second order system with a repeated
    root: ``x''(t) + 2*x'(t) + x(t) = u(t)``

    >>> from scipy import signal
    >>> system = ([1.0], [1.0, 2.0, 1.0])
    >>> t, y = signal.impulse(system)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y)

    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('impulse can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_ss()
    if X0 is None:
        X = squeeze(sys.B)
    else:
        X = squeeze(sys.B + X0)
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = asarray(T)

    _, h, _ = lsim(sys, 0., T, X, interp=False)
    return T, h


def step(system, X0=None, T=None, N=None):
    """Step response of continuous-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector (default is zero).
    T : array_like, optional
        Time points (computed if not given).
    N : int, optional
        Number of time points to compute if `T` is not given.

    Returns
    -------
    T : 1D ndarray
        Output time points.
    yout : 1D ndarray
        Step response of system.


    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> lti = signal.lti([1.0], [1.0, 1.0])
    >>> t, y = signal.step(lti)
    >>> plt.plot(t, y)
    >>> plt.xlabel('Time [s]')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Step response for 1. Order Lowpass')
    >>> plt.grid()

    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('step can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_ss()
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = asarray(T)
    U = ones(T.shape, sys.A.dtype)
    vals = lsim(sys, U, T, X0=X0, interp=False)
    return vals[0], vals[1]


def bode(system, w=None, n=100):
    """
    Calculate Bode magnitude and phase data of a continuous-time system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    w : array_like, optional
        Array of frequencies (in rad/s). Magnitude and phase data is calculated
        for every value in this array. If not given a reasonable set will be
        calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.

    Returns
    -------
    w : 1D ndarray
        Frequency array [rad/s]
    mag : 1D ndarray
        Magnitude array [dB]
    phase : 1D ndarray
        Phase array [deg]

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> sys = signal.TransferFunction([1], [1, 1])
    >>> w, mag, phase = signal.bode(sys)

    >>> plt.figure()
    >>> plt.semilogx(w, mag)    # Bode magnitude plot
    >>> plt.figure()
    >>> plt.semilogx(w, phase)  # Bode phase plot
    >>> plt.show()

    """
    w, y = freqresp(system, w=w, n=n)

    mag = 20.0 * np.log10(abs(y))
    phase = np.unwrap(np.arctan2(y.imag, y.real)) * 180.0 / np.pi

    return w, mag, phase


def freqresp(system, w=None, n=10000):
    r"""Calculate the frequency response of a continuous-time system.

    Parameters
    ----------
    system : an instance of the `lti` class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)

    w : array_like, optional
        Array of frequencies (in rad/s). Magnitude and phase data is
        calculated for every value in this array. If not given, a reasonable
        set will be calculated.
    n : int, optional
        Number of frequency points to compute if `w` is not given. The `n`
        frequencies are logarithmically spaced in an interval chosen to
        include the influence of the poles and zeros of the system.

    Returns
    -------
    w : 1D ndarray
        Frequency array [rad/s]
    H : 1D ndarray
        Array of complex magnitude values

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    Generating the Nyquist plot of a transfer function

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Construct the transfer function :math:`H(s) = \frac{5}{(s-1)^3}`:

    >>> s1 = signal.ZerosPolesGain([], [1, 1, 1], [5])

    >>> w, H = signal.freqresp(s1)

    >>> plt.figure()
    >>> plt.plot(H.real, H.imag, "b")
    >>> plt.plot(H.real, -H.imag, "r")
    >>> plt.show()
    """
    if isinstance(system, lti):
        if isinstance(system, (TransferFunction, ZerosPolesGain)):
            sys = system
        else:
            sys = system._as_zpk()
    elif isinstance(system, dlti):
        raise AttributeError('freqresp can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_zpk()

    if sys.inputs != 1 or sys.outputs != 1:
        raise ValueError("freqresp() requires a SISO (single input, single "
                         "output) system.")

    if w is not None:
        worN = w
    else:
        worN = n

    if isinstance(sys, TransferFunction):
        # In the call to freqs(), sys.num.ravel() is used because there are
        # cases where sys.num is a 2-D array with a single row.
        w, h = freqs(sys.num.ravel(), sys.den, worN=worN)

    elif isinstance(sys, ZerosPolesGain):
        w, h = freqs_zpk(sys.zeros, sys.poles, sys.gain, worN=worN)

    return w, h


# This class will be used by place_poles to return its results
# see https://code.activestate.com/recipes/52308/
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def _valid_inputs(A, B, poles, method, rtol, maxiter):
    """
    Check the poles come in complex conjugate pairs
    Check shapes of A, B and poles are compatible.
    Check the method chosen is compatible with provided poles
    Return update method to use and ordered poles

    """
    poles = np.asarray(poles)
    if poles.ndim > 1:
        raise ValueError("Poles must be a 1D array like.")
    # Will raise ValueError if poles do not come in complex conjugates pairs
    poles = _order_complex_poles(poles)
    if A.ndim > 2:
        raise ValueError("A must be a 2D array/matrix.")
    if B.ndim > 2:
        raise ValueError("B must be a 2D array/matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if len(poles) > A.shape[0]:
        raise ValueError("maximum number of poles is %d but you asked for %d" %
                         (A.shape[0], len(poles)))
    if len(poles) < A.shape[0]:
        raise ValueError("number of poles is %d but you should provide %d" %
                         (len(poles), A.shape[0]))
    r = np.linalg.matrix_rank(B)
    for p in poles:
        if sum(p == poles) > r:
            raise ValueError("at least one of the requested pole is repeated "
                             "more than rank(B) times")
    # Choose update method
    update_loop = _YT_loop
    if method not in ('KNV0','YT'):
        raise ValueError("The method keyword must be one of 'YT' or 'KNV0'")

    if method == "KNV0":
        update_loop = _KNV0_loop
        if not all(np.isreal(poles)):
            raise ValueError("Complex poles are not supported by KNV0")

    if maxiter < 1:
        raise ValueError("maxiter must be at least equal to 1")

    # We do not check rtol <= 0 as the user can use a negative rtol to
    # force maxiter iterations
    if rtol > 1:
        raise ValueError("rtol can not be greater than 1")

    return update_loop, poles


def _order_complex_poles(poles):
    """
    Check we have complex conjugates pairs and reorder P according to YT, ie
    real_poles, complex_i, conjugate complex_i, ....
    The lexicographic sort on the complex poles is added to help the user to
    compare sets of poles.
    """
    ordered_poles = np.sort(poles[np.isreal(poles)])
    im_poles = []
    for p in np.sort(poles[np.imag(poles) < 0]):
        if np.conj(p) in poles:
            im_poles.extend((p, np.conj(p)))

    ordered_poles = np.hstack((ordered_poles, im_poles))

    if poles.shape[0] != len(ordered_poles):
        raise ValueError("Complex poles must come with their conjugates")
    return ordered_poles


def _KNV0(B, ker_pole, transfer_matrix, j, poles):
    """
    Algorithm "KNV0" Kautsky et Al. Robust pole
    assignment in linear state feedback, Int journal of Control
    1985, vol 41 p 1129->1155
    https://la.epfl.ch/files/content/sites/la/files/
        users/105941/public/KautskyNicholsDooren

    """
    # Remove xj form the base
    transfer_matrix_not_j = np.delete(transfer_matrix, j, axis=1)
    # If we QR this matrix in full mode Q=Q0|Q1
    # then Q1 will be a single column orthogonal to
    # Q0, that's what we are looking for !

    # After merge of gh-4249 great speed improvements could be achieved
    # using QR updates instead of full QR in the line below

    # To debug with numpy qr uncomment the line below
    # Q, R = np.linalg.qr(transfer_matrix_not_j, mode="complete")
    Q, R = s_qr(transfer_matrix_not_j, mode="full")

    mat_ker_pj = np.dot(ker_pole[j], ker_pole[j].T)
    yj = np.dot(mat_ker_pj, Q[:, -1])

    # If Q[:, -1] is "almost" orthogonal to ker_pole[j] its
    # projection into ker_pole[j] will yield a vector
    # close to 0.  As we are looking for a vector in ker_pole[j]
    # simply stick with transfer_matrix[:, j] (unless someone provides me with
    # a better choice ?)

    if not np.allclose(yj, 0):
        xj = yj/np.linalg.norm(yj)
        transfer_matrix[:, j] = xj

        # KNV does not support complex poles, using YT technique the two lines
        # below seem to work 9 out of 10 times but it is not reliable enough:
        # transfer_matrix[:, j]=real(xj)
        # transfer_matrix[:, j+1]=imag(xj)

        # Add this at the beginning of this function if you wish to test
        # complex support:
        #    if ~np.isreal(P[j]) and (j>=B.shape[0]-1 or P[j]!=np.conj(P[j+1])):
        #        return
        # Problems arise when imag(xj)=>0 I have no idea on how to fix this


def _YT_real(ker_pole, Q, transfer_matrix, i, j):
    """
    Applies algorithm from YT section 6.1 page 19 related to real pairs
    """
    # step 1 page 19
    u = Q[:, -2, np.newaxis]
    v = Q[:, -1, np.newaxis]

    # step 2 page 19
    m = np.dot(np.dot(ker_pole[i].T, np.dot(u, v.T) -
        np.dot(v, u.T)), ker_pole[j])

    # step 3 page 19
    um, sm, vm = np.linalg.svd(m)
    # mu1, mu2 two first columns of U => 2 first lines of U.T
    mu1, mu2 = um.T[:2, :, np.newaxis]
    # VM is V.T with numpy we want the first two lines of V.T
    nu1, nu2 = vm[:2, :, np.newaxis]

    # what follows is a rough python translation of the formulas
    # in section 6.2 page 20 (step 4)
    transfer_matrix_j_mo_transfer_matrix_j = np.vstack((
            transfer_matrix[:, i, np.newaxis],
            transfer_matrix[:, j, np.newaxis]))

    if not np.allclose(sm[0], sm[1]):
        ker_pole_imo_mu1 = np.dot(ker_pole[i], mu1)
        ker_pole_i_nu1 = np.dot(ker_pole[j], nu1)
        ker_pole_mu_nu = np.vstack((ker_pole_imo_mu1, ker_pole_i_nu1))
    else:
        ker_pole_ij = np.vstack((
                                np.hstack((ker_pole[i],
                                           np.zeros(ker_pole[i].shape))),
                                np.hstack((np.zeros(ker_pole[j].shape),
                                                    ker_pole[j]))
                                ))
        mu_nu_matrix = np.vstack(
            (np.hstack((mu1, mu2)), np.hstack((nu1, nu2)))
            )
        ker_pole_mu_nu = np.dot(ker_pole_ij, mu_nu_matrix)
    transfer_matrix_ij = np.dot(np.dot(ker_pole_mu_nu, ker_pole_mu_nu.T),
                             transfer_matrix_j_mo_transfer_matrix_j)
    if not np.allclose(transfer_matrix_ij, 0):
        transfer_matrix_ij = (np.sqrt(2)*transfer_matrix_ij /
                              np.linalg.norm(transfer_matrix_ij))
        transfer_matrix[:, i] = transfer_matrix_ij[
            :transfer_matrix[:, i].shape[0], 0
            ]
        transfer_matrix[:, j] = transfer_matrix_ij[
            transfer_matrix[:, i].shape[0]:, 0
            ]
    else:
        # As in knv0 if transfer_matrix_j_mo_transfer_matrix_j is orthogonal to
        # Vect{ker_pole_mu_nu} assign transfer_matrixi/transfer_matrix_j to
        # ker_pole_mu_nu and iterate. As we are looking for a vector in
        # Vect{Matker_pole_MU_NU} (see section 6.1 page 19) this might help
        # (that's a guess, not a claim !)
        transfer_matrix[:, i] = ker_pole_mu_nu[
            :transfer_matrix[:, i].shape[0], 0
            ]
        transfer_matrix[:, j] = ker_pole_mu_nu[
            transfer_matrix[:, i].shape[0]:, 0
            ]


def _YT_complex(ker_pole, Q, transfer_matrix, i, j):
    """
    Applies algorithm from YT section 6.2 page 20 related to complex pairs
    """
    # step 1 page 20
    ur = np.sqrt(2)*Q[:, -2, np.newaxis]
    ui = np.sqrt(2)*Q[:, -1, np.newaxis]
    u = ur + 1j*ui

    # step 2 page 20
    ker_pole_ij = ker_pole[i]
    m = np.dot(np.dot(np.conj(ker_pole_ij.T), np.dot(u, np.conj(u).T) -
               np.dot(np.conj(u), u.T)), ker_pole_ij)

    # step 3 page 20
    e_val, e_vec = np.linalg.eig(m)
    # sort eigenvalues according to their module
    e_val_idx = np.argsort(np.abs(e_val))
    mu1 = e_vec[:, e_val_idx[-1], np.newaxis]
    mu2 = e_vec[:, e_val_idx[-2], np.newaxis]

    # what follows is a rough python translation of the formulas
    # in section 6.2 page 20 (step 4)

    # remember transfer_matrix_i has been split as
    # transfer_matrix[i]=real(transfer_matrix_i) and
    # transfer_matrix[j]=imag(transfer_matrix_i)
    transfer_matrix_j_mo_transfer_matrix_j = (
        transfer_matrix[:, i, np.newaxis] +
        1j*transfer_matrix[:, j, np.newaxis]
        )
    if not np.allclose(np.abs(e_val[e_val_idx[-1]]),
                              np.abs(e_val[e_val_idx[-2]])):
        ker_pole_mu = np.dot(ker_pole_ij, mu1)
    else:
        mu1_mu2_matrix = np.hstack((mu1, mu2))
        ker_pole_mu = np.dot(ker_pole_ij, mu1_mu2_matrix)
    transfer_matrix_i_j = np.dot(np.dot(ker_pole_mu, np.conj(ker_pole_mu.T)),
                              transfer_matrix_j_mo_transfer_matrix_j)

    if not np.allclose(transfer_matrix_i_j, 0):
        transfer_matrix_i_j = (transfer_matrix_i_j /
            np.linalg.norm(transfer_matrix_i_j))
        transfer_matrix[:, i] = np.real(transfer_matrix_i_j[:, 0])
        transfer_matrix[:, j] = np.imag(transfer_matrix_i_j[:, 0])
    else:
        # same idea as in YT_real
        transfer_matrix[:, i] = np.real(ker_pole_mu[:, 0])
        transfer_matrix[:, j] = np.imag(ker_pole_mu[:, 0])


def _YT_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol):
    """
    Algorithm "YT" Tits, Yang. Globally Convergent
    Algorithms for Robust Pole Assignment by State Feedback
    https://hdl.handle.net/1903/5598
    The poles P have to be sorted accordingly to section 6.2 page 20

    """
    # The IEEE edition of the YT paper gives useful information on the
    # optimal update order for the real poles in order to minimize the number
    # of times we have to loop over all poles, see page 1442
    nb_real = poles[np.isreal(poles)].shape[0]
    # hnb => Half Nb Real
    hnb = nb_real // 2

    # Stick to the indices in the paper and then remove one to get numpy array
    # index it is a bit easier to link the code to the paper this way even if it
    # is not very clean. The paper is unclear about what should be done when
    # there is only one real pole => use KNV0 on this real pole seem to work
    if nb_real > 0:
        #update the biggest real pole with the smallest one
        update_order = [[nb_real], [1]]
    else:
        update_order = [[],[]]

    r_comp = np.arange(nb_real+1, len(poles)+1, 2)
    # step 1.a
    r_p = np.arange(1, hnb+nb_real % 2)
    update_order[0].extend(2*r_p)
    update_order[1].extend(2*r_p+1)
    # step 1.b
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 1.c
    r_p = np.arange(1, hnb+1)
    update_order[0].extend(2*r_p-1)
    update_order[1].extend(2*r_p)
    # step 1.d
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 2.a
    r_j = np.arange(2, hnb+nb_real % 2)
    for j in r_j:
        for i in range(1, hnb+1):
            update_order[0].append(i)
            update_order[1].append(i+j)
    # step 2.b
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 2.c
    r_j = np.arange(2, hnb+nb_real % 2)
    for j in r_j:
        for i in range(hnb+1, nb_real+1):
            idx_1 = i+j
            if idx_1 > nb_real:
                idx_1 = i+j-nb_real
            update_order[0].append(i)
            update_order[1].append(idx_1)
    # step 2.d
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)
    # step 3.a
    for i in range(1, hnb+1):
        update_order[0].append(i)
        update_order[1].append(i+hnb)
    # step 3.b
    if hnb == 0 and np.isreal(poles[0]):
        update_order[0].append(1)
        update_order[1].append(1)
    update_order[0].extend(r_comp)
    update_order[1].extend(r_comp+1)

    update_order = np.array(update_order).T-1
    stop = False
    nb_try = 0
    while nb_try < maxiter and not stop:
        det_transfer_matrixb = np.abs(np.linalg.det(transfer_matrix))
        for i, j in update_order:
            if i == j:
                assert i == 0, "i!=0 for KNV call in YT"
                assert np.isreal(poles[i]), "calling KNV on a complex pole"
                _KNV0(B, ker_pole, transfer_matrix, i, poles)
            else:
                transfer_matrix_not_i_j = np.delete(transfer_matrix, (i, j),
                                                    axis=1)
                # after merge of gh-4249 great speed improvements could be
                # achieved using QR updates instead of full QR in the line below

                #to debug with numpy qr uncomment the line below
                #Q, _ = np.linalg.qr(transfer_matrix_not_i_j, mode="complete")
                Q, _ = s_qr(transfer_matrix_not_i_j, mode="full")

                if np.isreal(poles[i]):
                    assert np.isreal(poles[j]), "mixing real and complex " + \
                        "in YT_real" + str(poles)
                    _YT_real(ker_pole, Q, transfer_matrix, i, j)
                else:
                    assert ~np.isreal(poles[i]), "mixing real and complex " + \
                        "in YT_real" + str(poles)
                    _YT_complex(ker_pole, Q, transfer_matrix, i, j)

        det_transfer_matrix = np.max((np.sqrt(np.spacing(1)),
                                  np.abs(np.linalg.det(transfer_matrix))))
        cur_rtol = np.abs(
            (det_transfer_matrix -
             det_transfer_matrixb) /
            det_transfer_matrix)
        if cur_rtol < rtol and det_transfer_matrix > np.sqrt(np.spacing(1)):
            # Convergence test from YT page 21
            stop = True
        nb_try += 1
    return stop, cur_rtol, nb_try


def _KNV0_loop(ker_pole, transfer_matrix, poles, B, maxiter, rtol):
    """
    Loop over all poles one by one and apply KNV method 0 algorithm
    """
    # This method is useful only because we need to be able to call
    # _KNV0 from YT without looping over all poles, otherwise it would
    # have been fine to mix _KNV0_loop and _KNV0 in a single function
    stop = False
    nb_try = 0
    while nb_try < maxiter and not stop:
        det_transfer_matrixb = np.abs(np.linalg.det(transfer_matrix))
        for j in range(B.shape[0]):
            _KNV0(B, ker_pole, transfer_matrix, j, poles)

        det_transfer_matrix = np.max((np.sqrt(np.spacing(1)),
                                  np.abs(np.linalg.det(transfer_matrix))))
        cur_rtol = np.abs((det_transfer_matrix - det_transfer_matrixb) /
                       det_transfer_matrix)
        if cur_rtol < rtol and det_transfer_matrix > np.sqrt(np.spacing(1)):
            # Convergence test from YT page 21
            stop = True

        nb_try += 1
    return stop, cur_rtol, nb_try


def place_poles(A, B, poles, method="YT", rtol=1e-3, maxiter=30):
    """
    Compute K such that eigenvalues (A - dot(B, K))=poles.

    K is the gain matrix such as the plant described by the linear system
    ``AX+BU`` will have its closed-loop poles, i.e the eigenvalues ``A - B*K``,
    as close as possible to those asked for in poles.

    SISO, MISO and MIMO systems are supported.

    Parameters
    ----------
    A, B : ndarray
        State-space representation of linear system ``AX + BU``.
    poles : array_like
        Desired real poles and/or complex conjugates poles.
        Complex poles are only supported with ``method="YT"`` (default).
    method: {'YT', 'KNV0'}, optional
        Which method to choose to find the gain matrix K. One of:

            - 'YT': Yang Tits
            - 'KNV0': Kautsky, Nichols, Van Dooren update method 0

        See References and Notes for details on the algorithms.
    rtol: float, optional
        After each iteration the determinant of the eigenvectors of
        ``A - B*K`` is compared to its previous value, when the relative
        error between these two values becomes lower than `rtol` the algorithm
        stops.  Default is 1e-3.
    maxiter: int, optional
        Maximum number of iterations to compute the gain matrix.
        Default is 30.

    Returns
    -------
    full_state_feedback : Bunch object
        full_state_feedback is composed of:
            gain_matrix : 1-D ndarray
                The closed loop matrix K such as the eigenvalues of ``A-BK``
                are as close as possible to the requested poles.
            computed_poles : 1-D ndarray
                The poles corresponding to ``A-BK`` sorted as first the real
                poles in increasing order, then the complex conjugates in
                lexicographic order.
            requested_poles : 1-D ndarray
                The poles the algorithm was asked to place sorted as above,
                they may differ from what was achieved.
            X : 2-D ndarray
                The transfer matrix such as ``X * diag(poles) = (A - B*K)*X``
                (see Notes)
            rtol : float
                The relative tolerance achieved on ``det(X)`` (see Notes).
                `rtol` will be NaN if it is possible to solve the system
                ``diag(poles) = (A - B*K)``, or 0 when the optimization
                algorithms can't do anything i.e when ``B.shape[1] == 1``.
            nb_iter : int
                The number of iterations performed before converging.
                `nb_iter` will be NaN if it is possible to solve the system
                ``diag(poles) = (A - B*K)``, or 0 when the optimization
                algorithms can't do anything i.e when ``B.shape[1] == 1``.

    Notes
    -----
    The Tits and Yang (YT), [2]_ paper is an update of the original Kautsky et
    al. (KNV) paper [1]_.  KNV relies on rank-1 updates to find the transfer
    matrix X such that ``X * diag(poles) = (A - B*K)*X``, whereas YT uses
    rank-2 updates. This yields on average more robust solutions (see [2]_
    pp 21-22), furthermore the YT algorithm supports complex poles whereas KNV
    does not in its original version.  Only update method 0 proposed by KNV has
    been implemented here, hence the name ``'KNV0'``.

    KNV extended to complex poles is used in Matlab's ``place`` function, YT is
    distributed under a non-free licence by Slicot under the name ``robpole``.
    It is unclear and undocumented how KNV0 has been extended to complex poles
    (Tits and Yang claim on page 14 of their paper that their method can not be
    used to extend KNV to complex poles), therefore only YT supports them in
    this implementation.

    As the solution to the problem of pole placement is not unique for MIMO
    systems, both methods start with a tentative transfer matrix which is
    altered in various way to increase its determinant.  Both methods have been
    proven to converge to a stable solution, however depending on the way the
    initial transfer matrix is chosen they will converge to different
    solutions and therefore there is absolutely no guarantee that using
    ``'KNV0'`` will yield results similar to Matlab's or any other
    implementation of these algorithms.

    Using the default method ``'YT'`` should be fine in most cases; ``'KNV0'``
    is only provided because it is needed by ``'YT'`` in some specific cases.
    Furthermore ``'YT'`` gives on average more robust results than ``'KNV0'``
    when ``abs(det(X))`` is used as a robustness indicator.

    [2]_ is available as a technical report on the following URL:
    https://hdl.handle.net/1903/5598

    References
    ----------
    .. [1] J. Kautsky, N.K. Nichols and P. van Dooren, "Robust pole assignment
           in linear state feedback", International Journal of Control, Vol. 41
           pp. 1129-1155, 1985.
    .. [2] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
           pole assignment by state feedback", IEEE Transactions on Automatic
           Control, Vol. 41, pp. 1432-1452, 1996.

    Examples
    --------
    A simple example demonstrating real pole placement using both KNV and YT
    algorithms.  This is example number 1 from section 4 of the reference KNV
    publication ([1]_):

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> A = np.array([[ 1.380,  -0.2077,  6.715, -5.676  ],
    ...               [-0.5814, -4.290,   0,      0.6750 ],
    ...               [ 1.067,   4.273,  -6.654,  5.893  ],
    ...               [ 0.0480,  4.273,   1.343, -2.104  ]])
    >>> B = np.array([[ 0,      5.679 ],
    ...               [ 1.136,  1.136 ],
    ...               [ 0,      0,    ],
    ...               [-3.146,  0     ]])
    >>> P = np.array([-0.2, -0.5, -5.0566, -8.6659])

    Now compute K with KNV method 0, with the default YT method and with the YT
    method while forcing 100 iterations of the algorithm and print some results
    after each call.

    >>> fsf1 = signal.place_poles(A, B, P, method='KNV0')
    >>> fsf1.gain_matrix
    array([[ 0.20071427, -0.96665799,  0.24066128, -0.10279785],
           [ 0.50587268,  0.57779091,  0.51795763, -0.41991442]])

    >>> fsf2 = signal.place_poles(A, B, P)  # uses YT method
    >>> fsf2.computed_poles
    array([-8.6659, -5.0566, -0.5   , -0.2   ])

    >>> fsf3 = signal.place_poles(A, B, P, rtol=-1, maxiter=100)
    >>> fsf3.X
    array([[ 0.52072442+0.j, -0.08409372+0.j, -0.56847937+0.j,  0.74823657+0.j],
           [-0.04977751+0.j, -0.80872954+0.j,  0.13566234+0.j, -0.29322906+0.j],
           [-0.82266932+0.j, -0.19168026+0.j, -0.56348322+0.j, -0.43815060+0.j],
           [ 0.22267347+0.j,  0.54967577+0.j, -0.58387806+0.j, -0.40271926+0.j]])

    The absolute value of the determinant of X is a good indicator to check the
    robustness of the results, both ``'KNV0'`` and ``'YT'`` aim at maximizing
    it.  Below a comparison of the robustness of the results above:

    >>> abs(np.linalg.det(fsf1.X)) < abs(np.linalg.det(fsf2.X))
    True
    >>> abs(np.linalg.det(fsf2.X)) < abs(np.linalg.det(fsf3.X))
    True

    Now a simple example for complex poles:

    >>> A = np.array([[ 0,  7/3.,  0,   0   ],
    ...               [ 0,   0,    0,  7/9. ],
    ...               [ 0,   0,    0,   0   ],
    ...               [ 0,   0,    0,   0   ]])
    >>> B = np.array([[ 0,  0 ],
    ...               [ 0,  0 ],
    ...               [ 1,  0 ],
    ...               [ 0,  1 ]])
    >>> P = np.array([-3, -1, -2-1j, -2+1j]) / 3.
    >>> fsf = signal.place_poles(A, B, P, method='YT')

    We can plot the desired and computed poles in the complex plane:

    >>> t = np.linspace(0, 2*np.pi, 401)
    >>> plt.plot(np.cos(t), np.sin(t), 'k--')  # unit circle
    >>> plt.plot(fsf.requested_poles.real, fsf.requested_poles.imag,
    ...          'wo', label='Desired')
    >>> plt.plot(fsf.computed_poles.real, fsf.computed_poles.imag, 'bx',
    ...          label='Placed')
    >>> plt.grid()
    >>> plt.axis('image')
    >>> plt.axis([-1.1, 1.1, -1.1, 1.1])
    >>> plt.legend(bbox_to_anchor=(1.05, 1), loc=2, numpoints=1)

    """
    # Move away all the inputs checking, it only adds noise to the code
    update_loop, poles = _valid_inputs(A, B, poles, method, rtol, maxiter)

    # The current value of the relative tolerance we achieved
    cur_rtol = 0
    # The number of iterations needed before converging
    nb_iter = 0

    # Step A: QR decomposition of B page 1132 KN
    # to debug with numpy qr uncomment the line below
    # u, z = np.linalg.qr(B, mode="complete")
    u, z = s_qr(B, mode="full")
    rankB = np.linalg.matrix_rank(B)
    u0 = u[:, :rankB]
    u1 = u[:, rankB:]
    z = z[:rankB, :]

    # If we can use the identity matrix as X the solution is obvious
    if B.shape[0] == rankB:
        # if B is square and full rank there is only one solution
        # such as (A+BK)=inv(X)*diag(P)*X with X=eye(A.shape[0])
        # i.e K=inv(B)*(diag(P)-A)
        # if B has as many lines as its rank (but not square) there are many
        # solutions and we can choose one using least squares
        # => use lstsq in both cases.
        # In both cases the transfer matrix X will be eye(A.shape[0]) and I
        # can hardly think of a better one so there is nothing to optimize
        #
        # for complex poles we use the following trick
        #
        # |a -b| has for eigenvalues a+b and a-b
        # |b a|
        #
        # |a+bi 0| has the obvious eigenvalues a+bi and a-bi
        # |0 a-bi|
        #
        # e.g solving the first one in R gives the solution
        # for the second one in C
        diag_poles = np.zeros(A.shape)
        idx = 0
        while idx < poles.shape[0]:
            p = poles[idx]
            diag_poles[idx, idx] = np.real(p)
            if ~np.isreal(p):
                diag_poles[idx, idx+1] = -np.imag(p)
                diag_poles[idx+1, idx+1] = np.real(p)
                diag_poles[idx+1, idx] = np.imag(p)
                idx += 1  # skip next one
            idx += 1
        gain_matrix = np.linalg.lstsq(B, diag_poles-A, rcond=-1)[0]
        transfer_matrix = np.eye(A.shape[0])
        cur_rtol = np.nan
        nb_iter = np.nan
    else:
        # step A (p1144 KNV) and beginning of step F: decompose
        # dot(U1.T, A-P[i]*I).T and build our set of transfer_matrix vectors
        # in the same loop
        ker_pole = []

        # flag to skip the conjugate of a complex pole
        skip_conjugate = False
        # select orthonormal base ker_pole for each Pole and vectors for
        # transfer_matrix
        for j in range(B.shape[0]):
            if skip_conjugate:
                skip_conjugate = False
                continue
            pole_space_j = np.dot(u1.T, A-poles[j]*np.eye(B.shape[0])).T

            # after QR Q=Q0|Q1
            # only Q0 is used to reconstruct  the qr'ed (dot Q, R) matrix.
            # Q1 is orthogonal to Q0 and will be multiplied by the zeros in
            # R when using mode "complete". In default mode Q1 and the zeros
            # in R are not computed

            # To debug with numpy qr uncomment the line below
            # Q, _ = np.linalg.qr(pole_space_j, mode="complete")
            Q, _ = s_qr(pole_space_j, mode="full")

            ker_pole_j = Q[:, pole_space_j.shape[1]:]

            # We want to select one vector in ker_pole_j to build the transfer
            # matrix, however qr returns sometimes vectors with zeros on the
            # same line for each pole and this yields very long convergence
            # times.
            # Or some other times a set of vectors, one with zero imaginary
            # part and one (or several) with imaginary parts. After trying
            # many ways to select the best possible one (eg ditch vectors
            # with zero imaginary part for complex poles) I ended up summing
            # all vectors in ker_pole_j, this solves 100% of the problems and
            # is a valid choice for transfer_matrix.
            # This way for complex poles we are sure to have a non zero
            # imaginary part that way, and the problem of lines full of zeros
            # in transfer_matrix is solved too as when a vector from
            # ker_pole_j has a zero the other one(s) when
            # ker_pole_j.shape[1]>1) for sure won't have a zero there.

            transfer_matrix_j = np.sum(ker_pole_j, axis=1)[:, np.newaxis]
            transfer_matrix_j = (transfer_matrix_j /
                                 np.linalg.norm(transfer_matrix_j))
            if ~np.isreal(poles[j]):  # complex pole
                transfer_matrix_j = np.hstack([np.real(transfer_matrix_j),
                                               np.imag(transfer_matrix_j)])
                ker_pole.extend([ker_pole_j, ker_pole_j])

                # Skip next pole as it is the conjugate
                skip_conjugate = True
            else:  # real pole, nothing to do
                ker_pole.append(ker_pole_j)

            if j == 0:
                transfer_matrix = transfer_matrix_j
            else:
                transfer_matrix = np.hstack((transfer_matrix, transfer_matrix_j))

        if rankB > 1:  # otherwise there is nothing we can optimize
            stop, cur_rtol, nb_iter = update_loop(ker_pole, transfer_matrix,
                                                  poles, B, maxiter, rtol)
            if not stop and rtol > 0:
                # if rtol<=0 the user has probably done that on purpose,
                # don't annoy them
                err_msg = (
                    "Convergence was not reached after maxiter iterations.\n"
                    f"You asked for a tolerance of {rtol}, we got {cur_rtol}."
                    )
                warnings.warn(err_msg, stacklevel=2)

        # reconstruct transfer_matrix to match complex conjugate pairs,
        # ie transfer_matrix_j/transfer_matrix_j+1 are
        # Re(Complex_pole), Im(Complex_pole) now and will be Re-Im/Re+Im after
        transfer_matrix = transfer_matrix.astype(complex)
        idx = 0
        while idx < poles.shape[0]-1:
            if ~np.isreal(poles[idx]):
                rel = transfer_matrix[:, idx].copy()
                img = transfer_matrix[:, idx+1]
                # rel will be an array referencing a column of transfer_matrix
                # if we don't copy() it will changer after the next line and
                # and the line after will not yield the correct value
                transfer_matrix[:, idx] = rel-1j*img
                transfer_matrix[:, idx+1] = rel+1j*img
                idx += 1  # skip next one
            idx += 1

        try:
            m = np.linalg.solve(transfer_matrix.T, np.dot(np.diag(poles),
                                                          transfer_matrix.T)).T
            gain_matrix = np.linalg.solve(z, np.dot(u0.T, m-A))
        except np.linalg.LinAlgError as e:
            raise ValueError("The poles you've chosen can't be placed. "
                             "Check the controllability matrix and try "
                             "another set of poles") from e

    # Beware: Kautsky solves A+BK but the usual form is A-BK
    gain_matrix = -gain_matrix
    # K still contains complex with ~=0j imaginary parts, get rid of them
    gain_matrix = np.real(gain_matrix)

    full_state_feedback = Bunch()
    full_state_feedback.gain_matrix = gain_matrix
    full_state_feedback.computed_poles = _order_complex_poles(
        np.linalg.eig(A - np.dot(B, gain_matrix))[0]
        )
    full_state_feedback.requested_poles = poles
    full_state_feedback.X = transfer_matrix
    full_state_feedback.rtol = cur_rtol
    full_state_feedback.nb_iter = nb_iter

    return full_state_feedback


def dlsim(system, u, t=None, x0=None):
    """
    Simulate output of a discrete-time linear system.

    Parameters
    ----------
    system : tuple of array_like or instance of `dlti`
        A tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1: (instance of `dlti`)
            * 3: (num, den, dt)
            * 4: (zeros, poles, gain, dt)
            * 5: (A, B, C, D, dt)

    u : array_like
        An input array describing the input at each time `t` (interpolation is
        assumed between given times).  If there are multiple inputs, then each
        column of the rank-2 array represents an input.
    t : array_like, optional
        The time steps at which the input is defined.  If `t` is given, it
        must be the same length as `u`, and the final value in `t` determines
        the number of steps returned in the output.
    x0 : array_like, optional
        The initial conditions on the state vector (zero by default).

    Returns
    -------
    tout : ndarray
        Time values for the output, as a 1-D array.
    yout : ndarray
        System response, as a 1-D array.
    xout : ndarray, optional
        Time-evolution of the state-vector.  Only generated if the input is a
        `StateSpace` system.

    See Also
    --------
    lsim, dstep, dimpulse, cont2discrete

    Examples
    --------
    A simple integrator transfer function with a discrete time step of 1.0
    could be implemented as:

    >>> import numpy as np
    >>> from scipy import signal
    >>> tf = ([1.0,], [1.0, -1.0], 1.0)
    >>> t_in = [0.0, 1.0, 2.0, 3.0]
    >>> u = np.asarray([0.0, 0.0, 1.0, 1.0])
    >>> t_out, y = signal.dlsim(tf, u, t=t_in)
    >>> y.T
    array([[ 0.,  0.,  0.,  1.]])

    """
    # Convert system to dlti-StateSpace
    if isinstance(system, lti):
        raise AttributeError('dlsim can only be used with discrete-time dlti '
                             'systems.')
    elif not isinstance(system, dlti):
        system = dlti(*system[:-1], dt=system[-1])

    # Condition needed to ensure output remains compatible
    is_ss_input = isinstance(system, StateSpace)
    system = system._as_ss()

    u = np.atleast_1d(u)

    if u.ndim == 1:
        u = np.atleast_2d(u).T

    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(np.floor(stoptime / system.dt)) + 1

    # Pre-build output arrays
    xout = np.zeros((out_samples, system.A.shape[0]))
    yout = np.zeros((out_samples, system.C.shape[0]))
    tout = np.linspace(0.0, stoptime, num=out_samples)

    # Check initial condition
    if x0 is None:
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        xout[0, :] = np.asarray(x0)

